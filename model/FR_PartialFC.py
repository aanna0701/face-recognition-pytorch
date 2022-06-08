import torchmetrics
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import importlib
import torch
from torch import nn
from utils.logger import print_log
import torch.optim as optim
import numpy as np
import time
from einops import rearrange
from utils.eval import performance
from easydict import EasyDict as edict
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from utils.amp import MaxClipGradScaler
from torch.nn.utils import clip_grad_norm_


def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")
        
        
class Model(nn.Module):
    global LOGGER

    def __init__(self, conf:str, logger:str=None, stage:str='stage'):
        super().__init__()
        # # turn off automatic optimization
        # self.automatic_optimization = False
        
        # ---------------------------------------
        # writers
        # ---------------------------------------
        
        self.conf = conf
        self.logger_ = logger
        self.epoch = 0
        self.val_msg = edict()
        self.idx_save_image = 0
        self.lr = conf.lr
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.security_level = self.conf.security_level
        self.max_level = self.conf.max_level
        self.min_level = self.conf.min_level
        
        for val_dataset_name in conf.val_dataset:
            self.val_msg[f'{val_dataset_name}'] = edict()
        
        # ---------------------------------------
        # model
        # ---------------------------------------
    
        # Encoder
        if 'ResNet' in conf.network:
            self.encoder = importlib.import_module(f"nets.resnet").Encoder(conf=conf)
            
        if conf.transfer_learning:
            print('Transferring Weight')
            pre_weight = torch.load(conf.pretrained_dir, map_location='cpu')
            self.encoder.load_state_dict(pre_weight['model_state_dict'], strict=True)
            del pre_weight
            print('Finished')
            
        # Classifier
        if stage=='train':
            # DDP Setting for Encoder
            self.encoder = self.encoder.to(conf.local_rank)
            self.encoder = DDP(self.encoder, broadcast_buffers=False, device_ids=[conf.local_rank])
            
            if conf.local_rank == 0:
                print(self.encoder)
            print_peak_memory("Max Memory Afer DDP", conf.local_rank)
            
            # Loading PartialFC loss
            self.loss = nn.ModuleList()
            for i in range(len(conf.train_dataset)):
                self.loss.append(importlib.import_module(f"nets.{conf.loss}").PartialFC(conf=conf, 
                                                                                        n_classes=conf.n_classes[i],
                                                                                        idx=i))
                
            # Initializing Optimizers and Schedulers
            self.opts, self.schs = self.configure_optimizers()
            
        else:
            weight = torch.load(str(Path(conf.weight_path)), map_location='cpu')
            self.encoder.load_state_dict(weight, strict=True)
            del weight
            
        # ---------------------------------------
        # Criterion
        # ---------------------------------------
        
        self.criterion = nn.CrossEntropyLoss()
        
        
        # ---------------------------------------
        # Metrics
        # ---------------------------------------
        
        self.train_acc = torchmetrics.Accuracy()
        
        # ---------------------------------------
        # Save Path
        # ---------------------------------------
        
        self.save_path = Path(logger).parent
        
        # ---------------------------------------
        # Mixed precision
        # ---------------------------------------
        
        self.grad_amp = MaxClipGradScaler(conf.b, 128 * conf.b) if conf.mixed_precision else None
        
        
        
    # --------------------------------------------
    # forward
    # --------------------------------------------
    
    def forward(self, x):
        feat = self.encoder(x) 
        return feat


    # --------------------------------------------
    # training
    # --------------------------------------------
    
    def training_step(self, batch, train_set_idx):
        # inputs
        img, id_ = batch
        img, id_ = img.to(self.conf.local_rank), id_.to(self.conf.local_rank)
        
        self.opts.encoder.zero_grad()
        self.opts.loss[train_set_idx].zero_grad()
        
        # Encdoer forward
        self.encoder.train()
        feat = F.normalize(self.forward(img))
        
        # PartialFC forward and backward
        grad, loss, logits_exp = self.loss[train_set_idx].forward_backward(id_, feat, self.opts.loss[train_set_idx])
        
        # Encoder backward
        if self.conf.mixed_precision:
            feat.backward(self.grad_amp.scale(grad))
            self.grad_amp.unscale_(self.opts.encoder)
            clip_grad_norm_(self.encoder.parameters(), max_norm=5, norm_type=2)
            self.grad_amp.step(self.opts.encoder)
            self.grad_amp.update()
            
        else:
            feat.backward(grad)
            clip_grad_norm_(self.encoder.parameters(), max_norm=5, norm_type=2)
            self.opts.encoder.step()

        # PartialFC Update
        self.opts.loss[train_set_idx].step()
        self.loss[train_set_idx].update()
        
        return {
            'loss': loss.cpu()
        }


    def _shared_eval_step(self, batch, dataset_name, prefix):
        # batch input
        pair, label = batch
        pair = rearrange(pair, 'b p c h w -> (b p) c h w')
        
        # inference time counter
        torch.cuda.empty_cache() if prefix == 'val' else None
        start = self.starter.record() if prefix == 'val' else time.time()
        
        # inference
        self.encoder.eval()
        
        with torch.no_grad():
            feat = self.forward(pair)
            embedding = F.normalize(feat)
        
        embedding_1, embedding_2 = embedding[0::2], embedding[1::2]
        
        # inference time measurement
        self.ender.record() if prefix == 'val' else None
        torch.cuda.synchronize() if prefix == 'val' else None
        infer_time = self.starter.elapsed_time(self.ender) if prefix == 'val' else time.time() - start
        
        return {
            f'{dataset_name}_embedding_1': embedding_1.cpu().numpy(),
            f'{dataset_name}_embedding_2': embedding_2.cpu().numpy(),
            f'{dataset_name}_infer_time': infer_time,
            f'{dataset_name}_label_list': label.cpu().numpy()
            }


    # --------------------------------------------
    # Validation Step
    # --------------------------------------------
    
    def validation_step(self, batch, dataset_idx):
        
        dataset_name = self.conf.val_dataset[dataset_idx]
        eval_step = self._shared_eval_step(batch, dataset_name, 'val')
        
        return {
            f'{dataset_name}_embedding_1': eval_step[f'{dataset_name}_embedding_1'], 
            f'{dataset_name}_embedding_2': eval_step[f'{dataset_name}_embedding_2'],
            f'{dataset_name}_infer_time': eval_step[f'{dataset_name}_infer_time'], 
            f'{dataset_name}_label_list': eval_step[f'{dataset_name}_label_list'],
            'dataset_name': dataset_name
        }
        
    def validation_epoch_end(self, outputs):
        infer_time_list = list()         
        label_list = list()         
        embedding_1_list = list()
        embedding_2_list = list()
        
        for output in outputs:
            dataset_name = output['dataset_name']
            infer_time_list.append(output[f'{dataset_name}_infer_time'])
            label_list.append(output[f'{dataset_name}_label_list'])
            embedding_1_list.append(output[f'{dataset_name}_embedding_1'])
            embedding_2_list.append(output[f'{dataset_name}_embedding_2'])
            
        infer_time= np.array(infer_time_list).mean()
        labels = np.concatenate(label_list)
        embedding_1 = np.concatenate(embedding_1_list)
        embedding_2 = np.concatenate(embedding_2_list)
        
        roc, acc = performance(embedding_1, embedding_2, labels, min_level=self.min_level, max_level=self.max_level)
        
        self.val_msg[f'{dataset_name}'].acc = acc
        self.val_msg[f'{dataset_name}'].infer_time = infer_time
        self.val_msg[f'{dataset_name}'].roc = roc
        
        
    # --------------------------------------------
    # training epoch end
    # --------------------------------------------
    
    def training_epoch_end(self, outputs, train_set_idx, running_t=None):
        ## Print train results
        # train loss
        train_loss = np.stack([x["loss"] for x in outputs]).mean()
        # learning rate
        lr = self.schs.encoder.get_last_lr()[0]
        # epoch
        epoch = self.epoch + 1
        
        # evaluation
        msg = '='*50 
        msg += f'\n[Training with "{self.conf.train_dataset[train_set_idx]}"]\n' + \
                f'- Epoch {epoch}/{self.conf.num_epoch}\n' + \
                f'- Learning Rate: {lr}\n' + \
                f'- Train Loss: {train_loss:.4f}\n'
        if running_t is not None:
            msg += f'- Training Time per Epoch: {running_t:.2f}s'
        
        val_acc = None
        
        if epoch % self.conf.valid_freq == 0:
            val_acc = edict()
            for val_dataset_name in self.val_msg:
                val_acc[val_dataset_name] = self.val_msg[val_dataset_name].acc
                
                msg += '\n'.join([
                            f'\n\n[Validation with "{val_dataset_name}"]',
                            f'- Val Accuracy: {self.val_msg[val_dataset_name].acc:.2f}%',
                            f'- Val Inference Time: {self.val_msg[val_dataset_name].infer_time:.2f}ms\n'
                            ])
                msg += self.val_msg[val_dataset_name].roc
        
        msg += '='*50 + '\n'
        # print & log train log
        print(msg) if self.logger_ is None else print_log(self.logger_, msg)
            
        # tensorboard log
        
        self.epoch += 1
        
        self.schs.encoder.step()
        self.schs.loss[train_set_idx].step()
        
        return {
                    'lr': lr,
                    'train_loss': train_loss,
                    'val_acc': val_acc
                }
        

    # --------------------------------------------
    # optimization
    # --------------------------------------------
    
    def configure_optimizers(self): 
        opt = edict()
        opt.loss = list()
        
        sch = edict()
        sch.loss = list()
        
        ## Optimizer
        if self.conf.optimizer == 'Adam':
            opt.encoder = torch.optim.Adam([{'params': self.encoder.parameters()}], 
                                            lr=self.lr, 
                                            weight_decay=self.conf.wd)
                    
        elif self.conf.optimizer == 'SGD':
            opt.encoder = torch.optim.SGD([{'params': self.encoder.parameters()}], 
                                            lr=self.lr, 
                                            momentum=self.conf.mom, 
                                            weight_decay=self.conf.wd)
            
        ## Scheduler
        if self.conf.lr_scheduler == 'CosineAnnealingWarmupRestarts':
            sch.encoder = importlib.import_module("utils.scheduler").CosineAnnealingWarmupRestarts(
                                                                                                    opt.encoder,
                                                                                                    first_cycle_steps=self.conf.num_epoch, 
                                                                                                    warmup_steps=self.conf.warmup_steps,
                                                                                                    min_lr=self.conf.min_lr,
                                                                                                    max_lr=self.lr)
            
        elif self.conf.lr_scheduler == 'MultiStep':
            sch.encoder = optim.lr_scheduler.MultiStepLR(opt.encoder, milestones=self.conf.lr_decay_epoch, gamma=self.conf.lr_decay_ratio)
            
        elif self.conf.lr_scheduler == 'StepLR':
            sch.encoder = optim.lr_scheduler.StepLR(opt.encoder, step_size=self.conf.lr_decay_epoch_size, gamma=self.conf.lr_decay_ratio)
            
        elif self.conf.lr_scheduler == 'CosineAnnealingLR':
            sch.encoder = importlib.import_module("utils.scheduler").CosineAnnealingWarmupRestarts(
                                                                                                    opt.encoder,
                                                                                                    first_cycle_steps=self.conf.num_epoch+1, 
                                                                                                    warmup_steps=self.conf.warmup_steps,
                                                                                                    min_lr=self.conf.min_lr,
                                                                                                    max_lr=self.lr)
                                                    
        for loss in self.loss:
            ## Optimizer
            if self.conf.optimizer == 'Adam':
                opt.loss.append(torch.optim.Adam([{'params': loss.parameters()}], 
                                                    lr=self.lr, 
                                                    weight_decay=self.conf.wd))
                
            elif self.conf.optimizer == 'SGD':
                opt.loss.append(torch.optim.SGD([{'params': loss.parameters()}], 
                                                    lr=self.lr, 
                                                    momentum=self.conf.mom, 
                                                    weight_decay=self.conf.wd))
            
            ## Scheduler
            if self.conf.lr_scheduler == 'CosineAnnealingWarmupRestarts':
                sch.loss.append(importlib.import_module("utils.scheduler").CosineAnnealingWarmupRestarts(
                                                                                                        opt.loss[-1],
                                                                                                        first_cycle_steps=self.conf.num_epoch, 
                                                                                                        warmup_steps=self.conf.warmup_steps,
                                                                                                        min_lr=self.conf.min_lr,
                                                                                                        max_lr=self.lr))
                
            elif self.conf.lr_scheduler == 'MultiStep':
                sch.loss.append(optim.lr_scheduler.MultiStepLR(opt.loss[-1], milestones=self.conf.lr_decay_epoch, gamma=self.conf.lr_decay_ratio))
                
            elif self.conf.lr_scheduler == 'StepLR':
                sch.loss.append(optim.lr_scheduler.StepLR(opt.loss[-1], step_size=self.conf.lr_decay_epoch_size, gamma=self.conf.lr_decay_ratio))
                
            elif self.conf.lr_scheduler == 'CosineAnnealingLR':
                sch.loss.append(importlib.import_module("utils.scheduler").CosineAnnealingWarmupRestarts(
                                                                                                        opt.loss[-1],
                                                                                                        first_cycle_steps=self.conf.num_epoch+1, 
                                                                                                        warmup_steps=self.conf.warmup_steps,
                                                                                                        min_lr=self.conf.min_lr,
                                                                                                        max_lr=self.lr))
        
        msg = '\n'+ '='*50 + '\n'
        msg += '* Optimizer and Scheduler *\n'
        msg += f'- The Number of Optimizers: {len(opt.loss)}\n'
        msg += f'- The Number of Schedulers: {len(sch.loss)}\n'
        msg += '='*50 + '\n'
        print(msg) if self.logger_ is None else print_log(self.logger_, msg)
        
        return opt, sch
    
    
