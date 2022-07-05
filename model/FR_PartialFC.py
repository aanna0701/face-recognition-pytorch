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
from utils.eval import performance_roc, cross_score, pair_score, performance_acc
from easydict import EasyDict as edict
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torchsummary import summary

def print_peak_memory(prefix, device):
    if device == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")
        
        
class Model(nn.Module):

    def __init__(self, conf:str, logger:str =None, stage:str = 'train'):
        super().__init__()
        
        # ---------------------------------------
        # writers
        # ---------------------------------------
        self.conf = conf
        self.logger_ = logger
        self.epoch = 0
        self.idx_save_image = 0
        self.lr = conf.lr
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.security_level = self.conf.security_level
        self.max_level = self.conf.max_level
        self.min_level = self.conf.min_level
        
        if stage == 'train':
            self.val_msg = edict()
            for val_dataset_name in conf.val_dataset:
                self.val_msg[f'{val_dataset_name}'] = edict()
        
        elif stage == 'test':
            self.test_msg = edict()
            if conf.test_type == 'cross':
                for test_dataset_name in conf.cross_test_dataset:
                    self.test_msg[f'{test_dataset_name}'] = edict()
            
            elif conf.test_type == 'pair':
                for test_dataset_name in conf.test_dataset:
                    self.test_msg[f'{test_dataset_name}'] = edict()
        
        
        # ---------------------------------------
        # model
        # ---------------------------------------
        # Encoder
        if 'ResNet' in conf.network:
            self.encoder = importlib.import_module(f"nets.resnet").Encoder(conf=conf)
            
        elif 'AlterNet' in conf.network:
            self.encoder = importlib.import_module(f"nets.AlterNet_SwinV2_FAN").Encoder(conf=conf)
            
        elif 'Swin' in conf.network:
            self.encoder = importlib.import_module(f"nets.SwinV2").Encoder(conf=conf)
            
        elif 'EffiAlter' in conf.network:
            self.encoder = importlib.import_module(f"nets.EffiAlterNet_SwinV2_FAN").Encoder(conf=conf)
            
        self.encoder = self.encoder.to(conf.local_rank)
        
        if conf.ckpt_path is not None:
            from collections import OrderedDict
            
            print('Transferring Weight')
            pre_weight = torch.load(conf.ckpt_path)['model_state_dict']
            
            new_pre_weight = OrderedDict()
            for k, v in pre_weight.items():
                name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
                new_pre_weight[name] = v
                
            self.encoder.load_state_dict(new_pre_weight, strict=True)
            del pre_weight
            del new_pre_weight
            print('Finished')
            
            
        # Classifier
        if stage=='train':
            # DDP Setting for Encoder
            
            print_peak_memory("Max memory allocated after creating local encoder", conf.local_rank)
            self.encoder = DDP(self.encoder, broadcast_buffers=False, find_unused_parameters=True,device_ids=[conf.local_rank])
            print_peak_memory("Max memory allocated after creating DDP", conf.local_rank)
            
            # Loading PartialFC loss          
            if conf.optimizer == 'SGD':
                self.loss = importlib.import_module(f"nets.{conf.loss}").PartialFC( conf=conf, 
                                                                                    num_classes=conf.n_classes
                                                                                    )
            elif conf.optimizer == 'AdamW':
                self.loss = importlib.import_module(f"nets.{conf.loss}").PartialFCAdamW(conf=conf, 
                                                                                        num_classes=conf.n_classes
                                                                                        )

            self.loss.train().to(conf.local_rank)
                    
            if conf.local_rank == 0:
                print()
                summary(self.encoder, (3, conf.img_size, conf.img_size))
                # print(self.encoder)
                print()
                print(self.loss)
                print()
            
            # Initializing Optimizers and Schedulers
            self.opt, self.sch = self.configure_optimizers()
            
            
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
            self.save_path = Path(logger).parent if conf.local_rank == 0 else None
            
            
            # ---------------------------------------
            # Mixed precision
            # ---------------------------------------
            if conf.mixed_precision:
                self.amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
                print('Mixed Precision !!!\n') if conf.local_rank == 0 else None
            
            
    # --------------------------------------------
    # forward
    # --------------------------------------------
    def forward(self, x):
        feat = self.encoder(x) 
        return feat


    # --------------------------------------------
    # training
    # --------------------------------------------
    def training_step(self, batch):
        # inputs
        img, id_ = batch
        img, id_ = img.to(self.conf.local_rank), id_.to(self.conf.local_rank)
        
        self.opt.zero_grad()
        
        # Encdoer forward
        self.encoder.train()
        feat = F.normalize(self.forward(img))
        
        # PartialFC forward and backward
        self.loss.train().cuda()
        loss = self.loss(feat, id_, self.opt)
        
        # Encoder backward
        if self.conf.mixed_precision:
            self.amp.scale(loss).backward()
            self.amp.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
            self.amp.step(self.opt)
            self.amp.update()
            
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 5)
            self.opt.step()
        
        
        return {
            'loss': loss.cpu().detach().numpy()
        }


    def _shared_eval_step(self, batch, dataset_name, prefix):
        # batch input
        pair, label = batch
        pair, label = pair.to(self.conf.local_rank), label.to(self.conf.local_rank)
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
        
        hist_genuine, hist_imposter, score_list = pair_score(embedding_1, embedding_2, labels)
        
        roc, eer_th = performance_roc(hist_genuine, hist_imposter, min_level=self.min_level, max_level=self.max_level)
        acc = performance_acc(score_list, labels, eer_th)
        
        self.val_msg[f'{dataset_name}'].acc = acc
        self.val_msg[f'{dataset_name}'].infer_time = infer_time
        self.val_msg[f'{dataset_name}'].roc = roc
        
        
    # --------------------------------------------
    # training epoch end
    # --------------------------------------------
    
    def training_epoch_end(self, outputs, running_t=None):
        ## Print train results
        if self.conf.local_rank == 0:
        # train loss
            train_loss = np.stack([x["loss"] for x in outputs]).mean()
            # learning rate
            lr = self.sch.get_last_lr()[0]
            # epoch
            epoch = self.epoch + 1
        
            # evaluation
            msg = '='*50 
            msg += f'\n[Training with "{self.conf.train_dataset}"]\n' + \
                    f'- Epoch {epoch}/{self.conf.num_epoch}\n' + \
                    f'- Learning Rate: {lr}\n' + \
                    f'- Train Loss: {train_loss:.4f}\n'
            if running_t is not None:
                msg += f'- Training Time per Epoch: {running_t:.2f}s\n'
            
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
        
        self.sch.step()
        
        if self.conf.local_rank == 0:
            return {
                        'lr': lr,
                        'train_loss': train_loss,
                        'val_acc': val_acc
                    }
            
            
    # --------------------------------------------
    # Test Step
    # --------------------------------------------
    
    def test_step(self, batch, dataset_idx):
        
        dataset_name = self.conf.test_dataset[dataset_idx]
        eval_step = self._shared_eval_step(batch, dataset_name, 'test')
        
        return {
            f'{dataset_name}_embedding_1': eval_step[f'{dataset_name}_embedding_1'], 
            f'{dataset_name}_embedding_2': eval_step[f'{dataset_name}_embedding_2'],
            f'{dataset_name}_infer_time': eval_step[f'{dataset_name}_infer_time'], 
            f'{dataset_name}_label_list': eval_step[f'{dataset_name}_label_list'],
            'dataset_name': dataset_name
        }
        
    def test_epoch_end(self, outputs):
        infer_time_list = list()         
        label_list = list()         
        embedding_1_list = list()
        embedding_2_list = list()
        
        for output in outputs:
            ##########################
            dataset_name = output['dataset_name']
            infer_time_list.append(output[f'{dataset_name}_infer_time'])
            label_list.append(output[f'{dataset_name}_label_list'])
            embedding_1_list.append(output[f'{dataset_name}_embedding_1'])
            embedding_2_list.append(output[f'{dataset_name}_embedding_2'])
            
        infer_time= np.array(infer_time_list).mean()
        labels = np.concatenate(label_list)
        embedding_1 = np.concatenate(embedding_1_list)
        embedding_2 = np.concatenate(embedding_2_list)
        
        s_t = time.time()
        hist_genuine, hist_imposter, score_list = pair_score(embedding_1, embedding_2, labels)
        
        roc, eer_th = performance_roc(hist_genuine, hist_imposter, min_level=self.min_level, max_level=self.max_level)
        acc = performance_acc(score_list, labels, eer_th)
        
        self.test_msg[f'{dataset_name}'].acc = acc
        # self.test_msg[f'{dataset_name}'].infer_time = infer_time
        self.test_msg[f'{dataset_name}'].infer_time = time.time()-s_t
        self.test_msg[f'{dataset_name}'].roc = roc
        
    # --------------------------------------------
    # Cross-matching Test Step
    # --------------------------------------------        
    
    def cross_test_step(self, batch, dataset_idx):
        
        dataset_name = self.conf.cross_test_dataset[dataset_idx]
        
        img, label = batch
        img, label = img.to(self.conf.local_rank), label.to(self.conf.local_rank)
        
        start = time.time()
        
        with torch.no_grad():
            feat = self.forward(img)
            embedding = F.normalize(feat)
        
        infer_time = time.time() - start
        
        return {
            f'{dataset_name}_embedding': embedding.cpu(), 
            f'{dataset_name}_infer_time': infer_time, 
            f'{dataset_name}_label_list': label.cpu(),
            'dataset_name': dataset_name
        }
        
    def cross_test_epoch_end(self, outputs):
        
        infer_time_list = list()         
        label_list = list()         
        embed_list = list()
        
        for output in outputs:
            dataset_name = output['dataset_name']
            embed_list.append(output[f'{dataset_name}_embedding'])
            label_list.append(output[f'{dataset_name}_label_list'])
            infer_time_list.append(output[f'{dataset_name}_infer_time'])
            
        infer_time= np.array(infer_time_list).mean()
        labels = np.concatenate(label_list)
        embeds = np.concatenate(embed_list)
        
        hist_genuine, hist_imposter, score_list, label_list = cross_score(
                                                                            embeds, 
                                                                            labels
                                                                            )
        
        roc, eer_th = performance_roc(hist_genuine, hist_imposter, min_level=self.min_level, max_level=self.max_level)
        acc = performance_acc(score_list, label_list, eer_th)
        
        self.test_msg[f'{dataset_name}'].acc = acc
        self.test_msg[f'{dataset_name}'].infer_time = infer_time
        self.test_msg[f'{dataset_name}'].roc = roc
        
        
        
    # --------------------------------------------
    # optimization
    # --------------------------------------------
    def configure_optimizers(self): 
        
        ## Optimizer
        if self.conf.optimizer == 'AdamW':
            opt = torch.optim.AdamW(    [{'params': self.encoder.parameters()}, {'params': self.loss.parameters()}], 
                                        lr=self.lr,
                                        weight_decay=self.conf.wd,
                                        eps=self.conf.eps,
                                        betas=self.conf.betas
                                        )
            
        elif self.conf.optimizer == 'SGD':
            opt = torch.optim.SGD(  [{'params': self.encoder.parameters()}, {'params': self.loss.parameters()}], 
                                    lr=self.lr, 
                                    momentum=self.conf.mom, 
                                    weight_decay=self.conf.wd   )
    
        ## Scheduler
        if self.conf.lr_scheduler == 'CosineAnnealingWarmupRestarts':
            sch = importlib.import_module("utils.scheduler").CosineAnnealingWarmupRestarts(
                                                                                            opt,
                                                                                            first_cycle_steps=self.conf.num_epoch, 
                                                                                            warmup_steps=self.conf.warmup_steps,
                                                                                            min_lr=self.conf.min_lr,
                                                                                            max_lr=self.lr)
            
        elif self.conf.lr_scheduler == 'MultiStep':
            sch = optim.lr_scheduler.MultiStepLR(opt, milestones=self.conf.lr_decay_epoch, gamma=self.conf.lr_decay_ratio)
            
        elif self.conf.lr_scheduler == 'StepLR':
            sch = optim.lr_scheduler.StepLR(opt, step_size=self.conf.lr_decay_epoch_size, gamma=self.conf.lr_decay_ratio)
            
        if self.conf.local_rank == 0:
            msg = '\n'+ '='*50 + '\n'
            msg += '* Optimizer and Scheduler *\n'
            msg += f'- Optimizers: {opt}\n'
            msg += f'- Schedulers: {sch}\n'
            msg += '='*50 + '\n'
            print(msg) if self.logger_ is None else print_log(self.logger_, msg)
        
        return opt, sch
    
    