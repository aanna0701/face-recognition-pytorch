import torchmetrics
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import importlib
import torch
from torch import nn
import pytorch_lightning as pl
from utils.logger import print_log
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import numpy as np
import time
import cv2
from utils.data import DATA_Module
from einops import rearrange
from utils.eval import performance
from easydict import EasyDict as edict


class FR(pl.LightningModule):
    global LOGGER

    def __init__(self, conf:str, logger:str =None, stage:str = 'fit'):
        super().__init__()
        # # turn off automatic optimization
        # self.automatic_optimization = False
        
        # ---------------------------------------
        # writers
        # ---------------------------------------
        self.conf = conf
        self.logger_ = logger
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
        # Classifier
        if stage=='fit':
            self.classifier = nn.ModuleList()
            for i in range(len(conf.train_dataset)):
                if not conf.loss == 'PartialFC':
                    self.classifier.append(importlib.import_module(f"nets.{conf.loss}").Classifier(conf=conf, 
                                                                                                    n_classes=conf.n_classes[i]))
                else:
                    self.classifier.append(importlib.import_module(f"nets.{conf.loss}").PartialFCAMSoftmax(conf=conf, 
                                                                                                    n_classes=conf.n_classes[i]))
                
        # ---------------------------------------
        # Loss
        # ---------------------------------------
        self.criterion = nn.CrossEntropyLoss()
        
        
        # ---------------------------------------
        # Metrics
        # ---------------------------------------
        self.train_acc = torchmetrics.Accuracy()
        
        # ---------------------------------------
        # save path
        # ---------------------------------------
        self.save_path = Path(logger).parent
        
    # --------------------------------------------
    # forward
    # --------------------------------------------
    def forward(self, x):
        feat = self.encoder(x) 
        return feat

    # --------------------------------------------
    # training
    # --------------------------------------------
    def training_step(self, batch, batch_idx):
        img, id_ = batch[0]
        feat = self(img)
        logits = self.classifier[0](feat, id_)
        loss = self.criterion(logits, id_)
        acc = self.train_acc(logits.clone(), id_)
        
        
        return {
            'loss': loss.cpu(),
            'acc': acc.cpu()
        }

    # --------------------------------------------
    # training epoch end
    # --------------------------------------------
    def training_epoch_end(self, outputs):
        ## Print train results
        # train loss
        train_loss = np.stack([x["loss"] for x in outputs]).mean()
        # learning rate
        lr = self.lr_schedulers().get_last_lr()[0]
        # epoch
        epoch = self.current_epoch + 1
        # train acc
        train_acc = np.stack([x["acc"] for x in outputs]).mean()
        
        # evaluation
        if (self.current_epoch+1) % self.conf.print_epoch == 0:
            msg = '='*50 
            msg += f'\n[Training with "{self.conf.train_dataset[0]}"]\n' + \
                    f'- Epoch {epoch}/{self.conf.num_epoch}\n' + \
                    f'- Learning Rate: {lr}\n' + \
                    f'- Train Loss: {train_loss:.4f}\n' + \
                    f'- Train ACC: {train_acc*100:.2f}%\n'
                        
            for val_dataset_name in self.val_msg:
                self.log(f"{val_dataset_name} Verification Accuracy", self.val_msg[val_dataset_name].acc)
                
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
        self.log("Learning Rate", lr)
        self.log("Average Train Loss", train_loss)
        self.log("Average Train Classification Accuracy", train_acc)
        

    # --------------------------------------------
    # optimization
    # --------------------------------------------
    def configure_optimizers(self): 
        opt = list()
        sch = list()
        
        ## Optimizer
        for classifier in self.classifier:
            if self.conf.optimizer == 'Adam':
                opt.append(torch.optim.Adam([
                        {'params': self.encoder.parameters()}, {'params': classifier.parameters()}
                    ], lr=self.lr, weight_decay=self.conf.wd))
                
            elif self.conf.optimizer == 'SGD':
                opt.append(torch.optim.SGD([
                        {'params': self.encoder.parameters()}, {'params': classifier.parameters()}
                    ], lr=self.lr, momentum=self.conf.mom, weight_decay=self.conf.wd))
            
            ## Scheduler
            if self.conf.lr_scheduler == 'CosineAnnealingWarmupRestarts':
                sch.append(importlib.import_module("utils.scheduler").CosineAnnealingWarmupRestarts(
                                                                opt[-1],
                                                                first_cycle_steps=self.conf.num_epoch, 
                                                                warmup_steps=self.conf.warmup_steps,
                                                                min_lr=self.conf.min_lr,
                                                                max_lr=self.lr))
            elif self.conf.lr_scheduler == 'MultiStep':
                sch.append(optim.lr_scheduler.MultiStepLR(opt[-1], milestones=self.conf.lr_decay_epoch, gamma=self.conf.lr_decay_ratio))
            elif self.conf.lr_scheduler == 'StepLR':
                sch.append(optim.lr_scheduler.StepLR(opt[-1], step_size=self.conf.lr_decay_epoch_size, gamma=self.conf.lr_decay_ratio))
            elif self.conf.lr_scheduler == 'CosineAnnealingLR':
                sch.append(importlib.import_module("utils.scheduler").CosineAnnealingWarmupRestarts(
                                                                opt[-1],
                                                                first_cycle_steps=self.conf.num_epoch+1, 
                                                                warmup_steps=self.conf.warmup_steps,
                                                                min_lr=self.conf.min_lr,
                                                                max_lr=self.lr))
        
        msg = '\n'+ '='*50 + '\n'
        msg += '* Optimizer and Scheduler *\n'
        msg += f'- The Number of Optimizers: {len(opt)}\n'
        msg += f'- The Number of Schedulers: {len(sch)}\n'
        msg += '='*50 + '\n'
        print(msg) if self.logger_ is None else print_log(self.logger_, msg)
        
        return opt, sch
    
    def _shared_eval_step(self, batch, batch_idx, dataset_name, prefix):
        # batch input
        pair, label = batch
        pair = rearrange(pair, 'b p c h w -> (b p) c h w')
        
        # inference time counter
        torch.cuda.empty_cache() if prefix == 'val' else None
        start = self.starter.record() if prefix == 'val' else time.time()
        
        # inference
        feat = self(pair)
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
    
    def validation_step(self, batch, batch_idx, dataset_idx):
        dataset_name = self.conf.val_dataset[dataset_idx]
        eval_step = self._shared_eval_step(batch, batch_idx, dataset_name, 'val')
        
        return {
            f'{dataset_name}_embedding_1': eval_step[f'{dataset_name}_embedding_1'], 
            f'{dataset_name}_embedding_2': eval_step[f'{dataset_name}_embedding_2'],
            f'{dataset_name}_infer_time': eval_step[f'{dataset_name}_infer_time'], 
            f'{dataset_name}_label_list': eval_step[f'{dataset_name}_label_list'],
            'dataset_name': dataset_name
        }
        
    def validation_epoch_end(self, outputs):
        for outputs_per_dataset in outputs: 
            infer_time_list = list()         
            label_list = list()         
            embedding_1_list = list()
            embedding_2_list = list()
            for output in outputs_per_dataset:
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
    # prediction mode
    # --------------------------------------------
    def predict_step(self, batch, batch_idx):  
        
        depth = batch
        
        feat = self.encoder(depth)
        logit = self.classifier(feat)
        score = F.softmax(logit, dim=1).squeeze()[1]
        
        return score
    