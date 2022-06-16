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
from torchsummary import summary
from torch.distributed.optim import ZeroRedundancyOptimizer
import itertools
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    
    def __init__(self, conf, SAVE_DIR, mode='train', LOGGER=None):
        # super().__init__()
        assert mode in ['train', 'test'], 'Invalid Mode !!!'

        self.conf = conf
        if mode == 'train':
            self.world_size = conf.world_size
            self.local_rank = conf.local_rank
            torch.cuda.set_device(self.local_rank)
        
            # ===========================================================
            # Addional Settings (Tracking, Summary, etc)
            # ===========================================================
            self.save_dir = SAVE_DIR
            
            # TBoard
            self.writer = None
            if self.local_rank == 0:
                self.writer = SummaryWriter(str(Path.cwd().parent / "TBLog" / SAVE_DIR.name))
                str_val = ''
                for k, v in conf.items(): 
                    str_val += '{} : {}  \n'.format(k, v)
                self.writer.add_text('Config', str_val, 0)    
        
        self.logger_ = LOGGER
        
    def train(self, model, train_dm, val_dm):
        
        train_dm.setup(stage='train')
        
        train_loader, train_sampler = train_dm.train_dataloader()
        
        if self.conf.local_rank == 0:
            val_dm.setup(stage='val')
            val_loaders = val_dm.val_dataloader()
        
        start_epoch = 0
        
        for epoch in range(start_epoch, self.conf.num_epoch):
            
            # Time counter
            runnig_t = 0
            
            # Train outputs container
            train_outputs = list()
            
            # Randomly shuffle the sample index every epoch
            train_sampler.set_epoch(epoch)
            
            # Mini-batch training
            print(f'{epoch+1} Epoch Traning') if self.conf.local_rank == 0 else train_loader
            train_loader = tqdm(train_loader) if self.conf.local_rank == 0 else train_loader
            for batch in train_loader:
                start_t = time.time()
                train_outputs.append(model.training_step(batch))
                runnig_t += time.time() - start_t
            
            if self.local_rank == 0:
                # Validation step
                if (epoch+1) % self.conf.valid_freq == 0:
                    print(f'{epoch+1} Epoch Validation')
                    for v_idx, val_loader in enumerate(val_loaders):
                        val_outputs = list()
                        print(f'{v_idx+1}th Validation Dataset') 
                        # Valdiation per Val Dataset
                        for batch in  tqdm(val_loader):
                            val_outputs.append(model.validation_step(batch, v_idx))
                            
                        # Summary Validation Results of Each Dataset
                        model.validation_epoch_end(val_outputs)
                
            # Summary Results per Epoch
            results = model.training_epoch_end(train_outputs, runnig_t)
            
            if self.local_rank == 0:
                # Tensorboard logging
                self.writer.add_scalar(f'{train_dm.train_dataset_name}/Learning Rate', results['lr'], epoch)
                self.writer.add_scalar(f'{train_dm.train_dataset_name}/Train Loss', results['train_loss'], epoch)
                if results['val_acc'] is not None:
                    for val_name in results['val_acc']:
                        self.writer.add_scalar(f'{train_dm.train_dataset_name}/{val_name} Validation ACC', results['val_acc'][val_name], epoch)
                
                if (epoch+1) % self.conf.save_epoch == 0:
                    # Save Checkpoint
                    # Encoder
                    encoder_save_dir = self.save_dir
                    torch.save({
                                'model_state_dict': model.encoder.state_dict(),
                                'epoch': epoch+1,
                                'name': self.conf.network
                                }, str(encoder_save_dir / f'{epoch+1}_epoch_encoder.pth'))


    def test(self, model, test_dm):
            
            test_dm.setup(stage='test')
            test_loader = test_dm.test_dataloader()
            # test step
                
            for test_idx, test_loader in enumerate(test_loader):
                
                test_outputs = list()
                print(f'{test_idx+1}th Test Dataset') 
                # Valdiation per Val Dataset
                for batch in tqdm(test_loader):
                    test_outputs.append(model.test_step(batch, test_idx))
                    
                # Summary Validation Results of Each Dataset
                model.test_epoch_end(test_outputs)
                
            
            # evaluation
            msg = '='*50         
            test_acc = None
            
            test_acc = edict()
            for test_dataset_name in model.test_msg:
                test_acc[test_dataset_name] = model.test_msg[test_dataset_name].acc
                
                msg += '\n'.join([
                            f'\n\n[Test with "{test_dataset_name}"]',
                            f'- Test Accuracy: {model.test_msg[test_dataset_name].acc:.2f}%',
                            f'- Test Inference Time: {model.test_msg[test_dataset_name].infer_time:.2f}ms\n'
                            ])
                msg += model.test_msg[test_dataset_name].roc
            
            msg += '='*50 + '\n'
            print(msg) if self.logger_ is None else print_log(self.logger_, msg)