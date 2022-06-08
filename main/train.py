import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import torch
import numpy as np
import torch.distributed as dist 
from torch.utils.tensorboard import SummaryWriter
#import mlflow
#from mlflow import log_metric, log_param, log_artifact
import argparse
import os
import pandas as pd
import importlib
from shutil import copyfile
import time
import torch.distributed as dist
import pandas as pd
import torch

from utils.data_partial import DATA_Module
from model.FR_PartialFC import Model
import time
from utils.logger import print_log
from shutil import copyfile
import os

os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
now = time.localtime()
conf = None
args = None

# =========================================== Arguments ===========================================

def get_wm_size(path):
    df = pd.read_csv(path)
    ids = df.iloc[:, 1]
    ids = list(ids)
    unq_ids = np.unique(ids)
    sz = np.shape(unq_ids)[0]
    return sz
    

def parse_args():
    """ Arguments for training config file """

    parser = argparse.ArgumentParser(description='train the face recognition network')
    parser.add_argument('--config', default="lfw", help='name of config file without file extension')
    parser.add_argument('--local_rank', default=0, type=int, help='Local Rank')
    parser.add_argument('--mode', default="train", choices=['train', 'test'], help='mode of this script')
    parser.add_argument('--network', default="ResNet50", type=str, help='Backbone network')
    parser.add_argument('--loss', default="PartialFC", type=str, help='Embedding space')
    parser.add_argument('--optimizer', default="SGD", type=str, help='optimizer')
    parser.add_argument('--lr', default="1e-1", type=float, help='learning rate')
    parser.add_argument('--mixed_precision', action='store_true')

    args = parser.parse_args()

    return args



# ============================================ Training ============================================

class Trainer:
    
    def __init__(self, conf, mode='train'):
        # super().__init__()
        assert mode in ['train', 'pred'], 'Invalid Mode !!!'

        self.conf = conf
        self.world_size = conf.world_size
        self.rank = conf.rank
        self.local_rank = conf.local_rank
        torch.cuda.set_device(self.local_rank)
        
        # ===========================================================
        # Define Save Folder and Parameters
        # ===========================================================
        ## Save directories
        SAVE_DIR = Path.cwd().parent / 'save' / f'{args.mode}_{now.tm_mon}-{now.tm_mday}_{now.tm_hour}h{now.tm_min}m-{now.tm_sec}s'
        SAVE_DIR = SAVE_DIR.parent / ('_'.join(
                                            [
                                                SAVE_DIR.name, 
                                                conf.network, 
                                                conf.loss, 
                                                conf.optimizer, 
                                                f'lr_{str(args.lr)}'
                                                ]
                                            ))
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        self.save_dir = SAVE_DIR
        self.logger = str(SAVE_DIR / 'log.txt')
        
        # ===========================================================
        # Addional Settings (Tracking, Summary, etc)
        # ===========================================================
        
        # TBoard
        self.writer = None
        if self.local_rank == 0:
            self.writer = SummaryWriter(str(Path.cwd().parent / "TBLog"), filename_suffix=SAVE_DIR.name)
            str_val = ''
            for k, v in conf.items(): 
                str_val += '{} : {}  \n'.format(k, v)
            self.writer.add_text('Config', str_val, 0)    
        
        
        
    def train(self, model, train_dm, val_dm):
        
        train_dm.setup(stage='train')
        train_loaders, train_sampler = train_dm.train_dataloader()
        
        val_dm.setup(stage='val')
        val_loaders = val_dm.val_dataloader()
        
        start_epoch = 0
        
        for epoch in range(start_epoch, self.conf.num_epoch):
            
            for t_idx, train_loader in enumerate(train_loaders):
                # Time counter
                runnig_t = 0
                
                # Train outputs container
                train_outputs = list()
                
                # Randomly shuffle the sample index every epoch
                train_sampler[t_idx].set_epoch(epoch)
                
                # Mini-batch training
                for batch in train_loader:
                    start_t = time.time()
                    train_outputs.append(model.training_step(batch, t_idx))
                    runnig_t += time.time() - start_t
                
                # Validation step
                if (epoch+1) % self.conf.valid_freq == 0:
                    
                    for v_idx, val_loader in enumerate(val_loaders):
                        val_outputs = list()
                        
                        # Valdiation per Val Dataset
                        for batch in val_loader:
                            val_outputs.append(model.validation_step(batch, v_idx))
                            
                        # Summary Validation Results of Each Dataset
                        model.validation_epoch_end(val_outputs)
                
                # Summary Results per Epoch
                results = model.training_epoch_end(train_outputs, t_idx, runnig_t)
                
                if self.local_rank == 0:
                    # Tensorboard logging
                    self.writer.add_scalar(f'{train_dm.train_dataset_name[t_idx]}/Learning Rate', results['lr'], epoch)
                    self.writer.add_scalar(f'{train_dm.train_dataset_name[t_idx]}/Train Loss', results['train_loss'], epoch)
                    if results['val_acc'] is not None:
                        for val_name in results['val_acc']:
                            self.writer.add_scalar(f'{train_dm.train_dataset_name[t_idx]}/{val_name} Validation Loss', results['val_acc'][val_name], epoch)
                    
                    if (epoch+1) % self.conf.save_epoch == 0:
                        # Save Checkpoint
                        # Encoder
                        encoder_save_dir = self.save_dir / f'{t_idx}_Dataset'
                        encoder_save_dir.mkdir(parents=True, exist_ok=True)
                        torch.save({
                                    'model_state_dict': model.encoder.state_dict(),
                                    'optimizer_state_dict': model.opts.encoder.state_dict(),
                                    'scheduler_state_dict': model.schs.encoder.state_dict(),
                                    'epoch': epoch+1,
                                    'name': self.conf.network
                                    }, str(encoder_save_dir / f'{epoch+1}_epoch_encoder.pth'))
                    

# ============================================== Main ==============================================

def main():
    
    torch.backends.cudnn.benchmark = True

    # ===========================================================
    # Arguments
    # ===========================================================
    
    global args
    args = parse_args()
    
    
    
    # ===========================================================
    # DDP settings
    # ===========================================================
    
    try:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        dist.init_process_group('nccl')
        
    except KeyError:
        print("Training single node single GPUs\n")
        world_size = 1
        rank = 0
        dist.init_process_group(
                                backend='nccl', 
                                init_method="tcp://127.0.0.1:12584", 
                                rank=rank, 
                                world_size=world_size
                                )
    
    
    
    # ===========================================================
    # Configurations
    # ===========================================================
    
    config = importlib.import_module(f"configs.{args.config}")
    global conf
    conf = config.conf
    conf.network = args.network
    assert conf.network in config.NETWORK, 'Invalid model !!!'
    conf.loss = args.loss
    assert conf.loss in config.LOSS, 'Invalid loss !!!'
    conf.optimizer = args.optimizer
    assert conf.optimizer in config.OPTIMIZER, 'Invalid optimizer !!!'
    conf.lr = args.lr
    conf.world_size = world_size
    conf.rank = rank
    conf.local_rank = args.local_rank
    conf.mixed_precision = args.mixed_precision
    
    config.generate_config(conf.network, conf.loss, conf.optimizer, conf.lr_scheduler)
    
    
    
    # ===========================================================
    # Save directories
    # ===========================================================   
    
    SAVE_DIR = Path.cwd().parent / 'save' / f'{args.mode}_{now.tm_mon}-{now.tm_mday}_{now.tm_hour}h{now.tm_min}m-{now.tm_sec}s'
    SAVE_DIR = SAVE_DIR.parent / ('_'.join(
                                        [
                                            SAVE_DIR.name, 
                                            conf.network, 
                                            conf.loss, 
                                            conf.optimizer, 
                                            f'lr_{str(args.lr)}'
                                            ]
                                        ))
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER = str(SAVE_DIR / 'log.txt')
    
    
    
    # ===========================================================
    # Print configurations
    # ===========================================================
    
    msg_conf = '\n' + '='*50 + '\n'
    msg_conf += '* Configuration *\n\n'
    for k in conf: msg_conf += f"{k} = {conf[k]}" + "\n"
    msg_conf += '='*50
    print_log(LOGGER, msg_conf)
    del msg_conf
    copyfile(Path.cwd().parent / 'configs' / f'{args.config}.py', SAVE_DIR / f'{args.config}.py')

    
    
    # ===========================================================
    # Data Module
    # ===========================================================
    train_dm = DATA_Module(conf, LOGGER)
    val_dm = DATA_Module(conf, LOGGER)

    
    
    # ===========================================================
    # Model
    # ===========================================================
    model = Model(conf, LOGGER, 'train')
    
    
    # ===========================================================
    # Run Jobs
    # ===========================================================
    
    trainer = Trainer(conf, 'train')
    trainer.train(model, train_dm, val_dm)
    
    
    
if __name__ == '__main__':

    main()

