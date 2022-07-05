import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
import torch
import torch.distributed as dist 
import torch.multiprocessing as mp
import argparse
import os
import importlib
from utils.data_partial import DATA_Module
from model.FR_PartialFC import Model
import time
from utils.logger import print_log
from utils.trainer import Trainer

os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
now = time.localtime()
conf = None
args = None


# ===========================================================
# Arguments
# ===========================================================

def parse_args():
    """ Arguments for training config file """

    parser = argparse.ArgumentParser(description='train the face recognition network')
    parser.add_argument('--config', default="ms1m_arcface_122", help='name of config file without file extension')
    parser.add_argument('--mode', default="train", choices=['train', 'test'], help='mode of this script')
    parser.add_argument('--network', default="ResNet50", type=str, help='Backbone network')
    parser.add_argument('--loss', default="PartialFC", type=str, help='Embedding space')
    parser.add_argument('--optimizer', default="SGD", type=str, help='optimizer')
    parser.add_argument('--lr', default="1e-1", type=float, help='learning rate')
    parser.add_argument('--no_mixed_precision', action='store_false')
    parser.add_argument('--sample_rate', default=0.3, type=float)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--test_type', default='pair', type=str, choices=['pair', 'cross'])

    args = parser.parse_args()

    return args



# ===========================================================
# Training
# ===========================================================

def train(rank, world_size, args):
    LOGGER = None
    torch.backends.cudnn.benchmark = True

    
    # ===========================================================
    # DDP Setup
    # ===========================================================
    
    world_size = world_size
    # rank = rank
    local_rank = rank

    if world_size > 1:     
        print(f"Training {world_size} node single GPUs\n") if local_rank == 0 else None
        
    else:
        print("Training single node single GPUs\n")
        
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
    conf = config.conf
    conf.network = args.network
    assert conf.network in config.NETWORK, 'Invalid model !!!'
    conf.loss = args.loss
    assert conf.loss in config.LOSS, 'Invalid loss !!!'
    conf.optimizer = args.optimizer
    assert conf.optimizer in config.OPTIMIZER, 'Invalid optimizer !!!'
    conf.lr = args.lr
    # conf.num_workers = conf.num_workers / args.n_gpus
    
    config.generate_config(conf.network, conf.loss, conf.optimizer, conf.lr_scheduler)
    
    if conf.lr_scheduler == "CosineAnnealingWarmupRestarts":
        conf.min_lr = args.lr / 1000
        
    conf.local_rank = local_rank
    conf.world_size = world_size
    conf.mixed_precision = args.no_mixed_precision
    conf.sample_rate = args.sample_rate
    conf.ckpt_path = args.ckpt_path
    conf.img_size = 192 if 'AlterNet' in conf.network else 112

    # ===========================================================
    # Save directories
    # ===========================================================   
    
    SAVE_DIR = Path.cwd().parent / 'save' / f'{args.mode}' / f'{now.tm_mon}-{now.tm_mday}_{now.tm_hour}h{now.tm_min}m-{now.tm_sec}s'
    SAVE_DIR = SAVE_DIR.parent / ('_'.join(
                                        [
                                            SAVE_DIR.name, 
                                            conf.network, 
                                            conf.loss, 
                                            conf.optimizer, 
                                            f'lr_{str(args.lr)}'
                                            ]
                                        ))

    if local_rank == 0:
        
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
        
        with open(SAVE_DIR / f'{args.config}_config.txt', "w") as file:
            file.write(msg_conf)
            
        del msg_conf
    
    
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
    
    trainer = Trainer(conf, SAVE_DIR, 'train')
    trainer.train(model, train_dm, val_dm)
    
    
    # ===========================================================
    # DDP Cleanup
    # ===========================================================
    
    dist.destroy_process_group()
    
# ===========================================================
# Test
# ===========================================================

def test(args):
    LOGGER = None
    # ===========================================================
    # Configurations
    # ===========================================================
    
    config = importlib.import_module(f"configs.{args.config}")
    conf = config.conf
    conf.local_rank = 0
    conf.network = args.network
    conf.ckpt_path = args.ckpt_path
    assert conf.network in config.NETWORK, 'Invalid model !!!'
    conf.test_type = args.test_type
    
    config.generate_config(conf.network, conf.loss, conf.optimizer, conf.lr_scheduler)
    
    
    # ===========================================================
    # Save directories
    # ===========================================================   
    
    SAVE_DIR = Path.cwd().parent / 'save' / f'{args.test_type}_{args.mode}' / f'{now.tm_mon}-{now.tm_mday}_{now.tm_hour}h{now.tm_min}m-{now.tm_sec}s'
    ckpt_path = conf.ckpt_path[:-4].split('/')
    SAVE_DIR = SAVE_DIR / '_'.join(ckpt_path)
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
    
    with open(SAVE_DIR / f'{args.config}_config.txt', "w") as file:
            file.write(msg_conf)
            
    del msg_conf
    
    
    # ===========================================================
    # Data Module
    # ===========================================================

    test_dm = DATA_Module(conf, LOGGER)
    
    
    # ===========================================================
    # Model
    # ===========================================================
    model = Model(conf, LOGGER, 'test')
    
    
    # ===========================================================
    # Run Jobs
    # ===========================================================
    
    trainer = Trainer(conf, SAVE_DIR, 'test', LOGGER)
    trainer.test(model, test_dm)
    
    
    
if __name__ == '__main__':
    
    # ===========================================================
    # Arguments
    # ===========================================================
    args = parse_args()
    
    
    # ===========================================================
    # Main
    # ===========================================================

    if args.mode == 'train':
        world_size = torch.cuda.device_count()
        
        if world_size > 1:
            mp.spawn(train,
                    args=(world_size, args),
                    nprocs=world_size,
                    join=True
                    )
        
        else:
            train(0, 1, args)
        
    elif args.mode == 'test':
        test(args)

