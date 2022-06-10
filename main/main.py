import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import Trainer
import importlib
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from utils.data import DATA_Module
from model.FR import FR
import time
from utils.logger import print_log
from shutil import copyfile
import os
from utils.data import DATA_Module

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

now = time.localtime()


# =========================================== Arguments ===========================================

def parse_args():
    """ Arguments for training config file """

    parser = argparse.ArgumentParser(description='train the face recognition network')
    parser.add_argument('--config', default="lfw", help='name of config file without file extension')
    parser.add_argument('--mode', default="train", choices=['train', 'test'], help='mode of this script')
    
    parser.add_argument('--network', default="ResNet50", type=str, help='Backbone network')
    parser.add_argument('--loss', default="ArcFace", type=str, help='Embedding space')
    parser.add_argument('--optimizer', default="SGD", type=str, help='optimizer')
    parser.add_argument('--lr', default="1e-1", type=float, help='learning rate')
    
    args = parser.parse_args()

    return args


def main():
    
    # --------------------------------------------
    # train arguments
    # --------------------------------------------
    
    global args
    args = parse_args()
    
    # --------------------------------------------
    # Configurations
    # --------------------------------------------   
    print(sys.path)
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
    
    config.generate_config(conf.network, conf.loss, conf.optimizer, conf.lr_scheduler)
    
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
    LOGGER = str(SAVE_DIR / 'log.txt')
    
    ## Print configurations
    msg_conf = '\n' + '='*50 + '\n'
    msg_conf += '* Configuration *\n\n'
    for k in conf: msg_conf += f"{k} = {conf[k]}" + "\n"
    msg_conf += '='*50
    print_log(LOGGER, msg_conf)
    del msg_conf
    copyfile(Path.cwd().parent / 'configs' / f'{args.config}.py', SAVE_DIR / f'{args.config}.py')
    
    # --------------------------------------------
    # Dataset
    # --------------------------------------------
    dm = DATA_Module(conf, LOGGER)
    
    # --------------------------------------------
    # Model
    # --------------------------------------------
    model = FR(conf, LOGGER)
    
    # --------------------------------------------
    # Callbacks
    # --------------------------------------------
    
    # checkpoint
    
    checkpoint_callback = [
        ModelCheckpoint(dirpath=str(SAVE_DIR), monitor='epoch', mode='max', filename='{epoch}')
        ]
    
    # tensorboard
    tensorboard_logger = TensorBoardLogger(str(Path.cwd().parent), name="TBLog", version=SAVE_DIR.name)
    
    # --------------------------------------------
    # Training
    # --------------------------------------------
    
    if args.mode == 'train':
        trainer = Trainer(gpus=1, 
                        max_epochs=conf.num_epoch, 
                        enable_progress_bar=False,
                        callbacks=[*checkpoint_callback], 
                        logger=tensorboard_logger, 
                        log_every_n_steps=1,
                        strategy = DDPStrategy(find_unused_parameters=False),
                        # precision=16 if conf.mixed_precision else None
                        )
        trainer.fit(model, dm)
    
    # --------------------------------------------
    # Test
    # --------------------------------------------

    elif args.mode == 'test':
        trainer = Trainer(logger=tensorboard_logger)
        trainer.test(model, ckpt_path=conf.pretrained_model)

if __name__ == "__main__":
    main()
