from easydict import EasyDict as edict
from pathlib import Path

conf = edict()

NETWORK = [
            'ResNet100', 'ResNet200', 'ResNet34', 'ResNet50',
            'AlterNet50'
            ]
LOSS = ['ArcFace', 'PartialFC']
METRIC = ['ArcFace']
OPTIMIZER = ['SGD', 'AdamW']
DATA_DIR = '/workspace/dataset/FR'
TRAIN_DATA = ['webface42m', 'lfw', 'ms1m_arcface_122']
N_CLASSESE = {
            'webface42m': 2059906,
            'lfw': 5749,
            'ms1m_arcface_122': 86690 
            }
VAL_DATA = ['lfw', 'agedb_30', 'cfp_fp', 'cfp_ff', 'calfw', 'cplfw']

# --------------------------------------------
# Default network
# --------------------------------------------
conf.network = 'AlterNet50'
assert conf.network in NETWORK, 'Invalid model !!!'
conf.resume = False
conf.pretrained = False
conf.transfer_learning = False
conf.pretrained_dir = './models/'
conf.use_checkpoint = False
conf.checkpoint_path = './model_backup/LResNet200E-IR+ArcFace_MP_MMAAG_Baseline_MP_2021-07-23_09_34_37_epochs_2_metric.pth'
conf.security_level=3
conf.max_level=6
conf.min_level=1
assert conf.security_level >= conf.min_level and conf.security_level <= conf.max_level
# --------------------------------------------
# Default loss and optimizer
# --------------------------------------------
conf.loss = 'PartialFC'
assert conf.loss in LOSS, 'Invalid loss !!!'
conf.metric = 'ArcFace'
assert METRIC, 'Invalid metric !!!'
conf.optimizer = 'SGD'
assert conf.optimizer in OPTIMIZER, 'Invalid optimizer !!!'

# --------------------------------------------
# Default dataset
# --------------------------------------------
conf.train_dataset = ['lfw']
for train_dataset in conf.train_dataset:
    assert train_dataset in TRAIN_DATA, 'Invalid dataset !!!'
    
conf.n_classes = [ N_CLASSESE[name] for name in conf.train_dataset]

conf.db_pose = ['all']

conf.val_dataset = ['lfw']
for val_dataset in conf.val_dataset:
    assert val_dataset in VAL_DATA, 'Invalid dataset !!!'
    
conf.val_byte = True

# --------------------------------------------
# Default directory
# --------------------------------------------
conf.train_dataset_dir = [str(Path(DATA_DIR) / 'train' / name) for name in conf.train_dataset]
conf.val_dataset_dir = [str(Path(DATA_DIR) / 'validation' / name) for name in conf.val_dataset]
conf.test_dataset_dir = str(Path(DATA_DIR) / 'test')
conf.model_dir = './models'

# --------------------------------------------
# Default hyperparameters
# --------------------------------------------
conf.b = 64 # mini-batch size
conf.lr = 0.05 # learning rate
conf.k = 1
conf.sample_rate = 1.0
conf.use_ddp = False
conf.num_workers = 20
conf.num_epoch = 300 # end of epoch
conf.device = 'cuda'
conf.valid_freq = 1
conf.save_epoch = 99999
conf.matching_type = 'euclidean'
conf.data_augmentation = [
                            "RandomHorizontalFlip", 
                            "RandomGammaContrast", 
                            "RandomMotionBlur", 
                            "ISONoise", 
                            "RandomErasing"
                            ]
conf.label_smooth = False
conf.mixed_precision = True
conf.lr_scheduler = "CosineAnnealingWarmupRestarts"

# --------------------------------------------
# Data Augmentation
# --------------------------------------------
conf.img_augmenation = edict()
if "RandomGammaContrast" in conf.data_augmentation:
    conf.img_augmenation.gamma_s = (80, 120)
    conf.img_augmenation.gamma_p = 0.5
if "RandomMotionBlur" in conf.data_augmentation:
    conf.img_augmenation.blur_p = 0.5
if "ISONoise" in conf.data_augmentation:
    conf.img_augmenation.c_shift = (0, 0.05)
    conf.img_augmenation.intensity = (0, 0.3)
    conf.img_augmenation.iso_p = 0.5
if "RandomErasing" in conf.data_augmentation:
    conf.img_augmenation.erase_p = 0.5
    conf.img_augmenation.erase_min_holes = 1
    conf.img_augmenation.erase_max_holes = 1
    conf.img_augmenation.erase_max_h = 20
    conf.img_augmenation.erase_max_w = 20
    
# ===================================== Network configuration =====================================
network = edict()
# --------------------------------------------
# Residual Network (ResNet) configurations
# --------------------------------------------
# ResNet100
network.ResNet100 = edict()
network.ResNet100.network_name = 'ResNet100'
# ResNet200
network.ResNet200 = edict()
network.ResNet200.network_name = 'ResNet200'
# ResNet34
network.ResNet34 = edict()
network.ResNet34.network_name = 'ResNet34'
# ResNet50
network.ResNet50 = edict()
network.ResNet50.network_name = 'ResNet50'

# --------------------------------------------
# AlterNet configurations
# --------------------------------------------
# AlterNet101
network.AlterNet101 = edict()
network.AlterNet101.network_name = 'AlterNet101'
# AlterNet151
network.AlterNet151 = edict()
network.AlterNet151.network_name = 'AlterNet151'
# AlterNet34
network.AlterNet34 = edict()
network.AlterNet34.network_name = 'AlterNet34'
# AlterNet50
network.AlterNet50 = edict()
network.AlterNet50.network_name = 'AlterNet50'



# ================================== Loss function configuration ==================================
loss = edict()
# --------------------------------------------
# ArcFace loss configurations: s*[cos(m1*theta + m2) - m3]
# --------------------------------------------
loss.ArcFace = edict()
loss.ArcFace.loss_name = 'ArcFace'
loss.ArcFace.emd_size = 512
loss.ArcFace.loss_s = 30.0
loss.ArcFace.loss_m = 0.35
loss.ArcFace.easy_margin = False

# --------------------------------------------
# ArcFace loss configurations: s*[cos(m1*theta + m2) - m3]
# --------------------------------------------
loss.PartialFC = edict()
loss.PartialFC.loss_name = 'PartialFC'
loss.PartialFC.emd_size = 512
loss.PartialFC.loss_s = 30.0
loss.PartialFC.loss_m = 0.35
loss.PartialFC.sample_rate = 1.0
loss.PartialFC.WM_path = '.'


# ==================================== Optimizer configuration ====================================
optimizer = edict()
# --------------------------------------------
# SGD configurations
# --------------------------------------------
optimizer.SGD = edict()
optimizer.SGD.optimizer_name = 'SGD'
optimizer.SGD.wd = 0.0005
optimizer.SGD.mom = 0.9
# --------------------------------------------
# Adam configurations
# --------------------------------------------
optimizer.Adam = edict()
optimizer.Adam.optimizer_name = 'Adam'
optimizer.Adam.wd = 0.0005
optimizer.SGD.mom = 0.9
# --------------------------------------------
# AdamW configurations
# --------------------------------------------
optimizer.AdamW = edict()
optimizer.AdamW.optimizer_name = 'AdamW'
optimizer.AdamW.wd = 0.0005
optimizer.AdamW.eps = 1e-8
optimizer.AdamW.betas = (0.9, 0.999)

# ==================================== Scheduler configuration ====================================
scheduler = edict()
# --------------------------------------------
# CosineAnnealingWarmupRestarts configurations
# --------------------------------------------
scheduler.CosineAnnealingWarmupRestarts = edict()
scheduler.CosineAnnealingWarmupRestarts.warmup_steps = 5
scheduler.CosineAnnealingWarmupRestarts.min_lr = conf.lr / 1000
# --------------------------------------------
# CosineAnnealingLR configurations
# --------------------------------------------
scheduler.CosineAnnealingLR = edict()
scheduler.CosineAnnealingLR.warmup_steps = 0
scheduler.CosineAnnealingLR.min_lr = conf.lr / 1000
# --------------------------------------------
# MultiStepLR configurations
# --------------------------------------------
scheduler.MultiStep = edict()
scheduler.MultiStep.lr_decay_epoch = [8, 12, 16, 20, 40]
scheduler.MultiStep.lr_decay_ratio = 0.95
# --------------------------------------------
# StepLR configurations
# --------------------------------------------
scheduler.StepLR = edict()
scheduler.StepLR.lr_decay_epoch_size  = 500
scheduler.StepLR.lr_decay_ratio  = 0.5

# =================================================================================================
def generate_config(_network, _loss, _optimizer, _scheduler):
    for k, v in loss[_loss].items():
        conf[k] = v
    for k, v in optimizer[_optimizer].items():
        conf[k] = v
    for k, v in network[_network].items():
        conf[k] = v
    for k,v in scheduler[_scheduler].items():
        conf[k] = v

# generate_config(conf.network, conf.loss, conf.optimizer, conf.lr_scheduler)


