
from pathlib import Path
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as alb
import albumentations.pytorch as alp
from utils.logger import print_log
import bcolz
from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.folder import DatasetFolder, default_loader, IMG_EXTENSIONS
import time
from torch.utils.data.distributed import DistributedSampler


class VAL_DATASET_BYTE(Dataset):
    def __init__(self, data_dir: str, conf: str=None):
        super().__init__()
        
        self.pair_arr = bcolz.carray(rootdir = data_dir, mode='r')
        N, C, H, W = np.shape(self.pair_arr)
        self.pair_arr = np.reshape(self.pair_arr, [N//2, 2, C, H, W])
        
        self.label_arr = np.load(f'{data_dir}_list.npy')
        permute = list(range(len(self.label_arr)))
        random.shuffle(permute)
        self.pair_arr = self.pair_arr[permute]
        self.label_arr = self.label_arr[permute]
        
        assert np.shape(self.pair_arr)[0] == np.shape(self.label_arr)[0], 'Not match size of patch and label !!!'
        
    def __len__(self):
        return len(self.pair_arr)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        pair = torch.Tensor(self.pair_arr[idx])
        
        label = self.label_arr[idx]
        
        return pair, label
    
    
class CustomImageFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        conf,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            str(Path(root) / 'imgs'),
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            is_valid_file=is_valid_file,
        )
        self.conf = conf
        self.imgs = self.samples
        self.transform = self.make_transform(conf)
        
        msg_aug = '='*50 + '\n'
        msg_aug += f'* Data Augmentation *\n\n'
        msg_aug += f'{self.transform}\n'
        msg_aug += '='*50 + '\n'
        print(msg_aug)
        
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = self.transform(image=np.array(sample))["image"]
        
        return (sample, target)
    
    
    def __len__(self) -> int:
        return len(self.samples)
    
    
    def make_transform(self, conf):
        img_transform = list()
        
        if "RandomGammaContrast" in conf.data_augmentation:
            img_transform.append(alb.RandomGamma(gamma_limit=conf.img_augmenation.gamma_s, p=conf.img_augmenation.gamma_p))
        
        if "RandomMotionBlur" in conf.data_augmentation:
            img_transform.append(alb.MotionBlur(p=conf.img_augmenation.blur_p))  
        
        if "ISONoise" in conf.data_augmentation:
            img_transform.append(alb.ISONoise(p=conf.img_augmenation.iso_p, color_shift=conf.img_augmenation.c_shift, intensity=conf.img_augmenation.intensity))
        
        img_transform.append(alb.Resize(112, 112))
        
        if "RandomHorizontalFlip" in conf.data_augmentation:
            img_transform.append(alb.HorizontalFlip()) 
        
        img_transform.append(alb.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        
        if "RandomErasing" in conf.data_augmentation:
            img_transform.append(alb.CoarseDropout(p=conf.img_augmenation.erase_p,
                                                    min_holes=conf.img_augmenation.erase_min_holes, 
                                                    max_holes=conf.img_augmenation.erase_max_holes,
                                                    max_height=conf.img_augmenation.erase_max_h, 
                                                    max_width=conf.img_augmenation.erase_max_w))
        
        img_transform.append(alp.transforms.ToTensorV2())
        
        img_transform = alb.Compose(img_transform)
        
        return img_transform
    
    
    
class DATA_Module:
    
    def __init__(self, conf: str=None, logger: str=None):
        self.conf = conf
        self.logger_ = logger
        self.n_classes = list()

    def setup(self, stage: str='train'):

        if stage == "train":
            
            # ==================
            # Train Dataset
            # ==================
            
            self.train_dataset = list()
            self.train_dataset_name = list()
            msg = '='*50 + '\n'
            msg += '* Train Dataset info *\n'
            
            start_t = time.time()
            idx = 0
            for data_dir in self.conf.train_dataset_dir:
                train_dataset = CustomImageFolder(root=data_dir, conf=self.conf)
                self.train_dataset.append(train_dataset)
                self.n_classes.append(self.conf.n_classes[idx])
                self.train_dataset_name.append(Path(data_dir).name)                
                
                msg += f'- The Number of Training Images of "{self.train_dataset_name[-1]}": {self.train_dataset[-1].__len__()}\n'
                msg += f'- The Number of Training Classes of "{self.train_dataset_name[-1]}": {self.n_classes[-1]} \n' 
                
                idx += 1
                
            print(msg) if self.logger_ is None else print_log(self.logger_, msg)
            print_log(self.logger_, f"Loading time: {time.time() - start_t:.4f}s\n")
            
        elif stage == 'val':
            # ==================
            # Validation Dataset
            # ==================
            
            self.val_dataset = list()
            self.val_dataset_name = list()
            msg = '* Validation Dataset info *\n'
            
            start_t = time.time()
            for data_dir in self.conf.val_dataset_dir:
                self.val_dataset.append(VAL_DATASET_BYTE(data_dir=data_dir, conf=self.conf))
                self.val_dataset_name.append(Path(data_dir).name)
                
                msg += f'- The Number of Validation Pairs of "{self.val_dataset_name[-1]}": {self.val_dataset[-1].__len__()} ' + '\n'
            
            print(msg) if self.logger_ is None else print_log(self.logger_, msg)
            print_log(self.logger_, f"Loading time: {time.time() - start_t:.4f}s")
            print_log(self.logger_, '='*50 + '\n')
            

    def train_dataloader(self):
        train_loader_list = list()
        train_sampler_list = list()
        for train_datatset in self.train_dataset:
            train_sampler = DistributedSampler(train_datatset)
            train_sampler_list.append(train_sampler)
            train_loader_list.append(DataLoader(train_datatset, batch_size=self.conf.b, num_workers=self.conf.num_workers, 
                                                shuffle=False, pin_memory=True, drop_last=True, sampler=train_sampler))
        
        return train_loader_list, train_sampler_list
    
    def val_dataloader(self):
        val_loader_list = list()
        for val_datatset in self.val_dataset:
            val_loader_list.append(DataLoader(val_datatset, batch_size=self.conf.b, num_workers=self.conf.num_workers, shuffle=False, pin_memory=False))
        
        return val_loader_list

