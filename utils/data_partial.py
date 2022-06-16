
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


# class TEST_DATASET(Dataset):
#     def __init__(self, data_dir: str, transform=None):
#         super().__init__()
        
#         self.data_info = pd.read_csv(str(Path(data_dir / 'suprema.csv')), header=None)
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_arr = np.asarray(self.data_info.iloc[:, 0])
#         self.label_arr = np.asarray(self.data_info.iloc[:, 1])


#     def __len__(self):
#         return len(self.data_info)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         # image
#         img_path = Path(self.root_dir, self.image_arr[idx])
#         image = io.imread(str(img_path))
#         image = self.transform(image)

#         #label
#         label = self.label_arr[idx]

#         return image, label

    


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

        print(msg_aug)  if conf.local_rank == 0 else None
        
        
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
            
            msg = '='*50 + '\n'
            msg += '* Train Dataset info *\n'
            
            start_t = time.time()

            self.train_dataset = CustomImageFolder(root=self.conf.train_dataset_dir, conf=self.conf)
            self.n_classes = self.conf.n_classes
            self.train_dataset_name = Path(self.conf.train_dataset_dir).name   
                
            msg += f'- The Number of Training Images of "{self.train_dataset_name}": {self.train_dataset.__len__()}\n'
            msg += f'- The Number of Training Classes of "{self.train_dataset_name}": {self.n_classes} \n' 
                

            if self.conf.local_rank == 0:
                if self.logger_ is None:
                    print(msg)   
                    print(f"Loading time: {time.time() - start_t:.4f}s\n")   
                    
                else:
                    print_log(self.logger_, msg)
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
            
            if self.conf.local_rank == 0:
                if self.logger_ is None:
                    print(msg)   
                    print(f"Loading time: {time.time() - start_t:.4f}s")   
                    print('='*50 + '\n')   

                else:
                    print_log(self.logger_, msg)
                    print_log(self.logger_, f"Loading time: {time.time() - start_t:.4f}s")
                    print_log(self.logger_, '='*50 + '\n')

        elif stage == 'test':
            # ==================
            # Test Dataset
            # ==================
            
            self.test_dataset = list()
            self.test_dataset_name = list()
            msg = '* Test Dataset info *\n'
            
            start_t = time.time()
            for data_dir in self.conf.test_dataset_dir:
                self.test_dataset.append(VAL_DATASET_BYTE(data_dir=data_dir, conf=self.conf))
                self.test_dataset_name.append(Path(data_dir).name)
                
                msg += f'- The Number of Test Pairs of "{self.test_dataset_name[-1]}": {self.test_dataset[-1].__len__()} ' + '\n'
            
        
            if self.logger_ is None:
                print(msg)   
                print(f"Loading time: {time.time() - start_t:.4f}s")   
                print('='*50 + '\n')   

            else:
                print_log(self.logger_, msg)
                print_log(self.logger_, f"Loading time: {time.time() - start_t:.4f}s")
                print_log(self.logger_, '='*50 + '\n')
            

    def train_dataloader(self):
        
        train_sampler = DistributedSampler(self.train_dataset)
        
        train_loader = DataLoader(  self.train_dataset, batch_size=self.conf.b, num_workers=self.conf.num_workers, 
                                    shuffle=False, pin_memory=True, drop_last=True, sampler=train_sampler)
        
        return train_loader, train_sampler
    
    def val_dataloader(self):
        val_loader_list = list()
        for val_datatset in self.val_dataset:
            val_loader_list.append(DataLoader(val_datatset, batch_size=self.conf.b, num_workers=self.conf.num_workers, 
                                                shuffle=False, pin_memory=True))
        
        return val_loader_list
    
    def test_dataloader(self):
        test_loader_list = list()
        for test_datatset in self.test_dataset:
            test_loader_list.append(DataLoader( test_datatset, batch_size=1000, num_workers=self.conf.num_workers, 
                                                shuffle=False, pin_memory=True))
        
        return test_loader_list

