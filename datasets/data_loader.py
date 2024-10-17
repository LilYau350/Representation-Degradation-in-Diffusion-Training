import os
import math
import random
import torch
from PIL import Image, PngImagePlugin, ImageFile
import blobfile as bf
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
import torch.distributed as dist
import h5py
from tools.dist_util import is_main_process
import torch.distributed as dist

Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * (2 ** 20)  # 1024MB
PngImagePlugin.MAX_TEXT_MEMORY = 128 * (2 ** 20)  # 128MB

# Helper functions for cropping
def center_crop_arr(arr, image_size):
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y:crop_y + image_size, crop_x:crop_x + image_size]

def random_crop_arr(arr, image_size):
    crop_y = random.randint(0, arr.shape[0] - image_size)
    crop_x = random.randint(0, arr.shape[1] - image_size)
    return arr[crop_y:crop_y + image_size, crop_x:crop_x + image_size]

# Encoded ImageNet HDF5 Dataset
class EncodedImageNet(Dataset):
    def __init__(self, h5_file, dataset_type="train", image_size=32, random_crop=False, random_flip=True):
        super().__init__()
        with h5py.File(h5_file, 'r') as f:
            self.images = f[f'{dataset_type}_latents'][:]
            self.labels = f[f'{dataset_type}_labels'][:]
        self.image_size = image_size
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # If random crop is enabled
        if self.random_crop:
            img = random_crop_arr(img, self.image_size)
        else:
            img = center_crop_arr(img, self.image_size)

        # Apply random flip
        if self.random_flip and random.random() < 0.5:
            img = np.flip(img, axis=1)

        return img, label

# ImageNet Dataset
def load_imagenet(data_dir, image_size, random_crop, random_flip):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size) if random_crop else transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

    return train_dataset, val_dataset

# CIFAR10 Dataset
def load_cifar10(data_dir, image_size, random_crop, random_flip):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size) if random_crop else transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    if is_main_process():
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        
    if dist.is_initialized():
        dist.barrier()

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)    
    
    return train_dataset, test_dataset

# LSUN Bedroom Dataset
def load_lsun_bedroom(data_dir, image_size, random_crop, random_flip):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size) if random_crop else transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_dataset = datasets.LSUN(root=data_dir, classes='bedroom_train', transform=transform)
    val_dataset = datasets.LSUN(root=data_dir, classes='bedroom_val', transform=transform)

    return train_dataset, val_dataset

# HDF5 Encoded ImageNet Loader
def load_encoded_imagenet(data_dir, image_size, random_crop, random_flip):
    h5_file = os.path.join(data_dir, 'ImageNet.h5')
    train_dataset = EncodedImageNet(h5_file=h5_file, dataset_type='train', image_size=image_size, random_crop=random_crop, random_flip=random_flip)
    val_dataset = EncodedImageNet(h5_file=h5_file, dataset_type='val', image_size=image_size, random_crop=random_crop, random_flip=random_flip)
    
    return train_dataset, val_dataset

# Unified Dataset Loader
def load_dataset(data_dir, dataset_name, batch_size=128, image_size=None, class_cond=False, deterministic=False, random_crop=False, random_flip=True, num_workers=4, shuffle=True):
    if dataset_name == 'CIFAR-10':
        train_dataset, test_dataset = load_cifar10(data_dir, image_size, random_crop, random_flip)
        input_channels = 3
        image_size = 32 if image_size is None else image_size

    elif dataset_name == 'ImageNet':
        if image_size not in [64, 128, 256]:
            raise ValueError("ImageNet's image size must be one of 64, 128, or 256.")
        train_dataset, test_dataset = load_imagenet(data_dir, image_size, random_crop, random_flip)
        input_channels = 3

    elif dataset_name == 'LSUN_Bedroom':
        train_dataset, test_dataset = load_lsun_bedroom(data_dir, image_size, random_crop, random_flip)
        input_channels = 3
        image_size = 256 if image_size is None else image_size
        
    elif dataset_name == 'Encoded_ImageNet':
        train_dataset, test_dataset = load_encoded_imagenet(data_dir, image_size, random_crop, random_flip)
        input_channels = 4  # 32x32x4 encoded ImageNet dataset
        image_size = 32 if image_size is None else image_size
        
    else:
        raise ValueError("Unsupported dataset")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    return train_loader, test_loader, input_channels, image_size
