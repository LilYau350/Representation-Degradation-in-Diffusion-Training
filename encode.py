import argparse
from diffusers import AutoencoderKL
import torch
import os
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
import h5py
from PIL import Image, PngImagePlugin, ImageFile

Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * (2 ** 20)  # 1024MB
PngImagePlugin.MAX_TEXT_MEMORY = 128 * (2 ** 20)  # 128MB

'''
ImageNet.h5
├── train_latents  # Shape: (num_train_samples, latent_dim)
├── train_labels   # Shape: (num_train_samples,)
├── val_latents    # Shape: (num_val_samples, latent_dim)
└── val_labels     # Shape: (num_val_samples,)
'''

# Load the AutoencoderKL model
def initialize_vae():
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").cuda()
    vae.eval()  # Set model to evaluation mode
    return vae

def load_imagenet(input, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Load datasets with transformations
    train_dataset = datasets.ImageFolder(root=f"{input}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{input}/val", transform=transform)
    
    # Create DataLoader for batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def compress_batch(images, device, vae):
    images = images.to(device)
    
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.mean * 0.18215
    return latents

def save_compressed_latents(data_loader, f, dataset_name, device, vae):
    latents_dataset = None  
    labels_dataset = None
    
    for batch_idx, (images, labels) in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Compressing {dataset_name}"):
        latents = compress_batch(images, device, vae)
        
        if latents_dataset is None:
            num_latents = len(data_loader.dataset)
            latents_shape = latents.shape[1:]  # e.g., (D,)
            
            latents_dataset = f.create_dataset(
                f'{dataset_name}_latents', (num_latents, *latents_shape), dtype='float32'
            )
            
            labels_dataset = f.create_dataset(
                f'{dataset_name}_labels', (num_latents,), dtype='int64'  
            )
        
        start_idx = batch_idx * data_loader.batch_size
        end_idx = start_idx + latents.size(0)
        
        latents_dataset[start_idx:end_idx] = latents.cpu().numpy()
        labels_dataset[start_idx:end_idx] = labels.cpu().numpy()
        f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder Image Compression/Decompression")
    parser.add_argument("--input", type=str, required=True, help="Input folder path or latent file")
    parser.add_argument("--output", type=str, required=True, help="Output folder path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing images")
    parser.add_argument("--image_size", type=int, default=256, help="Image size for processing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the VAE model once at the start
    vae = initialize_vae()

    # Load train and val datasets
    train_loader, val_loader = load_imagenet(args.input, args.image_size, args.batch_size)

    # Save compressed latents and labels to HDF5
    h5_file = os.path.join(args.output, "ImageNet.h5")
    with h5py.File(h5_file, 'w') as f:
        save_compressed_latents(train_loader, f, "train", device, vae)
        save_compressed_latents(val_loader, f, "val", device, vae)
