'''
Loads data from the STGOvary dataset for training and validation, 
from pre-prepared classification data file.
Data size: 2323x2323 pixels per image.
Classification Labels: 1 (with ploidy cells), 0 (no ploidy cells)
'''
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
from torchvision.transforms import InterpolationMode
import json
import torch
from PIL import Image

class STGOvaryDataset(Dataset):
    def __init__(self, root_dir, indices=None, augmentation=None):
        self.root_dir = root_dir
        self.augmentation = v2.Compose(augmentation) if augmentation else None
        self.image_dir = os.path.join(root_dir, 'Data', 'STGOvary', 'Segmentation_Data')
        
        # Efficiently list and sort image and mask files using os.scandir
        files = sorted([entry.name for entry in os.scandir(self.image_dir) if entry.is_file()])
        image_files = [file for file in files if os.path.basename(file).split('.')[0].isdigit() == True]
        
        # If indices are provided, subset the data
        if indices is not None:
            self.image_files = [image_files[i] for i in indices]
        else:
            self.image_files = image_files
        # self.mask_files = [file.replace('.png', '_mask.png') for file in self.image_files]

        # image_names = sorted([os.path.basename(file).split('.')[0] for file in self.image_files])
        # mask_names = sorted([os.path.basename(file).split('.')[0][:-5] for file in self.mask_files])
        # assert set(image_names) == set(mask_names), "Image and mask names do not match"

        # Transformation normalisations
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.image_norm = v2.Compose([v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        mask_name = img_name.replace('.png', '_mask.png')
        
        # Read jpeg images
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        
        # Apply transformations
        image, mask = self.to_tensor(image, mask)
        if self.augmentation:
            image, mask = self.augmentation(image, mask)

        image = self.image_norm(image)
        return image, mask


def load_stg_ovary_data(root_dir, batch_size=32, shuffle=True, num_workers=1, augmentations=[None], seed=42):
    # Load the dataset
    full_dataset = STGOvaryDataset(root_dir=root_dir)

    # Set the random seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(full_dataset))
    validation_size = len(full_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, validation_size], generator=generator)
    train_indices = train_dataset.indices
    validation_indices = validation_dataset.indices
    

    train_dataset = STGOvaryDataset(
        root_dir=root_dir, 
        indices=train_indices, 
        augmentation = augmentations)  # Apply augmentations
    validation_dataset = STGOvaryDataset(
        root_dir=root_dir, 
        indices=validation_indices, 
        augmentation = None)  # No augmentations for validation

    # Create data loaders for train and validation datasets with pin_memory for faster GPU transfers
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, pin_memory=True, prefetch_factor=4)  
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=shuffle, 
        num_workers=num_workers, pin_memory=True, prefetch_factor=4)

    return train_loader, validation_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create the data loader
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    train_loader, val_loader = load_stg_ovary_data(root_dir)
    
    # Get the first batch
    images, labels = next(iter(val_loader))
    
    # Convert the first image and its mask to numpy arrays
    import numpy as np
    first_image_np = images[0].permute(1, 2, 0).numpy()
    print(np.max(first_image_np), np.min(first_image_np))
    print(np.mean(first_image_np), np.std(first_image_np))

    # Revert the images from [-1, 1] to [0, 1] for visualization
    images = (images + 1) / 2
    
    # Display the image and mask side by side using matplotlib
    plt.imshow(first_image_np)
    plt.title(f'First Image with label {labels[0]}')
    plt.axis('off')
    
    plt.show()
