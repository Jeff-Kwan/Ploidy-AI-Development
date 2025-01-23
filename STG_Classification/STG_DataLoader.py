'''
Loads data from the STGOvary dataset for training and validation.
Images: 3-channel RGB images with pixel values in the range [-1, 1]
Masks: 1-channel binary masks with values [0, 1]
'''
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as v2
import tifffile
import torch

class STGOvaryDataset(Dataset):
    def __init__(self, root_dir, indices=None, augmentation=None):
        self.root_dir = root_dir
        self.augmentation = v2.Compose(augmentation) if augmentation else None
        self.image_dir = os.path.join(root_dir, 'Data', 'STGOvary', 'Original')
        self.mask_dir = os.path.join(root_dir, 'Data', 'STGOvary', 'Ground Truth')
        
        # Efficiently list and sort image and mask files using os.scandir
        self.image_files = sorted([entry.name for entry in os.scandir(self.image_dir) if entry.is_file()])
        self.mask_files = sorted([entry.name for entry in os.scandir(self.mask_dir) if entry.is_file()])
        
        # Ensure that the number of images and masks is the same
        assert len(self.image_files) == len(self.mask_files), "Number of images and masks should be the same"
        
        # Ensure that the filenames match between images and masks
        for img_file, mask_file in zip(self.image_files, self.mask_files):
            assert os.path.splitext(img_file)[0] == os.path.splitext(mask_file)[0], \
                f"Image and mask filenames do not match: {img_file} and {mask_file}"
        
        # If indices are provided, subset the data
        if indices is not None:
            self.image_files = [self.image_files[i] for i in indices]
            self.mask_files = [self.mask_files[i] for i in indices]

        # Transformation normalisations
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.image_norm = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.ensure_mask = EnsureBinaryMask()
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        mask_name = os.path.join(self.mask_dir, self.mask_files[idx])
        
        # Read images using tifffile to handle more than 8-bit color depth
        image = tifffile.imread(img_name)
        mask = tifffile.imread(mask_name)
        
        # Apply transformations
        image, mask = self.to_tensor(image, mask)
        if self.augmentation:
            image, mask = self.augmentation(image, mask)
            mask = self.ensure_mask(mask)
        image = self.image_norm(image)
        
        return image, mask
    
class EnsureBinaryMask:
    def __call__(self, x):
        return (x > 0.5).float()


def load_cvc_clinicdb_data(root_dir, batch_size=32, shuffle=True, num_workers=1, augmentations=[None], seed=42):
    # Load the dataset
    full_dataset = STGOvaryDataset(root_dir=root_dir)

    # Calculate the sizes for train and validation splits
    train_size = int(0.8 * len(full_dataset))
    validation_size = len(full_dataset) - train_size

    # Split the dataset with a fixed random seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_subset, validation_subset = torch.utils.data.random_split(
        full_dataset, [train_size, validation_size], generator=generator)

    # Extract indices from subsets
    train_indices = train_subset.indices
    validation_indices = validation_subset.indices

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
        num_workers=num_workers, pin_memory=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=1, shuffle=shuffle, 
        num_workers=num_workers, pin_memory=True)   # Too eaasy to crash if using gradient accumulation

    return train_loader, validation_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create the data loader
    root_dir = os.path.dirname(__file__)
    train_loader, val_loader = load_cvc_clinicdb_data(root_dir)
    
    # Get the first batch
    images, masks = next(iter(val_loader))
    
    # Revert the images from [-1, 1] to [0, 1] for visualization
    images = (images + 1) / 2
    
    # Convert the first image and its mask to numpy arrays
    first_image_np = images[0].permute(1, 2, 0).numpy()
    first_mask_np = masks[0].squeeze().numpy()
    
    # Display the image and mask side by side using matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(first_image_np)
    axs[0].set_title('First Image')
    axs[0].axis('off')
    
    axs[1].imshow(first_mask_np, cmap='gray')
    axs[1].set_title('First Mask')
    axs[1].axis('off')
    
    plt.show()
