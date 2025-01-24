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
        self.image_dir = os.path.join(root_dir, 'Data', 'STGOvary', 'Filtered_Classification_Data')
        
        # Efficiently list and sort image and mask files using os.scandir
        image_files = sorted([entry.name for entry in os.scandir(self.image_dir) if entry.is_file() and entry.name.endswith('.png')])
        labels = json.load(open(os.path.join(self.image_dir, 'labels.json')))
        
        # If indices are provided, subset the data
        if indices is not None:
            self.image_files = [image_files[i] for i in indices]
            self.labels = {image_files[i].split('.')[0]: labels[str(i)] for i in indices}
        else:
            self.image_files = image_files
            self.labels = labels

        # Transformation normalisations
        self.to_tensor = v2.Compose([v2.ToImage(), 
                                    #  v2.Resize(1024, interpolation=InterpolationMode.BILINEAR),
                                     v2.ToDtype(torch.float32, scale=True)
                                     ])
        self.image_norm = v2.Compose([v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        img_idx = self.image_files[idx].split('.')[0]
        
        # Read jpeg images
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.labels[img_idx], dtype=torch.float32)
        
        # Apply transformations
        image = self.to_tensor(image)
        if self.augmentation:
            image = self.augmentation(image)
        image = self.image_norm(image)
        
        return image, label


def load_stg_ovary_data(root_dir, batch_size=32, shuffle=True, num_workers=1, augmentations=[None], seed=42):
    # Load the dataset
    full_dataset = STGOvaryDataset(root_dir=root_dir)

    # Extract indices for positive and negative samples
    positive_indices = [int(i) for i, label in full_dataset.labels.items() if label == 1]
    negative_indices = [int(i) for i, label in full_dataset.labels.items() if label == 0]

    # Set the random seed for reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Randomly select 50 positives and 50 negatives for validation
    validation_positive_indices = torch.tensor(positive_indices).tolist()
    validation_negative_indices = torch.tensor(negative_indices).tolist()
    validation_positive_indices = torch.utils.data.random_split(validation_positive_indices, [50, len(validation_positive_indices) - 50], generator=generator)[0]
    validation_negative_indices = torch.utils.data.random_split(validation_negative_indices, [50, len(validation_negative_indices) - 50], generator=generator)[0]

    # Combine validation indices
    validation_indices = validation_positive_indices + validation_negative_indices

    # The rest are for training
    train_indices = list(set(range(len(full_dataset))) - set(validation_indices))

    # print number of positives and negatives in training and validation sets
    # train_labels = [full_dataset.labels[str(i)] for i in train_indices]
    # validation_labels = [full_dataset.labels[str(i)] for i in validation_indices]
    # print(f"Train: {train_labels.count(1)} positives, {train_labels.count(0)} negatives")
    # print(f"Validation: {validation_labels.count(1)} positives, {validation_labels.count(0)} negatives")

    train_dataset = STGOvaryDataset(
        root_dir=root_dir, 
        indices=train_indices, 
        augmentation = augmentations)  # Apply augmentations
    validation_dataset = STGOvaryDataset(
        root_dir=root_dir, 
        indices=validation_indices, 
        augmentation = None)  # No augmentations for validation

    # Print number of positives and negatives in the train and validation dataset labels
    # train_dataset_labels = list(train_dataset.labels.values())
    # validation_dataset_labels = list(validation_dataset.labels.values())
    # print(f"Train: {train_dataset_labels.count(1)} positives, {train_dataset_labels.count(0)} negatives")
    # print(f"Validation: {validation_dataset_labels.count(1)} positives, {validation_dataset_labels.count(0)} negatives")
    # exit()

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
