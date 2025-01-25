'''
Training script for VisionCAT model.
'''
import torch
import torch.utils
import torchvision.transforms.v2 as v2
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
import os
from datetime import datetime

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Model.Classification.Swin_TransformerV2_Seg import SwinTransformerV2 as SwinTransformer
from STG_Segmentation.STG_DataLoader import load_stg_ovary_data
from STG_Segmentation.STG_Train import train_STG_model, crunch_STG_batch

def dataloaders(batch_size=128, shuffle=True, workers=1):
    augmentations = [
        # Augmentations
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        # v2.RandomRotation(degrees=20, interpolation=InterpolationMode.BILINEAR),
        v2.RandomCrop(1024, padding=64),
        # v2.RandomAffine(degrees=10, scale=(0.9, 1.1), shear=(0.1, 0.1), interpolation=InterpolationMode.BILINEAR),
        # v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ]
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_loader, val_loader = load_stg_ovary_data(root_dir, batch_size, shuffle, augmentations=augmentations, num_workers=workers)
    return train_loader, val_loader


class Config:
    def __init__(self, model_args):
        # Identifiers
        self.name = 'SwinTransformer'
        self.data = 'STG_ClinicDB'
        self.timestamp = datetime.now().strftime("%H-%M")
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.comments = ['Swin Transformer on STGOvary dataset.']

        # Training parameters
        self.epochs = 100
        self.batch = 32
        self.minibatch = 2
        self.shuffle = True
        self.save_steps = None
        self.learning_rate = 1e-3
        self.weight_decay = 1e-2
        self.gamma = None
        self.loss = 'BCEWithLogitsLoss'
        self.data_augmentation = {}
        self.continue_training = False

        # Model Params
        self.model_args = model_args

        # Training variables
        self.at_epoch = 0
        self.b_iter = 0
        self.train_losses = []
        self.val_losses = []


if __name__ == '__main__':
    # Model Configurations
    model_args = {
        'img_size': 1024,
        'patch_size': 4,
        'in_chans': 3,
        'out_chans': 1,
        'embed_dim': 96,
        'depths': [2, 2, 4, 2], # note smaller 3rd stage (original 6)
        'num_heads': [3, 6, 12, 24],
        'window_size': 8,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'qk_scale': None,
        'drop': 0.2,
        'drop_path_rate': 0.0,
        'norm_layer': torch.nn.LayerNorm,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
        'fused_window_process': False   # Cannot use with torch compile
    }
    manual_seed = 36
    matmul_precision = 'highest'

    # Training Configurations
    config = Config(model_args)

    # Directories
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    now = datetime.now()
    timestamp = now.strftime("%H-%M")
    date = now.strftime("%Y-%m-%d")

    # Manual Seed
    if manual_seed:
        torch.manual_seed(manual_seed)

    # Precision
    torch.set_float32_matmul_precision(matmul_precision)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'Output', date, f'{timestamp}-{config.name}-{config.data}')

    # Load data
    train_loader, val_loader = dataloaders(batch_size=config.batch, workers=8)

    # Initialize the model, loss function, and optimizer
    model = SwinTransformer(**config.model_args).to(device)
    print(f"Initialized {config.name} - Size: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6} M")

    # Training 
    # crunch_STG_batch(model, config, train_loader, output_dir, steps=config.epochs, device=device)
    model, config = train_STG_model(model, config, train_loader, val_loader, config.save_steps, output_dir, device=device)