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
from Model.Segmentation.backbones.Swin_Transformer import SwinTransformer
from CVC_Demo.CVC_DataLoader import load_cvc_clinicdb_data
from CVC_Demo.CVC_Train import train_cvc_model, crunch_cvc_batch


def load_cvc_data(config):
    # Load data
    config.data_augmentation["horizontal_flip"] = True
    config.data_augmentation["vertical_flip"] = True
    config.data_augmentation["random_crop"] = {"height": 288, "width": 384, "padding": 16}
    config.data_augmentation["color_jitter"] = {"brightness": 0.3, "contrast": 0.2, "saturation": 0.2, "hue": 0.1}
    config.data_augmentation["affine"] = {"degrees": 30, "translate": 0.1, "scale": [0.8, 1.2], "shear": 0.1}
    DA = config.data_augmentation
    data_augmentations = [v2.RandomHorizontalFlip() if DA.get("horizontal_flip") else None,
                          v2.RandomVerticalFlip() if DA.get("horizontal_flip") else None,
                          v2.ColorJitter(brightness=DA["color_jitter"]["brightness"], contrast=DA["color_jitter"]["contrast"], saturation=DA["color_jitter"]["saturation"], hue=DA["color_jitter"]["hue"]) if DA.get("color_jitter") else None,
                          v2.RandomCrop((DA["random_crop"]["height"], DA["random_crop"]["width"]), padding=DA["random_crop"]["padding"]) if DA.get("random_crop") else None,
                          v2.RandomAffine(degrees=DA["affine"]["degrees"], translate=(DA["affine"]["translate"], 
                                        DA["affine"]["translate"]), scale=(DA["affine"]["scale"][0], 
                                        DA["affine"]["scale"][1]), shear=DA["affine"]["shear"],
                                        interpolation=InterpolationMode.BILINEAR) if DA.get("affine") else None]
    data_augmentations = [aug for aug in data_augmentations if aug is not None]  # Remove None values
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_loader, val_loader = load_cvc_clinicdb_data(root_dir, batch_size=config.batch, augmentations=data_augmentations, shuffle=True)
    return train_loader, val_loader

class Config:
    def __init__(self, model_args):
        # Identifiers
        self.name = 'SwinTransformer'
        self.data = 'CVC_ClinicDB'
        self.timestamp = datetime.now().strftime("%H-%M")
        self.date = datetime.now().strftime("%Y-%m-%d")
        self.comments = ['Swin Transformer on CVC-ClinicDB dataset.']

        # Training parameters
        self.epochs = 100
        self.batch = 8
        self.minibatch = None
        self.shuffle = True
        self.save_steps = None
        self.learning_rate = 1e-3
        self.weight_decay = 1e-2
        self.gamma = None
        self.loss = 'CrossEntropyLoss'
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
        'pretrain_img_size': 256, # irrelavant if ape false
        'patch_size': 4,
        'in_chans': 3,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'norm_layer': torch.nn.LayerNorm,
        'ape': False,
        'patch_norm': True,
        'out_indices': (0, 1, 2, 3),
        'frozen_stages': -1,
        'use_checkpoint': False,
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
    output_dir = os.path.join(root_dir, 'Output', date, f'{timestamp}-{config.name}-{config.data}')

    # Load data
    train_loader, val_loader = load_cvc_data(config)

    # Initialize the model, loss function, and optimizer
    model = SwinTransformer(**config.model_args).to(device)
    print(f"Initialized {config.name} - Size: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6} M")

    # Training 
    # crunch_cvc_batch(model, config, train_loader, output_dir, steps=config.epochs, device=device)
    model, config = train_cvc_model(model, config, train_loader, val_loader, config.save_steps, output_dir, device=device)