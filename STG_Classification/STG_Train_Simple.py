import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2, InterpolationMode
from tqdm import tqdm
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Model.Classification.Swin_Transformer import SwinTransformer
from STG_Classification.STG_DataLoader import load_stg_ovary_data
from sklearn.metrics import confusion_matrix



def dataloaders(batch_size=128, shuffle=True):
    augmentations = [
        # Augmentations
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        # v2.RandomCrop(32, padding=4),
        # v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(0.1, 0.1), interpolation=InterpolationMode.BILINEAR),
        # v2.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
    ]
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_loader, val_loader = load_stg_ovary_data(root_dir, batch_size, shuffle, augmentations=augmentations)
    return train_loader, val_loader


def train_loop(model, num_epochs, train_loader, test_loader, criterion, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Torch Compile
    # if 'linux' in sys.platform:
    #     model = torch.jit.script(model)


    for epoch in range(num_epochs):
        model.train()
        p_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader))
        for i, (x, y) in p_bar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred = model(x).squeeze()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            p_bar.set_postfix({'Loss': loss.item()})

        model.eval()
        all_preds = []
        all_labels = []
        with torch.inference_mode():
            p_bar = tqdm(enumerate(test_loader), desc=f"Validating", total=len(test_loader))
            for i, (x, y) in p_bar:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                _, predicted = torch.max(y_pred, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        print(f'\nEpoch {epoch} - Confusion Matrix:\n{cm}\n')

if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 20
    batch_size = 16
    learning_rate = 1e-3
    weight_decay = 1e-2
    
    # Load data
    train_loader, test_loader = dataloaders(batch_size=batch_size)

    # Model
    model_args = {
        'img_size': 512,
        'patch_size': 4,
        'in_chans': 3,
        'num_classes': 1,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 8,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'norm_layer': torch.nn.LayerNorm,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
        'fused_window_process': False   # Cannot use with torch compile
    }
    model = SwinTransformer(**model_args)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Initialized Swin Transformer with {sum(p.numel() for p in model.parameters())/1e6}M parameters")

    # Train
    train_loop(model, num_epochs, train_loader, test_loader, criterion, optimizer)

