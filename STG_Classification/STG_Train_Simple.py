import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2, InterpolationMode
from tqdm import tqdm
import os
import datetime
import json

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Model.Classification.Swin_Transformer_v2 import SwinTransformerV2 as SwinTransformer
from STG_Classification.STG_DataLoader import load_stg_ovary_data
from sklearn.metrics import confusion_matrix



def dataloaders(batch_size=128, shuffle=True, workers=1):
    augmentations = [
        # Augmentations
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomCrop(256, padding=32),
        # v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(0.1, 0.1), interpolation=InterpolationMode.BILINEAR),
        # v2.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
    ]
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_loader, val_loader = load_stg_ovary_data(root_dir, batch_size, shuffle, augmentations=augmentations, num_workers=workers)
    return train_loader, val_loader


def train_loop(model, num_epochs, aggregation, train_loader, test_loader, criterion, optimizer):
    # Output directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M")
    output_dir = os.path.join(root_dir, 'STG_Classification', 'Output', date, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Keep track
    results = {
            'training_losses': [0] * num_epochs,
            'validation_losses': [0] * num_epochs,
            'confusion_matrices': []
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Torch compile & matmul precision
    if 'linux' in sys.platform:
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)


    for epoch in range(num_epochs):
        model.train()
        p_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader))
        for i, (x, y) in p_bar:
            optimizer.zero_grad()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_pred = model(x).squeeze(-1)
            # print(y_pred, y)
            loss = criterion(y_pred, y)
            loss.backward()
            results['training_losses'][epoch] += loss.item()
            p_bar.set_postfix({'Loss': results['training_losses'][epoch]/(i+1)})
            if (i-1) % aggregation == 0:
                optimizer.step()

        model.eval()
        all_preds = []
        all_labels = []
        with torch.inference_mode():
            p_bar = tqdm(enumerate(test_loader), desc=f"Validating", total=len(test_loader))
            for i, (x, y) in p_bar:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                y_pred = model(x).squeeze(-1)
                results['validation_losses'][epoch] += criterion(y_pred, y).item()
                predicted = y_pred > 0.    # Binary classification for logits
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        results['cms'].append(cm)
        print(f'\nEpoch {epoch} - Confusion Matrix:\n{cm}\n')

        # Save the model & losses with performance
        torch.save(model.state_dict(), os.path.join(output_dir, f'{timestamp}_Swin.pth'))
        results['training_losses'][epoch] /= len(train_loader)
        results['validation_losses'][epoch] /= len(test_loader)
        with open(os.path.join(output_dir, f'{timestamp}_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        

if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 50
    batch_size = 48
    aggregation = 1     # Number of batches to aggregate gradients
    learning_rate = 1e-3
    weight_decay = 1e-2
    
    # Load data
    train_loader, test_loader = dataloaders(batch_size=batch_size, workers=8)

    # Model
    model_args = {
        'img_size': 256,
        'patch_size': 4,
        'in_chans': 3,
        'num_classes': 1,
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
    model = SwinTransformer(**model_args)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Initialized Swin Transformer with {sum(p.numel() for p in model.parameters())/1e6}M parameters")

    # Train
    train_loop(model, num_epochs, aggregation, train_loader, test_loader, criterion, optimizer)

