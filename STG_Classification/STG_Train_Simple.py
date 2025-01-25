import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2, InterpolationMode
from tqdm import tqdm
import os
import datetime
import json
import matplotlib.pyplot as plt

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
        # v2.RandomRotation(degrees=20, interpolation=InterpolationMode.BILINEAR),
        v2.RandomCrop(1024, padding=64),
        # v2.RandomAffine(degrees=10, scale=(0.9, 1.1), shear=(0.1, 0.1), interpolation=InterpolationMode.BILINEAR),
        # v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    ]
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_loader, val_loader = load_stg_ovary_data(root_dir, batch_size, shuffle, augmentations=augmentations, num_workers=workers)
    return train_loader, val_loader


def train_loop(model, num_epochs, aggregation, train_loader, val_loader, criterion, optimizer):
    # Output directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M")
    output_dir = os.path.join(root_dir, 'STG_Classification', 'Output', date, f"{timestamp}-Swin")
    os.makedirs(output_dir, exist_ok=True)

    # Keep track
    results = {
            'training_losses': [],
            'validation_losses': [],
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
        results['training_losses'].append(0)
        results['validation_losses'].append(0)
        p_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}", total=len(train_loader))
        optimizer.zero_grad(); norm = torch.tensor(0.0)
        for i, (x, y) in p_bar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_pred = model(x).squeeze(-1)
            loss = criterion(y_pred, y)
            loss.backward()
            results['training_losses'][epoch] += loss.item()
            p_bar.set_postfix({'Loss': results['training_losses'][epoch]/(i+1), 'Norm': norm.item()})
            if ((i+1) % aggregation == 0) or (i == len(train_loader)-1):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        all_preds = []
        all_labels = []
        with torch.inference_mode():
            p_bar = tqdm(enumerate(val_loader), desc=f"Validating", total=len(val_loader))
            for i, (x, y) in p_bar:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                y_pred = model(x).squeeze(-1)
                results['validation_losses'][epoch] += criterion(y_pred, y).item()
                predicted = torch.sigmoid(y_pred) > 0.5
                all_preds.extend(predicted.detach().cpu().numpy())
                all_labels.extend(y.detach().cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        results['confusion_matrices'].append(cm.tolist())
        results['training_losses'][epoch] /= len(train_loader)
        results['validation_losses'][epoch] /= len(val_loader)
        print(f'\nEpoch {epoch+1} - Validation Loss {results['validation_losses'][epoch]}; Confusion Matrix:\n{cm}\n')

        # Save the model & losses with performance
        torch.save(model.state_dict(), os.path.join(output_dir, f'{timestamp}_Swin.pth'))
        with open(os.path.join(output_dir, f'{timestamp}_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        plt.plot(results['training_losses'], label='Training Loss')
        plt.plot(results['validation_losses'], label='Validation Loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{timestamp}_losses.png'))
        plt.clf()
        

if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 100
    batch_size = 4
    aggregation = 16     # Number of batches to aggregate gradients
    learning_rate = 1e-4
    weight_decay = 1e-2
    
    # Load data
    train_loader, val_loader = dataloaders(batch_size=batch_size, workers=8)

    # Model
    model_args = {
        'img_size': 1024,
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

    # Ratio of positive : Negative samples in train labels
    positives = list(train_loader.dataset.labels.values()).count(1)
    print(f'Number of positive samples in training labels: {positives}')
    negatives = list(train_loader.dataset.labels.values()).count(0)
    print(f'Number of negative samples in training labels: {negatives}')
    ratio = negatives / positives
    print(f'Negative : Positive ratio in training labels: {ratio} -> Use as pos_weight in training BCE')
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Initialized Swin Transformer with {sum(p.numel() for p in model.parameters())/1e6}M parameters")

    # Train
    train_loop(model, num_epochs, aggregation, train_loader, val_loader, criterion, optimizer)

