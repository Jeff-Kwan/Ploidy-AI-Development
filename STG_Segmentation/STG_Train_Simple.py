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
from Model.Classification.Swin_TransformerV2_Seg import SwinTransformerV2 as SwinTransformer
from STG_Segmentation.STG_DataLoader import load_stg_ovary_data
from torcheval.metrics import BinaryConfusionMatrix


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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Output directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%H-%M")
    output_dir = os.path.join(root_dir, 'STG_Segmentation', 'Output', date, f"{timestamp}-Swin")
    os.makedirs(output_dir, exist_ok=True)

    # Keep track
    results = {
            'training_losses': [],
            'validation_losses': [],
            'confusion_matrices': [],
            'F1_scores': []
        }
    confusion_matrix = BinaryConfusionMatrix(device=device)

    model.to(device)

    # Torch compile & matmul precision
    if 'linux' in sys.platform:
        print("Compiling model...")
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
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            results['training_losses'][epoch] += loss.item()
            p_bar.set_postfix({'Loss': results['training_losses'][epoch]/(i+1), 'Norm': norm.item()})
            if ((i+1) % aggregation == 0) or (i == len(train_loader)-1):
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                break

        model.eval()
        cm_list = []
        with torch.inference_mode():
            p_bar = tqdm(enumerate(val_loader), desc=f"Validating", total=len(val_loader))
            for i, (x, y) in p_bar:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                y_pred = model(x)
                results['validation_losses'][epoch] += criterion(y_pred, y).item()
                predicted = torch.sigmoid(y_pred) > 0.5
                confusion_matrix.update(predicted.flatten(), y.flatten())
                cm = confusion_matrix.compute()
                cm_list.append(cm)

        # Save model and results
        torch.save(model.state_dict(), os.path.join(output_dir, f'{timestamp}_Swin.pth'))
        results['training_losses'][epoch] /= len(train_loader)
        results['validation_losses'][epoch] /= len(val_loader)
        cm = sum(cm_list) / len(cm_list)
        results['confusion_matrices'].append(cm.cpu().numpy().tolist())
        results['F1_scores'].append(F1_score(cm))
        print(f'\nEpoch {epoch+1} - Validation Loss {results['validation_losses'][epoch]}; Confusion Matrix:\n{cm}\n')
        save_results(results, output_dir, timestamp)


def save_results(results, output_dir, timestamp):
    # Results JSON
    with open(os.path.join(output_dir, f'{timestamp}_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(results['training_losses'], label='Training Loss', color='tab:blue')
    ax1.plot(results['validation_losses'], label='Validation Loss', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('F1 Score', color='tab:green')
    ax2.plot(results['F1_scores'], label='F1 Score', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower left')
    plt.title('Training and Validation Losses and F1 Score')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{timestamp}_metrics.png'))
    plt.clf()
        
def F1_score(cm):
    precision = cm[1,1] / (cm[1,1] + cm[0,1])
    recall = cm[1,1] / (cm[1,1] + cm[1,0])
    return 2 * precision * recall / (precision + recall)

if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 100
    batch_size = 2
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
        'out_chans': 1,
        'embed_dim': 96,
        'depths': [2, 2, 6, 2], # note smaller 3rd stage (original 6)
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
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([[[5.0]]], device=torch.device('cuda')))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Initialized Swin Transformer with {sum(p.numel() for p in model.parameters())/1e6}M parameters")

    # Train
    train_loop(model, num_epochs, aggregation, train_loader, val_loader, criterion, optimizer)

