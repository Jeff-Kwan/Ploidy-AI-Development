import torch
from torch import nn
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
from STG_Classification.STG_DataLoader import load_stg_ovary_data
from sklearn.metrics import confusion_matrix



def dataloaders(batch_size=128, shuffle=True, workers=1):
    augmentations = [
        # Augmentations
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomCrop(1024, padding=32),
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
    output_dir = os.path.join(root_dir, 'STG_Classification', 'Output', date, f"{timestamp}-SimpleCNN")
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
        p_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader))
        for i, (x, y) in p_bar:
            optimizer.zero_grad()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_pred = model(x).squeeze(-1)
            loss = criterion(y_pred, y)
            loss.backward()
            results['training_losses'][epoch] += loss.item()
            p_bar.set_postfix({'Loss': results['training_losses'][epoch]/(i+1)})
            if (i+1) % aggregation == 0:
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
                predicted = torch.sigmoid(y_pred) > 0.5
                all_preds.extend(predicted.detach().cpu().numpy())
                all_labels.extend(y.detach().cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        results['confusion_matrices'].append(cm.tolist())
        print(f'\nEpoch {epoch} - Confusion Matrix:\n{cm}\n')

        # Save the model & losses with performance
        torch.save(model.state_dict(), os.path.join(output_dir, f'{timestamp}_SimpleCNN.pth'))
        results['training_losses'][epoch] /= len(train_loader)
        results['validation_losses'][epoch] /= len(test_loader)
        with open(os.path.join(output_dir, f'{timestamp}_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        plt.plot(results['training_losses'], label='Training Loss')
        plt.plot(results['validation_losses'], label='Validation Loss')
        plt.legend()
        plt.title('Training and Validation Losses')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{timestamp}_losses.png'))
        plt.clf()
        
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.SiLU = nn.SiLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = self.bn(x + self.conv2(self.SiLU(self.conv1(x))))
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(3, 16, 1, 1, 0), ResidualBlock(16))
        self.convs = nn.ModuleList([
            ResidualBlock(32),
            ResidualBlock(64),
            ResidualBlock(128),
        ])
        self.downs = nn.ModuleList([
            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0, bias=False),
        ])
        self.acts = nn.ModuleList([nn.SiLU() for _ in range(len(self.convs))])
        self.norms = nn.ModuleList([nn.BatchNorm2d(32*2**i) for i in range(len(self.convs))])
        self.fc = nn.Linear(128, 1) 

    def forward(self, x):
        x = self.in_conv(x)
        for conv, down, act, norm in zip(self.convs, self.downs, self.acts, self.norms):
            x = down(x)
            x = norm(x + act(conv(x)))
        x = self.fc(torch.mean(x, dim=[2, 3]))
        return x

if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 100
    batch_size = 8
    aggregation = 2     # Number of batches to aggregate gradients
    learning_rate = 1e-3
    weight_decay = 1e-2
    
    # Load data
    train_loader, test_loader = dataloaders(batch_size=batch_size, workers=8)

    # Model
    model = SimpleCNN()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Initialized SimpleCNN with {sum(p.numel() for p in model.parameters())/1e6}M parameters")

    # Train
    train_loop(model, num_epochs, aggregation, train_loader, test_loader, criterion, optimizer)

