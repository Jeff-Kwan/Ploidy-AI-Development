import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2, InterpolationMode
from tqdm import tqdm
import sys

from Model.Classification.Swin_Transformer import SwinTransformer


def dataloaders(batch_size=128, shuffle=True):
    train_transform = v2.Compose([
        # To Tensor
        v2.ToImage(), 
        # Augmentations
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomCrop(32, padding=4),
        # v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(0.1, 0.1), interpolation=InterpolationMode.BILINEAR),
        # v2.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),

        # Normalize
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    test_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),  v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    train_data = datasets.CIFAR10(root='./Data', train=True, download=False, transform=train_transform)
    test_data = datasets.CIFAR10(root='./Data', train=False, download=False, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader

def train_loop(model, num_epochs, train_loader, test_loader, criterion, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    for epoch in range(num_epochs):
        model.train()
        p_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader))
        for i, (x, y) in p_bar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            p_bar.set_postfix({'Loss': loss.item()})

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                y_pred = model(x)
                _, predicted = torch.max(y_pred, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        print(f'Epoch {epoch} Test Accuracy: {correct/total*100:.2f}%')

if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 10
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 1e-2
    
    # Load data
    train_loader, test_loader = dataloaders(batch_size=batch_size)

    # Model
    model_args = {
        'img_size': 32,
        'patch_size': 2,
        'in_chans': 3,
        'num_classes': 10,
        'embed_dim': 96,
        'depths': [2, 2, 2],
        'num_heads': [3, 6, 12],
        'window_size': 4,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'qk_scale': None,
        'drop_rate': 0.0,
        'drop_path_rate': 0.0,
        'norm_layer': torch.nn.LayerNorm,
        'ape': False,
        'patch_norm': True,
        'use_checkpoint': False,
        'fused_window_process': False   # False (default) learns faster at same computational speed
    }
    model = SwinTransformer(**model_args)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print(f"Initialized Swin Transformer with {sum(p.numel() for p in model.parameters())/1e6}M parameters")

    # Torch Compile
    # if 'linux' in sys.platform:
    #     model = torch.jit.script(model)

    # Train
    train_loop(model, num_epochs, train_loader, test_loader, criterion, optimizer)

