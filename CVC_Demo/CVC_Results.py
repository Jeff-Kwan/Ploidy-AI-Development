'''
Assumes model always outputs logits?
'''
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import random
from PIL import Image
import numpy as np


class LoadModel(nn.Module):
    def __init__(self, model_path, model_instance, device):
        super(LoadModel, self).__init__()
        self.model = model_instance
        self.model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        self.model.to(device)
        self.model.eval()
    
    def forward(self, x):
        with torch.inference_mode():
            return self.model(x)
        
def validate_model(model, val_loader, criterion, device='cuda', steps=None):
    '''Validate the model on the CVC-ClinicDB dataset.
    Returns: Accuracy, average loss, precision, recall.'''
    steps = steps if steps else len(val_loader)
    if steps < len(val_loader):
        steps = 10       # Don't validate too long!
    model.eval()
    av_loss = 0.0
    s = 0
    with torch.inference_mode():
        p_bar = tqdm(val_loader, desc="Validating")
        for images, masks in p_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, masks) # Normalize masks to [0, 1] range for BCE
            av_loss += loss.item()
            # Early stop
            s += 1
            if steps:
                if s >= steps:
                    break
    av_loss /= len(val_loader)
    return av_loss
        

def get_image_mask_pred(image, mask, output):
    # Convert logits to numpy probabilities
    # To [0, 1] range (logits if discriminative, natural range if diffusion)
    pred = torch.sigmoid(output).detach().squeeze().cpu().numpy()
    true_mask = mask.detach().cpu().squeeze().numpy()

    # Original image
    image_pil = Image.fromarray(((image.detach().squeeze().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8))  # Adjusted for [-1, 1] range
    
    # Predicted mask prediction as grescale
    predicted_int = (pred * 255).astype(np.uint8) 
    predicted_pil = Image.fromarray(predicted_int)
    
    # Overlay true mask on predicted mask with different colors for matches and mismatches
    # Green: True positive, Red: False positive, Amber: False negative
    # Ground truth is always green or amber
    # Model prediction is green or red (maximize green, minimize red)
    overlay = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    
    # True positive (correct match): Green
    overlay[:,:, 1] = (((pred > 0.5) & (true_mask == 1)) * 255).astype(np.uint8)
    
    # False positive (predicted mask=1, ground truth=0): Red
    overlay[:,:, 0] = (((pred > 0.5) & (true_mask == 0)) * 255).astype(np.uint8)
    
    # False negative (predicted=0, truth=1): Amber (Red + Green)
    overlay[:,:, 0] += (((pred <= 0.5) & (true_mask == 1)) * 255).astype(np.uint8)
    overlay[:,:, 1] += (((pred <= 0.5) & (true_mask == 1)) * 255).astype(np.uint8)
    
    overlay_pil = Image.fromarray(overlay)
    return image_pil, predicted_pil, overlay_pil


def save_results(model, val_loader, config, path, device, results=5):
    '''Run model on 5 random val images, compare results horizontally, stack 5 examples vertically'''
    random_indices = random.sample(range(len(val_loader.dataset)), results)
    fig, axes = plt.subplots(results, 3, figsize=(15, 25))

    for i, idx in enumerate(random_indices):
        image, mask = val_loader.dataset[idx]
        image = image.unsqueeze(0).to(device)
        with torch.inference_mode():
            output = model(image)

        image_pil, predicted_pil, overlay_pil = get_image_mask_pred(image, mask, output)

        # Original image
        axes[i, 0].imshow(image_pil, cmap='gray')
        axes[i, 0].set_title('Original Image', fontsize=24)
        axes[i, 0].axis('off')
        
        # Predicted mask prediction as grescale
        axes[i, 1].imshow(predicted_pil, cmap='gray')
        axes[i, 1].set_title('Predicted Probabilities', fontsize=24)
        axes[i, 1].axis('off')
        
        # Overlay true mask on predicted mask with different colors for matches and mismatches
        axes[i, 2].imshow(overlay_pil)
        axes[i, 2].set_title('Overlay (G:T, Y:FN, R:FP)', fontsize=24)
        axes[i, 2].axis('off')

    plt.suptitle(f'{config.name} Results - 5 Random Validation Images', fontsize=30)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'{config.timestamp}-{config.name}-{config.data}-Results.png'))        # Save the results
    plt.close(fig) 



def plot_losses(config, output_dir):
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(config.train_losses)+1), config.train_losses, label='Training Loss')
    plt.plot(range(1, len(config.val_losses)+1), config.val_losses, label='Validation Loss')
    plt.xlabel('Checkpoints')
    plt.ylabel('Loss')
    plt.title(f'{config.name}-{config.data} Losses Over Training')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{config.timestamp}-{config.name}-{config.data}-losses.png'))
    plt.close()
    

def plot_PRC(model, val_loader, config, path, device):
    '''Plot the PRC curve for the model on the validation set.
    Computes the AUPRC.'''
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    av_loss = 0.0
    all_labels = []
    all_preds = []
    total_correct = 0
    total_pixels = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    with torch.inference_mode():
        for images, masks in tqdm(val_loader, desc="Plot PRC:"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            outputs = model(images) # [-1, 1] target range
            loss = criterion(outputs, masks) # Normalize masks to [0, 1] range for BCE
            av_loss += loss.item()

            all_labels.append(masks.cpu().numpy().flatten())
            all_preds.append(outputs.cpu().numpy().flatten())

            # Compute predictions and accuracy
            predicted = outputs > 0.0    # Binary Mask
            total_correct += (predicted == masks).sum().item()
            total_pixels += masks.numel()

            # Compute precision and recall components
            true_positives += ((predicted == 1) & (masks == 1)).sum().item()
            false_positives += ((predicted == 1) & (masks == 0)).sum().item()
            false_negatives += ((predicted == 0) & (masks == 1)).sum().item()
        av_loss /= len(val_loader)

    all_labels = np.concatenate(all_labels).astype(int)
    all_preds = np.concatenate(all_preds).astype(int)
    accuracy = total_correct / total_pixels
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    P, R, _ = precision_recall_curve(all_labels, all_preds, drop_intermediate=True)
    auprc = auc(R, P)

    plt.figure(figsize=(8, 5))
    plt.plot(R, P, marker='.')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'{config.timestamp}-{config.name} Precision-Recall Curve (AUPRC: {auprc:.5f})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'{config.timestamp}-{config.name}-{config.data}PRC.png'))
    plt.close()
    return accuracy, precision, recall, auprc, av_loss


def save_model_desc(output_dir, config, model):
    config.save(os.path.join(output_dir, f'{config.timestamp}-{config.name}-{config.data}-config.json'))
    train_time = config.train_time
    epoch = config.at_epoch
    i = config.b_iter; l = config.tot_batches
    train_losses = config.train_losses[-1]
    val_losses = config.val_losses[-1]
    min_val_loss = min(config.val_losses)
    comments = config.comments
    
    train_time_h = int(train_time // 3600)
    train_time_m = int((train_time % 3600) // 60)
    train_time_s = int(train_time % 60)
    train_time_str = f"{train_time_h:02d}:{train_time_m:02d}:{train_time_s:02d}"

    with open(os.path.join(output_dir, f'{config.timestamp}-{config.name}-{config.data}-description.txt'), 'w') as f:
        f.write(f"Training Results for {config.name}-{config.data} on {config.date}\n\n")
        f.write(f"Saved at Epoch {epoch}, batch iteration {i}/{l}, training time: {train_time_str}\n")
        f.write(f"Model Size: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6} M\n")
        f.write(f"Final Training Loss: {train_losses:.5f}\n")
        f.write(f"Final Validation Loss: {val_losses:.5f}\n")
        f.write(f"Min Validation Loss: {min_val_loss:.5f}\n\n")
        if comments:
            for comment in comments:
                f.write(f"{comment}\n")
        f.write(f"\nConfigurations:\n")
        for key, value in config.get_dict().items():
            f.write(f"{key}: {value}\n")
        f.write(f"\n")
        f.write(f"Model: {model}")






