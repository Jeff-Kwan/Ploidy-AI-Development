import torch
from time import time
from tqdm import tqdm
import os, sys
import matplotlib.pyplot as plt

from CVC_Demo.CVC_Results import validate_model, plot_losses, save_model_desc, save_results, plot_PRC, get_image_mask_pred

def get_loss(loss):
    if loss == 'BCEWithLogitsLoss':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif loss == 'CrossEntropyLoss':
        criterion = torch.nn.CrossEntropyLoss()
    elif loss == 'MSELoss':
        criterion = torch.nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss function {loss} not implemented.")
    return criterion

def model_forward(model, batch_x, batch_y, criterion, device):
    '''Forward pass of the model'''
    logits = model(batch_x)
    print([l.size() for l in logits], batch_y.size())
    exit()
    loss = criterion(logits, batch_y)
    if model.training:
        loss.backward()
    return model, loss


def print_fro_stats(model):
    W = model.embed.weight
    W = W.view(W.size(0), -1)
    sv = torch.linalg.svdvals(W)
    frob_norm = torch.linalg.norm(W, ord='fro')
    print('Embedding:', round(frob_norm.item(),5), round(torch.mean(sv).item(),5), 
            round(torch.max(sv).item(),5), round(torch.min(sv).item(),5))
    W = model.out.weight
    W = W.view(W.size(0), -1)
    sv = torch.linalg.svdvals(W)
    frob_norm = torch.linalg.norm(W, ord='fro')
    print('Out Proj:', round(frob_norm.item(),5), round(torch.mean(sv).item(),5), 
            round(torch.max(sv).item(),5), round(torch.min(sv).item(),5))
    return

def train_cvc_model(model, config, train_loader, val_loader, save_steps, output_dir, device='cuda'):
    '''Train the segmentation model on the STGOvary dataset.
    Returns: Trained model, training losses, validation losses, training time.'''
    # Admin work
    os.makedirs(output_dir, exist_ok=True)
    assert config.minibatch is None or config.batch > config.minibatch or config.batch % config.minibatch == 0, "Invalid batch-minibatch configuration."

    criterion = get_loss(config.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader), gamma=config.gamma) if config.gamma else None
    # Consider ReduceLRonPlateau for more advanced learning rate scheduling

    # Compile the model
    if sys.platform.startswith('linux'):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Train the model
    config.tot_batches = len(train_loader)
    config.comments.append('')
    start_time = time()
    for epoch in range(config.epochs - config.at_epoch):
        running_loss = 0.0
        epoch_loss = 0.0
        i = 0.0
        config.b_iter = 0
        config.at_epoch += 1
        p_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_x, batch_y in p_bar:
            model.train()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache() if device == 'cuda' else None
            # Minibatching
            if config.minibatch:
                for mb in range((len(batch_x)+config.minibatch-1)//config.minibatch):
                    mini_x = batch_x[mb*config.minibatch:(mb+1)*config.minibatch]
                    mini_y = batch_y[mb*config.minibatch:(mb+1)*config.minibatch]
                    model, loss = model_forward(model, mini_x, mini_y, criterion, device)
            else:
                model, loss = model_forward(model, batch_x, batch_y, criterion, device)

            # Clip norm and backpropagate
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            scheduler.step()    if config.gamma else None

            p_bar.set_postfix({'Norm': norm.item(), 'Loss': loss.item()})
            i += 1
            config.b_iter += 1
            if save_steps and (config.b_iter < config.tot_batches):
                if i == save_steps:
                    torch.save(model.state_dict(), os.path.join(output_dir, f'{config.timestamp}-{config.name}-{config.data}-model.pt'))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, f'{config.timestamp}-{config.name}-{config.data}-optimizer.pt')) if config.continue_training else None
                    config.train_losses.append(running_loss / i)
                    accuracy, precision, recall, auprc, av_loss = plot_PRC(model, val_loader, config, output_dir, device)
                    config.train_time += time() - start_time ; start_time = time()
                    config.val_losses.append(av_loss)
                    config.comments[-1] = f"Accuracy: {accuracy*100:.3f}%, Precision: {precision:.3f}, Recall: {recall:.3f}, AUPRC: {auprc:.5f}, Average Loss: {av_loss:.5f}"
                    save_model_desc(output_dir, config, model)
                    plot_losses(config, output_dir)
                    save_results(model, val_loader, config, output_dir, device, results=5)
                    running_loss = 0
                    i = 0
                    print_fro_stats(model)
        
        torch.save(model.state_dict(), os.path.join(output_dir, f'{config.timestamp}-{config.name}-{config.data}-model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, f'{config.timestamp}-{config.name}-{config.data}-optimizer.pt')) if config.continue_training else None
        config.train_losses.append(running_loss / i)
        accuracy, precision, recall, auprc, av_loss = plot_PRC(model, val_loader, config, output_dir, device)
        config.train_time += time() - start_time ; start_time = time()
        config.val_losses.append(av_loss)
        config.comments[-1] = f"Accuracy: {accuracy*100:.3f}%, Precision: {precision:.3f}, Recall: {recall:.3f}, AUPRC: {auprc:.5f}, Average Loss: {av_loss:.5f}"
        save_model_desc(output_dir, config, model)
        plot_losses(config, output_dir)
        save_results(model, val_loader, config, output_dir, device, results=5)
        torch.cuda.empty_cache() if device == 'cuda' else None
        tqdm.write(f"Epoch [{epoch+1}/{config.epochs}], Loss: {config.train_losses[-1]:.5f}, Val Loss: {config.val_losses[-1]:.5f}")
        print_fro_stats(model)

    print(f"Trained {config.name} Model Size: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    print(f"Achieved Accuracy: {accuracy*100:.3f}%, Precision: {precision:.3f}, Recall: {recall:.3f}, AUPRC: {auprc:.5f}, Average Loss: {av_loss:.5f}")
    return model, config


def crunch_cvc_batch(model, config, train_loader, path, steps=50, device='cuda'):
    '''Crunch a single batch of images and masks through the model.'''
    os.makedirs(path, exist_ok=True)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None
    criterion = get_loss(config.loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=(device=='cuda'))

    print("Crunching through the first sample!\n")
    images, masks = next(iter(train_loader))
    images = images[:1].to(device, non_blocking=True)
    masks = masks[:1].to(device, non_blocking=True)
    start_time = time()
    for i in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # Forward pass with mixed precision

        #with torch.amp.autocast(device_type=str(device), enabled=config.autocast):
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(f"Iter {i+1}/{steps}, Norm: {norm.item():.3f}, Loss: {loss.item():.5f}")
    print(f"Time taken: {time()-start_time:.5f} s")
    print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1e6:.2f} MB") if torch.cuda.is_available() else None

    # Show crunched image(s)
    B = len(images)
    fig, axes = plt.subplots(B, 3, figsize=(12, 6))
    for i in range(B):
        image, pred, overlay = get_image_mask_pred(images[0], masks[0], outputs[0])
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image', fontsize=24)
        axes[0].axis('off')
        
        # Predicted mask prediction as grescale
        axes[1].imshow(pred, cmap='gray')
        axes[1].set_title('Predicted Probabilities', fontsize=24)
        axes[1].axis('off')
        
        # Overlay true mask on predicted mask with different colors for matches and mismatches
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (G:T, Y:FN, R:FP)', fontsize=24)
        axes[2].axis('off')
        plt.suptitle(f'{config.name} Crunching Results', fontsize=30)
    plt.tight_layout()
    plt.savefig(os.path.join(path, f'{config.timestamp}-{config.name}-{config.data}-Crunch.png'))        # Save the results
    plt.close(fig) 
    