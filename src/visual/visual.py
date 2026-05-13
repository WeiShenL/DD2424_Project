import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import os


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def visualize_batch(dataloader, class_names, num_images=8, title_prefix="Batch"):
    """
    Visualize a batch of images from a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        class_names: List of class names
        num_images: Number of images to display (default: 8)
        title_prefix: Prefix for the title (default: "Batch")
    """
    inputs, labels = next(iter(dataloader))
    out = torchvision.utils.make_grid(inputs[:num_images])
    titles = [class_names[x] for x in labels[:num_images]]
    imshow(out, title=f"{title_prefix}: {', '.join(titles)}")
    plt.show()


def visualize_predictions(model, dataloader, class_names, num_images=8, device='cpu'):
    """
    Visualize model predictions vs ground truth on a batch.
    
    Args:
        model: PyTorch model
        dataloader: PyTorch DataLoader
        class_names: List of class names
        num_images: Number of images to display (default: 8)
        device: Device to run model on ('cpu', 'cuda', or 'mps')
    """
    model.eval()
    inputs, labels = next(iter(dataloader))
    inputs = inputs.to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
    
    out = torchvision.utils.make_grid(inputs[:num_images].cpu())
    titles = [f"GT: {class_names[labels[i]]}\nPred: {class_names[predictions[i]]}" 
              for i in range(min(num_images, len(labels)))]
    imshow(out, title="Predictions vs Ground Truth")
    plt.show()


def save_batch_images(dataloader, class_names, save_path, num_images=8):
    """
    Save a batch of images to disk.
    
    Args:
        dataloader: PyTorch DataLoader
        class_names: List of class names
        save_path: Path to save the image grid
        num_images: Number of images to save (default: 8)
    """
    inputs, labels = next(iter(dataloader))
    out = torchvision.utils.make_grid(inputs[:num_images])
    titles = [class_names[x] for x in labels[:num_images]]
    imshow(out, title=f"Batch: {', '.join(titles)}")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Batch image saved to {save_path}")
    plt.close()

def plot_training_history(history, save_dir="results/Images", filename="training_history.png"):
    """
    Plots training and validation accuracy and loss, and saves the plot.
    
    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists.
        save_dir (str): Directory to save the plot.
        filename (str): Name of the saved image file.
    """
    # Ensure the target directory exists
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)

    # Create a figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue', marker='o', markersize=4)
    ax1.plot(epochs, history['val_loss'], label='Val Loss', color='red', marker='x', markersize=4)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Accuracy
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', color='blue', marker='o', markersize=4)
    ax2.plot(epochs, history['val_acc'], label='Val Accuracy', color='red', marker='x', markersize=4)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n📈 Training history plot saved to {save_path}")
    plt.close()