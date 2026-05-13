import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


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