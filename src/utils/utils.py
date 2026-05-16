import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import f1_score
import random

# --- Transforms ---
baseline_train_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    # transforms.Resize(256),       # 1. Scale shortest edge to 256 (preserves aspect ratio)
    # transforms.CenterCrop(224),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

augmented_train_transform = transforms.Compose([
    # transforms.Resize(256), 
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # add: small size scaling + crop
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomRotation(15),          
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # optional but helpful
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),       
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def setup_device():
    """Detect and return the best available device (GPU > MPS > CPU)."""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")
    return device

def setup_dataloaders(data_root, train_transform, fraction=1.0, batch_size=32, binary=False):
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=os.path.join(data_root, "val"),   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(root=os.path.join(data_root, "test"),  transform=eval_transform)

    if binary:
        target_transform = make_binary_target_transform(train_dataset.classes)
        train_dataset.target_transform = target_transform
        val_dataset.target_transform   = target_transform
        test_dataset.target_transform  = target_transform
    
    if fraction < 1.0:
        train_dataset = get_stratified_subset(train_dataset, fraction)
    
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    
    return train_loader, val_loader, test_loader

# --- Target Mapping ---
def make_binary_target_transform(classes: list[str]):
    """Map 37-breed ImageFolder labels → 0 (cat) or 1 (dog)."""
    cat_idxs = {i for i, name in enumerate(classes) if name[0].isupper()}
    return lambda idx: 0 if idx in cat_idxs else 1

# --- Training Helpers ---
def run_epoch(model, loader, optimizer, criterion, device, training: bool):
    """Run one epoch; return (avg_loss, accuracy %)."""
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, f1

def get_stratified_subset(dataset, fraction=1.0):
    """Returns a stratified subset of the dataset."""
    if fraction >= 1.0:
        return dataset
        
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    else:
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    
    n_classes = len(set(targets))
    n_samples = int(len(targets) * fraction)
    
    # One shot
    if n_samples < n_classes:
        print(f"  ⚠️ Warning: fraction={fraction} gives {n_samples} samples, but there are {n_classes} classes.")
        print(f"  ⚠️ Overriding to One-Shot Learning: selecting exactly 1 image per class ({n_classes} images total).")
        
        targets_np = np.array(targets)
        subset_indices = []
        rng = np.random.default_rng(42)
        
        for class_idx in np.unique(targets_np):
            # Find all image indices belonging to this specific breed
            class_indices = np.where(targets_np == class_idx)[0]
            # Randomly pick exactly 1 image from this breed
            chosen_idx = rng.choice(class_indices, size=1, replace=False)[0]
            subset_indices.append(chosen_idx)
            
        return Subset(dataset, subset_indices)

    # Stratified split to ensure every class maintains its ratio
    subset_indices, _ = train_test_split(
        np.arange(len(targets)),
        train_size=fraction,
        stratify=targets,
        random_state=42 # Fixed seed for reproducible experiments
    )
    
    return Subset(dataset, subset_indices)

def set_seed(seed=42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False