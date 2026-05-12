import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ─────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────
# 37 class breed tree built by src/data/build_splits.py.
data_root = "data/folders"

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# pin_memory only helps on CUDA; causes a warning on MPS
pin_memory = device.type == "cuda"


# ─────────────────────────────────────────
# 2. TRANSFORMS
# ─────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────
# 3. BINARY LABEL MAPPING
# ─────────────────────────────────────────
def make_binary_target_transform(classes: list[str]):
    """Map 37-breed ImageFolder labels → 0 (cat) or 1 (dog).
    """
    cat_idxs = {i for i, name in enumerate(classes) if name[0].isupper()}
    return lambda idx: 0 if idx in cat_idxs else 1


# ─────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────
def build_model() -> nn.Module:
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


# ─────────────────────────────────────────
# 5. TRAINING HELPERS
# ─────────────────────────────────────────
def run_epoch(model, loader: DataLoader, optimizer, criterion, training: bool) -> tuple[float, float]:
    """Run one epoch; return (avg_loss, accuracy %)."""
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

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

    return total_loss / len(loader), 100.0 * correct / total


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":

    # Data loaders, point at the 37-class tree, remap labels to 0/1 on the fly
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=os.path.join(data_root, "val"),   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(root=os.path.join(data_root, "test"),  transform=eval_transform)

    to_binary = make_binary_target_transform(train_dataset.classes)
    train_dataset.target_transform = to_binary
    val_dataset.target_transform   = to_binary
    test_dataset.target_transform  = to_binary

    # num_workers=0 avoids macOS multiprocessing spawn issues
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0, pin_memory=pin_memory)

    n_cat_classes = sum(1 for c in train_dataset.classes if c[0].isupper())
    n_dog_classes = len(train_dataset.classes) - n_cat_classes
    print(f"Dataset sizes, train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    print(f"Breed classes: {len(train_dataset.classes)} total "
          f"({n_cat_classes} cat / {n_dog_classes} dog) → remapped to {{0=cat, 1=dog}}")

    # Model, optimiser, loss
    model     = build_model()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    NUM_EPOCHS   = 10
    best_val_acc = 0.0
    checkpoint_dir = "results/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = os.path.join(checkpoint_dir, "resnet34_binary_best.pth")

    print(f"\nStarting Phase 1, Binary Sanity Check on {device}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, training=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   optimizer, criterion, training=False)

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)
            print(f"  → New best val acc: {best_val_acc:.2f}% .")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")

    # Final test evaluation
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, test_acc = run_epoch(model, test_loader, optimizer, criterion, training=False)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    if test_acc >= 99.0:
        print("✓ Sanity check PASSED, target ≥99% achieved!")
    else:
        print("✗ Sanity check not yet passed")
