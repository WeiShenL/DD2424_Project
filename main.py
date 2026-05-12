import os
import shutil

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ─────────────────────────────────────────
# 1. SETUP
# ─────────────────────────────────────────
data_root  = "Data/Data/images"
output_dir = "Data/Data/binary_task"

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
# 3. DATA ORGANISATION  (runs only once)
# ─────────────────────────────────────────
def create_binary_split(csv_path: str, subset_name: str) -> None:
    """Copy images into binary_task/<subset>/{cat,dog}/ folders."""
    dest_root = os.path.join(output_dir, subset_name)
    if os.path.exists(dest_root):
        print(f"[skip] {dest_root} already exists.")
        return

    df = pd.read_csv(csv_path)
    copied, missing = 0, 0
    for _, row in df.iterrows():
        img_name   = f"{row['name']}.jpg"
        source     = os.path.join(data_root, img_name)
        label      = "cat" if int(row['species']) == 1 else "dog"
        target_dir = os.path.join(dest_root, label)
        os.makedirs(target_dir, exist_ok=True)
        if os.path.exists(source):
            shutil.copy(source, os.path.join(target_dir, img_name))
            copied += 1
        else:
            missing += 1
    print(f"[{subset_name}] copied={copied}, missing={missing}")


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
# MAIN — required on macOS to avoid
#         multiprocessing spawn errors
# ─────────────────────────────────────────
if __name__ == "__main__":

    # Data organisation
    create_binary_split("splits/train.csv", "train")
    create_binary_split("splits/val.csv",   "val")
    create_binary_split("splits/test.csv",  "test")
    print("Binary split complete.")

    # Data loaders
    # num_workers=0 avoids macOS multiprocessing spawn issues
    train_dataset = datasets.ImageFolder(root=os.path.join(output_dir, "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(root=os.path.join(output_dir, "val"),   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(root=os.path.join(output_dir, "test"),  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0, pin_memory=pin_memory)

    print(f"Dataset sizes — train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # Model, optimiser, loss
    model     = build_model()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    NUM_EPOCHS   = 10
    best_val_acc = 0.0
    checkpoint   = "resnet34_binary_best.pth"

    print(f"\nStarting Phase 1 — Binary Sanity Check on {device}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, training=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   optimizer, criterion, training=False)

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)
            print(f"  → New best val acc: {best_val_acc:.2f}% — checkpoint saved.")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")

    # Final test evaluation — only run once using best checkpoint
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, test_acc = run_epoch(model, test_loader, optimizer, criterion, training=False)
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    if test_acc >= 99.0:
        print("✓ Sanity check PASSED — target ≥99% achieved!")
    else:
        print("✗ Sanity check not yet passed — consider more epochs or unfreezing more layers.")