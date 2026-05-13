import torch
from torchvision import transforms

# --- Transforms ---
train_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    transforms.Resize(256),       # 1. Scale shortest edge to 256 (preserves aspect ratio)
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.Resize(256),       # 1. Scale shortest edge to 256 (preserves aspect ratio)
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

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