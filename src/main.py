# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

# Import from your new modules
from models.models import build_model, build_resnet18_model, build_model_finetune, unfreeze_layer
from utils.utils import baseline_train_transform, eval_transform, augmented_train_transform,make_binary_target_transform, run_epoch
from visual.visual import visualize_predictions, save_batch_images, plot_training_history

def binary_classification(data_type=baseline_train_transform):
    # 1. Setup Device & Paths
    data_root = "data/folders"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")
    pin_memory = device.type == "cuda"

    # 2. Data Loaders
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=data_type)
    val_dataset   = datasets.ImageFolder(root=os.path.join(data_root, "val"),   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(root=os.path.join(data_root, "test"),  transform=eval_transform)

    # Remap labels to binary (For Phase 1)
    to_binary = make_binary_target_transform(train_dataset.classes)
    train_dataset.target_transform = to_binary
    val_dataset.target_transform   = to_binary
    test_dataset.target_transform  = to_binary

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0, pin_memory=pin_memory)

    # 3. Model, Optimizer, Loss
    model = build_model(device, num_classes=2)
    # Adam optimizer
    # optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    # NAG optimizer
    optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    NUM_EPOCHS = 25
    best_val_acc = 0.0
    checkpoint_dir = "results/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = os.path.join(checkpoint_dir, "resnet34_binary_best.pth")

    print(f"\nStarting Phase 1, Binary Sanity Check on {device}\n")

    start_time = time.time()
    early_stopping_patience = 5
    epochs_without_improvement = 0
    early_stop_threshold = 0.01

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device, training=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   optimizer, criterion, device, training=False)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Adjust learning rate based on validation accuracy
        scheduler.step(val_acc)  

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc + early_stop_threshold:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
            print(f"  → New best val acc: {best_val_acc:.2f}% .")
        else:
            epochs_without_improvement += 1
            print(f"  → No improvement for {epochs_without_improvement} epoch(s).")
        
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered! No improvement in {early_stopping_patience} epochs.")
            break # Exit the loop

        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    end_time = time.time()
    plot_training_history(history, save_dir="results/Images", filename="binary_training_history.png")

    # 5. Final Test Evaluation
    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, test_acc = run_epoch(model, test_loader, optimizer, criterion, device, training=False)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

def multiclass_classification(data_type=baseline_train_transform):
    # 1. Setup Device & Paths
    data_root = "data/folders"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")
    pin_memory = device.type == "cuda"

    # 2. Data Loaders (NOTICE: No binary target transforms here!)
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=data_type)
    val_dataset   = datasets.ImageFolder(root=os.path.join(data_root, "val"),   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(root=os.path.join(data_root, "test"),  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0, pin_memory=pin_memory)

    # 3. Model, Optimizer, Loss (NOTICE: num_classes=37)
    model = build_model(device, num_classes=37)
    
    # Adam Optimizer
    optimizer = optim.Adam(model.fc.parameters(), lr=3e-4)
    # We still use NAG and 1e-2 because we are still doing Linear Probing (frozen backbone)
    # optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    NUM_EPOCHS = 25
    best_val_acc = 0.0
    checkpoint_dir = "results/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # NOTICE: Changed checkpoint name so it doesn't overwrite your binary success!
    checkpoint = os.path.join(checkpoint_dir, "resnet34_multiclass_baseline.pth") 

    print(f"\nStarting Multi-Class Linear Probing Baseline on {device}\n")

    start_time = time.time()

    early_stopping_patience = 5
    epochs_without_improvement = 0
    early_stop_threshold = 0.01
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device, training=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   optimizer, criterion, device, training=False)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        scheduler.step(val_acc)  

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc + early_stop_threshold:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
            print(f"  → New best val acc: {best_val_acc:.2f}% .")
        else:
            epochs_without_improvement += 1
            print(f"  → No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered! No improvement in {early_stopping_patience} epochs.")
            break # Exit the loop

    end_time = time.time()
    plot_training_history(history, save_dir="results/Images", filename="multiclass_training_history.png")

    # 5. Final Test Evaluation
    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, test_acc = run_epoch(model, test_loader, optimizer, criterion, device, training=False)
    print(f"Final Test Accuracy (Linear Probing Baseline): {test_acc:.2f}%")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

def strategy1_finetune(l=1, data_type=baseline_train_transform):
    data_root = "data/folders"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    pin_memory = device.type == "cuda"

    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=data_type)
    val_dataset   = datasets.ImageFolder(root=os.path.join(data_root, "val"),   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(root=os.path.join(data_root, "test"),  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0, pin_memory=pin_memory)

    model = build_model_finetune(device, num_classes=37, l=l)

    # Use differential LR: lower for pretrained layers, higher for fc
    pretrained_params = [p for name, p in model.named_parameters() 
                        if p.requires_grad and 'fc' not in name]
    fc_params = model.fc.parameters()

    optimizer = optim.Adam([
        {'params': pretrained_params, 'lr': 1e-4},
        {'params': fc_params,         'lr': 3e-4},
    ])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    NUM_EPOCHS = 25
    best_val_acc = 0.0
    checkpoint_dir = "results/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = os.path.join(checkpoint_dir, f"resnet34_finetune_l{l}.pth")

    print(f"\nStrategy 1: Fine-tuning last {l} layer(s) on {device}\n")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    early_stopping_patience = 5
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device, training=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   optimizer, criterion, device, training=False)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        scheduler.step(val_acc)

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc + 0.01:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint)
            print(f"  → New best val acc: {best_val_acc:.2f}%")
        else:
            epochs_without_improvement += 1
            print(f"  → No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered!")
            break

    end_time = time.time()
    plot_training_history(history, save_dir="results/Images", filename=f"strategy1_l{l}.png")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, test_acc = run_epoch(model, test_loader, optimizer, criterion, device, training=False)
    print(f"Final Test Accuracy (l={l}): {test_acc:.2f}%")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

def strategy2_gradual_unfreeze(epochs_per_stage=5, data_type=baseline_train_transform):
    # 1. Setup Device & Paths
    data_root = "data/folders"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")
    pin_memory = device.type == "cuda"

    # 2. Data Loaders
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"), transform=data_type)
    val_dataset   = datasets.ImageFolder(root=os.path.join(data_root, "val"),   transform=eval_transform)
    test_dataset  = datasets.ImageFolder(root=os.path.join(data_root, "test"),  transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0, pin_memory=pin_memory)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=0, pin_memory=pin_memory)

    # 3. Model — start fully frozen except fc (same as linear probing)
    model = build_model(device, num_classes=37)

    # Unfreeze schedule: after each stage, unfreeze the next layer inward
    # Stage 0: fc only
    # Stage 1: layer4 + fc
    # Stage 2: layer3 + layer4 + fc
    # Stage 3: layer2 + layer3 + layer4 + fc
    # Stage 4: layer1 + ... + fc
    unfreeze_schedule = ['layer4', 'layer3', 'layer2', 'layer1']

    checkpoint_dir = "results/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = os.path.join(checkpoint_dir, "resnet34_strategy2_best.pth")

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    criterion = nn.CrossEntropyLoss()

    print(f"\nStrategy 2: Gradual Unfreezing on {device}\n")
    start_time = time.time()

    # Run through each stage
    for stage in range(len(unfreeze_schedule) + 1):
        if stage == 0:
            print(f"\n--- Stage {stage}: Training fc only ---")
        else:
            layer_name = unfreeze_schedule[stage - 1]
            unfreeze_layer(model, layer_name)
            print(f"\n--- Stage {stage}: Unfreezing {layer_name} ---")

        # Rebuild optimizer each stage to include newly unfrozen params
        # Use differential LR: lower for pretrained layers, higher for fc
        pretrained_params = [p for name, p in model.named_parameters()
                             if p.requires_grad and 'fc' not in name]
        fc_params = list(model.fc.parameters())

        if pretrained_params:
            optimizer = optim.Adam([
                {'params': pretrained_params, 'lr': 1e-4},
                {'params': fc_params,         'lr': 3e-4},
            ])
        else:
            optimizer = optim.Adam(fc_params, lr=3e-4)

        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        # Train for epochs_per_stage epochs at this stage
        for epoch in range(1, epochs_per_stage + 1):
            global_epoch = stage * epochs_per_stage + epoch

            train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device, training=True)
            val_loss,   val_acc   = run_epoch(model, val_loader,   optimizer, criterion, device, training=False)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            scheduler.step(val_acc)

            print(f"  Epoch [{global_epoch:02d}] Stage {stage} Epoch [{epoch}/{epochs_per_stage}]  "
                  f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
                  f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc + 0.01:
                best_val_acc = val_acc
                torch.save(model.state_dict(), checkpoint)
                print(f"    → New best val acc: {best_val_acc:.2f}%")

    end_time = time.time()
    plot_training_history(history, save_dir="results/Images", filename="strategy2_gradual_unfreeze.png")

    # Final Test Evaluation
    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, test_acc = run_epoch(model, test_loader, optimizer, criterion, device, training=False)
    print(f"Final Test Accuracy (Strategy 2): {test_acc:.2f}%")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    # binary_classification(baseline_train_transform)
    # multiclass_classification(baseline_train_transform)

    for l in range(1, 5):
        strategy1_finetune(l=l, data_type=baseline_train_transform)
    
    strategy2_gradual_unfreeze(epochs_per_stage=10, data_type=baseline_train_transform)