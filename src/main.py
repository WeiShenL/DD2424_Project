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
from models.models import build_model, build_model_finetune, unfreeze_layer
from utils.utils import baseline_train_transform, eval_transform, augmented_train_transform, make_binary_target_transform, run_epoch, get_stratified_subset, setup_device, setup_dataloaders, set_seed
from visual.visual import visualize_predictions, save_batch_images, plot_training_history

def make_optimizer(model, optimizer_type="Adam", pretrained_lr=1e-4, fc_lr=3e-4, weight_decay=0.0):
    pretrained_params = [p for name, p in model.named_parameters()
                         if p.requires_grad and 'fc' not in name]
    fc_params = list(model.fc.parameters())

    if pretrained_params:
        param_groups = [
            {'params': pretrained_params, 'lr': pretrained_lr, 'weight_decay': weight_decay},
            {'params': fc_params,         'lr': fc_lr,         'weight_decay': weight_decay},
        ]
    else:
        # If the backbone is frozen, only optimize the fc layer
        param_groups = [
            {'params': fc_params, 'lr': fc_lr, 'weight_decay': weight_decay}
        ]

    # 3. Return the requested optimizer
    if optimizer_type.lower() == 'nag':
        # NAG is mathematically SGD with momentum and nesterov=True
        return optim.SGD(param_groups, momentum=0.9, nesterov=True)
    else:
        # Default to Adam
        return optim.Adam(param_groups)

def binary_classification(data_type=baseline_train_transform, model_type="resnet34"):
    # 1. Setup Device & Paths
    data_root = "data/folders"
    set_seed(42)
    device = setup_device()
    pin_memory = device.type == "cuda"

    # 2. Data Loaders
    train_loader, val_loader, test_loader = setup_dataloaders(
        data_root, train_transform=data_type, fraction=1.0, batch_size=32, binary=True
    )

    # 3. Model, Optimizer, Loss
    model = build_model(device, num_classes=2, model_type=model_type)
    # Adam optimizer
    # optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)
    # optimizer = make_optimizer(model, fc_lr=1e-3)
    # NAG optimizer
    # optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    optimizer = make_optimizer(model, optimizer_type="NAG", fc_lr=1e-2)
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
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, optimizer, criterion, device, training=True)
        val_loss,   val_acc, val_f1   = run_epoch(model, val_loader,   optimizer, criterion, device, training=False)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        # Adjust learning rate based on validation accuracy
        scheduler.step(val_acc)  

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  Train F1: {train_f1:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  Val F1: {val_f1:.4f}")

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
    _, test_acc, test_f1 = run_epoch(model, test_loader, optimizer, criterion, device, training=False)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Final Test F1 Score: {test_f1:.2f}")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

def multiclass_classification(data_type=baseline_train_transform, model_type="resnet34"):
    # 1. Setup Device & Paths
    data_root = "data/folders"
    set_seed(42)
    device = setup_device()

    # 2. Data Loaders (NOTICE: No binary target transforms here!)
    train_loader, val_loader, test_loader = setup_dataloaders(
        data_root, train_transform=data_type, fraction=1.0, batch_size=32, binary=False
    )

    # 3. Model, Optimizer, Loss (NOTICE: num_classes=37)
    model = build_model(device, num_classes=37, model_type=model_type)
    
    # Adam Optimizer
    # optimizer = optim.Adam(model.fc.parameters(), lr=3e-4)
    optimizer = make_optimizer(model, optimizer_type="Adam", fc_lr=3e-4)
    # We still use NAG and 1e-2 because we are still doing Linear Probing (frozen backbone)
    # optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9, nesterov=True)
    # optimizer = make_optimizer(model, optimizer_type="NAG", fc_lr=1e-2)
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
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, optimizer, criterion, device, training=True)
        val_loss,   val_acc, val_f1   = run_epoch(model, val_loader,   optimizer, criterion, device, training=False)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        scheduler.step(val_acc)  

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  Train F1: {train_f1:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  Val F1: {val_f1:.4f}")

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
    _, test_acc, test_f1 = run_epoch(model, test_loader, optimizer, criterion, device, training=False)
    print(f"Final Test Accuracy (Linear Probing Baseline): {test_acc:.2f}%  Test F1: {test_f1:.4f}")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

def strategy1_finetune(l=1, data_type=baseline_train_transform, fraction=1.0, weight_decay=0.0, model_type="resnet34"):
    data_root = "data/folders"
    set_seed(42)
    device = setup_device()
    # pin_memory = device.type == "cuda"

    train_loader, val_loader, test_loader = setup_dataloaders(
        data_root, train_transform=data_type, fraction=fraction, batch_size=32, binary=False
    )

    model = build_model_finetune(device, num_classes=37, l=l, model_type=model_type)

    # Use differential LR: lower for pretrained layers, higher for fc
    # pretrained_params = [p for name, p in model.named_parameters() 
    #                     if p.requires_grad and 'fc' not in name]
    # fc_params = model.fc.parameters()

    # optimizer = optim.Adam([
    #     {'params': pretrained_params, 'lr': 1e-4},
    #     {'params': fc_params,         'lr': 3e-4},
    # ])
    optimizer = make_optimizer(model, optimizer_type="Adam", pretrained_lr=3e-5, fc_lr=3e-4, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    NUM_EPOCHS = 25
    best_val_acc = 0.0
    checkpoint_dir = "results/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    # checkpoint = os.path.join(checkpoint_dir, f"resnet34_finetune_l{l}.pth")
    aug_tag = "aug" if data_type is augmented_train_transform else "base"
    wd_tag = f"_wd{weight_decay}" if weight_decay > 0 else ""
    checkpoint = os.path.join(checkpoint_dir, f"resnet34_s1_l{l}_frac{fraction}_{aug_tag}{wd_tag}.pth")

    print(f"\nStrategy 1: Fine-tuning last {l} layer(s) on {device}\n")

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}
    early_stopping_patience = 5
    epochs_without_improvement = 0
    start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, optimizer, criterion, device, training=True)
        val_loss,   val_acc, val_f1   = run_epoch(model, val_loader,   optimizer, criterion, device, training=False)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        scheduler.step(val_acc)

        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  Train F1: {train_f1:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%  Val F1: {val_f1:.4f}")

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
    plot_training_history(history, save_dir="results/Images/run1", filename=f"strategy1_l{l}_frac{fraction}_{aug_tag}{wd_tag}.png")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, test_acc, test_f1 = run_epoch(model, test_loader, optimizer, criterion, device, training=False)
    print(f"Final Test Accuracy (l={l}): {test_acc:.2f}%    Test F1: {test_f1:.4f}")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

def strategy2_gradual_unfreeze(max_epochs_per_stage=25, data_type=baseline_train_transform, fraction=1.0, weight_decay=0.0, model_type="resnet34"):
    # 1. Setup Device & Paths
    data_root = "data/folders"
    set_seed(42)
    device = setup_device()


    # 2. Data Loaders
    train_loader, val_loader, test_loader = setup_dataloaders(
        data_root, train_transform=data_type, fraction=fraction, batch_size=32, binary=False
    )

    # 3. Model — start fully frozen except fc (same as linear probing)
    model = build_model(device, num_classes=37, model_type=model_type)

    # Unfreeze schedule: after each stage, unfreeze the next layer inward
    # Stage 0: fc only
    # Stage 1: layer4 + fc
    # Stage 2: layer3 + layer4 + fc
    # Stage 3: layer2 + layer3 + layer4 + fc
    # Stage 4: layer1 + ... + fc
    unfreeze_schedule = ['layer4', 'layer3', 'layer2', 'layer1']

    checkpoint_dir = "results/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    # checkpoint = os.path.join(checkpoint_dir, "resnet34_strategy2_best.pth")
    aug_tag = "aug" if data_type is augmented_train_transform else "base"
    wd_tag = f"_wd{weight_decay}" if weight_decay > 0 else ""
    checkpoint = os.path.join(checkpoint_dir, f"resnet34_s2_frac{fraction}_{aug_tag}{wd_tag}.pth")

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': []}
    criterion = nn.CrossEntropyLoss()

    print(f"\nStrategy 2: Gradual Unfreezing on {device}\n")
    start_time = time.time()

    global_epoch = 0 

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
        # pretrained_params = [p for name, p in model.named_parameters()
        #                      if p.requires_grad and 'fc' not in name]
        # fc_params = list(model.fc.parameters())

        # if pretrained_params:
        #     optimizer = optim.Adam([
        #         {'params': pretrained_params, 'lr': 1e-4},
        #         {'params': fc_params,         'lr': 3e-4},
        #     ])
        # else:
        #     optimizer = optim.Adam(fc_params, lr=3e-4)

        optimizer = make_optimizer(model, pretrained_lr=3e-5, fc_lr=3e-4, weight_decay=weight_decay)

        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        early_stopping_patience = 5
        epochs_without_improvement = 0
        # best_stage_val_loss = float('inf')
        epoch = 0

        # Train for epochs_per_stage epochs at this stage

        while epoch < max_epochs_per_stage:
            epoch += 1
            global_epoch += 1

            train_loss, train_acc, train_f1 = run_epoch(model, train_loader, optimizer, criterion, device, training=True)
            val_loss,   val_acc, val_f1   = run_epoch(model, val_loader,   optimizer, criterion, device, training=False)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)

            scheduler.step(val_acc)

            print(f"  Epoch [{global_epoch:02d}] Stage {stage} Epoch [{epoch}/{max_epochs_per_stage}]  "
                  f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  Train F1: {train_f1:.4f}  |"
                  f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%   Val F1: {val_f1:.4f}")

            # Improvement threshold of 0.01 in val acc
            if val_acc > best_val_acc + 0.01:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint)
                print(f"    → New best val acc: {best_val_acc:.2f}%")
            else:
                epochs_without_improvement += 1
                print(f"    → No improvement in val acc for {epochs_without_improvement} epoch(s).")
            
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\n  Early stopping triggered for Stage {stage}!")
                break

    end_time = time.time()
    plot_training_history(history, save_dir="results/Images/run1", filename=f"strategy2_frac{fraction}_{aug_tag}{wd_tag}.png")

    # Final Test Evaluation
    print(f"\nTraining complete. Best val acc: {best_val_acc:.2f}%")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, test_acc, test_f1 = run_epoch(model, test_loader, optimizer, criterion, device, training=False)
    print(f"Final Test Accuracy (Strategy 2): {test_acc:.2f}%   Test F1: {test_f1:.4f}")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    model = "resnet34" 

    # Phase 1 sanity check (non augmented data)
    # binary_classification(baseline_train_transform)

    # Phase 2: Multiclass linear probing (non augmented data)
    # multiclass_classification(baseline_train_transform)

    # Strategy 1: Full data, varying l from 1 to 4, no augmentation, no weight decay
    print(f"\n\n--- Running Strategy 1: Fine-tuning last l layers ---")
    for l in range(1, 5):
        strategy1_finetune(l=l, data_type=baseline_train_transform, model_type=model)
    
    # Strategy 2: Full data, gradual unfreeze every 25 epochs, no augmentation, no weight decay
    # print(f"\n\n--- Running Strategy 2: Gradual Unfreeze ---")
    strategy2_gradual_unfreeze(max_epochs_per_stage=25, data_type=baseline_train_transform, model_type=model)

    # Running stratified dataset (10% and 1%) experiments for both strategies, no augmentation, no weight decay
    for fraction in [0.1, 0.01]:
        print(f"\n\n--- Running data experiment with fraction={fraction} ---")
        strategy1_finetune(l=2, data_type=baseline_train_transform, fraction=fraction, model_type=model)
        strategy2_gradual_unfreeze(max_epochs_per_stage=25, data_type=baseline_train_transform, fraction=fraction, model_type=model)
    
    # Augmentation benefit across dataset sizes
    print(f"\n\n--- Running data experiment with augmented data ---")
    for fraction in [1.0, 0.1, 0.01]:
        strategy1_finetune(l=2, data_type=augmented_train_transform, fraction=fraction, model_type=model)
        strategy2_gradual_unfreeze(max_epochs_per_stage=25, data_type=augmented_train_transform, fraction=fraction, model_type=model)

    # L2 regularisation + varying data size comparison
    print(f"\n\n--- Running L2 regularization + varying data size comparison ---")
    for fraction in [1.0, 0.1, 0.01]:
        for weight_decay in [1e-3, 1e-2]:  
            strategy1_finetune(l=2, data_type=baseline_train_transform, fraction=fraction, weight_decay=weight_decay, model_type=model)
            strategy2_gradual_unfreeze(max_epochs_per_stage=25, data_type=baseline_train_transform, fraction=fraction, weight_decay=weight_decay, model_type=model)