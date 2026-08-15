import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    # ---------------------------------------------------------
    # 1. Device Setup & Optimization Flags
    # ---------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"Device: {device} | {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    print(f"==================================================\n")

    # Benchmark speeds up convolutions for fixed input sizes on Ampere GPUs
    torch.backends.cudnn.benchmark = True

    # ---------------------------------------------------------
    # 2. Paths & Hyperparameters
    # ---------------------------------------------------------
    data_dir = r"C:\Users\prajw\Downloads\archive\PlantVillage\dataset_split"
    batch_size = 64  # Efficient size for RTX 3070 Ti 8GB VRAM
    num_epochs = 15
    lr = 3e-4
    img_size = 224

    # ---------------------------------------------------------
    # 3. Robust Augmentation & Preprocessing Pipeline
    # ---------------------------------------------------------
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value='random')
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    # ---------------------------------------------------------
    # 4. DataLoaders
    # ---------------------------------------------------------
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    # num_workers=4 with pin_memory accelerates Host-to-GPU memory transfer
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Detected {num_classes} Classes: {class_names}\n")

    # Save class mapping to JSON for inference later
    with open("class_labels.json", "w") as f:
        json.dump(class_names, f, indent=4)

    # ---------------------------------------------------------
    # 5. Model Architecture (Pretrained ConvNeXt-Tiny / MobileNetV2)
    # ---------------------------------------------------------
    # EfficientNet-B0 or MobileNetV3 are lightweight and highly accurate
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)

    # Modify final classifier layer
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    # ---------------------------------------------------------
    # 6. Loss, Optimizer, Mixed Precision & Scheduler
    # ---------------------------------------------------------
    # Label smoothing prevents overconfidence on noisy leaf backgrounds
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Automatic Mixed Precision (FP16) - utilizes RTX 3070 Ti Tensor Cores
    scaler = torch.amp.GradScaler('cuda')

    # ---------------------------------------------------------
    # 7. Training & Evaluation Loop
    # ---------------------------------------------------------
    best_acc = 0.0
    start_total_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]  |  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 45)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            loop = tqdm(dataloaders[phase], desc=f"{phase.upper():<5}", leave=False)

            for inputs, labels in loop:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == 'train'):
                    # FP16 autocast for faster matrix math
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                loop.set_postfix(loss=f"{loss.item():.4f}")

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

            print(f"  {phase.upper():<5} Loss: {epoch_loss:.4f}  |  Acc: {epoch_acc * 100:.2f}%")

            # Checkpoint saving
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'class_names': class_names,
                    'accuracy': best_acc
                }, "best_tomato_model.pth")
                print(f"  >>> New Best Model Saved! (Validation Accuracy: {best_acc * 100:.2f}%)")

        print()

    total_time = time.time() - start_total_time
    print(f"==================================================")
    print(f"Training Complete in: {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Highest Validation Accuracy: {best_acc * 100:.2f}%")
    print(f"Weights saved to: best_tomato_model.pth")
    print(f"Labels saved to:  class_labels.json")
    print(f"==================================================")


if __name__ == '__main__':
    main()