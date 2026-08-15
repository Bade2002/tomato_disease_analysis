import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# =====================================================================
# 1. Custom CNN Architecture from Scratch
# =====================================================================
class CustomTomatoCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomTomatoCNN, self).__init__()

        # Block 1: Input (3, 224, 224) -> Output (32, 112, 112)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2: Input (32, 112, 112) -> Output (64, 56, 56)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3: Input (64, 56, 56) -> Output (128, 28, 28)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 4: Input (128, 28, 28) -> Output (256, 14, 14)
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Global Pooling & Fully Connected Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Squeezes to (256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# =====================================================================
# 2. Main Training Function
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"Training Custom CNN on: {device} ({torch.cuda.get_device_name(0)})")
    print(f"==================================================\n")

    torch.backends.cudnn.benchmark = True
    data_dir = r"C:\Users\prajw\Downloads\archive\PlantVillage\dataset_split"
    batch_size = 64
    num_epochs = 20
    lr = 0.001
    img_size = 224

    # Data Pipelines
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    # Initialize Custom Model
    model = CustomTomatoCNN(num_classes=num_classes).to(device)

    # Calculate trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Initialized! Total Trainable Parameters: {total_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    start_total_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}] | LR: {optimizer.param_groups[0]['lr']:.6f}")
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

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

            print(f"  {phase.upper():<5} Loss: {epoch_loss:.4f}  |  Acc: {epoch_acc * 100:.2f}%")

            if phase == 'val':
                scheduler.step(epoch_acc)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'class_names': class_names,
                        'accuracy': best_acc
                    }, "custom_tomato_cnn_best.pth")
                    print(f"  >>> Best Custom CNN Saved! (Accuracy: {best_acc * 100:.2f}%)")

        print()

    total_time = time.time() - start_total_time
    print(f"==================================================")
    print(f"Custom CNN Training Finished in: {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best Validation Accuracy: {best_acc * 100:.2f}%")
    print(f"Model saved as: custom_tomato_cnn_best.pth")
    print(f"==================================================")


if __name__ == '__main__':
    main()