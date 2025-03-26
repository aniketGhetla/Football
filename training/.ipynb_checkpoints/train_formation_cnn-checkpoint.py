import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Paths and Device
data_dir = "dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load base dataset without transform
base_dataset = datasets.ImageFolder(data_dir)
class_names = base_dataset.classes

# Train/Val/Test split
train_size = int(0.7 * len(base_dataset))
val_size = int(0.15 * len(base_dataset))
test_size = len(base_dataset) - train_size - val_size

dataset_split = random_split(base_dataset, [train_size, val_size, test_size],
                              generator=torch.Generator().manual_seed(42))

# Apply transforms to each subset
train_dataset = Subset(datasets.ImageFolder(data_dir, transform=train_transforms), dataset_split[0].indices)
val_dataset = Subset(datasets.ImageFolder(data_dir, transform=val_test_transforms), dataset_split[1].indices)
test_dataset = Subset(datasets.ImageFolder(data_dir, transform=val_test_transforms), dataset_split[2].indices)

# Dataloaders with larger batch size
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model Setup
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, len(class_names))
)
model = model.to(device)

# Optimizer, Loss, Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Train Function with Early Stopping
def train_model(model, num_epochs=10, patience=2):
    best_acc = 0.0
    best_loss = float('inf')
    counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), "best_formation_model.pth")
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping triggered.")
                        plot_losses(train_losses, val_losses)
                        return

    plot_losses(train_losses, val_losses)
    print(f"\nBest Val Accuracy: {best_acc:.4f}")

# Plotting loss curves
def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.show()

# Evaluation on Test Set
def test_model():
    model.load_state_dict(torch.load("best_formation_model.pth"))
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nTest Set Classification Report:\n")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0
    ))

# Run Training and Testing
train_model(model, num_epochs=10, patience=2)
test_model()
