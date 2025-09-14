import numpy as np
import pandas as pd
from scipy.fftpack import idct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from collections import Counter
import torch.nn.functional as F  # Import for functional operations

# --- Load and Preprocess Data for PyTorch ---
class CombinedDataset(Dataset):
    def __init__(self, train_csv, dev_csv, transform=None):
        self.train_df = pd.read_csv(train_csv, skiprows=1, header=None)
        self.dev_df = pd.read_csv(dev_csv, skiprows=1, header=None)
        self.combined_df = pd.concat([self.train_df, self.dev_df], ignore_index=True)
        print(f"Combined dataset shape: {self.combined_df.shape}")
        self.labels = self.combined_df.iloc[:, 0].values
        self.dct_coeffs = self.combined_df.iloc[:, 1:].values
        self.transform = transform

    def __len__(self):
        return len(self.combined_df)

    def __getitem__(self, idx):
        coeffs = self.dct_coeffs[idx]
        r_dct = coeffs[0:1024].reshape(32, 32)
        g_dct = coeffs[1024:2048].reshape(32, 32)
        b_dct = coeffs[2048:3072].reshape(32, 32)
        r = idct(idct(r_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        g = idct(idct(g_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        b = idct(idct(b_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        rgb_image = np.stack([r, g, b], axis=-1).astype('float32')

        if self.transform:
            rgb_image = self.transform(rgb_image)
        else:
            rgb_image = torch.tensor(rgb_image).permute(2, 0, 1)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return rgb_image, label

class BreastDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file, skiprows=1, header=None)
        print(self.df.shape)
        self.labels = self.df.iloc[:, 0].values
        self.dct_coeffs = self.df.iloc[:, 1:].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        coeffs = self.dct_coeffs[idx]
        r_dct = coeffs[0:1024].reshape(32, 32)
        g_dct = coeffs[1024:2048].reshape(32, 32)
        b_dct = coeffs[2048:3072].reshape(32, 32)
        r = idct(idct(r_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        g = idct(idct(g_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        b = idct(idct(b_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
        rgb_image = np.stack([r, g, b], axis=-1).astype('float32')

        if self.transform:
            rgb_image = self.transform(rgb_image)
        else:
            rgb_image = torch.tensor(rgb_image).permute(2, 0, 1) # Convert to tensor if no transform

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return rgb_image, label

# --- Data Augmentation ---
# train_transforms = transforms.Compose([
#     transforms.ToTensor(), # Convert NumPy array to PyTorch Tensor
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.RandomVerticalFlip(p=0.5),
#     transforms.RandomRotation(degrees=15),
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.98, 1.02)),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Load the Datasets ---
print("\nLoading Combined Training and Dev Data")
combined_dataset = CombinedDataset("../../../../FullData/train.csv", "../../../../FullData/dev.csv", transform=train_transforms)
combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True, num_workers=4)

print("\nLoading Training Data")
train_dataset = BreastDataset("../../../../FullData/train.csv", transform=eval_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)

print("\nLoading Dev Data")
dev_dataset = BreastDataset("../../../../FullData/dev.csv", transform=eval_transforms)
dev_loader = DataLoader(dev_dataset, batch_size=64, shuffle=False, num_workers=4)

print("\nLoading Evaluation Data")
eval_dataset = BreastDataset("../../../../FullData/eval.csv", transform=eval_transforms)
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, num_workers=4)


# --- Define a 4-Layer CNN Model with Batch Normalization and Dropout ---
class BetterCNN(nn.Module):
    def __init__(self, num_classes, kernel_size=5):  # Added kernel_size parameter
        super(BetterCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=kernel_size, padding=2)
        self.bn1 = nn.BatchNorm2d(64)  # Changed to 64
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=2)
        self.bn2 = nn.BatchNorm2d(128)  # Changed to 128
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=kernel_size, padding=2)
        self.bn3 = nn.BatchNorm2d(256) # Changed to 256
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.3)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=kernel_size, padding=2)
        self.bn4 = nn.BatchNorm2d(512)  # Changed to 512
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.3)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 2 * 2, 256)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.5) # 0.4
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        x = self.flatten(x)
        x = self.relu5(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)
        return x

# --- Initialize Model, Loss Function, and Optimizer ---
num_classes = 9
model = BetterCNN(num_classes, kernel_size=5)

# Class Weighted Loss
train_labels = [label.item() for _, label in combined_loader.dataset]
class_counts = Counter(train_labels)
total_samples = len(train_labels)
class_weights = [total_samples / class_counts[i] for i in range(num_classes)]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)


optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# --- Training the Model ---
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(combined_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(combined_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0
    scheduler.step(running_loss/len(combined_loader))
    print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {running_loss/len(combined_loader):.4f}')

# --- Evaluate the model on the evaluation data ---
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for images, labels in eval_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

avg_test_loss = test_loss / len(eval_loader)
avg_test_acc = 100 * correct / total
print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.2f} %')

# --- Make Predictions and Save to CSV ---
def predict_and_save(dataloader, model, filename):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for images, _ in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())

    df_predictions = pd.DataFrame({'label': all_predictions})
    df_predictions.to_csv(filename, index=False, header=True)
    print(f"Predictions saved to {filename}")

predict_and_save(train_loader, model, "./predictions/train_predictions.csv")
predict_and_save(dev_loader, model, "./predictions/dev_predictions.csv")
predict_and_save(eval_loader, model, "./predictions/eval_predictions.csv")

print("Done!")
