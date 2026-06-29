# face.py
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# Settings
# -----------------------------
TRAIN_DIR = "archive/train"
TEST_DIR = "archive/test"
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)

# -----------------------------
# Dataset
# -----------------------------
class FolderDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []
        for idx, emotion in enumerate(EMOTIONS):
            folder = os.path.join(root_dir, emotion)
            for file in os.listdir(folder):
                self.data.append(os.path.join(folder, file))
                self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
        img = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0)/255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

train_dataset = FolderDataset(TRAIN_DIR)
test_dataset = FolderDataset(TEST_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# -----------------------------
# CNN Model
# -----------------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*22*22, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = CNN(NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training
# -----------------------------
print("Training CNN...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss:.4f}")

# -----------------------------
# Testing
# -----------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total*100:.2f}%")

# -----------------------------
# Save model
# -----------------------------
MODEL_PATH = "emotion_cnn.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
