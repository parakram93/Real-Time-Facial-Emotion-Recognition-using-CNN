import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TEST_DIR = "archive/test"
MODEL_PATH = "emotion_cnn.pth"
IMG_SIZE = 48
BATCH_SIZE = 32

EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

NUM_CLASSES = len(EMOTIONS)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FolderDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        self.labels = []

        for idx, emotion in enumerate(EMOTIONS):
            folder = os.path.join(root_dir, emotion)

            if not os.path.exists(folder):
                raise FileNotFoundError(f"Dataset folder not found: {folder}")

            for file in os.listdir(folder):
                path = os.path.join(folder, file)

                if os.path.isfile(path):
                    self.data.append(path)
                    self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert("L").resize(
            (IMG_SIZE, IMG_SIZE)
        )

        img = torch.tensor(
            np.array(img),
            dtype=torch.float32
        ).unsqueeze(0) / 255.0

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return img, label


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 22 * 22, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


test_dataset = FolderDataset(TEST_DIR)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

model = CNN(NUM_CLASSES).to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

true_labels = []
predicted_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        true_labels.extend(labels.numpy())
        predicted_labels.extend(predictions.cpu().numpy())

accuracy = accuracy_score(
    true_labels,
    predicted_labels
)

precision = precision_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

recall = recall_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print(f"Test Samples: {len(test_dataset)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)
