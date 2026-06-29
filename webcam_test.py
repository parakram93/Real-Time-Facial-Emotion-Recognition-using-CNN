# webcam_test.py
import cv2
import torch
import numpy as np
from torch import nn

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = 48
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH = "emotion_cnn.pth"

# -----------------------------
# CNN Model (same as training)
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

# -----------------------------
# Load trained model
# -----------------------------
model = CNN(len(EMOTIONS)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Loaded trained model.")

# -----------------------------
# Start webcam
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray_frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face_tensor = torch.tensor(face_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)/255.0
        face_tensor = face_tensor.to(device)
        with torch.no_grad():
            pred = model(face_tensor)
            emo_id = pred.argmax().item()
            emotion_name = EMOTIONS[emo_id]

        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame, emotion_name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        print("Emotion:", emotion_name)

    cv2.imshow("Webcam Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
