import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import threading
from torchvision.models import mobilenet_v3_small
import time
import os
from collections import deque
import torch.nn.functional as F

class EnhancedGestureRecognitionModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base_model = mobilenet_v3_small(pretrained=True)
        self.base_model.features[0][0] = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.base_model.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.base_model(x)

class EnhancedEmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.res1 = ResidualBlock(64, 128)
        self.res2 = ResidualBlock(128, 256)
        self.res3 = ResidualBlock(256, 512)
        self.attention = SpatialAttention()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes))
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.attention(x) * x
        x = self.gap(x).squeeze()
        return self.fc(x)

class EnhancedActionRecognitionModel(nn.Module):
    def __init__(self, input_size=33*3, num_classes=10):
        super().__init__()
        self.proj = nn.Linear(input_size, 96)  # Project to divisible dimension
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=96, nhead=8, dim_feedforward=512, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
        self.fc = nn.Sequential(
            nn.Linear(96, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes))
    
    def forward(self, x):
        x = self.proj(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        return self.fc(x)

# Helper Components
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(combined))

# Main Class Implementation
class NeuraStyle:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize MediaPipe components
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3)
        self.mp_hands = mp.solutions.hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.mp_pose = mp.solutions.pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Initialize models
        self.gesture_model = EnhancedGestureRecognitionModel().to(self.device)
        self.emotion_model = EnhancedEmotionRecognitionModel().to(self.device)
        self.action_model = EnhancedActionRecognitionModel().to(self.device)
        
        # Class labels
        self.gesture_classes = ["fist", "open", "pointing", "peace", "thumbs_up", 
                               "thumbs_down", "ok", "rock", "scissors", "paper"]
        self.emotion_classes = ["angry", "disgust", "fear", "happy", 
                              "sad", "surprise", "neutral"]
        self.action_classes = ["walking", "running", "jumping", "sitting", 
                             "standing", "waving", "clapping", "pointing", 
                             "dancing", "falling"]
        
        # Preprocessing transforms
        self.gesture_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.emotion_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Tracking and processing
        self.action_buffer = deque(maxlen=30)
        self.tracked_data = {
            "face_landmarks": [], "hand_landmarks": [], "pose_landmarks": [],
            "detected_gestures": [], "detected_emotions": [], "detected_actions": []
        }
        self.running = False
        self.process_thread = None

    def start_processing(self, video_source=0):
        if self.running:
            print("Already running")
            return
        self.running = True
        self.process_thread = threading.Thread(target=self._process_video, args=(video_source,))
        self.process_thread.daemon = True
        self.process_thread.start()
        print(f"Started processing video source {video_source}")

    def stop_processing(self):
        self.running = False
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
            print("Processing stopped")

    def _process_video(self, video_source):
        cap = cv2.VideoCapture(video_source)
        frame_count = 0
        skip_frames = 2  # Process every 3rd frame

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % (skip_frames + 1) != 0:
                continue

            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._process_frame(rgb_frame, frame)
            cv2.imshow('NeuraStyle', frame)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        self.running = False

    def _process_frame(self, rgb_frame, display_frame):
        # Face processing
        face_results = self.mp_face_mesh.process(rgb_frame)
        if face_results.multi_face_landmarks:
            self.tracked_data["face_landmarks"] = face_results.multi_face_landmarks
            # Emotion detection logic
            # ... (face cropping and emotion prediction)
            if face_results.multi_face_landmarks:
                self.tracked_data["face_landmarks"] = face_results.multi_face_landmarks
    
                # Crop face region
                h, w, _ = rgb_frame.shape
                face_landmarks = face_results.multi_face_landmarks[0]  # Take first face
                mesh_points = np.array([[p.x * w, p.y * h] for p in face_landmarks.landmark])
                x, y, w_rect, h_rect = cv2.boundingRect(mesh_points.astype(np.int32))
                face_img = rgb_frame[y:y+h_rect, x:x+w_rect]
    
                # Predict emotion
                emotion = self._predict_emotion(face_img)
                self.tracked_data["detected_emotions"].append(emotion)

        # Hand processing
        hand_results = self.mp_hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            self.tracked_data["hand_landmarks"] = hand_results.multi_hand_landmarks
            # Gesture detection logic
            # ... (hand cropping and gesture prediction)
            if hand_results.multi_hand_landmarks:
                self.tracked_data["hand_landmarks"] = hand_results.multi_hand_landmarks
                
                # Crop hand region (example for first hand)
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                mesh_points = np.array([[p.x * w, p.y * h] for p in hand_landmarks.landmark])
                x, y, w_rect, h_rect = cv2.boundingRect(mesh_points.astype(np.int32))
                hand_img = rgb_frame[y:y+h_rect, x:x+w_rect]
                
                # Predict gesture
                gesture = self._predict_gesture(hand_img)
                self.tracked_data["detected_gestures"].append(gesture)
                        

        # Pose processing
        pose_results = self.mp_pose.process(rgb_frame)
        if pose_results.pose_landmarks:
            self.tracked_data["pose_landmarks"] = pose_results.pose_landmarks
            # Action detection logic
            # ... (pose sequence buffering and action prediction)
            if pose_results.pose_landmarks:
                self.tracked_data["pose_landmarks"] = pose_results.pose_landmarks
                
                # Extract pose landmarks (33 points)
                pose_data = np.array([[p.x, p.y, p.z] for p in pose_results.pose_landmarks.landmark]).flatten()
                self.action_buffer.append(pose_data)  # Buffer for action prediction
                
                # Predict action when buffer is full
                if len(self.action_buffer) >= 30:
                    action = self._predict_action()
                    self.tracked_data["detected_actions"].append(action)
                        

    # Prediction methods
    def _predict_gesture(self, hand_img):
        input_tensor = self.gesture_transform(hand_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.gesture_model(input_tensor)
            _, pred = torch.max(outputs, 1)
        return self.gesture_classes[pred.item()]

    def _predict_emotion(self, face_img):
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        input_tensor = self.emotion_transform(face_gray).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.emotion_model(input_tensor)
            _, pred = torch.max(outputs, 1)
        return self.emotion_classes[pred.item()]

    def _predict_action(self):
        if len(self.action_buffer) < 30:
            return "none"
        sequence = torch.FloatTensor(self.action_buffer).unsqueeze(1).to(self.device)
        with torch.no_grad():
            outputs = self.action_model(sequence)
            _, pred = torch.max(outputs, 1)
        return self.action_classes[pred.item()]

    # Utility methods
    def get_current_data(self):
        return self.tracked_data

    def save_models(self, path="models"):
        os.makedirs(path, exist_ok=True)
        torch.save(self.gesture_model.state_dict(), f"{path}/gesture.pth")
        torch.save(self.emotion_model.state_dict(), f"{path}/emotion.pth")
        torch.save(self.action_model.state_dict(), f"{path}/action.pth")

    def load_models(self, path="models"):
        self.gesture_model.load_state_dict(torch.load(f"{path}/gesture.pth", map_location=self.device))
        self.emotion_model.load_state_dict(torch.load(f"{path}/emotion.pth", map_location=self.device))
        self.action_model.load_state_dict(torch.load(f"{path}/action.pth", map_location=self.device))
        self.gesture_model.eval()
        self.emotion_model.eval()
        self.action_model.eval()

# Usage Example
if __name__ == "__main__":
    ai = NeuraStyle()
    ai.save_models()  
    ai.load_models()
    ai.start_processing(0)
    
    try:
        while ai.running:
            time.sleep(1)
            print(ai.get_current_data()["detected_gestures"][-1:])
    except KeyboardInterrupt:
        ai.stop_processing()