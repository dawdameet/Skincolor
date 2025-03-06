import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ----- Dataset Definition -----
class SkinToneDataset(Dataset):
    """
    A dataset that returns (image, skin_tone) pairs.
    Assumes a CSV file with columns: image_path, R, G, B.
    """
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Read image
        img_path = self.data.iloc[idx]['image_path']
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Ground truth skin tone as a tensor of shape (3,)
        skin_tone = self.data.iloc[idx][['R', 'G', 'B']].values.astype(np.float32)
        # Normalize ground truth to [0,1]
        skin_tone = torch.tensor(skin_tone / 255.0)
        return image, skin_tone

# ----- Model Definition -----
class SkinToneNet(nn.Module):
    """
    A simple CNN that outputs an RGB triplet (normalized between 0 and 1).
    """
    def __init__(self):
        super(SkinToneNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # [B,32,H/2,W/2]
            nn.ReLU(),
            nn.MaxPool2d(2),                                       # [B,32,H/4,W/4]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # [B,64,H/4,W/4]
            nn.ReLU(),
            nn.MaxPool2d(2)                                        # [B,64,H/8,W/8]
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),  # assuming input image size 128x128; adjust accordingly
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()  # outputs in [0,1]
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# ----- Training Loop -----
def train_model(model, dataloader, criterion, optimizer, num_epochs=10, device="cpu"):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    return model

# ----- Main Function -----
if __name__ == "__main__":
    # Hyperparameters and settings
    csv_path = "skin_tone_data.csv"  # CSV file with image paths and ground truth skin tone
    num_epochs = 20
    batch_size = 16
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Create dataset and dataloader
    dataset = SkinToneDataset(csv_file=csv_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model, criterion, and optimizer
    model = SkinToneNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    print("Starting training...")
    model = train_model(model, dataloader, criterion, optimizer, num_epochs=num_epochs, device=device)
    
    # Save the trained model
    model_save_path = "skin_tone_net.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
