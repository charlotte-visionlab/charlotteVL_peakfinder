import torch
import torch.nn as nn
    
# CNN Encoder-Decoder Model
class SignalToImageCNN(nn.Module):
    def __init__(self):
        super(SignalToImageCNN, self).__init__()
        
        # Encoder: 1D CNN to extract features
        self.encoder = nn.Sequential(
            nn.Conv1d(15, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        
        # Flatten and reshape into spatial feature maps
        self.fc1 = nn.Linear(256 * 63, 1024)  # Latent space
        self.fc2 = nn.Linear(1024, 64 * 64)  # Map to image

    def forward(self, x):
        x = self.encoder(x)  # 1D CNN Encoder
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 64, 64)  # Reshape to 64×64
        return x  # No extra channel dimension