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
    
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Complex CNN-Based Encoder-Decoder Model
class ComplexSignalToImageCNN(nn.Module):
    def __init__(self):
        super(ComplexSignalToImageCNN, self).__init__()

        # 1D CNN Encoder (Extract temporal-spatial features)
        self.encoder_1d = nn.Sequential(
            nn.Conv1d(in_channels=15, out_channels=64, kernel_size=5, stride=2, padding=2),  # (batch, 64, 250)
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),  # (batch, 128, 125)
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),  # (batch, 256, 63)
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),  # (batch, 512, 32)
            nn.ReLU()
        )

        # Fully Connected Layers (Latent Representation)
        self.fc1 = nn.Linear(512 * 32, 4096)  # Map to 2D feature map
        self.fc2 = nn.Linear(4096, 16 * 16 * 128)  # Reshape to 16×16 with 128 channels

        # 2D CNN Decoder (Upsample to 64×64)
        self.decoder_2d = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 64, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch, 32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # (batch, 1, 64, 64) -> Single-channel image
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        x = self.encoder_1d(x)  # 1D CNN Encoder
        x = x.view(x.shape[0], -1)  # Flatten to (batch, 512*32)
        x = self.fc1(x)  # Latent Space
        x = self.fc2(x)  # Map to 16×16×128
        
        # Reshape for 2D CNN Decoder
        x = x.view(-1, 128, 16, 16)  # (batch, 128, 16, 16)
        x = self.decoder_2d(x)  # 2D CNN Decoder
        x = x.view(-1, 64, 64)  # Final output (batch, 64, 64)
        return x

# Define Residual Block for 1D CNN
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False) \
                    if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Residual connection
        out = self.relu(out)
        return out

# Define the Complete Model
class ResNetSignalToImage(nn.Module):
    def __init__(self):
        super(ResNetSignalToImage, self).__init__()

        # 1D CNN Encoder with Residual Blocks
        self.encoder = nn.Sequential(
            ResBlock1D(15, 64),
            nn.MaxPool1d(2),  # Reduce time resolution
            ResBlock1D(64, 128),
            nn.MaxPool1d(2),
            ResBlock1D(128, 256),
            nn.MaxPool1d(2)
        )

        # Fully Connected Latent Representation
        self.fc1 = nn.Linear(256 * (500 // 8), 1024)
        self.fc2 = nn.Linear(1024, 8 * 8 * 128)

        # 2D CNN Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output normalized between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)  # 1D CNN Encoder
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc1(x)  # Latent Space
        x = self.fc2(x)  # Map to 16×16×128

        # Reshape for 2D CNN Decoder
        x = x.view(-1, 128, 8, 8)
        x = self.decoder(x)  # 2D CNN Decoder
        x = x.view(-1, 64, 64)  # Final output
        return x

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, pool_size=2):
        super(TransformerBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(pool_size)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x


class XformNetSignalToImage(nn.Module):
    def __init__(self):
        super(XformNetSignalToImage, self).__init__()
        # Transformer blocks
        # 1D CNN Encoder with Residual Blocks
        self.encoder = nn.Sequential(
            TransformerBlock(15, 64),
            TransformerBlock(64, 128),
            TransformerBlock(128, 256)
        )

        # Fully Connected Latent Representation
        self.fc1 = nn.Linear(256 * (500 // 8), 1024)
        self.fc2 = nn.Linear(1024, 8 * 8 * 128)

        # 2D CNN Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output normalized between 0 and 1
        )


    def forward(self, x):
        x = self.encoder(x)  # 1D CNN Encoder
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc1(x)  # Latent Space
        x = self.fc2(x)  # Map to 16×16×128

        # Reshape for 2D CNN Decoder
        x = x.view(-1, 128, 8, 8)
        x = self.decoder(x)  # 2D CNN Decoder
        x = x.view(-1, 64, 64)  # Final output
        return x