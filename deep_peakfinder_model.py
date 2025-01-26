import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

# create some sample input data
# x = torch.randn(1, 3, 256, 256)

# generate predictions for the sample data
# y = MyPyTorchModel()(x)

# generate a model architecture visualization
# make_dot(y.mean(),
#          params=dict(MyPyTorchModel().named_parameters()),
#          show_attrs=True,
#          show_saved=True).render("MyPyTorchModel_torchviz", format="png")

class ResidualBlock1D(nn.Module):
    """
    A 1D residual block consisting of sequential convolutional layers with residual connections.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=nn.ReLU):
        super(ResidualBlock1D, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.activation = activation()
        self.downsample = None

        # Downsample if input and output dimensions do not match
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.activation(out)
        return out


class ResNet1D(nn.Module):
    """
    A configurable 1D ResNet.
    """
    def __init__(self, input_channels, output_size, block_configs):
        super(ResNet1D, self).__init__()
        # self.bn1 = nn.BatchNorm1d(input_channels)

        self.blocks = nn.ModuleList()

        # Construct the ResNet from block_configs
        for config in block_configs:
            block = ResidualBlock1D(
                in_channels=config["in_channels"],
                out_channels=config["out_channels"],
                kernel_size=config["kernel_size"],
                stride=config["stride"],
                activation=config.get("activation", nn.ReLU)
            )
            self.blocks.append(block)

        # Fully connected layer for classification/regression (adjustable)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        self.fc = nn.Linear(block_configs[-1]["out_channels"], output_size)  # Output a single value

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Remove the last dimension
        x = self.fc(x)
        return x


class PeakDetectionNet(nn.Module):
    def __init__(self):
        super(PeakDetectionNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, stride=1, padding=2)  # (32, 400)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)  # (64, 400)

        # Downsampling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # (64, 200)

        # Third convolutional layer
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # (128, 200)

        # Upsampling layer
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)  # (128, 400)

        # Final convolutional layer
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)  # (1, 400)

    def forward(self, x):
        # Forward pass through the network
        x = F.relu(self.conv1(x))  # Conv1
        x = F.relu(self.conv2(x))  # Conv2
        x = self.pool(x)           # MaxPool
        x = F.relu(self.conv3(x))  # Conv3
        x = self.upsample(x)       # Upsample
        x = torch.sigmoid(self.conv4(x))  # Conv4 + Sigmoid for binary output
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


class PeakDetectionXformNet(nn.Module):
    def __init__(self, input_channels=2, output_length=400, num_classes=2):
        super(PeakDetectionXformNet, self).__init__()
        # Transformer blocks
        self.num_classes = num_classes
        self.block1 = TransformerBlock(input_channels, 32)
        self.block2 = TransformerBlock(32, 64)
        self.block3 = TransformerBlock(64, 128)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(128 * (output_length // 8), 256)  # Adjust based on downsampled size
        self.fc2 = nn.Linear(256, output_length * num_classes)  # Output (2 * 400)


    def forward(self, x):
        # Pass through transformer blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation here; softmax applied during evaluation
        x = x.view(x.shape[0], self.num_classes, -1)
        # x = F.softmax(x, dim=1)
        return x
