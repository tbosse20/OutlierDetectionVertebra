import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple3DCNN(nn.Module):
    def __init__(self):
        super(Simple3DCNN, self).__init__()
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(in_channels=2, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Max-pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layers
        # Ensure the dimensions match after pooling and flattening
        # self.fc1 = nn.Linear(128 * 26 * 26 * 26, 256)  # Adjust based on final feature map size
        # self.fc2 = nn.Linear(256, 64)
        # self.fc3 = nn.Linear(64, 1)  # Binary classification

        self.fc_in = 128 * 15 ** 3
        self.fc = nn.Linear(self.fc_in, 1)
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Output: (N, 32, D/2, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (N, 64, D/4, H/4, W/4)
        x = self.pool(F.relu(self.conv3(x)))  # Output: (N, 128, D/8, H/8, W/8)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, self.fc_in)  # Adjust based on output size after convolution
        
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        
        x = torch.sigmoid(self.fc(x))  # Binary output
        
        return x

# # Example usage
# model = Simple3DCNN()