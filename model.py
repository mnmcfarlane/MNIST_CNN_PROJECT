
import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Detect simple visual patterns, such as edges.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)

        # Combine simple patterns into more complex features.
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Reduce image dimensions.
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Help prevent overfitting.
        self.dropout = nn.Dropout(0.3)

        # Classify the extracted image features.
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))

        return self.fc2(x)
