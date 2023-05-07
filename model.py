import torch
import torch.nn as nn


class SpeechEmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        conv_layers = []

        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        conv_layers += [self.conv1, self.relu1, self.bn1]

        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(128)
        conv_layers += [self.conv2, self.relu2, self.bn2]

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(256)
        conv_layers += [self.conv3, self.relu3, self.bn3]

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(512)
        conv_layers += [self.conv4, self.relu4, self.bn4]

        self.conv = nn.Sequential(*conv_layers)

        self.adapt_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=512, out_features=8)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.adapt_avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x