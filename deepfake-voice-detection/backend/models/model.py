import torch
import torch.nn as nn
from torchvision import models

class ResNetDeepFake(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # 1. Load a pre-trained ResNet18 (trained on ImageNet)
        # We use weights='DEFAULT' or pretrained=True depending on torchvision version
        try:
            self.resnet = models.resnet18(weights='DEFAULT')
        except:
            self.resnet = models.resnet18(pretrained=True)
        
        # 2. HACK: Modify the first layer (Input)
        # Standard ResNet expects 3 channels (RGB: Red, Green, Blue).
        # Our Spectrogram only has 1 channel (Grayscale intensity).
        # We take the average of the original RGB weights to keep the "smartness".
        original_weights = self.resnet.conv1.weight.data.clone()
        # Sum across the 3 channels to make it 1 channel
        new_weights = original_weights.sum(dim=1, keepdim=True)
        
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.conv1.weight.data = new_weights
        
        # 3. HACK: Modify the last layer (Output)
        # Standard ResNet outputs 1000 classes (Dog, Cat, Car...).
        # We only want 2 classes: (0: Real, 1: Fake).
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        # x shape: [Batch_Size, 1, Freq, Time]
        logits = self.resnet(x)
        return logits