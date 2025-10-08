import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    """
    Plain 3D CNN (no Batch/Group/InstanceNorm), stable for batch_size=1.
    Global avg pool avoids hard-coded shapes.
    """
    def __init__(self, in_ch=1, num_classes=2, dropout=0.20):
        super().__init__()
        act = nn.LeakyReLU(0.1, inplace=True)

        self.features = nn.Sequential(
            nn.Conv3d(in_ch, 32, 3, padding=1), act, nn.MaxPool3d(2),   # /2
            nn.Conv3d(32, 64, 3, padding=1),   act, nn.MaxPool3d(2),   # /4
            nn.Conv3d(64, 128, 3, padding=1),  act, nn.MaxPool3d(2),   # /8
            nn.Conv3d(128, 192, 3, padding=1), act,
            nn.AdaptiveAvgPool3d(1),
            nn.Dropout(p=dropout),
        )
        self.head = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)  # logits


