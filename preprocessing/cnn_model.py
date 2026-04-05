import torch
import torch.nn as nn

class ECGCNN1D(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        self.features = nn.Sequential(

            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):

        x = self.features(x)     # (batch, 64, 1)
        x = self.classifier(x)   # (batch, 1)

        return x