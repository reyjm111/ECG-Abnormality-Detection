import torch
import torch.nn as nn

class ECGCNN1D(nn.Module):

    def __init__(self):
        
        super().__init__()
        
        # 1D CNN Model
        self.features = nn.Sequential(

            nn.Conv1d(1, 16, kernel_size=5, padding=2), # 16 filters over 5 time points, more primitive features are learned
            nn.BatchNorm1d(16), # normalize activations
            nn.ReLU(inplace=True), # adds nonlinearity by making negative values into 0 
            nn.MaxPool1d(kernel_size=2), # reduces temporal resolution and computation

            nn.Conv1d(16, 32, kernel_size=5, padding=2), # learning more complex features
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1), # learning higher level features
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1) # reduces numbers of parameters to a fixed length
        )

        self.classifier = nn.Sequential(
            nn.Flatten(), # (batch, 64, 1) to (batch, 64)
            nn.Dropout(0.2), # reduce overfitting
            nn.Linear(64, 1) # output binary
        )

    def forward(self, x):

        x = self.features(x)     # obtain features (batch, 64, 1)
        x = self.classifier(x)   # classify based on features (batch, 1)

        return x