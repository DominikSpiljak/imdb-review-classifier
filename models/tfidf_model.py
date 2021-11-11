import torch
import torch.nn as nn


class TfIdfModel(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1),
        )
        self.device = device
        self.to(self.device)

    def forward(self, X):
        return self.model.forward(X)
