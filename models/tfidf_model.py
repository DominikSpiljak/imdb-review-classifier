from tqdm import tqdm

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


def train(model, dataloader, num_epochs, loss_fn, optimizer, device):

    for epoch in range(num_epochs):
        with tqdm(dataloader, unit="batch") as pbar:
            for batch in pbar:
                pbar.set_description(f"Epochs {epoch}")

                X, y = batch
                X = X.to(device)
                y = y.to(device)

                preds = model(X)

                loss = loss_fn(preds, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                pbar.set_postfix(loss=loss.item())


def eval(model, dataloader, device):
    # TODO: Implement eval loop
    pass
