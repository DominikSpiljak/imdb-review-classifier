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


class RNNModel(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_size,
        hidden_size,
        bidirectional,
        dropout,
        device,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=embedding_size,
            padding_idx=0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            batch_first=True,
        )

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.kaiming_normal_(param)

        self.classification_layer = nn.Linear(hidden_size, 1)
        self.device = device
        self.to(self.device)

    def forward(self, X):
        embedded = self.embedding(X)
        output, hidden = self.rnn(self.dropout(embedded))
        return self.classification_layer(output[:, -1, :])


def train(
    model,
    dataloader,
    val_dataloader,
    metrics,
    num_epochs,
    loss_fn,
    optimizer,
    device,
):
    model.train()

    for epoch in range(num_epochs):

        for metric in metrics:
            metric.initialize()

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

                probs = torch.sigmoid(preds)

                for metric in metrics:
                    metric.log_batch(probs, y)

                pbar.set_postfix(loss=loss.item())

        print(f"Evaluation results for Train:")
        for metric in metrics:
            metric.compute()

        evaluate(model, val_dataloader, metrics, "Validation", device)


def evaluate(model, dataloader, metrics, eval_name, device):
    model.eval()

    for metric in metrics:
        metric.initialize()

    with tqdm(dataloader, unit="batch") as pbar:
        for batch in pbar:

            X, y = batch
            X = X.to(device)
            y = y.to(device)

            preds = model(X)

            probs = torch.sigmoid(preds)

            for metric in metrics:
                metric.log_batch(probs, y)

    print(f"Evaluation results for {eval_name}:")
    for metric in metrics:
        metric.compute()

    model.train()
