import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

label_mapping = {
    "positive": 1,
    "negative": 0,
}


def get_datasets(dataset_path, train_to_test_ratio, train_to_val_ratio, preprocessors):
    print("Reading csv...")
    with dataset_path.open() as handle:
        reader = csv.reader(handle)
        next(reader, None)  # skip header
        reviews = []
        labels = []
        for line in reader:
            reviews.append(line[0])
            labels.append(label_mapping[line[1]])

    train_reviews, test_reviews, train_labels, test_labels = train_test_split(
        reviews, labels, test_size=train_to_test_ratio, shuffle=True
    )
    train_reviews, val_reviews, train_labels, val_labels = train_test_split(
        train_reviews, train_labels, test_size=train_to_val_ratio, shuffle=True
    )

    for preprocessor in preprocessors:
        train_reviews, val_reviews, test_reviews = preprocessor(
            train_reviews, val_reviews, test_reviews
        )

    return (
        IMDBDataset(
            train_reviews,
            torch.Tensor(train_labels).unsqueeze(1),
        ),
        IMDBDataset(
            val_reviews,
            torch.Tensor(val_labels).unsqueeze(1),
        ),
        IMDBDataset(
            test_reviews,
            torch.Tensor(test_labels).unsqueeze(1),
        ),
    )


def collate_fn(batch):
    reviews = torch.stack([entry["review"] for entry in batch])
    labels = torch.stack([entry["label"] for entry in batch])

    return reviews, labels


def padding_collate_fn(batch):
    reviews = nn.utils.rnn.pad_sequence(
        [torch.tensor(entry["review"]) for entry in batch], batch_first=True
    )
    labels = torch.stack([entry["label"] for entry in batch])

    return reviews, labels


def get_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    collate_fn,
    batch_size,
    padding,
    num_workers=4,
):
    print("Initializing dataloaders...")
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=padding_collate_fn if padding else collate_fn,
    )
    dataloader_val = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=padding_collate_fn if padding else collate_fn,
    )
    dataloader_test = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=padding_collate_fn if padding else collate_fn,
    )

    return dataloader_train, dataloader_val, dataloader_test


class IMDBDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels
        print("Dataset initialized...")

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        return {"review": self.reviews[index], "label": self.labels[index]}
