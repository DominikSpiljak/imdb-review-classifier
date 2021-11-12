import csv

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

label_mapping = {
    "positive": 1,
    "negative": 0,
}


def csr_matrix_to_sparse_tensor(csr_matrix):
    matrix_coo = csr_matrix.tocoo()

    return torch.sparse.FloatTensor(
        torch.LongTensor([matrix_coo.row.tolist(), matrix_coo.col.tolist()]),
        torch.FloatTensor(matrix_coo.data.astype(np.float)),
    )


def get_datasets(dataset_path, train_to_test_ratio, preprocessors):
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

    for preprocessor in preprocessors:
        train_reviews, test_reviews = preprocessor(train_reviews, test_reviews)

    return IMDBDataset(
        csr_matrix_to_sparse_tensor(train_reviews),
        torch.Tensor(train_labels).unsqueeze(1),
    ), IMDBDataset(
        csr_matrix_to_sparse_tensor(test_reviews),
        torch.Tensor(test_labels).unsqueeze(1),
    )


def collate_fn(batch):
    reviews = torch.stack([entry["review"] for entry in batch])
    labels = torch.stack([entry["label"] for entry in batch])

    return reviews, labels


def get_dataloaders(train_dataset, test_dataset, collate_fn, batch_size, num_workers=4):
    print("Initializing dataloaders...")
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    dataloader_test = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return dataloader_train, dataloader_test


class IMDBDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels
        print("Dataset initialized...")
        print(f"Reviews shape: {self.reviews.shape}, Labels shape {self.labels.shape}")

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        return {"review": self.reviews[index], "label": self.labels[index]}
