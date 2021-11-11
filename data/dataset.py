import csv
from re import I

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

label_mapping = {
    "positive": 1,
    "negative": 0,
}


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

    return IMDBDataset(train_reviews, train_labels), IMDBDataset(
        test_reviews, test_labels
    )


class IMDBDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, index):
        return self.reviews[index], self.labels[index]
