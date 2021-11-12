from datetime import datetime

import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

from data.dataset import get_datasets, get_dataloaders, collate_fn
from data.data_preprocessing import TfIdfPreprocessor
from models.tfidf_model import TfIdfModel, train, eval
from argument_parser import parse_args


def main():
    args = parse_args()

    device = torch.device(args.device)

    # Dataset loading
    print("Loading dataset...")
    train_dataset, test_dataset = get_datasets(
        dataset_path=args.dataset,
        train_to_test_ratio=args.train_to_test_ratio,
        preprocessors=[TfIdfPreprocessor(max_features=args.tfidf_max_features)],
    )

    # Dataloader initialization
    train_dataloader, test_dataloader = get_dataloaders(
        train_dataset,
        test_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model initialization
    print("Initializing model...")
    model = TfIdfModel(args.tfidf_max_features, device=device)

    # Optimizer initialization
    print("Initializing optimizer...")
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Loss initialization
    print("Initializing loss")
    loss_fn = BCEWithLogitsLoss()

    print(f"Starting training loop for {args.num_epochs} epochs...")
    train(
        model=model,
        dataloader=train_dataloader,
        num_epochs=args.num_epochs,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    print("Saving model state dict...")
    args.save_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        args.save_path / f"model_{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}.pth",
    )


if __name__ == "__main__":
    main()
