from datetime import datetime

import torch
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

from data.dataset import get_datasets, get_dataloaders, collate_fn
from data.data_preprocessing import TfIdfPreprocessor, RNNPreprocessor
from metrics.metrics import Accuracy, Recall, Precision
from models.models import TfIdfModel, RNNModel, train, evaluate
from argument_parser import parse_args


def train_model(model_name):
    args = parse_args()

    device = torch.device(args.device)

    preprocessors = (
        [TfIdfPreprocessor(max_features=args.vocabulary)]
        if model_name == "tfidf"
        else [
            RNNPreprocessor(max_features=args.vocabulary, max_seq_len=args.max_seq_len)
        ]
    )

    # Dataset loading
    print("Loading dataset...")
    train_dataset, val_dataset, test_dataset = get_datasets(
        dataset_path=args.dataset,
        train_to_test_ratio=args.train_to_test_ratio,
        train_to_val_ratio=args.train_to_val_ratio,
        preprocessors=preprocessors,
    )

    if args.num_workers > 0 and model_name == "tfidf":
        print(
            "Pytorch multithread sparse tensors are currently not implemented, setting num workers to 0"
        )
        args.num_workers = 0

    # Dataloader initialization
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        padding=not model_name == "tfidf",
    )

    # Model initialization
    print("Initializing model...")
    model = (
        TfIdfModel(args.vocabulary, device=device)
        if model_name == "tfidf"
        else RNNModel(
            args.vocabulary + 2,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            bidirectional=args.bidirectional,
            dropout=args.dropout,
            device=device,
        )
    )

    # Optimizer initialization
    print("Initializing optimizer...")
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # Loss initialization
    print("Initializing loss...")
    loss_fn = BCEWithLogitsLoss()

    print(f"Starting training loop for {args.num_epochs} epochs...")
    train(
        model=model,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        metrics=[Accuracy(), Recall(), Precision()],
        num_epochs=args.num_epochs,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
    )

    print("Saving model state dict...")
    args.save_path.mkdir(parents=True, exist_ok=True)

    id_ = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    torch.save(
        model.state_dict(),
        args.save_path / f"{model_name}_model_{id_}.pth",
    )

    for preprocessor in preprocessors:
        preprocessor.save(args.save_path, id_)

    evaluate(
        model=model,
        dataloader=test_dataloader,
        metrics=[Accuracy(), Recall(), Precision()],
        eval_name="Test",
        device=device,
    )
