from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser()

    data = parser.add_argument_group("data")
    model = parser.add_argument_group("model")
    training = parser.add_argument_group("training")

    data.add_argument("--dataset", help="Path to dataset file", type=Path)
    data.add_argument(
        "--train-to-test-ratio",
        help="Ratio of train to test size when splitting",
        type=float,
        default=0.3,
    )
    data.add_argument(
        "--train-to-val-ratio",
        help="Ratio of train to test size when splitting",
        type=float,
        default=0.1,
    )

    model.add_argument(
        "--vocabulary",
        help="Max features to be used during tfidf vectorization",
        type=int,
        default=20_000,
    )
    model.add_argument(
        "--embedding-size",
        help="Embedding size to be used in rnn model",
        type=int,
        default=64,
    )
    model.add_argument(
        "--hidden-size",
        help="Hidden size to be used in rnn model",
        type=int,
        default=64,
    )
    model.add_argument(
        "--bidirectional",
        help="Whether to use bidirectinal rnn model",
        action="store_true",
    )
    model.add_argument(
        "--dropout",
        help="Dropout for rnn model",
        type=float,
        default=0.3,
    )

    training.add_argument(
        "--max-seq-len",
        help="Maximum sequence length when training rnn",
        type=int,
        default=128,
    )
    training.add_argument(
        "--device",
        help="Device for running the training",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    training.add_argument(
        "--batch-size", help="Batch size for training", type=int, default=64
    )
    training.add_argument(
        "--num-workers",
        help="Number of workers for parallel data loading",
        type=int,
        default=4,
    )
    training.add_argument(
        "--learning-rate", help="Learning rate for training", type=float, default=1e-3
    )
    training.add_argument(
        "--num-epochs", help="Number of epochs", type=int, default=100
    )
    training.add_argument(
        "--save-path",
        help="Path to save models",
        type=Path,
        default=Path("model_checkpoints/"),
    )

    return parser.parse_args()
