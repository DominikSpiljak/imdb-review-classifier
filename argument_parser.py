from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser()

    data = parser.add_argument_group("data")
    model = parser.add_argument_group("model")

    data.add_argument("--dataset", help="Path to dataset file", type=Path)
    data.add_argument(
        "--train-to-test-ratio",
        help="Ratio of train to test size when splitting",
        type=float,
        default=0.3,
    )

    return parser.parse_args()
