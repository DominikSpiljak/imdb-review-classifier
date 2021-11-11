import torch

from data.dataset import get_datasets
from data.data_preprocessing import TfIdfPreprocessor
from models.tfidf_model import TfIdfModel
from argument_parser import parse_args


def main():
    args = parse_args()

    # Dataset loading
    print("Loading dataset...")
    train_dataset, test_dataset = get_datasets(
        dataset_path=args.dataset,
        train_to_test_ratio=args.train_to_test_ratio,
        preprocessors=[TfIdfPreprocessor()],
    )

    # Model initialization
    print("Initializing model...")
    model = TfIdfModel(5, device=torch.device("cpu"))


if __name__ == "__main__":
    main()
