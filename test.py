from pathlib import Path
import torch

from models.models import TfIdfModel
from data.data_preprocessing import TfIdfPreprocessor


def main():
    preprocessor = TfIdfPreprocessor.load_from_checkpoint(
        list(Path("test_checkpoints").glob("*.pkl"))[0]
    )
    model_checkpoint = torch.load(list(Path("test_checkpoints").glob("*.pth"))[0])

    model = TfIdfModel(input_dim=5000, device="cpu")
    model.load_state_dict(model_checkpoint)
    model.eval()

    while True:
        review = input(">> ")
        vectorized = preprocessor.transform([review])

        print(
            f"The review is {torch.sigmoid(model(vectorized)).item() * 100}% positive."
        )


if __name__ == "__main__":
    main()
