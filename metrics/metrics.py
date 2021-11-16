from .metrics_interface import Metric


class Accuracy(Metric):
    def initialize(self):
        self.correct = 0
        self.total = 0

    def log_batch(self, predicted, ground_truth):
        self.correct += ((predicted >= 0.5).int() == ground_truth).sum().item()
        self.total += len(ground_truth)

    def compute(self):
        print(f"Accuracy: {self.correct / self.total}")


# TODO: Add more metrics
