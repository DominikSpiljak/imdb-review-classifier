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


class Recall(Metric):
    def initialize(self):
        self.tp = 0
        self.fn = 0

    def log_batch(self, predicted, ground_truth):
        predicted = (predicted >= 0.5).int()
        for p, gt in zip(predicted, ground_truth):
            if p.item() == 1 and int(gt.item()) == 1:
                self.tp += 1
            elif p.item() == 0 and int(gt.item()) == 1:
                self.fn += 1

    def compute(self):
        print(f"Recall: {self.tp / (self.tp + self.fn)}")


class Precision(Metric):
    def initialize(self):
        self.tp = 0
        self.fp = 0

    def log_batch(self, predicted, ground_truth):
        predicted = (predicted >= 0.5).int()
        for p, gt in zip(predicted, ground_truth):
            if p.item() == 1 and int(gt.item()) == 1:
                self.tp += 1
            elif p.item() == 1 and int(gt.item()) == 0:
                self.fp += 1

    def compute(self):
        print(f"Precision: {self.tp / (self.tp + self.fp)}")


# TODO: Add more metrics
