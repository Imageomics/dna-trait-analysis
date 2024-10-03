from abc import ABC, abstractmethod
from collections import defaultdict

import torch


class TrainingTracker(ABC):
    def __init__(self, options=None):
        self.options = options
        self.data_storage = defaultdict(list)

    @abstractmethod
    def record_train_batch(self, *args, **kwargs):
        pass

    @abstractmethod
    def record_test_batch(self, *args, **kwargs):
        pass

    def reset_data_storage(self):
        self.data_storage = defaultdict(list)


class BasicTrainingTracker(TrainingTracker):
    def record_train_batch(self, output, batch, loss):
        """Records training stats per batch

        Args:
            output (_type_): expected model output
            batch (_type_): expected data input batch
            loss (_type_): expected loss object
        """

        self.data_storage["training_loss"].append(loss.item())

    def record_test_batch(self, output, batch, loss):
        """Records testing stats per batch

        Args:
            output (_type_): expected model output
            batch (_type_): expected data input batch
            loss (_type_): expected loss object
        """

        self.data_storage["testing_loss"].append(loss.item())


class DNATrainingTracker(BasicTrainingTracker):
    def record_train_batch(self, output, batch, loss):
        """Records training stats per batch

        Args:
            output (_type_): expected model output
            batch (_type_): expected data input batch
            loss (_type_): expected loss object
        """
        super().record_train_batch(output, batch, loss)
        self.data_storage["training_rmse"].append(torch.sqrt(loss).item())

    def record_test_batch(self, output, batch, loss):
        """Records testing stats per batch

        Args:
            output (_type_): expected model output
            batch (_type_): expected data input batch
            loss (_type_): expected loss object
        """
        super().record_test_batch(output, batch, loss)
        self.data_storage["testing_rmse"].append(torch.sqrt(loss).item())
