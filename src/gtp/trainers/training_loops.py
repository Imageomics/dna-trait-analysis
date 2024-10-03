from abc import ABC, abstractmethod


class TrainingLoop(ABC):
    def __init__(self, options=None):
        self.options = options

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def test(self, *args, **kwargs):
        pass


class BasicTrainingLoop(TrainingLoop):
    def train(self, train_dataloader, model, loss_fn, optimizer, tracker):
        """Trains the model once through an entire dataloader

        Args:
            train_dataloader (_type_): Standard dataloader for training
            model (_type_): model with forward implemented.
            loss_fn (_type_): loss function given output and batch
            optimizer (_type_): optimizer for the model
            tracker (_type_): data tracker
        """

        for batch in train_dataloader:
            output = model(batch)
            loss = loss_fn(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tracker.record_train_batch(output, batch, loss)

    def test(self, test_dataloader, model, loss_fn, tracker):
        """Runs the model once through an entire dataloader

        Args:
            test_dataloader (_type_): Standard dataloader for training
            model (_type_): model with forward implemented.
            loss_fn (_type_): loss function given output and batch
            tracker (_type_): data tracker
        """

        for batch in test_dataloader:
            output = model(batch)
            loss = loss_fn(output, batch)
            tracker.record_test_batch(output, batch, loss)
