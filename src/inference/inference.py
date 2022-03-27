from abc import ABC, abstractmethod

import torch


class Inference(ABC):
    @abstractmethod
    def fit(self, train_loader, val_loader, epochs, lr):
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns predicted class probabilities.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        pass

    @property
    @abstractmethod
    def num_params(self) -> int:
        pass
