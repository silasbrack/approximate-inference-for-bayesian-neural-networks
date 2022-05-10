from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader


class Inference(ABC):
    name: str = None

    @abstractmethod
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        lr: float,
    ):
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor, aggregate: bool = True) -> torch.Tensor:
        """Returns predicted class probabilities."""
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

    def update_prior(self) -> None:
        pass
