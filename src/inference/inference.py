from abc import ABC, abstractmethod

import torch


class Inference(ABC):
    @abstractmethod
    def fit(self, train_loader):
        pass
    
    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass
    
    # @abstractmethod
    # def save():
    #     pass

    # @property
    # @abstractmethod
    # def num_params():
    #     pass
