import logging

import torch
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd

from src.inference.nn import NeuralNetwork

logger = logging.getLogger(__name__)


class McDropout(NeuralNetwork):
    def __init__(
        self, model, device: str, prior: bool, posterior_samples: int
    ):
        super().__init__(model, device, prior)
        self.posterior_samples = posterior_samples

        if not any(
            isinstance(layer, _DropoutNd) for layer in self.model.children()
        ):
            logging.warn("No Dropout layers detected in model.")

    def predict(self, x, aggregate=True):
        x = x.to(self.device)
        self.model.train()
        logits = torch.stack(
            [self.model(x) for _ in range(self.posterior_samples)]
        )
        if aggregate:
            logits = logits.mean(dim=0)
        return F.softmax(logits, dim=-1)
