import os

import torch
from torch.nn import functional as F

from src.inference.inference import Inference


class NeuralNetwork(Inference):
    def __init__(self, model, device):
        self.model = model()
        self.device = device

    def fit(self, train_loader, val_loader, epochs, lr):
        optim = torch.optim.Adam(self.model.parameters(), lr)
        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optim.zero_grad()
                logits = self.model(x)
                loss = F.nll_loss(logits, y)
                loss.backward()
                optim.step()
    
    def predict(self, x):
        x = x.to(self.device)
        logits = self.model(x)
        return F.softmax(logits, dim=-1)
    
    def save(folder):
        torch.save(
            model.state_dict(),
            os.path.join(folder, "state_dict.pt"),
        )
