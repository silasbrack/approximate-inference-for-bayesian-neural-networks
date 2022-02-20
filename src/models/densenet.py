import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torch
import pytorch_lightning as pl


class Densenet(nn.Module):
    def __init__(self, num_class=1, pretrained=False):
        super().__init__()
        self.channels = 1664
        densenet_169 = models.densenet169(pretrained=pretrained)
        for params in densenet_169.parameters():
            params.requires_grad_(False)
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=4)
        self.features = nn.Sequential(*list(densenet_169.features.children()))
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(self.channels, num_class)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        features = self.features(x)
        out = self.relu(features)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(-1, self.channels)
        return self.sigmoid(self.fc1(out))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
