import os
import torch
from pytorch_lightning import LightningModule
import hydra
from omegaconf import DictConfig

from src.models import MNISTModel


def load_model(cfg: DictConfig, experiment_location: str) -> LightningModule:
    model = MNISTModel(
        cfg.paths.data,
        cfg.params.lr,
        cfg.params.batch_size,
    )
    state_dict_path = experiment_location + cfg.paths.model + cfg.files.state_dict
    model.load_state_dict(torch.load(state_dict_path))
    return model


@hydra.main(config_path="../conf", config_name="mnist")
def predict_model(model_input: torch.Tensor, model: LightningModule):
    model = load_model()
    return model(model_input)
