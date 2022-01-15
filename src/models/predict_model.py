import torch
from pytorch_lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from src.models import MNISTModel


def predict_model(
    path: str,  # Path to log folder, e.g., outputs/2022-01-15/14-36/59
    model_input,
):
    cfg: DictConfig = OmegaConf.load(f"{path}/models/logs/hparams.yaml")
    model: LightningModule = MNISTModel(**cfg)
    state_dict_path = f"{path}/models/state_dict.pt"
    model.load_state_dict(torch.load(state_dict_path))

    return model(model_input)
