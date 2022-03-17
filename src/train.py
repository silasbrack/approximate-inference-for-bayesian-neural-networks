import logging

import hydra
import torch
from omegaconf import DictConfig

from src.models import DenseNet
from src.data.helper import load_and_setup_data
from src.evaluate import evaluate
from src.inference import VariationalInference, NeuralNetwork, DeepEnsemble


@hydra.main(config_path="../conf", config_name="deep_ensemble")
def train(cfg: DictConfig):
    if cfg.training.seed:
        torch.manual_seed(cfg.training.seed)

    data = load_and_setup_data(cfg.training.dataset, cfg.paths.data, cfg.training.batch_size, cfg.hardware.num_workers)
    device = torch.device("cuda" if cfg.hardware.gpus else "cpu")

    model = DenseNet
    inference = NeuralNetwork(model, device, cfg.num_ensembles)
    inference.fit(data.train_dataloader(), data.val_dataloader(), cfg.training.epochs, cfg.training.lr)
    print(evaluate(inference, data.test_dataloader(), "mnist", 10))
    # inference.save(path)


if __name__ == "__main__":
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.INFO)

    train()
