import logging

import hydra
import torch
from omegaconf import DictConfig

from src.models import DenseNet
from src.data.helper import load_and_setup_data
from src.evaluate import evaluate
from src.inference import VariationalInference, NeuralNetwork, DeepEnsemble


@hydra.main(config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    if cfg.training.seed:
        torch.manual_seed(cfg.training.seed)
    logging.captureWarnings(True)
    logging.getLogger().setLevel(cfg.training.logging)

    data = hydra.utils.instantiate(cfg.data)
    data.setup()

    inference = hydra.utils.instantiate(cfg.inference)
    train_result = inference.fit(data.train_dataloader(), data.val_dataloader(), cfg.training.epochs, cfg.training.lr)
    eval_result = evaluate(inference, data.test_dataloader(), "mnist", 10)
    print(train_result, eval_result)
    # inference.save(path)


if __name__ == "__main__":

    train()
