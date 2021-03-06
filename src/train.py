import logging
import pickle

import hydra
import torch
from omegaconf import DictConfig

from src.evaluate import evaluate, print_dict


@hydra.main(config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    if cfg.training.seed:
        torch.manual_seed(cfg.training.seed)

    data = hydra.utils.instantiate(cfg.data)
    data.setup()

    inference = hydra.utils.instantiate(cfg.inference)
    train_result = inference.fit(
        data.train_dataloader(),
        data.val_dataloader(),
        cfg.training.epochs,
        cfg.training.lr,
    )
    print_dict(train_result)
    with open("training.pkl", "wb") as f:
        pickle.dump(train_result, f)
    eval_result = evaluate(
        inference, data.test_dataloader(), data.name, data.n_classes
    )
    print_dict(eval_result)
    with open("results.pkl", "wb") as f:
        pickle.dump(eval_result, f)
    inference.save(cfg.training.model_path)


if __name__ == "__main__":
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.INFO)
    train()
