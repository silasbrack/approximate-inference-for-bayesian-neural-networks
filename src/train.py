import logging

import hydra
import torch
from omegaconf import DictConfig

from src.evaluate import evaluate


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
    print(train_result)
    inference.save(cfg.training.model_path)
    eval_result = evaluate(inference,
                           data.test_dataloader(),
                           data.name,
                           data.n_classes)
    accuracy = eval_result["Accuracy"]
    logging.info(f"{accuracy=:.3f}")
    print(eval_result)


if __name__ == "__main__":
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.INFO)
    train()
