import logging

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

    # TODO: Clean this part up, doesn't seem right.
    recursive = False \
        if cfg.inference["_target_"] in ["src.inference.DeepEnsemble",
                                         "src.inference.MultiSwag"] \
        else True
    inference = hydra.utils.instantiate(cfg.inference, _recursive_=recursive)
    train_result = inference.fit(
        data.train_dataloader(),
        data.val_dataloader(),
        cfg.training.epochs,
        cfg.training.lr,
    )
    print_dict(train_result)
    inference.save(cfg.training.model_path)
    eval_result = evaluate(
        inference, data.test_dataloader(), data.name, data.n_classes
    )
    print_dict(eval_result)


if __name__ == "__main__":
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.INFO)
    train()
