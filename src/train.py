import logging
import os
import pickle

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import wandb

from src.evaluate import evaluate, print_dict


@hydra.main(config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    wandb.init(
        project=cfg.project, settings=wandb.Settings(start_method="thread")
    )
    wandb.config.update({"Experiment directory": os.getcwd()})
    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

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
    for metric, val in train_result.items():
        if not isinstance(val, torch.Tensor) and not isinstance(
            val, np.ndarray
        ):
            wandb.summary[metric] = val
    with open("training.pkl", "wb") as f:
        pickle.dump(train_result, f)
    eval_result = evaluate(
        inference, data.test_dataloader(), data.name, data.n_classes
    )
    for metric, val in eval_result.items():
        if not isinstance(val, torch.Tensor) and not isinstance(
            val, np.ndarray
        ):
            wandb.summary[metric] = val

    n_classes = eval_result["Test probabilities"].shape[-1]
    table = wandb.Table(
        columns=["Test targets"],
        data=[np.expand_dims(eval_result["Test targets"], axis=0).tolist()],
    )
    for i in range(n_classes):
        table.add_column(
            f"Class {i} probability",
            data=np.expand_dims(
                eval_result["Test probabilities"][:, i], axis=0
            ).tolist(),
        )
    wandb.summary["Predictions"] = table

    print_dict(eval_result)
    with open("results.pkl", "wb") as f:
        pickle.dump(eval_result, f)
    inference.save(cfg.training.model_path)
    wandb.finish()


if __name__ == "__main__":
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.INFO)
    train()
