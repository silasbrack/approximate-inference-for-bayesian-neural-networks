import logging
import os

import fire
import hydra
from omegaconf import DictConfig, OmegaConf

from src.evaluate import evaluate, print_dict
from src.inference.inference import Inference


def load_model(path: str, device="cpu"):
    config_path = os.path.join(path, ".hydra", "config.yaml")
    cfg = DictConfig(OmegaConf.load(config_path))
    cfg.data.data_dir = os.path.join(os.getcwd(), "data/")
    cfg.inference.device = device

    recursive = (
        False
        if cfg.inference["_target_"]
        in ["src.inference.DeepEnsemble", "src.inference.MultiSwag"]
        else True
    )
    inference: Inference = hydra.utils.instantiate(
        cfg.inference, _recursive_=recursive
    )
    inference.load(path)

    data = hydra.utils.instantiate(cfg.data)
    data.setup()
    return inference, data


def predict(path: str):
    config_path = os.path.join(path, ".hydra", "config.yaml")
    cfg = DictConfig(OmegaConf.load(config_path))
    cfg.data.data_dir = os.path.join(os.getcwd(), "data/")

    inference: Inference = hydra.utils.instantiate(cfg.inference)
    inference.load(path)

    data = hydra.utils.instantiate(cfg.data)
    data.setup()
    eval_result = evaluate(
        inference, data.test_dataloader(), data.name, data.n_classes
    )
    print_dict(eval_result)


if __name__ == "__main__":
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.INFO)
    fire.Fire(predict)
