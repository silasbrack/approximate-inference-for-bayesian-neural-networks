import logging
import os

import fire
import hydra
from omegaconf import DictConfig, OmegaConf

from src.evaluate import evaluate
from src.inference.inference import Inference


def predict(path: str):
    config_path = os.path.join(path, ".hydra", "config.yaml")
    cfg = DictConfig(OmegaConf.load(config_path))
    cfg.data.data_dir = "/home/silas/Documents/university/approximate" \
                        "-inference-for-bayesian-neural-networks/data/"

    model: Inference = hydra.utils.instantiate(cfg.inference)
    model.load(path)

    data = hydra.utils.instantiate(cfg.data)
    data.setup()
    eval_result = evaluate(model, data.test_dataloader(), "mnist", 10)
    accuracy = eval_result["Accuracy"]
    print(f"{accuracy=:.3f}")


if __name__ == "__main__":
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.INFO)
    fire.Fire(predict)
