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
    eval_result = evaluate(model,
                           data.test_dataloader(),
                           data.name,
                           data.n_classes)
    accuracy = eval_result["Accuracy"]
    logging.info(f"{accuracy=:.3f}")

    # ensemble_accuracies = [evaluate(ensemble,
    #                                 data.test_dataloader(),
    #                                 data.name,
    #                                 data.n_classes)["Accuracy"]
    #                        for ensemble in model.ensembles]
    # logging.info(ensemble_accuracies)



if __name__ == "__main__":
    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.INFO)
    fire.Fire(predict)
