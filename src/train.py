import hydra
import torch
from omegaconf import DictConfig


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


if __name__ == "__main__":
    train()
