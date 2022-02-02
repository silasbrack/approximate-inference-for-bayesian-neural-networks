import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import fire
from src.data.fashion_mnist import FashionMNISTData

from src.models import MNISTModel
from src import data as d


def mnist(
    path: str,  # Path to log folder, e.g., outputs/2022-02-02/15-15-26/
    checkpoint_file: str,  # epoch=14-step=6449.ckpt
):
    # hparams: DictConfig = DictConfig(
    #     OmegaConf.load(f"{path}/logs/mnist_model/version_0/hparams.yaml")
    # )
    # model: MNISTModel = MNISTModel(**hparams)
    # state_dict_path = f"{path}/models/state_dict.pt"
    # model.load_state_dict(torch.load(state_dict_path))

    cfg: DictConfig = DictConfig(OmegaConf.load(f"{path}/.hydra/config.yaml"))
    model: MNISTModel = MNISTModel.load_from_checkpoint(
        checkpoint_path=f"{path}/{cfg.paths.checkpoints}/{checkpoint_file}"
    )
    data: FashionMNISTData = d.FashionMNISTData(
        "data/", cfg.params.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    trainer = pl.Trainer()
    trainer.test(model, data.test_dataloader())


if __name__ == "__main__":
    fire.Fire()
