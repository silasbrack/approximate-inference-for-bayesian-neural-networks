import copy

import hydra
import torch
from omegaconf import DictConfig
from torch.nn.functional import nll_loss
from torch.optim.swa_utils import AveragedModel

from src import data as d
from src.models import MNISTModel


def train_swa(cfg: DictConfig):

    data_dict = {
        "mnist": d.MNISTData,
        "fashionmnist": d.FashionMNISTData,
        "cifar": d.CIFARData,
        "svhn": d.SVHNData,
    }
    data = data_dict[cfg.training.dataset](
        cfg.paths.data, cfg.training.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    model = MNISTModel(cfg.training.lr)
    swa_model = AveragedModel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=100)
    loss_fn = nll_loss
    swa_start = int(cfg.swa_start_thresh * cfg.training.epochs)
    # swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    state_dicts = []

    for epoch in range(cfg.training.epochs):
        for image, target in data.train_dataloader():
            optimizer.zero_grad()
            loss_fn(model(image), target).backward()
            optimizer.step()
        if epoch > swa_start:
            swa_model.update_parameters(model)
            state_dicts.append(copy.deepcopy(model.state_dict()))
            # swa_scheduler.step()
        # else:
        #     scheduler.step()

    # accuracy_calculator = Accuracy()
    # for image, target in data.test_dataloader():
    #     logits = model(image)
    #     accuracy_calculator(logits, target)
    # accuracy = accuracy_calculator.compute()
    # print(f"Test accuracy for normal = {100*accuracy:.2f}")
    #
    # accuracy_calculator = Accuracy()
    # for image, target in data.test_dataloader():
    #     logits = swa_model(image)
    #     accuracy_calculator(logits, target)
    # accuracy = accuracy_calculator.compute()
    # print(f"Test accuracy for SWA = {100*accuracy:.2f}")

    state_dicts = {
        k: torch.stack([sd[k] for sd in state_dicts]) for k in state_dicts[0]
    }

    return state_dicts


@hydra.main(config_path="../../conf", config_name="swa")
def run(cfg: DictConfig):
    train_swa(cfg)


if __name__ == "__main__":
    run()
