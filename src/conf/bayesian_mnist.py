from dataclasses import dataclass


@dataclass
class Params:
    epochs: int = 3
    lr: float = 5e-3
    batch_size: int = 1024


@dataclass
class Hardware:
    gpus: int = 0
    num_workers: int = 16


@dataclass
class Paths:
    #   project: ${hydra:runtime.cwd}
    #   data: ${hydra:runtime.cwd}/data/processed/
    checkpoints: str = "models/checkpoints/weights/"
    model: str = "models/"
    logs: str = "logs/"


@dataclass
class Files:
    train: str = "train.csv"
    test: str = "test.csv"
    state_dict: str = "state_dict.pt"
