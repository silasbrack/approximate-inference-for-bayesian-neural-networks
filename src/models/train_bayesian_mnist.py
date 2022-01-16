import hydra
from omegaconf import DictConfig

import pyro
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro import poutine
from tqdm.auto import tqdm
from src.data.mnist import MNISTData

from src.models.bayesian_mnist import BayesianMnistModel


# TODO: currently, the results seem to get worse as you train past 1 epoch
@hydra.main(config_path="../conf", config_name="bayesian_mnist")
def train_model(cfg: DictConfig):

    model = BayesianMnistModel(cfg.params.lr)
    data = MNISTData(
        cfg.paths.data, cfg.params.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    guide = AutoDiagonalNormal(poutine.block(model, hide=["obs"]))
    # guide = AutoDiagonalNormal(model)  # Mean-field
    adam = pyro.optim.Adam({"lr": cfg.params.lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    x, y = next(iter(data.val_dataloader()))
    prediction = Predictive(model, guide=guide, num_samples=512)(x)
    preds = prediction["obs"].mode(dim=0).values  # MAP prediction
    acc = model.accuracy(preds, y)
    print(f"Accuracy before training: {100*acc:.2f}%")

    train_loader = data.train_dataloader()
    pyro.clear_param_store()
    for epoch in range(cfg.params.epochs):
        bar = tqdm(train_loader)
        for batch in bar:
            loss = svi.step(*batch)
            bar.set_postfix(loss=f"{loss / batch[0].shape[0]:.3f}")

    x, y = next(iter(data.val_dataloader()))
    prediction = Predictive(model, guide=guide, num_samples=512)(x)
    preds = prediction["obs"].mode(dim=0).values  # MAP prediction
    acc = model.accuracy(preds, y)
    print(f"Accuracy after training: {100*acc:.2f}%")


if __name__ == "__main__":
    train_model()
