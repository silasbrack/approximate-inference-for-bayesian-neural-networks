import hydra
from omegaconf import DictConfig

import pyro
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro import poutine
import torchmetrics as tm

from tqdm.auto import tqdm
import time

from src import data as d
from src import guides

from src.models.bayesian_mnist import BayesianMnistModel


@hydra.main(config_path="../conf", config_name="bayesian_mnist")
def train_model(cfg: DictConfig):

    model = BayesianMnistModel(cfg.params.lr)
    data = d.MNISTData(
        cfg.paths.data, cfg.params.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    blocked_model = poutine.block(model, hide=["obs"])
    # guide = guides.laplace(blocked_model)
    guide = guides.mean_field(blocked_model)
    # guide = guides.low_rank(blocked_model, rank=10)
    # guide = guides.full_rank(blocked_model)
    # guide = guides.AutoRadial(blocked_model)
    adam = pyro.optim.Adam({"lr": cfg.params.lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    x, y = next(iter(data.val_dataloader()))
    prediction = Predictive(model, guide=guide, num_samples=512)(x)
    preds = prediction["obs"].mode(dim=0).values  # MAP prediction
    acc = model.accuracy(preds, y)
    print(f"Prior accuracy: {100*acc:.2f}%")

    t0 = time.perf_counter()
    train_loader = data.train_dataloader()
    pyro.clear_param_store()
    for epoch in range(cfg.params.epochs):
        bar = tqdm(train_loader)
        for batch in bar:
            X, y = batch
            loss = svi.step(X, y)
            bar.set_postfix(loss=f"{loss / batch[0].shape[0]:.3f}")
    elapsed = time.perf_counter() - t0

    x, y = next(iter(data.val_dataloader()))
    prediction = Predictive(model, guide=guide, num_samples=512)(x)
    preds = prediction["obs"].mode(dim=0).values  # MAP prediction
    logits = prediction["logits"].mode(dim=0).values.squeeze(0)
    acc = model.accuracy(preds, y)
    print(logits.shape, y.shape)
    auroc = tm.AUROC(num_classes=10)(logits, y)
    print(f"Posterior accuracy: {100*acc:.2f}%")
    print(f"Training wall clock time: {elapsed}")
    print(f"AUROC: {auroc}")


if __name__ == "__main__":
    train_model()
