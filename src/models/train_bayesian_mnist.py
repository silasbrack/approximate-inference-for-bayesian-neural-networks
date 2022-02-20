import time

import hydra
import pyro
import torchmetrics as tm
from omegaconf import DictConfig
from pyro import poutine
from pyro.infer import SVI, Predictive, Trace_ELBO
from torch import nn
from tqdm.auto import tqdm

from src import data as d

from src import guides
from src.models.bayesian_mnist import BayesianMnistModel

# from src.guides.radial import AutoDiagonalNormal

pyro.enable_validation(True)
pyro.clear_param_store()


@hydra.main(config_path="../conf", config_name="bayesian_mnist")
def train_model(cfg: DictConfig):

    model = BayesianMnistModel(cfg.params.lr)
    data = d.FashionMNISTData(
        cfg.paths.data, cfg.params.batch_size, cfg.hardware.num_workers
    )
    data.setup()

    blocked_model = poutine.block(model, hide=["obs"])
    # guide = guides.laplace(blocked_model)
    guide = guides.mean_field(blocked_model)
    # guide = guides.low_rank(blocked_model, rank=10)
    # guide = guides.full_rank(blocked_model)
    # guide = guides.AutoRadial(blocked_model)
    # guide = model.guide
    # guide = AutoDiagonalNormal(model)
    adam = pyro.optim.Adam({"lr": cfg.params.lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO(num_particles=32),)

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

    # print(pyro.get_param_store().named_parameters())
    # print(pyro.get_param_store().get_all_param_names())

    # for name in pyro.get_param_store().get_all_param_names():
    #     print(name, pyro.param(name).data.numpy())

    # x, y = next(iter(data.val_dataloader()))
    # with poutine.trace() as tr:
    #     guide(x)
    # for site in tr.trace.nodes.values():
    #     print(site["type"], site["name"], site["value"].shape)

    trace_elbo = Trace_ELBO()
    elbo_sum = 0
    m = 0
    accuracy = tm.Accuracy()
    auroc = tm.AUROC(num_classes=10)
    nll = nn.NLLLoss(reduction="sum")
    nll_sum = 0
    # confidence_sum = 0
    n = 0
    for x, y in data.val_dataloader():
        prediction = Predictive(model, guide=guide, num_samples=128)(x)
        preds = prediction["obs"].mode(dim=0).values  # MAP prediction
        logits = prediction["logits"].mode(dim=0).values.squeeze(0)
        # probs = nn.functional.softmax(logits, dim=-1)

        # confidence_sum = torch.sum(torch.max(probs, dim=-1)[0].detach())
        elbo_sum += trace_elbo.loss(model, guide, x, y)
        acc = accuracy(preds, y)
        auroc(logits, y)
        nll_sum += nll(logits, y)
        n += y.shape[0]
        m += 1

    [print(key, val) for key, val in pyro.get_param_store().items()]

    guide_params = sum(val.shape.numel() for _, val in pyro.get_param_store().items())
    elbo_val = elbo_sum / m
    nll = nll_sum / n
    acc = accuracy.compute()
    accuracy.reset()
    auroc_val = auroc.compute()
    auroc.reset()
    # confidence = confidence_sum / n
    print(f"Number of parameters: {guide_params}")
    print(f"ELBO: {elbo_val:.2f}")
    print(f"Posterior accuracy: {100*acc:.2f}%")
    print(f"Training wall clock time: {elapsed:.2f} s")
    print(f"AUROC: {auroc_val:.3f}")
    print(f"NLL: {nll:.3f}")
    # print(f"Average confidence: {100*confidence:.2f}%")


if __name__ == "__main__":
    train_model()
