import torch
from torch import distributions as dist

from src.models.train_swag import train_model as train_swag


def run():
    num_ensembles = 10

    ensemble_params = torch.Tensor([list(train_swag()) for _ in range(num_ensembles)])  # [num_ensembles, 2, num_params]
    ensemble_locs = ensemble_params[:, 0, :].squeeze()  # [num_ensembles, n8iu]
    ensemble_scales = ensemble_params[:, 1, :].squeeze()

    mix = dist.Categorical(torch.ones_like(ensemble_params))  # In MultiSWAG we weigh each ensemble equally
    comp = dist.Independent(dist.Normal(ensemble_locs, ensemble_scales), 1)

    posterior = dist.MixtureSameFamily(mix, comp)  # MultiSWAG creates a GMM with {num_ensembles} components


if __name__ == "__main__":
    run()
