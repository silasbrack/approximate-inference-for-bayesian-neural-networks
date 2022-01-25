import torch
from torch.distributions import constraints
import pyro
from pyro import distributions as dist
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import Normalize
from pyro.infer.autoguide import AutoContinuous

# https://github.com/SebFar/radial_bnn/blob/master/radial_layers/distributions.py
# https://docs.pyro.ai/en/stable/infer.autoguide.html
# https://docs.pyro.ai/en/stable/contrib.bnn.html?highlight=pyro.distributions.Transfor
# https://docs.pyro.ai/en/1.7.0/distributions.html?highlight=radial#pyro.distributions.transforms.Radial
# https://docs.pyro.ai/en/1.7.0/_modules/pyro/distributions/transforms/radial.html#Radial


class AutoRadial(AutoContinuous):
    def get_posterior(self, *args, **kwargs):
        loc = pyro.param(
            "{}_loc".format(self.prefix), lambda: torch.zeros(self.latent_dim)
        )
        scale = pyro.param(
            "{}_scale".format(self.prefix),
            lambda: torch.ones(self.latent_dim),
            constraint=constraints.positive,
        )
        TransformedDistribution(
            dist.Normal(loc, scale).to_event(1), Normalize()
        )

    def _loc_scale(self, *args, **kwargs):
        loc = pyro.param("{}_loc".format(self.prefix))
        scale = pyro.param("{}_scale".format(self.prefix))
        return loc, scale


def radial(model):
    return AutoRadial(model)


# def radial(K: int):
#     with pyro.plate("K", K):
#         epsilon_mfvi = pyro.sample("epsilon_mfvi", dist.Normal(0.0, 1.0))

#     normalizing_factor = torch.norm(epsilon_mfvi)

#     distance = pyro.sample("distance", dist.Normal(0.0, 1.0))

#     direction = epsilon_mfvi / normalizing_factor
#     epsilon_radial = direction * distance

#     return epsilon_radial
