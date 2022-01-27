import torch
from pyro.nn import PyroParam
from torch.distributions import constraints
import pyro
from pyro import distributions as dist, nn
from pyro.distributions import TransformedDistribution
from pyro.distributions.transforms import Normalize
from pyro.infer.autoguide import AutoContinuous, init_to_median
from pyro.contrib.easyguide import EasyGuide
from pyro.distributions.constraints import softplus_positive

# https://github.com/SebFar/radial_bnn/blob/master/radial_layers/distributions.py
# https://docs.pyro.ai/en/stable/infer.autoguide.html
# https://docs.pyro.ai/en/stable/contrib.bnn.html?highlight=pyro.distributions.Transfor
# https://docs.pyro.ai/en/1.7.0/distributions.html?highlight=radial#pyro.distributions.transforms.Radial
# https://docs.pyro.ai/en/1.7.0/_modules/pyro/distributions/transforms/radial.html#Radial


# class AutoRadial(EasyGuide):
#     def guide(self, foo, bar):
#         group = self.group(match="*")


class AutoRadial(AutoContinuous):
    scale_constraint = softplus_positive

    def __init__(self, model, init_loc_fn=init_to_median, init_scale=0.1):
        if not isinstance(init_scale, float) or not (init_scale > 0):
            raise ValueError(
                "Expected init_scale > 0. but got {}".format(init_scale)
            )
        self._init_scale = init_scale
        super().__init__(model, init_loc_fn=init_loc_fn)

    def _setup_prototype(self, *args, **kwargs):
        super()._setup_prototype(*args, **kwargs)
        # Initialize guide params
        self.loc = nn.Parameter(self._init_loc())
        self.scale = PyroParam(
            self.loc.new_full((self.latent_dim,), self._init_scale),
            self.scale_constraint,
        )

    def get_base_dist(self):
        return dist.Normal(
            torch.zeros_like(self.loc), torch.ones_like(self.loc)
        ).to_event(1)

    def get_transform(self, *args, **kwargs):
        norm_transorm = dist.transforms.Normalize()
        affine_transform = dist.transforms.AffineTransform(
            self.loc, self.scale
        )
        return dist.transforms.ComposeTransform(
            [norm_transorm, affine_transform]
        )

    def get_posterior(self, *args, **kwargs):
        """
        Returns a diagonal Normal posterior distribution.
        """
        return dist.transforms.TransformedDistribution(
            dist.Normal(0.0, 1.0).to_event(1),
            Normalize(),
            dist.transforms.AffineTransform(self.loc, self.scale),
        )

    def _loc_scale(self, *args, **kwargs):
        return self.loc, self.scale


#     def get_posterior(self, *args, **kwargs):
#         loc = pyro.param(
#             "{}_loc".format(self.prefix), lambda: torch.zeros(self.latent_dim)
#         )
#         scale = pyro.param(
#             "{}_scale".format(self.prefix),
#             lambda: torch.ones(self.latent_dim),
#             constraint=constraints.positive,
#         )
#         TransformedDistribution(
#             dist.Normal(loc, scale).to_event(1), Normalize()
#         )

#     def _loc_scale(self, *args, **kwargs):
#         loc = pyro.param("{}_loc".format(self.prefix))
#         scale = pyro.param("{}_scale".format(self.prefix))
#         return loc, scale


# def radial(model):
#     return AutoRadial(model)


# def radial(K: int):
#     with pyro.plate("K", K):
#         epsilon_mfvi = pyro.sample("epsilon_mfvi", dist.Normal(0.0, 1.0))

#     normalizing_factor = torch.norm(epsilon_mfvi)

#     distance = pyro.sample("distance", dist.Normal(0.0, 1.0))

#     direction = epsilon_mfvi / normalizing_factor
#     epsilon_radial = direction * distance

#     return epsilon_radial
