from contextlib import ExitStack

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions.util import sum_rightmost
from pyro.infer import autoguide
from torch.distributions import biject_to
from torch.distributions.utils import _standard_normal


class RadialNormal(dist.Normal):
    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(
            shape, dtype=self.loc.dtype, device=self.loc.device
        )
        distance = torch.randn(1, device=self.loc.device)
        normalizing_factor = torch.norm(eps, p=2)
        direction = eps / normalizing_factor
        eps_radial = direction * distance
        return self.loc + eps_radial * self.scale

    #     if self._validate_args:
    #         self._validate_sample(value)

    #     var = self.scale**2
    #     log_scale = (
    #         math.log(self.scale)
    #         if isinstance(self.scale, Real)
    #         else self.scale.log()
    #     )
    #     return (
    #         -((value - self.loc) ** 2) / (2 * var)
    #         - log_scale
    #         - math.log(math.sqrt(2 * math.pi))
    #     )


class AutoRadial(autoguide.AutoNormal):
    def forward(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs``
        as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            transform = biject_to(site["fn"].support)

            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])

                site_loc, site_scale = self._get_loc_and_scale(name)
                unconstrained_latent = pyro.sample(
                    name + "_unconstrained",
                    RadialNormal(
                        site_loc,
                        site_scale,
                    ).to_event(self._event_dims[name]),
                    infer={"is_auxiliary": True},
                )

                value = transform(unconstrained_latent)
                log_density = transform.inv.log_abs_det_jacobian(
                    value, unconstrained_latent
                )
                log_density = sum_rightmost(
                    log_density,
                    log_density.dim() - value.dim() + site["fn"].event_dim,
                )
                delta_dist = dist.Delta(
                    value,
                    log_density=log_density,
                    event_dim=site["fn"].event_dim,
                )

                result[name] = pyro.sample(name, delta_dist)

        return result

    # @torch.distributions.kl.register_kl(RadialNormal, dist.Normal)
    # def _kl(q, p):
    #     log_posterior = -torch.sum(torch.log(q.scale))  # Entropy
    #     w = q()
    #     log_prior = p.log_likelihood(w)  # Cross-entropy
    #     return log_posterior - log_prior
