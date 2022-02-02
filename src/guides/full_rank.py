from pyro.infer.autoguide import AutoMultivariateNormal


def full_rank(*args, **kwargs):
    return AutoMultivariateNormal(*args, **kwargs)
