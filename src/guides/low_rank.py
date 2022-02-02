from pyro.infer.autoguide import AutoLowRankMultivariateNormal


def low_rank(*args, **kwargs):
    return AutoLowRankMultivariateNormal(*args, **kwargs)
