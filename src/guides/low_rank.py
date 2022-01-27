from pyro.infer.autoguide import AutoLowRankMultivariateNormal


def low_rank(model, rank):
    return AutoLowRankMultivariateNormal(model, rank=rank)
