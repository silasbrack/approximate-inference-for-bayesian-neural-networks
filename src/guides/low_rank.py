from functools import partial

from pyro.infer.autoguide import AutoLowRankMultivariateNormal


def low_rank(rank):
    return partial(AutoLowRankMultivariateNormal, rank=rank)
