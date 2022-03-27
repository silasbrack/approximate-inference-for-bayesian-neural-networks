from functools import partial

from pyro.infer.autoguide import AutoLowRankMultivariateNormal


class LowRank:
    def __init__(self, rank):
        self.name = "Low-rank"
        self.rank = rank

    def guide(self):
        return partial(AutoLowRankMultivariateNormal, rank=self.rank)
