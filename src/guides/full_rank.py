from pyro.infer.autoguide import AutoMultivariateNormal


def full_rank(model):
    return AutoMultivariateNormal(model)
