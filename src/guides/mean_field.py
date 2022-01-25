from pyro.infer.autoguide import AutoDiagonalNormal


def mean_field(model):
    return AutoDiagonalNormal(model)
