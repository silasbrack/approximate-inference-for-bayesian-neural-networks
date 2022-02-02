from pyro.infer.autoguide import AutoDiagonalNormal


def mean_field(*args, **kwargs):
    return AutoDiagonalNormal(*args, **kwargs)
