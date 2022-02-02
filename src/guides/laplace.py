from pyro.infer.autoguide import AutoLaplaceApproximation


def laplace(*args, **kwargs):
    return AutoLaplaceApproximation(*args, **kwargs)
