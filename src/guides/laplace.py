from pyro.infer.autoguide import AutoLaplaceApproximation


def laplace(model):
    return AutoLaplaceApproximation(model)
