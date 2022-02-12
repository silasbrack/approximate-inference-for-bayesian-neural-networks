from pyro.infer.autoguide import AutoNormal


def mean_field(*args, **kwargs):
    return AutoNormal(*args, **kwargs)
