from functools import partial

from pyro.distributions.transforms import iterated
from pyro.infer.autoguide import AutoNormalizingFlow


def normalizing_flow(model, num_flows, flow_type):
    transform_init = partial(iterated, num_flows, flow_type)
    return AutoNormalizingFlow(model, transform_init)
