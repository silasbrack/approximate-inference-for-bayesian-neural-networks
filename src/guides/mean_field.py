from functools import partial

import tyxe.guides


def mean_field(init_scale):
    return partial(tyxe.guides.AutoNormal, init_scale=init_scale)
