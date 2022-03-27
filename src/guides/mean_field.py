from functools import partial

import tyxe.guides


class MeanField:
    def __init__(self, init_scale):
        self.name = "Mean-field"
        self.init_scale = init_scale

    def guide(self):
        return partial(tyxe.guides.AutoNormal, init_scale=self.init_scale)
