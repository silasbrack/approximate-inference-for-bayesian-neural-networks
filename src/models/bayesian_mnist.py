import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.nn import PyroModule, PyroSample
from torch.distributions import constraints
from torch.nn import functional as F
from torchmetrics import Accuracy

softplus = nn.Softplus()


class BayesianMnistModel(PyroModule):
    def __init__(self, lr: float, hidden_size: int = 32):
        super().__init__()

        # Set our init args as class attributes
        self.hidden_size = hidden_size
        self.learning_rate = lr

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.input_size = channels * width * height

        self.flatten = nn.Flatten()
        self.fc1 = PyroModule[nn.Linear](self.input_size, hidden_size)
        self.fc1.weight = PyroSample(
            dist.Normal(0.0, 1.0).expand([hidden_size, self.input_size]).to_event(2)
        )
        self.fc1.bias = PyroSample(
            dist.Normal(0.0, 1.0).expand([hidden_size]).to_event(1)
        )
        self.fc2 = PyroModule[nn.Linear](hidden_size, hidden_size)
        self.fc2.weight = PyroSample(
            dist.Normal(0.0, 1.0).expand([hidden_size, hidden_size]).to_event(2)
        )
        self.fc2.bias = PyroSample(
            dist.Normal(0.0, 1.0).expand([hidden_size]).to_event(1)
        )
        self.fc3 = PyroModule[nn.Linear](hidden_size, self.num_classes)
        self.fc3.weight = PyroSample(
            dist.Normal(0.0, 1.0).expand([self.num_classes, hidden_size]).to_event(2)
        )
        self.fc3.bias = PyroSample(
            dist.Normal(0.0, 1.0).expand([self.num_classes]).to_event(1)
        )
        self.accuracy = Accuracy()

    def forward(self, x, y=None):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = pyro.deterministic("logits", F.log_softmax(self.fc3(x), dim=-1))
        with pyro.plate("data", x.shape[0]):
            pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits

    def guide(self, x, y=None):
        for var, dims in [
            ("fc1.weight", (self.hidden_size, self.input_size)),
            ("fc1.bias", (self.hidden_size,)),
            ("fc2.weight", (self.hidden_size, self.hidden_size)),
            ("fc2.bias", (self.hidden_size,)),
            ("fc3.weight", (self.num_classes, self.hidden_size)),
            ("fc3.bias", (self.num_classes,)),
        ]:
            mu = pyro.param(f"{var}_mu", torch.zeros(dims))
            sigma = pyro.param(
                f"{var}_sigma",
                torch.tensor(0.04) * torch.ones(dims),
                constraint=constraints.positive,
            )
            # r = pyro.sample(
            #     f"r_{var}",
            #     dist.Normal(0.0, 1.0),
            #     infer={"is_auxiliary": True}
            # )
            base_dist = dist.Normal(
                torch.zeros_like(mu), torch.ones_like(sigma)
            ).to_event(len(dims))
            transformed_dist = dist.TransformedDistribution(
                base_dist,
                [
                    # dist.transforms.Normalize(),
                    dist.transforms.AffineTransform(mu, sigma),
                    # dist.transforms.AffineTransform(0, r),
                ],
            )
            pyro.sample(var, transformed_dist)


# def guide(self, x, y=None):

#     prior_variance = 0.04

#     # r = pyro.sample(
#     #     "r", dist.Normal(0.0, 1.0), infer={"is_auxiliary": True}
#     # )
#     # base_dist = dist.Normal(
#     #     torch.zeros_like(fc1w_mu), torch.ones_like(fc1w_sigma)
#     # ).to_event(2)
#     # transformed_dist = dist.TransformedDistribution(
#     #     base_dist,
#     #     [
#     #         # dist.transforms.Normalize(),
#     #         dist.transforms.AffineTransform(fc1w_mu, fc1w_sigma),
#     #         dist.transforms.AffineTransform(0, r),
#     #     ],
#     # )
#     # fc1_weight = pyro.sample(
#     #     "fc1.weight", transformed_dist  # .expand([64, 784])
#     # )

#     dims = (self.hidden_size, self.input_size)
#     fc1w_mu = pyro.param("fc1w_mu", torch.zeros(dims))
#     fc1w_sigma = pyro.param(
#         "fc1w_sigma",
#         prior_variance * torch.ones(dims),
#         constraint=constraints.positive,
#     )
#     # fc1_eps = pyro.sample(
#     #     "fc1.eps",
#     #     dist.Normal(
#     #         torch.zeros_like(fc1w_mu), torch.ones_like(fc1w_sigma)
#     #     ).to_event(2),
#     # )
#     # fc1_weight = pyro.deterministic(
#     #     "fc1.weight", torch.addcmul(fc1w_mu, fc1w_sigma, fc1_eps)
#     # )
#     fc1_weight = pyro.sample(
#         "fc1.weight",
#         dist.Normal(fc1w_mu, fc1w_sigma).to_event(2),  # .expand([64, 784])
#     )

#     dims = (self.hidden_size,)
#     fc1b_mu = pyro.param("fc1b_mu", torch.zeros(dims))
#     fc1b_sigma = pyro.param(
#         "fc1b_sigma",
#         prior_variance * torch.ones(dims),
#         constraint=constraints.positive,
#     )
#     fc1_bias = pyro.sample(
#         "fc1.bias",
#         dist.Normal(fc1b_mu, fc1b_sigma).to_event(1),  # .expand([64])
#     )

#     dims = (self.hidden_size, self.hidden_size)
#     fc2w_mu = pyro.param("fc2w_mu", torch.zeros(dims))
#     fc2w_sigma = pyro.param(
#         "fc2w_sigma",
#         prior_variance * torch.ones(dims),
#         constraint=constraints.positive,
#     )
#     fc2_weight = pyro.sample(
#         "fc2.weight",
#         dist.Normal(fc2w_mu, fc2w_sigma).to_event(2),  # .expand([64, 784])
#     )

#     dims = (self.hidden_size,)
#     fc2b_mu = pyro.param("fc2b_mu", torch.zeros(dims))
#     fc2b_sigma = pyro.param(
#         "fc2b_sigma",
#         prior_variance * torch.ones(dims),
#         constraint=constraints.positive,
#     )
#     fc2_bias = pyro.sample(
#         "fc2.bias",
#         dist.Normal(fc2b_mu, fc2b_sigma).to_event(1),  # .expand([64])
#     )

#     dims = (self.num_classes, self.hidden_size)
#     fc3w_mu = pyro.param("fc3w_mu", torch.zeros(dims))
#     fc3w_sigma = pyro.param(
#         "fc3w_sigma",
#         prior_variance * torch.ones(dims),
#         constraint=constraints.positive,
#     )
#     fc3_weight = pyro.sample(
#         "fc3.weight",
#         dist.Normal(fc3w_mu, fc3w_sigma).to_event(2),  # .expand([64, 784])
#     )

#     dims = (self.num_classes,)
#     fc3b_mu = pyro.param("fc3b_mu", torch.zeros(dims))
#     fc3b_sigma = pyro.param(
#         "fc3b_sigma",
#         prior_variance * torch.ones(dims),
#         constraint=constraints.positive,
#     )
#     fc3_bias = pyro.sample(
#         "fc3.bias",
#         dist.Normal(fc3b_mu, fc3b_sigma).to_event(1),  # .expand([64])
#     )
