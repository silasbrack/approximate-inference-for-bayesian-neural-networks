from src.models.bayesian_mnist import BayesianMnistModel


def test_constructor():
    model = BayesianMnistModel(lr=0.01)
    assert type(model) == BayesianMnistModel
