import src.data as d


def load_and_setup_data(dataset: str, *args, **kwargs):
    data_dict = {
        "mnist": d.MNISTData,
        "fashionmnist": d.FashionMNISTData,
        "cifar": d.CIFARData,
        "svhn": d.SVHNData,
        "mura": d.MuraData,
    }
    data = data_dict[dataset](*args, **kwargs)
    data.setup()

    return data