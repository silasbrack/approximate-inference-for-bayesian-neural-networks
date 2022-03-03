import logging

from src.data import CIFARData, FashionMNISTData, MNISTData, SVHNData


def main():
    logger = logging.getLogger(__name__)

    logger.info("Downloading MNIST.")
    MNISTData("data/", 0, 0).prepare_data()
    logger.info("Downloading FashionMNIST.")
    FashionMNISTData("data/", 0, 0).prepare_data()
    logger.info("Downloading CIFAR10.")
    CIFARData("data/", 0, 0).prepare_data()
    logger.info("Downloading SVHN.")
    SVHNData("data/", 0, 0).prepare_data()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
