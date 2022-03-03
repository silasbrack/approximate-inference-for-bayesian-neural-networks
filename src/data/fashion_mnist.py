from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST


class FashionMNISTData(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()

        self.size = 60000
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_classes = 10

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        # download
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = FashionMNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = FashionMNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
        )


if __name__ == "__main__":
    data = FashionMNISTData("data/", 128, 12)
    data.setup()
    loader = data.train_dataloader()
    X, y = next(iter(loader))
    print(X.shape, y.shape)
