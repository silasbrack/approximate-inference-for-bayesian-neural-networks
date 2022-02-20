from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision.datasets as d


class CIFARData(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.eval_batch_size = 10000
        self.num_workers = num_workers
        self.n_classes = 10

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.df_train = None
        self.df_val = None
        self.df_test = None

    def prepare_data(self):
        # download
        d.CIFAR10(self.data_dir, train=True, download=True)
        d.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = d.CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.df_train, self.df_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.df_test = d.CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.df_train, num_workers=self.num_workers, batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.df_val, num_workers=self.num_workers, batch_size=self.eval_batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.df_test, num_workers=self.num_workers, batch_size=self.eval_batch_size,
        )
