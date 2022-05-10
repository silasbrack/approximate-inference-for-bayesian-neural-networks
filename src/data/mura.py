from os import path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.data.caching import cache_dataset, load_cached, save_cached


class MuraData(LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, cache_data):
        super().__init__()

        self.name = "MURA"
        self.size = 36808
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.eval_batch_size = 128
        self.num_workers = num_workers
        self.n_classes = 7
        self.resolution = 28
        self.channels = 1
        # self.resolution = 224
        # self.channels = 2
        self.train_val_test = [32000, 4808, 3197]

        self.cache_data = cache_data

        self.train_transform = transforms.Compose(
            [
                transforms.Resize(self.resolution),
                transforms.RandomCrop(self.resolution),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.456], [0.224]),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.Resize(self.resolution),
                transforms.RandomCrop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.456], [0.224]),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.mura_train = MuraDataset(
                f"{self.data_dir}/MURA-v1.1/train_image_paths.csv",
                self.data_dir,
                self.train_transform,
            )
            self.mura_train, self.mura_val = random_split(
                self.mura_train, self.train_val_test[:2]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mura_test = MuraDataset(
                f"{self.data_dir}/MURA-v1.1/valid_image_paths.csv",
                self.data_dir,
                self.train_transform,
            )

        if self.cache_data:
            cache_location = f"{self.data_dir}/mura_cached.pt"
            if path.exists(cache_location):
                load_cached(self.mura_train.dataset, cache_location)
            else:
                cache_dataset(self.mura_train.dataset)
                save_cached(self.mura_train.dataset, cache_location)

    def train_dataloader(self):
        return DataLoader(
            self.mura_train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mura_val,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mura_test,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
            drop_last=True,
        )


class MuraDataset(torch.utils.data.Dataset):
    def __init__(
        self, path: str, data_folder: str, transform=None, target="BodyPartIdx"
    ):
        df = pd.read_csv(path, header=None, names=["FilePath"])
        df["Label"] = df.apply(
            lambda x: 1 if "positive" in x.FilePath else 0, axis=1
        )
        df["BodyPart"] = df.apply(
            lambda x: x.FilePath.split("/")[2][3:], axis=1
        )
        df["StudyType"] = df.apply(
            lambda x: x.FilePath.split("/")[4][:6], axis=1
        )
        self.df = df
        self.data_folder = data_folder
        self.transform = transform
        self.target = target  # [BodyPartIdx, BodyPart, Label]
        body_parts = [
            "ELBOW",
            "FINGER",
            "FOREARM",
            "HAND",
            "HUMERUS",
            "SHOULDER",
            "WRIST",
        ]
        self.body_part_map = dict(zip(body_parts, range(len(body_parts))))
        self.df["BodyPartIdx"] = self.df["BodyPart"].map(self.body_part_map)

        self.cached_data = []
        self.cached_indices = {}
        self.n_cached = 0
        self.use_cache = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        if not self.use_cache:
            img_name = self.df.iloc[idx, 0]
            img = Image.open(f"{self.data_folder}/{img_name}").convert("LA")
            label = self.df.loc[idx, self.target]

            if self.transform:
                img = self.transform(img)
            img = img[0, :, :]  # Only if we're not using transforms I guess
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
            label = (
                torch.from_numpy(np.asarray(label))
                .double()
                .type(torch.LongTensor)
            )

            self.cached_data.append((img, label))
            self.cached_indices[idx] = self.n_cached
            self.n_cached += 1
        else:
            index_in_cache = self.cached_indices[idx]
            img, label = self.cached_data[index_in_cache]

        return img, label

    def set_use_cache(self, use_cache):
        self.use_cache = use_cache
