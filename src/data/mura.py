import torch
from PIL import Image
import numpy as np
import pandas as pd


class MuraDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, data_folder: str, transform=None):
        df = pd.read_csv(path, header=None, names=['FilePath'])
        df['Label'] = df.apply(
            lambda x: 1 if 'positive' in x.FilePath else 0, axis=1)
        df['BodyPart'] = df.apply(
            lambda x: x.FilePath.split('/')[2][3:], axis=1)
        df['StudyType'] = df.apply(
            lambda x: x.FilePath.split('/')[4][:6], axis=1)
        self.df = df
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img = Image.open(f"{self.data_folder}/{img_name}").convert('LA')
        label = self.df.iloc[idx, 1]

        if self.transform:
            img = self.transform(img)
        label = torch.from_numpy(np.asarray(label)).double().type(
            torch.LongTensor)
        return img, label
