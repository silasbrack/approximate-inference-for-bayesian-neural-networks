from pytorch_lightning import LightningDataModule


class MNISTData(LightningDataModule):
    def __init__(self, batch_size, num_workers, data_path):
        super(MNISTData).__init__()
        pass
