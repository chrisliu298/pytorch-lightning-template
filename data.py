import os

import numpy as np
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class DataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def prepare_data(self):
        # download data
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        indices = np.arange(len(self.train_dataset))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=True)
        self.train_dataset, self.val_dataset = (
            self.train_dataset[train_idx],
            self.train_dataset[val_idx],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )
