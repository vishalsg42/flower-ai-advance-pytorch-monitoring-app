import utils
import torch
from datasets import Dataset
from torch.utils.data import DataLoader


class DataClientLoader():
    def __init__(self, client_id: int):
        self.client_id = client_id

    def load_data(self):
        """Load and return the train and test datasets for the client."""
        trainset, testset = utils.load_partition(self.client_id)
        return trainset, testset

    def get_data_loaders(self, trainset, validation_split, batch_size):
        """Create and return the train and validation data loaders."""
        train_valid = trainset.train_test_split(validation_split, seed=42)
        trainset = train_valid["train"]
        valset = train_valid["test"]

        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=batch_size)

        return train_loader, val_loader

    def get_test_loader(self, testset):
        """Create and return the test data loader."""
        return DataLoader(testset, batch_size=16)
