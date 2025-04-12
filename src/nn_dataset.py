import torch
from torch.utils.data import Dataset
import pandas as pd

class FlightDataset(Dataset):
    def __init__(self, dataframe):
        self.X = dataframe.drop(columns=["totalFare"]).values.astype("float32")
        self.y = dataframe["totalFare"].values.astype("float32").reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
