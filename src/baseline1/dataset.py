from torch.utils.data import Dataset
import os

class B1Dataset(Dataset):
    def __init__(self, data_dir: str, target_dirs: list[int], transform=None):
        self.data_dir = data_dir
        self.target_dirs = target_dirs
        self.transform = transform

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass