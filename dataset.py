import torch
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence

class Dataset(BaseDataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = encodings["labels"]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])