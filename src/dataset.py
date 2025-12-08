import os
from typing import Optional, Callable, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class MITBIHArrhythmiaDataset(Dataset):
    """
    MIT-BIH Arrhythmia dataset wrapper.

    Kaggle CSV format:
      - 187 columns signal + 1 column label
    """

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()

        df = pd.read_csv(csv_path, header=None)

        # X: 187-length vector, y: class (0~4)
        X = df.iloc[:, :-1].values.astype(np.float32)
        y = df.iloc[:, -1].values.astype(np.int64)

        self.X = torch.from_numpy(X)  # (N, 187)
        self.y = torch.from_numpy(y)  # (N,)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]            # shape (187,)
        x = x.unsqueeze(0)         # → (1, 187) for CNN compatibility

        if self.transform:
            x = self.transform(x)

        label = self.y[idx]
        return x, label


def _train_val_split(dataset, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    n = len(dataset)
    idx = rng.permutation(n)

    split = int(n * (1 - val_ratio))
    train_idx, val_idx = idx[:split], idx[split:]

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    return train_set, val_set


def create_dataloaders(
    data_dir,
    batch_size=128,
    num_workers=2,
    transform=None,
    val_ratio=0.2,
    seed=42,
):
    train_csv = os.path.join(data_dir, "mitbih_train.csv")
    test_csv = os.path.join(data_dir, "mitbih_test.csv")

    train_dataset = MITBIHArrhythmiaDataset(train_csv, transform=transform)
    test_dataset = MITBIHArrhythmiaDataset(test_csv, transform=None)

    # split
    train_set, val_set = _train_val_split(train_dataset, val_ratio, seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
