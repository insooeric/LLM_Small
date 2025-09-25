import numpy as np, torch
from torch.utils.data import Dataset

class NpyTokensDataset(Dataset):
    def __init__(self, path: str, seq_len: int):
        self.path = str(path)
        self.seq_len = int(seq_len)
        arr = np.load(self.path, mmap_mode="r")
        self._ndim = arr.ndim
        self._shape = arr.shape
        assert self._ndim in (1,2), f"Expected 1D/2D, got {self._ndim}"
        if self._ndim == 1:  assert self._shape[0] > self.seq_len + 1
        else:                assert self._shape[1] > self.seq_len + 1
        self._arr = arr 

    def __getstate__(self):
        return {"path": self.path, "seq_len": self.seq_len, "_ndim": self._ndim, "_shape": self._shape}
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._arr = np.load(self.path, mmap_mode="r")

    def __len__(self):
        if self._ndim == 1:
            return max(4096, (self._shape[0] - 1) // self.seq_len)
        rows, L = self._shape
        return max(4096, rows * max(4, (L - 1) // self.seq_len))

    def __getitem__(self, idx):
        arr = self._arr
        if self._ndim == 1:
            L = self._shape[0]
            s = np.random.randint(0, L - self.seq_len - 1)
            x = arr[s:s+self.seq_len].astype(np.int64, copy=False)
            y = arr[s+1:s+self.seq_len+1].astype(np.int64, copy=False)
        else:
            rows, L = self._shape
            r = np.random.randint(0, rows)
            s = np.random.randint(0, L - self.seq_len - 1)
            row = arr[r]
            x = row[s:s+self.seq_len].astype(np.int64, copy=False)
            y = row[s+1:s+self.seq_len+1].astype(np.int64, copy=False)
        return torch.from_numpy(x), torch.from_numpy(y)