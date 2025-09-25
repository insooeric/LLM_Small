import numpy as np, torch
from torch.utils.data import Dataset

class NpySFTDataset(torch.utils.data.Dataset):
    def __init__(self, ids_path, mask_path, ctx):
        self.ids  = np.load(ids_path,  mmap_mode="r")
        self.mask = np.load(mask_path, mmap_mode="r")
        assert self.ids.shape == self.mask.shape
        self.ctx = ctx

    def __len__(self): 
        return self.ids.shape[0]

    def __getitem__(self, i):
        x_np = self.ids[i][:self.ctx]
        m_np = self.mask[i][:self.ctx]

        if x_np.dtype == object:
            x_np = np.asarray(x_np, dtype=np.int64)
            m_np = np.asarray(m_np, dtype=np.int64)

        y_np = x_np.copy().astype(np.int64, copy=False)
        y_np[m_np == 0] = -100

        x_t = torch.from_numpy(x_np.astype(np.int64, copy=False))
        y_t = torch.from_numpy(y_np)
        return x_t, y_t