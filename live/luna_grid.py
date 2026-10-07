"""Grid-based state partition (LUNA paper III-B1): each PCA dimension is split uniformly into m cells."""
import numpy as np


class GridPartition:
    def __init__(self, m):
        self.m = m

    def fit(self, Z):
        self.lo, self.hi = Z.min(0), Z.max(0)
        return self

    def predict(self, Z):
        idx = np.clip(((Z - self.lo) / (self.hi - self.lo + 1e-9) * self.m).astype(int), 0, self.m - 1)
        out = np.zeros(len(Z), dtype=np.int64)
        for j in range(Z.shape[1]): out = out * self.m + idx[:, j]
        return out
