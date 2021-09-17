from CTL._tree import _CausalTree
from CTL.causal_tree.sig_diff.sig_base import SigTreeBase
from CTL.causal_tree.sig_diff.sig_val import SigTreeVal
import numpy as np


class SigDiffTree(_CausalTree):

    def __init__(self, alpha=0.05, min_size=2, max_depth=-1, val=False, split_size=0.5, seed=724):
        super().__init__()

        params = {
            "alpha": alpha,
            "min_size": min_size,
            "max_depth": max_depth,
            "seed": seed,
        }
        if val:
            params["split_size"] = split_size
            self.tree = SigTreeVal(**params)
        else:
            self.tree = SigTreeBase(**params)

        self.column_num = 0
        self.fitted = False
        self.tree_depth = 0

        self.obj = 0

    def fit(self, x, y, t):
        self.column_num = x.shape[1]
        x = x.astype(np.float)
        y = y.astype(np.float)
        t = t.astype(np.float)
        self.tree.fit(x, y, t)
        self.fitted = True
        self.tree_depth = self.tree.tree_depth
        self.obj = self.tree.obj
