from CTL._tree import _CausalTree
from CTL.causal_tree.nn_pehe.base import *
from CTL.causal_tree.nn_pehe.val import *
from CTL.causal_tree.nn_pehe.honest import *


class PEHETree(_CausalTree):

    def __init__(self, min_size=2, max_depth=-1, k=1,
                 val=False, split_size=0.5,
                 honest=False,
                 seed=724):
        super().__init__()

        params = {
            "min_size": min_size,
            "max_depth": max_depth,
            "k": k,
            "seed": seed,
            "split_size": split_size
        }
        if val:
            self.tree = ValPEHE(**params)
        elif honest:
            self.tree = HonestPEHE(**params)
        else:
            self.tree = BasePEHE(**params)

        self.column_num = 0
        self.fitted = False
        self.tree_depth = 0

        self.obj = 0
        self.pehe = 0

    def fit(self, x, y, t):
        self.column_num = x.shape[1]
        x = x.astype(np.float)
        y = y.astype(np.float)
        t = t.astype(np.float)
        self.tree.fit(x, y, t)
        self.fitted = True
        self.tree_depth = self.tree.tree_depth
        self.obj = self.tree.obj
        self.pehe = self.tree.pehe
