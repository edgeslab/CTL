from CTL.causal_tree.ctl.adaptive import *
from CTL.causal_tree.ctl.honest import *
from CTL.causal_tree.ctl.ctl_base import *
from CTL.causal_tree.ctl.ctl_honest import *
from CTL.causal_tree.ctl.ctl_val_honest import *

from CTL.causal_tree.ctl_trigger.adaptive_trigger import *
from CTL.causal_tree.ctl_trigger.ctl_base_trigger import *
from CTL.causal_tree.ctl_trigger.ctl_honest_trigger import *
from CTL.causal_tree.ctl_trigger.ctl_val_honest_trigger import *
from CTL.causal_tree.ctl_trigger.honest_trigger import *

from CTL._tree import _CausalTree
from CTL.causal_tree.nn_pehe.base import *


class PEHETree(_CausalTree):

    def __init__(self, min_size=2, max_depth=-1, k=1,
                 seed=724):
        super().__init__()

        params = {
            "min_size": min_size,
            "max_depth": max_depth,
            "k": k,
            "seed": seed,
        }
        self.tree = BasePEHE(**params)

        self.column_num = 0
        self.fitted = False
        self.tree_depth = 0

    def fit(self, x, y, t):
        self.column_num = x.shape[1]
        x = x.astype(np.float)
        y = y.astype(np.float)
        t = t.astype(np.float)
        self.tree.fit(x, y, t)
        self.fitted = True
        self.tree_depth = self.tree.tree_depth