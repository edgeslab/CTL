from CTL.util import *
from sklearn.model_selection import train_test_split
import numpy as np
from abc import ABC, abstractmethod


class TriggerNode(ABC):

    def __init__(self, p_val=1.0, effect=0.0, node_depth=0, control_mean=0.0, treatment_mean=0.0, col=-1, value=-1,
                 is_leaf=False, leaf_num=-1, num_samples=0.0, obj=0.0, trigger=0.0):
        # not tree specific features (most likely added at creation)
        self.p_val = p_val
        self.effect = effect
        self.node_depth = node_depth
        self.control_mean = control_mean
        self.treatment_mean = treatment_mean

        self.trigger = trigger

        # during tree building
        self.obj = obj
        self.num_samples = num_samples

        # after building tree
        self.col = col
        self.value = value
        self.is_leaf = is_leaf
        self.leaf_num = leaf_num
        self.true_branch = None
        self.false_branch = None


class TriggerTree(ABC):

    def __init__(self, weight=0.5, val_split=0.5, max_depth=-1, min_size=2, quartile=False, seed=724):
        self.weight = weight
        self.val_split = val_split
        self.max_depth = max_depth
        self.min_size = min_size
        self.seed = seed

        self.quartile = quartile

        self.obj = 0.0

        # Haven't implemented "mse" yet
        self.mse = 0.0

        self.max_effect = 0.0
        self.min_effect = 0.0

        self.tree_depth = 0
        self.num_leaves = 0

        self.root = TriggerNode()

    @abstractmethod
    def fit(self, x, y, t):
        pass

    def predict(self, x):

        def _predict(node: TriggerNode, observation):
            if node.is_leaf:
                return node.effect
            else:
                v = observation[node.col]
                if v >= node.value:
                    branch = node.true_branch
                else:
                    branch = node.false_branch

            return _predict(branch, observation)

        if len(x.shape) == 1:
            prediction = _predict(self.root, x)
            return prediction

        num_test = x.shape[0]

        prediction = np.zeros(num_test)

        for i in range(num_test):
            test_example = x[i, :]
            prediction[i] = _predict(self.root, test_example)

        return prediction

    def prune(self, alpha=0.05):

        def _prune(node: TriggerNode):
            if node.true_branch is None or node.false_branch is None:
                return

            # recursive call for each branch
            if not node.true_branch.is_leaf:
                _prune(node.true_branch)
            if not node.false_branch.is_leaf:
                _prune(node.false_branch)

            # merge leaves (potentially)
            if node.true_branch.is_leaf and node.false_branch.is_leaf:
                # Get branches
                tb = node.true_branch
                fb = node.false_branch

                tb_pval = tb.p_val
                fb_pval = fb.p_val

                if tb_pval > alpha and fb_pval > alpha:
                    node.leaf_num = node.true_branch.leaf_num
                    node.true_branch = None
                    node.false_branch = None
                    self.num_leaves = self.num_leaves - 1

                    # ----------------------------------------------------------------
                    # Something about obj/mse? if that is added
                    #
                    # - can do a self function so that tree references itself/it's own type of node?
                    # ----------------------------------------------------------------
                    if tb.node_depth == self.tree_depth:
                        self.tree_depth = self.tree_depth - 1

        _prune(self.root)

    def save(self, filename):
        import pickle as pkl

        check_dir(filename)
        with open(filename, "wb") as file:
            pkl.dump(self, file)
