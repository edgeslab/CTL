# from CTL.causal_tree.util import *
from CTL.causal_tree.util_c import *
from CTL.causal_tree.causal_tree import *
import numpy as np


class CausalTreeLearnNode(CausalTreeNode):

    def __init__(self, p_val=1.0, effect=0.0, node_depth=0, control_mean=0.0, treatment_mean=0.0, col=-1, value=-1,
                 is_leaf=False, leaf_num=-1, num_samples=0.0, obj=0.0):
        super().__init__()
        # not tree specific features (most likely added at creation)
        self.p_val = p_val
        self.effect = effect
        self.node_depth = node_depth
        self.control_mean = control_mean
        self.treatment_mean = treatment_mean

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

        # after calling functions
        self.column_name = ""
        self.decision = ""


class CausalTreeLearn(CausalTree):

    def __init__(self, weight=0.5, split_size=0.5, max_depth=-1, min_size=2, seed=724):
        super().__init__()
        self.weight = weight
        self.val_split = split_size
        self.max_depth = max_depth
        self.min_size = min_size
        self.seed = seed

        self.max_effect = 0.0
        self.min_effect = 0.0

        self.features = None

        self.root = CausalTreeLearnNode()

    @abstractmethod
    def fit(self, x, y, t):
        pass

    def _eval(self, train_y, train_t, val_y, val_t):
        total_train = train_y.shape[0]
        total_val = val_y.shape[0]

        # return_val = (-np.inf, -np.inf)
        #
        # if total_train == 0 or total_val == 0:
        #     return return_val

        train_effect = ace(train_y, train_t)
        val_effect = ace(val_y, val_t)

        train_mse = (1 - self.weight) * total_train * (train_effect ** 2)
        cost = self.weight * total_val * np.abs(train_effect - val_effect)

        obj = (train_mse - cost) / (np.abs(total_train - total_val) + 1)
        mse = total_train * (train_effect ** 2)

        return obj, mse

    def predict(self, x):

        def _predict(node: CausalTreeLearnNode, observation):
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

    def get_groups(self, x):

        def _get_group(node: CausalTreeLearnNode, observation):
            if node.is_leaf:
                return node.leaf_num
            else:
                v = observation[node.col]
                if v >= node.value:
                    branch = node.true_branch
                else:
                    branch = node.false_branch

            return _get_group(branch, observation)

        if len(x.shape) == 1:
            return _get_group(self.root, x)
        num_test = x.shape[0]
        leaf_results = np.zeros(num_test)

        for i in range(num_test):
            test_example = x[i, :]
            leaf_results[i] = _get_group(self.root, test_example)

        return leaf_results

    def get_features(self, x):

        def _get_features(node: CausalTreeLearnNode, observation, features):
            if node.is_leaf:
                return features
            else:
                v = observation[node.col]
                if v >= node.value:
                    branch = node.true_branch
                else:
                    branch = node.false_branch

            features.append(node.decision)
            return _get_features(branch, observation, features)

        if len(x.shape) == 1:
            features = []
            return _get_features(self.root, x, features)
        num_test = x.shape[0]
        leaf_features = []

        for i in range(num_test):
            features = []
            test_example = x[i, :]
            leaf_features.append(_get_features(self.root, test_example, features))

        return leaf_features

    def prune(self, alpha=0.05):

        def _prune(node: CausalTreeLearnNode):
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
                    node.is_leaf = True

                    # ----------------------------------------------------------------
                    # Something about obj/mse? if that is added
                    #
                    # - can do a self function so that tree references itself/it's own type of node?
                    # ----------------------------------------------------------------
                    if tb.node_depth == self.tree_depth:
                        self.tree_depth = self.tree_depth - 1

                    _prune(tb)

        _prune(self.root)

    def get_triggers(self, x):
        pass

    def save(self, filename):
        import pickle as pkl

        check_dir(filename)
        with open(filename, "wb") as file:
            pkl.dump(self, file)
