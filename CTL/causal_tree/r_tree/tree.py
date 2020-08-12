try:
    from CTL.causal_tree.util_c import *
except:
    from CTL.causal_tree.util import *
from CTL.causal_tree.ct import *
import numpy as np
from scipy.spatial import cKDTree


# TODO: Add weighting on evaluations
# TODO: add weighting on k > 1 nearest neighbors?

def compute_nn_effect(x, y, t, k=1):
    kdtree = cKDTree(x)
    d, idx = kdtree.query(x, k=x.shape[0])
    idx = idx[:, 1:]
    treated = np.where(t == 1)[0]
    control = np.where(t == 0)[0]
    bool_treated = np.isin(idx, treated)
    bool_control = np.isin(idx, control)

    nn_effect = np.zeros(x.shape)
    for i in range(len(bool_treated)):
        i_treat_idx = np.where(bool_treated[i, :])[0][:k]
        i_control_idx = np.where(bool_control[i, :])[0][:k]

        i_treat_nn = y[idx[i, i_treat_idx]]
        i_cont_nn = y[idx[i, i_control_idx]]

        nn_effect[i] = np.mean(i_treat_nn) - np.mean(i_cont_nn)

    return nn_effect


class RNode(CTNode):

    def __init__(self, p_val=1.0, effect=0.0, node_depth=0, control_mean=0.0, treatment_mean=0.0, col=-1, value=-1,
                 is_leaf=False, leaf_num=-1, num_samples=0.0, obj=0.0, pehe=0.0):
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
        self.pehe = pehe

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


class RTree(CausalTree):

    def __init__(self, split_size=0.5, max_depth=-1, min_size=2, max_values=None, verbose=False,
                 k=1, use_propensity=False, propensity_model=None,
                 seed=724):
        super().__init__()
        self.val_split = split_size
        self.max_depth = max_depth
        self.min_size = min_size
        self.seed = seed

        self.max_values = max_values
        self.verbose = verbose

        self.max_effect = 0.0
        self.min_effect = 0.0

        self.features = None

        self.k = k
        self.num_training = 1
        self.pehe = 0
        self.use_propensity = use_propensity
        if use_propensity:
            if propensity_model is not None:
                self.proensity_model = propensity_model
            else:
                from sklearn.linear_model import LogisticRegression
                self.proensity_model = LogisticRegression()

        self.root = RNode()

    def compute_nn_effect(self, x, y, t, k=1):
        if self.use_propensity:
            self.proensity_model.fit(x, t)
            propensity = self.proensity_model.predict_proba(x)[:, 1:]
            kdtree = cKDTree(propensity)
            _, idx = kdtree.query(propensity, k=x.shape[0])
        else:
            kdtree = cKDTree(x)
            _, idx = kdtree.query(x, k=x.shape[0])
        idx = idx[:, 1:]
        treated = np.where(t == 1)[0]
        control = np.where(t == 0)[0]
        bool_treated = np.isin(idx, treated)
        bool_control = np.isin(idx, control)

        nn_effect = np.zeros(x.shape)
        for i in range(len(bool_treated)):
            i_treat_idx = np.where(bool_treated[i, :])[0][:k]
            i_control_idx = np.where(bool_control[i, :])[0][:k]

            i_treat_nn = y[idx[i, i_treat_idx]]
            i_cont_nn = y[idx[i, i_control_idx]]

            nn_effect[i] = np.mean(i_treat_nn) - np.mean(i_cont_nn)

        return nn_effect

    @abstractmethod
    def fit(self, x, y, t):
        pass

    def predict(self, x):

        def _predict(node: PEHENode, observation):
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

        def _get_group(node: PEHENode, observation):
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

        def _get_features(node: PEHENode, observation, features):
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

        def _prune(node: PEHENode):
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

        _prune(self.root)

    def get_triggers(self, x):
        pass

    def save(self, filename):
        import pickle as pkl

        check_dir(filename)
        with open(filename, "wb") as file:
            pkl.dump(self, file)
