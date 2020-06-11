from CTL.causal_tree.ctl.binary_ctl import *
from sklearn.model_selection import train_test_split


class AdaptiveNode(CTLearnNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.obj = obj


# ----------------------------------------------------------------
# Base causal tree (ctl, base objective)
# ----------------------------------------------------------------
class AdaptiveTree(CTLearn):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = AdaptiveNode()

    def adaptive_eval(self, train_y, train_t):
        total_train = train_y.shape[0]

        train_effect = ace(train_y, train_t)

        train_mse = total_train * (train_effect ** 2)

        obj = train_mse
        mse = total_train * (train_effect ** 2)

        return obj, mse

    def fit(self, x, y, t):
        if x.shape[0] == 0:
            return 0

        # ----------------------------------------------------------------
        # Seed
        # ----------------------------------------------------------------
        np.random.seed(self.seed)

        # ----------------------------------------------------------------
        # Verbosity?
        # ----------------------------------------------------------------

        # ----------------------------------------------------------------
        # Split data
        # ----------------------------------------------------------------

        self.root.num_samples = y.shape[0]
        # ----------------------------------------------------------------
        # effect and pvals
        # ----------------------------------------------------------------
        effect = tau_squared(y, t)
        p_val = get_pval(y, t)
        self.root.effect = effect
        self.root.p_val = p_val

        # ----------------------------------------------------------------
        # Not sure if i should eval in root or not
        # ----------------------------------------------------------------
        node_eval, mse = self.adaptive_eval(y, t)
        self.root.obj = node_eval

        # ----------------------------------------------------------------
        # Add control/treatment means
        # ----------------------------------------------------------------
        self.root.control_mean = np.mean(y[t == 0])
        self.root.treatment_mean = np.mean(y[t == 1])

        self.root.num_samples = x.shape[0]

        self._fit(self.root, x, y, t)

    def _fit(self, node: AdaptiveNode, train_x, train_y, train_t):

        if train_x.shape[0] == 0:
            return node

        if node.node_depth > self.tree_depth:
            self.tree_depth = node.node_depth

        if self.max_depth == self.tree_depth:
            self.num_leaves += 1
            node.leaf_num = self.num_leaves
            node.is_leaf = True
            return node

        best_gain = 0.0
        best_attributes = []
        best_tb_obj, best_fb_obj = (0.0, 0.0)

        column_count = train_x.shape[1]
        for col in range(0, column_count):
            unique_vals = np.unique(train_x[:, col])

            if self.max_values is not None:
                if self.max_values < 1:
                    idx = np.round(np.linspace(
                        0, len(unique_vals) - 1, self.max_values * len(unique_vals))).astype(int)
                    unique_vals = unique_vals[idx]
                else:
                    idx = np.round(np.linspace(
                        0, len(unique_vals) - 1, self.max_values)).astype(int)
                    unique_vals = unique_vals[idx]

            for value in unique_vals:

                # check training data size
                (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                    = divide_set(train_x, train_y, train_t, col, value)
                check1 = check_min_size(self.min_size, train_t1)
                check2 = check_min_size(self.min_size, train_t2)
                if check1 or check2:
                    continue

                tb_eval, tb_mse = self.adaptive_eval(train_y1, train_t1)
                fb_eval, fb_mse = self.adaptive_eval(train_y2, train_t2)

                split_eval = (tb_eval + fb_eval)
                gain = -node.obj + split_eval

                if gain > best_gain:
                    best_gain = gain
                    best_attributes = [col, value]
                    best_tb_obj, best_fb_obj = (tb_eval, fb_eval)

        if best_gain > 0:
            node.col = best_attributes[0]
            node.value = best_attributes[1]

            (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                = divide_set(train_x, train_y, train_t, node.col, node.value)

            y1 = train_y1
            y2 = train_y2
            t1 = train_t1
            t2 = train_t2

            best_tb_effect = ace(y1, t1)
            best_fb_effect = ace(y2, t2)
            tb_p_val = get_pval(y1, t1)
            fb_p_val = get_pval(y2, t2)

            self.obj = self.obj - node.obj + best_tb_obj + best_fb_obj

            # ----------------------------------------------------------------
            # Ignore "mse" here, come back to it later?
            # ----------------------------------------------------------------

            tb = AdaptiveNode(obj=best_tb_obj, effect=best_tb_effect, p_val=tb_p_val,
                              node_depth=node.node_depth + 1,
                              num_samples=y1.shape[0])
            fb = AdaptiveNode(obj=best_fb_obj, effect=best_fb_effect, p_val=fb_p_val,
                              node_depth=node.node_depth + 1,
                              num_samples=y2.shape[0])

            node.true_branch = self._fit(tb, train_x1, train_y1, train_t1)
            node.false_branch = self._fit(fb, train_x2, train_y2, train_t2)

            if node.effect > self.max_effect:
                self.max_effect = node.effect
            else:
                self.min_effect = node.effect

            return node

        else:
            if node.effect > self.max_effect:
                self.max_effect = node.effect
            if node.effect < self.min_effect:
                self.min_effect = node.effect

            self.num_leaves += 1
            node.leaf_num = self.num_leaves
            node.is_leaf = True
            return node
