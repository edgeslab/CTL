from CTL.causal_tree.sig_diff.sig import *


class BaseCausalTreeLearnNode(SigNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SigTreeBase(SigTree):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = BaseCausalTreeLearnNode()

    def fit(self, x, y, t):
        if x.shape[0] == 0:
            return 0

        # ----------------------------------------------------------------
        # Seed
        # ----------------------------------------------------------------
        np.random.seed(self.seed)

        train_x, train_y, train_t = x, y, t
        self.root.num_samples = train_y.shape[0]
        # ----------------------------------------------------------------
        # effect and pvals
        # ----------------------------------------------------------------
        effect = tau_squared(y, t)
        p_val = get_pval(y, t)
        self.root.effect = effect
        self.root.p_val = p_val

        self.root.obj = 0

        # ----------------------------------------------------------------
        # Add control/treatment means
        # ----------------------------------------------------------------
        self.root.control_mean = np.mean(y[t == 0])
        self.root.treatment_mean = np.mean(y[t == 1])

        self.root.num_samples = x.shape[0]

        self._fit(self.root, train_x, train_y, train_t)

    def _fit(self, node: BaseCausalTreeLearnNode, train_x, train_y, train_t):

        if train_x.shape[0] == 0:
            return node

        if node.node_depth > self.tree_depth:
            self.tree_depth = node.node_depth

        if self.max_depth == self.tree_depth:
            if node.effect > self.max_effect:
                self.max_effect = node.effect
            if node.effect < self.min_effect:
                self.min_effect = node.effect
            self.num_leaves += 1
            node.leaf_num = self.num_leaves
            node.is_leaf = True
            return node

        best_gain = 1.0
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

                t_stat, diff_pval = self._eval(train_y1, train_t1, train_y2, train_t2)

                gain = diff_pval

                if gain < best_gain and gain <= self.alpha:
                    best_gain = gain
                    best_attributes = [col, value]

        if best_gain <= self.alpha:
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

            tb = BaseCausalTreeLearnNode(obj=best_tb_obj, effect=best_tb_effect, p_val=tb_p_val,
                                         node_depth=node.node_depth + 1,
                                         num_samples=y1.shape[0])
            fb = BaseCausalTreeLearnNode(obj=best_fb_obj, effect=best_fb_effect, p_val=fb_p_val,
                                         node_depth=node.node_depth + 1,
                                         num_samples=y2.shape[0])

            node.true_branch = self._fit(tb, train_x1, train_y1, train_t1)
            node.false_branch = self._fit(fb, train_x2, train_y2, train_t2)

            if node.effect > self.max_effect:
                self.max_effect = node.effect
            if node.effect < self.min_effect:
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
