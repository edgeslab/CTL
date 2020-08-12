from CTL.causal_tree.r_tree.tree import *


class BaseNode(RNode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.obj = obj


# ----------------------------------------------------------------
# Base causal tree (ctl, base objective)
# ----------------------------------------------------------------
class BaseRTree(RTree):

    def __init__(self, eval2=False, **kwargs):
        super().__init__(**kwargs)
        self.root = BaseNode()
        self.eval2 = eval2

    def fit(self, x, y, t):
        if x.shape[0] == 0:
            return 0

        # ----------------------------------------------------------------
        # Seed
        # ----------------------------------------------------------------
        np.random.seed(self.seed)

        self.root.num_samples = y.shape[0]
        self.num_training = y.shape[0]

        # ----------------------------------------------------------------
        # NN_effect estimates
        # use the overall datasets for nearest neighbor for now
        # ----------------------------------------------------------------
        nn_effect = self.compute_nn_effect(x, y, t, k=self.k)

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
        nn_pehe = self._eval(y, t, nn_effect)
        self.root.pehe = nn_pehe
        self.pehe = self.root.pehe

        # ----------------------------------------------------------------
        # Add control/treatment means
        # ----------------------------------------------------------------
        self.root.control_mean = np.mean(y[t == 0])
        self.root.treatment_mean = np.mean(y[t == 1])

        self.root.num_samples = x.shape[0]

        self._fit(self.root, x, y, t, nn_effect)

        if self.num_leaves > 0:
            self.pehe = self.pehe / self.num_leaves

    def _eval(self, train_y, train_t, nn_effect):

        # treated = np.where(train_t == 1)[0]
        # control = np.where(train_t == 0)[0]
        # pred_effect = np.mean(train_y[treated]) - np.mean(train_y[control])
        pred_effect = ace(train_y, train_t)

        # nn_pehe = np.mean((nn_effect - pred_effect) ** 2)
        nn_pehe = np.sum((nn_effect - pred_effect) ** 2)

        return nn_pehe

    def _eval2(self, unique_vals, x, y, t, nn_effect, col, node_pehe):
        above_pehe = np.inf * np.ones(unique_vals.shape)
        below_pehe = np.inf * np.ones(unique_vals.shape)
        for i, val in enumerate(unique_vals):
            below_idx = (x[:, col] < val)
            above_idx = (x[:, col] >= val)
            below_treated_idx = y[below_idx & (t == 1)]
            below_control_idx = y[below_idx & (t == 0)]
            if len(below_treated_idx) < self.min_size or len(below_control_idx) < self.min_size:
                continue
            above_treated_idx = y[above_idx & (t == 1)]
            above_control_idx = y[above_idx & (t == 0)]
            if len(above_treated_idx) < self.min_size or len(above_control_idx) < self.min_size:
                continue
            below_treat_mean = np.mean(below_treated_idx)
            below_control_mean = np.mean(below_control_idx)

            above_treat_mean = np.mean(above_treated_idx)
            above_control_mean = np.mean(above_control_idx)

            below_pehe[i] = np.sum((nn_effect[below_idx] - (below_treat_mean - below_control_mean)) ** 2)
            above_pehe[i] = np.sum((nn_effect[above_idx] - (above_treat_mean - above_control_mean)) ** 2)
        sum_pehe = above_pehe + below_pehe
        # gain = node_pehe - sum_pehe

        # best_pehe_idx = np.argmax(gain)
        best_pehe_idx = np.argmin(sum_pehe)
        best_val = unique_vals[best_pehe_idx]
        best_pehe = sum_pehe[best_pehe_idx]
        best_tb_pehe = above_pehe[best_pehe_idx]
        best_fb_pehe = below_pehe[best_pehe_idx]

        return best_pehe, best_val, best_tb_pehe, best_fb_pehe

    def _fit(self, node: BaseNode, train_x, train_y, train_t, nn_effect):

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

        # print(self.tree_depth, self.obj)

        best_gain = 0.0
        # best_gain = node.pehe  # min amount
        best_attributes = []
        best_tb_obj, best_fb_obj = (0.0, 0.0)

        column_count = train_x.shape[1]
        for col in range(0, column_count):
            unique_vals = np.unique(train_x[:, col])

            for value in unique_vals:
                # check training data size
                (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                    = divide_set(train_x, train_y, train_t, col, value)
                check1 = check_min_size(self.min_size, train_t1)
                check2 = check_min_size(self.min_size, train_t2)
                if check1 or check2:
                    continue
                (_, _, nn_effect1, nn_effect2, _, _) \
                    = divide_set(train_x, nn_effect, train_t, col, value)

                tb_eval = self._eval(train_y1, train_t1, nn_effect1)
                fb_eval = self._eval(train_y2, train_t2, nn_effect2)

                split_eval = (tb_eval + fb_eval)
                gain = node.pehe - split_eval

                if gain > best_gain:
                    best_gain = gain
                    best_attributes = [col, value]
                    best_tb_obj, best_fb_obj = (tb_eval, fb_eval)
            # if self.eval2:
            #     split_eval, value, tb_eval, fb_eval = self._eval2(unique_vals, train_x, train_y, train_t, nn_effect,
            #                                                       col, node.pehe)
            #
            #     gain = node.pehe - split_eval
            #
            #     if gain > best_gain:
            #         best_gain = gain
            #         best_attributes = [col, value]
            #         best_tb_obj, best_fb_obj = (tb_eval, fb_eval)
            # else:
            #     for value in unique_vals:
            #         # check training data size
            #         (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
            #             = divide_set(train_x, train_y, train_t, col, value)
            #         check1 = check_min_size(self.min_size, train_t1)
            #         check2 = check_min_size(self.min_size, train_t2)
            #         if check1 or check2:
            #             continue
            #         (_, _, nn_effect1, nn_effect2, _, _) \
            #             = divide_set(train_x, nn_effect, train_t, col, value)
            #
            #         tb_eval = self._eval(train_y1, train_t1, nn_effect1)
            #         fb_eval = self._eval(train_y2, train_t2, nn_effect2)
            #
            #         split_eval = (tb_eval + fb_eval)
            #         gain = node.pehe - split_eval
            #
            #         if gain > best_gain:
            #             best_gain = gain
            #             best_attributes = [col, value]
            #             best_tb_obj, best_fb_obj = (tb_eval, fb_eval)

        if best_gain > 0:
            node.col = best_attributes[0]
            node.value = best_attributes[1]

            (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                = divide_set(train_x, train_y, train_t, node.col, node.value)
            (_, _, nn_effect1, nn_effect2, _, _) \
                = divide_set(train_x, nn_effect, train_t, node.col, node.value)

            y1 = train_y1
            y2 = train_y2
            t1 = train_t1
            t2 = train_t2

            best_tb_effect = ace(y1, t1)
            best_fb_effect = ace(y2, t2)
            tb_p_val = get_pval(y1, t1)
            fb_p_val = get_pval(y2, t2)

            self.pehe = self.pehe - node.pehe + best_tb_obj + best_fb_obj

            tb = BaseNode(obj=best_tb_obj, pehe=best_tb_obj, effect=best_tb_effect, p_val=tb_p_val,
                          node_depth=node.node_depth + 1,
                          num_samples=y1.shape[0])
            fb = BaseNode(obj=best_fb_obj, pehe=best_fb_obj, effect=best_fb_effect, p_val=fb_p_val,
                          node_depth=node.node_depth + 1,
                          num_samples=y2.shape[0])

            node.true_branch = self._fit(tb, train_x1, train_y1, train_t1, nn_effect1)
            node.false_branch = self._fit(fb, train_x2, train_y2, train_t2, nn_effect2)

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
