from CTL.causal_tree.nn_pehe.tree import *
from sklearn.model_selection import train_test_split


class HonestNode(PEHENode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.obj = obj


# ----------------------------------------------------------------
# Base causal tree (ctl, base objective)
# ----------------------------------------------------------------
class HonestPEHE(PEHETree):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = HonestNode()

    def fit(self, x, y, t):
        if x.shape[0] == 0:
            return 0

        # ----------------------------------------------------------------
        # Seed
        # ----------------------------------------------------------------
        np.random.seed(self.seed)

        # ----------------------------------------------------------------
        # Split data
        # ----------------------------------------------------------------
        x, est_x, y, est_y, t, est_t = train_test_split(x, y, t, random_state=self.seed, shuffle=True,
                                                        test_size=0.5)
        self.root.num_samples = est_y.shape[0]
        self.num_training = y.shape[0]

        # ----------------------------------------------------------------
        # NN_effect estimates
        # use the overall datasets for nearest neighbor for now
        # ----------------------------------------------------------------
        nn_effect = compute_nn_effect(x, y, t, k=self.k)
        # val_nn_effect = compute_nn_effect(est_x, est_y, est_t, k=self.k)

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
        self.root.obj = nn_pehe
        self.obj = self.root.obj

        # ----------------------------------------------------------------
        # Add control/treatment means
        # ----------------------------------------------------------------
        self.root.control_mean = np.mean(y[t == 0])
        self.root.treatment_mean = np.mean(y[t == 1])

        self.root.num_samples = x.shape[0]

        self._fit(self.root, x, y, t, nn_effect, est_x, est_y, est_t)

        if self.num_leaves > 0:
            self.obj = self.obj / self.num_leaves

    def _eval(self, train_y, train_t, nn_effect):

        # total_train = train_y.shape[0]

        # treated = np.where(train_t == 1)[0]
        # control = np.where(train_t == 0)[0]
        # pred_effect = np.mean(train_y[treated]) - np.mean(train_y[control])
        pred_effect = ace(train_y, train_t)

        # nn_pehe = np.mean((nn_effect - pred_effect) ** 2)
        nn_pehe = np.sum((nn_effect - pred_effect) ** 2)

        # val_effect = ace(val_y, val_t)
        # val_nn_pehe = np.sum((val_nn_effect - pred_effect) ** 2)
        # val_train_ratio = total_train / total_val
        # val_nn_pehe = val_nn_pehe * val_train_ratio
        # pehe_diff = np.abs(nn_pehe - val_nn_pehe)

        # cost = np.abs(total_train * pred_effect - total_train * val_effect)

        var_t, var_c = variance(train_y, train_t)

        return nn_pehe

    def _fit(self, node: HonestNode, train_x, train_y, train_t, nn_effect, est_x, est_y, est_t):

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
        best_attributes = []
        best_tb_obj, best_fb_obj = (0.0, 0.0)

        column_count = train_x.shape[1]
        for col in range(0, column_count):
            unique_vals = np.unique(train_x[:, col])

            for value in unique_vals:
                (est_x1, est_x2, est_y1, est_y2, est_t1, est_t2) \
                    = divide_set(est_x, est_y, est_t, col, value)

                # check est set size
                if check_min_size(self.min_size, est_t1) or check_min_size(self.min_size, est_t2):
                    continue

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
                gain = node.obj - split_eval

                if gain > best_gain:
                    best_gain = gain
                    best_attributes = [col, value]
                    best_tb_obj, best_fb_obj = (tb_eval, fb_eval)

                # print(tb_eval, fb_eval, gain, best_gain)

        if best_gain > 0:
            node.col = best_attributes[0]
            node.value = best_attributes[1]

            (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                = divide_set(train_x, train_y, train_t, node.col, node.value)
            (est_x1, est_x2, est_y1, est_y2, est_t1, est_t2) \
                = divide_set(est_x, est_y, est_t, node.col, node.value)
            (_, _, nn_effect1, nn_effect2, _, _) \
                = divide_set(train_x, nn_effect, train_t, node.col, node.value)

            # y1 = train_y1
            # y2 = train_y2
            # t1 = train_t1
            # t2 = train_t2
            # y1 = np.concatenate((train_y1, val_y1))
            # y2 = np.concatenate((train_y2, val_y2))
            # t1 = np.concatenate((train_t1, val_t1))
            # t2 = np.concatenate((train_t2, val_t2))
            y1 = est_y1
            y2 = est_y2
            t1 = est_t1
            t2 = est_t2

            best_tb_effect = ace(y1, t1)
            best_fb_effect = ace(y2, t2)
            tb_p_val = get_pval(y1, t1)
            fb_p_val = get_pval(y2, t2)

            self.obj = self.obj - node.obj + best_tb_obj + best_fb_obj

            tb = HonestNode(obj=best_tb_obj, pehe=best_tb_obj, effect=best_tb_effect, p_val=tb_p_val,
                            node_depth=node.node_depth + 1,
                            num_samples=train_y1.shape[0])
            fb = HonestNode(obj=best_fb_obj, pehe=best_fb_obj, effect=best_fb_effect, p_val=fb_p_val,
                            node_depth=node.node_depth + 1,
                            num_samples=train_y2.shape[0])

            node.true_branch = self._fit(tb, train_x1, train_y1, train_t1, nn_effect1, est_x1, est_y1, est_t1)
            node.false_branch = self._fit(fb, train_x2, train_y2, train_t2, nn_effect2, est_x2, est_y2, est_t2)

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
