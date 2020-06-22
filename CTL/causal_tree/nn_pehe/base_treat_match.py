from CTL.causal_tree.nn_pehe.tree import *


class BaseNode(PEHENode):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # self.obj = obj


# ----------------------------------------------------------------
# Base causal tree (ctl, base objective)
# ----------------------------------------------------------------
class BasePEHE(PEHETree):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = BaseNode()

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

            i_treat_nn = y[i_treat_idx]
            i_cont_nn = y[i_control_idx]

            nn_effect[i] = np.mean(i_treat_nn) - np.mean(i_cont_nn)

        return nn_effect

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

    def _fit(self, node: BaseNode, train_x, train_y, train_t, nn_effect):

        if train_x.shape[0] == 0:
            return node

        if node.node_depth > self.tree_depth:
            self.tree_depth = node.node_depth

        if self.max_depth == self.tree_depth:
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

                # print(tb_eval, fb_eval, gain, best_gain)

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
