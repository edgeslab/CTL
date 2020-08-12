from CTL.causal_tree.ctl.binary_ctl import *
from sklearn.model_selection import train_test_split


class HonestNode(CTLearnNode):

    def __init__(self, var=0.0, **kwargs):
        super().__init__(**kwargs)
        self.var = var
        # self.obj = obj


# ----------------------------------------------------------------
# Base causal tree (ctl, base objective)
# ----------------------------------------------------------------
class HonestTree(CTLearn):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = HonestNode()
        self.train_to_est_ratio = 1.0
        self.num_treated = 1.0
        self.num_samples = 1.0
        self.treated_share = 1.0

    def honest_eval(self, train_y, train_t):
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
        train_x, est_x, train_y, est_y, train_t, est_t = train_test_split(x, y, t, shuffle=True,
                                                                          random_state=self.seed, test_size=0.5)

        self.root.num_samples = est_y.shape[0]
        num_treat, _ = get_treat_size(est_t)
        self.num_treated = num_treat
        self.num_samples = est_x.shape[0]
        self.treated_share = self.num_treated / self.num_samples
        # ----------------------------------------------------------------
        # effect and pvals
        # ----------------------------------------------------------------
        effect = tau_squared(est_y, est_t)
        p_val = get_pval(est_y, est_t)
        self.root.effect = effect
        self.root.p_val = p_val

        # ----------------------------------------------------------------
        # Not sure if i should eval in root or not
        # ----------------------------------------------------------------
        node_eval, mse = self.honest_eval(train_y, train_t)
        self.train_to_est_ratio = est_x.shape[0] / train_x.shape[0]
        current_var_treat, current_var_control = variance(train_y, train_t)
        current_var = (1 * self.train_to_est_ratio) * (
                (current_var_treat / self.treated_share) + (current_var_control / (1 - self.treated_share)))

        self.root.var = current_var
        self.root.obj = node_eval - current_var

        # ----------------------------------------------------------------
        # Add control/treatment means
        # ----------------------------------------------------------------
        self.root.control_mean = np.mean(y[t == 0])
        self.root.treatment_mean = np.mean(y[t == 1])

        self.root.num_samples = est_x.shape[0]

        self._fit(self.root, train_x, train_y, train_t, est_x, est_y, est_t)

    def _fit(self, node: HonestNode, train_x, train_y, train_t, est_x, est_y, est_t):

        if train_x.shape[0] == 0 or est_x.shape[0] == 0:
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

        best_gain = 0.0
        best_attributes = []
        best_tb_obj, best_fb_obj = (0.0, 0.0)
        best_tb_var, best_fb_var = (0.0, 0.0)

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

                # check estimation treatment numbers
                (est_x1, est_x2, est_y1, est_y2, est_t1, est_t2) \
                    = divide_set(est_x, est_y, est_t, col, value)
                est_nt1, est_nc1, est_check1 = min_size_value_bool(self.min_size, est_t1)
                est_nt2, est_nc2, est_check2 = min_size_value_bool(self.min_size, est_t2)
                if est_check1 or est_check2:
                    continue

                # ----------------------------------------------------------------
                # Honest penalty
                # ----------------------------------------------------------------
                var_treat1, var_control1 = variance(train_y1, train_t1)
                var_treat2, var_control2 = variance(train_y2, train_t2)
                # tb_var = (1 + self.train_to_est_ratio) * (
                #         (var_treat1 / (train_nt1 + 1)) + (var_control1 / (train_nc1 + 1)))
                # fb_var = (1 + self.train_to_est_ratio) * (
                #         (var_treat2 / (train_nt2 + 1)) + (var_control2 / (train_nc2 + 1)))
                tb_var = (1 + self.train_to_est_ratio) * (
                        (var_treat1 / self.treated_share) + (var_control1 / (1 - self.treated_share)))
                fb_var = (1 + self.train_to_est_ratio) * (
                        (var_treat2 / self.treated_share) + (var_control2 / (1 - self.treated_share)))

                tb_eval, tb_mse = self.honest_eval(train_y1, train_t1)
                fb_eval, fb_mse = self.honest_eval(train_y2, train_t2)

                split_eval = (tb_eval + fb_eval) - (tb_var + fb_var)
                gain = -(node.obj - node.var) + split_eval

                if gain > best_gain:
                    best_gain = gain
                    best_attributes = [col, value]
                    best_tb_obj, best_fb_obj = (tb_eval, fb_eval)
                    best_tb_var, best_fb_var = (tb_var, fb_var)

        if best_gain > 0:
            node.col = best_attributes[0]
            node.value = best_attributes[1]

            (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                = divide_set(train_x, train_y, train_t, node.col, node.value)

            (est_x1, est_x2, est_y1, est_y2, est_t1, est_t2) \
                = divide_set(est_x, est_y, est_t, node.col, node.value)

            best_tb_effect = ace(est_y1, est_t1)
            best_fb_effect = ace(est_y2, est_t2)
            tb_p_val = get_pval(est_y1, est_t1)
            fb_p_val = get_pval(est_y2, est_t2)

            self.obj = self.obj - node.obj + best_tb_obj + best_fb_obj

            # ----------------------------------------------------------------
            # Ignore "mse" here, come back to it later?
            # ----------------------------------------------------------------

            tb = HonestNode(obj=best_tb_obj, effect=best_tb_effect, p_val=tb_p_val,
                            node_depth=node.node_depth + 1,
                            num_samples=est_y1.shape[0], var=best_tb_var)
            fb = HonestNode(obj=best_fb_obj, effect=best_fb_effect, p_val=fb_p_val,
                            node_depth=node.node_depth + 1,
                            num_samples=est_y2.shape[0], var=best_fb_var)

            node.true_branch = self._fit(tb, train_x1, train_y1, train_t1, est_x1, est_y1, est_t1)
            node.false_branch = self._fit(fb, train_x2, train_y2, train_t2, est_x2, est_y2, est_t2)

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
