from CTL.causal_tree.ctl.binary_ctl import *
from sklearn.model_selection import train_test_split


class HonestCausalTreeLearnNode(CausalTreeLearnNode):

    def __init__(self, var=0.0, **kwargs):
        super().__init__(**kwargs)

        self.var = var


# ----------------------------------------------------------------
# Honest causal tree learn (ctl, base objective - honest penalty)
# ----------------------------------------------------------------
class CausalTreeLearnHonest(CausalTreeLearn):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = HonestCausalTreeLearnNode()
        self.train_to_est_ratio = 1.0

        self.num_treated = 1.0
        self.num_samples = 1.0
        self.treated_share = 1.0

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
        train_x, val_x, train_y, val_y, train_t, val_t = train_test_split(x, y, t, random_state=self.seed, shuffle=True,
                                                                          test_size=self.val_split)
        # get honest/estimation portion
        train_x, est_x, train_y, est_y, train_t, est_t = train_test_split(train_x, train_y, train_t, shuffle=True,
                                                                          random_state=self.seed, test_size=0.5)

        num_treat, _ = get_treat_size(t)
        self.num_treated = num_treat
        self.num_samples = x.shape[0]
        self.treated_share = self.num_treated / self.num_samples
        self.root.num_samples = est_x.shape[0]
        # ----------------------------------------------------------------
        # effect and pvals
        # ----------------------------------------------------------------
        effect = tau_squared(est_y, est_t)
        p_val = get_pval(est_y, est_t)
        self.root.effect = effect
        self.root.p_val = p_val

        # TODO: est ratio is overall?
        self.train_to_est_ratio = est_x.shape[0] / train_x.shape[0]
        current_var_treat, current_var_control = variance(train_y, train_t)
        # num_treat, num_cont = get_treat_size(train_t)
        # current_var = (1 * self.train_to_est_ratio) * (
        #         (current_var_treat / num_treat) + (current_var_control / num_cont))
        current_var = (1 * self.train_to_est_ratio) * (
                (current_var_treat / self.treated_share) + (current_var_control / (1 - self.treated_share)))

        self.root.var = current_var
        # ----------------------------------------------------------------
        # Not sure if i should eval in root or not
        # ----------------------------------------------------------------
        node_eval, mse = self._eval(train_y, train_t, val_y, val_t)
        self.root.obj = node_eval - current_var

        # ----------------------------------------------------------------
        # Add control/treatment means
        # ----------------------------------------------------------------
        self.root.control_mean = np.mean(est_y[est_t == 0])
        self.root.treatment_mean = np.mean(est_y[est_t == 1])

        self.root.num_samples = est_x.shape[0]

        self._fit(self.root, train_x, train_y, train_t, val_x, val_y, val_t, est_x, est_y, est_t)

    def _fit(self, node: HonestCausalTreeLearnNode, train_x, train_y, train_t, val_x, val_y, val_t,
             est_x, est_y, est_t):

        if train_x.shape[0] == 0 or val_x.shape[0] == 0 or est_x.shape[0] == 0:
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

            try:
                if self.feature_batch_size is None:
                    split_obj, upper_obj, lower_obj, value = self._eval_fast_honest(train_x, train_y, train_t, val_x,
                                                                                    val_y, val_t, unique_vals, col,
                                                                                    est_x, est_y, est_t)

                    (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                        = divide_set(train_x, train_y, train_t, col, value)

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

                    # combine honest and our objective
                    split_eval = (upper_obj + lower_obj) - (tb_var + fb_var)
                    gain = -(node.obj - node.var) + split_eval

                    if gain > best_gain:
                        best_gain = gain
                        best_attributes = [col, value]
                        best_tb_obj, best_fb_obj = (upper_obj, lower_obj)
                        best_tb_var, best_fb_var = (tb_var, fb_var)
                else:
                    for x in batch(unique_vals, self.feature_batch_size):

                        return_val = self._eval_fast_honest(train_x, train_y, train_t, val_x, val_y, val_t, x, col,
                                                            est_x, est_y, est_t)

                        split_obj, upper_obj, lower_obj = return_val[0], return_val[1], return_val[2]
                        upper_var_treated, upper_var_control = return_val[3], return_val[4]
                        lower_var_treated, lower_var_control = return_val[5], return_val[6]
                        value = return_val[7]

                        tb_var = (1 + self.train_to_est_ratio) * ((upper_var_treated / self.treated_share) + (
                                    upper_var_control / (1 - self.treated_share)))
                        fb_var = (1 + self.train_to_est_ratio) * ((lower_var_treated / self.treated_share) + (
                                    lower_var_control / (1 - self.treated_share)))

                        # split_obj, upper_obj, lower_obj, value = self._eval_fast_honest(train_x, train_y, train_t,
                        #                                                                 val_x, val_y, val_t,
                        #                                                                 x, col,
                        #                                                                 est_x, est_y, est_t)
                        #
                        # (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                        #     = divide_set(train_x, train_y, train_t, col, value)
                        #
                        # # ----------------------------------------------------------------
                        # # Honest penalty
                        # # ----------------------------------------------------------------
                        # var_treat1, var_control1 = variance(train_y1, train_t1)
                        # var_treat2, var_control2 = variance(train_y2, train_t2)
                        # # tb_var = (1 + self.train_to_est_ratio) * (
                        # #         (var_treat1 / (train_nt1 + 1)) + (var_control1 / (train_nc1 + 1)))
                        # # fb_var = (1 + self.train_to_est_ratio) * (
                        # #         (var_treat2 / (train_nt2 + 1)) + (var_control2 / (train_nc2 + 1)))
                        # tb_var = (1 + self.train_to_est_ratio) * (
                        #         (var_treat1 / self.treated_share) + (var_control1 / (1 - self.treated_share)))
                        # fb_var = (1 + self.train_to_est_ratio) * (
                        #         (var_treat2 / self.treated_share) + (var_control2 / (1 - self.treated_share)))

                        # combine honest and our objective
                        # split_eval = (upper_obj + lower_obj) - (tb_var + fb_var)
                        # gain = -(node.obj - node.var) + split_eval
                        #
                        # if gain > best_gain:
                        #     best_gain = gain
                        #     best_attributes = [col, value]
                        #     best_tb_obj, best_fb_obj = (upper_obj, lower_obj)
                        #     best_tb_var, best_fb_var = (tb_var, fb_var)

                        split_eval = (upper_obj + lower_obj) - (tb_var + fb_var)
                        gain = -(node.obj - node.var) + split_eval

                        if gain > best_gain:
                            best_gain = gain
                            best_attributes = [col, value]
                            best_tb_obj, best_fb_obj = (upper_obj, lower_obj)
                            best_tb_var, best_fb_var = (tb_var, fb_var)

            except:
                for value in unique_vals:

                    (val_x1, val_x2, val_y1, val_y2, val_t1, val_t2) \
                        = divide_set(val_x, val_y, val_t, col, value)

                    # check validation size
                    val_size = self.val_split * self.min_size if self.val_split * self.min_size > 2 else 2
                    val_nt1, val_nc1, val_check1 = min_size_value_bool(val_size, val_t1)
                    val_nt2, val_nc2, val_check2 = min_size_value_bool(val_size, val_t2)
                    if val_check1 or val_check2:
                        continue

                    # check training size
                    (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                        = divide_set(train_x, train_y, train_t, col, value)
                    train_nt1, train_nc1, train_check1 = min_size_value_bool(self.min_size, train_t1)
                    train_nt2, train_nc2, train_check2 = min_size_value_bool(self.min_size, train_t2)
                    if train_check1 or train_check2:
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

                    # ----------------------------------------------------------------
                    # Regular objective
                    # ----------------------------------------------------------------
                    tb_eval, tb_mse = self._eval(train_y1, train_t1, val_y1, val_t1)
                    fb_eval, fb_mse = self._eval(train_y2, train_t2, val_y2, val_t2)

                    # combine honest and our objective
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

            (val_x1, val_x2, val_y1, val_y2, val_t1, val_t2) \
                = divide_set(val_x, val_y, val_t, node.col, node.value)

            (est_x1, est_x2, est_y1, est_y2, est_t1, est_t2) \
                = divide_set(est_x, est_y, est_t, node.col, node.value)

            best_tb_effect = ace(est_y1, est_t1)
            best_fb_effect = ace(est_y2, est_t2)
            tb_p_val = get_pval(est_y1, est_t1)
            fb_p_val = get_pval(est_y2, est_t2)

            self.obj = self.obj - (node.obj - node.var) + (best_tb_obj + best_fb_obj -
                                                           best_tb_var - best_fb_var)
            # ----------------------------------------------------------------
            # Ignore "mse" here, come back to it later?
            # ----------------------------------------------------------------

            tb = HonestCausalTreeLearnNode(obj=best_tb_obj, effect=best_tb_effect, p_val=tb_p_val,
                                           node_depth=node.node_depth + 1,
                                           var=best_tb_var, num_samples=est_x1.shape[0])
            fb = HonestCausalTreeLearnNode(obj=best_fb_obj, effect=best_fb_effect, p_val=fb_p_val,
                                           node_depth=node.node_depth + 1,
                                           var=best_tb_var, num_samples=est_x2.shape[0])

            node.true_branch = self._fit(tb, train_x1, train_y1, train_t1, val_x1, val_y1, val_t1,
                                         est_x1, est_y1, est_t1)
            node.false_branch = self._fit(fb, train_x2, train_y2, train_t2, val_x2, val_y2, val_t2,
                                          est_x2, est_y2, est_t2)

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
