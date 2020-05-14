from CTL.causal_tree.ctl_trigger.trigger_ctl import *
from sklearn.model_selection import train_test_split


class TriggerHonestNode(TriggerNode):

    def __init__(self, var=0.0, **kwargs):
        super().__init__(**kwargs)

        self.var = var


# ----------------------------------------------------------------
# Base causal tree (ctl, base objective)
# ----------------------------------------------------------------
class TriggerTreeHonest(TriggerTree):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.root = TriggerHonestNode()

        self.train_to_est_ratio = 1.0
        # self.num_treated = 1.0
        # self.num_samples = 1.0
        # self.treated_share = 1.0

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

        self.root.num_samples = y.shape[0]
        # ----------------------------------------------------------------
        # effect and pvals
        # ----------------------------------------------------------------
        _, trigger = tau_squared_trigger(y, t, self.min_size, self.quartile)
        effect = ace_trigger(est_y, est_t, trigger)
        p_val = get_pval_trigger(est_y, est_t, trigger)
        self.root.effect = effect
        self.root.p_val = p_val
        self.root.trigger = trigger
        # p_val = get_pval_trigger(y, t, trigger)
        # self.root.effect = effect
        # self.root.p_val = p_val
        # self.root.trigger = trigger

        # TODO: est ratio is overall?
        self.train_to_est_ratio = est_x.shape[0] / train_x.shape[0]
        current_var_treat, current_var_control = variance_trigger(train_y, train_t, trigger)
        num_treat, num_cont = get_treat_size(train_t, trigger)
        treated_share = num_treat / train_x.shape[0] if num_treat > 0 else 1.0
        control_share = 1 - treated_share if treated_share < 1 else 0.0
        current_var = (1 * self.train_to_est_ratio) * (
                (current_var_treat / treated_share) + (current_var_control / (1 - control_share)))

        self.root.var = current_var
        # ----------------------------------------------------------------
        # Not sure if i should eval in root or not
        # ----------------------------------------------------------------
        node_eval, trigger, mse = self._eval(train_y, train_t, val_y, val_t)
        self.root.obj = node_eval - current_var

        # ----------------------------------------------------------------
        # Add control/treatment means
        # ----------------------------------------------------------------
        self.root.control_mean = np.mean(y[t >= trigger])
        self.root.treatment_mean = np.mean(y[t < trigger])

        self.root.num_samples = x.shape[0]

        self._fit(self.root, train_x, train_y, train_t, val_x, val_y, val_t, est_x, est_y, est_t)

    def _fit(self, node: TriggerHonestNode, train_x, train_y, train_t, val_x, val_y, val_t, est_x, est_y, est_t):

        if train_x.shape[0] == 0 or val_x.shape[0] == 0:
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
        best_tb_trigger, best_fb_trigger = (0.0, 0.0)

        column_count = train_x.shape[1]
        for col in range(0, column_count):
            unique_vals = np.unique(train_x[:, col])

            # ----------------------------------------------------------------
            # TODO: Max values stuff
            # ----------------------------------------------------------------

            for value in unique_vals:

                (val_x1, val_x2, val_y1, val_y2, val_t1, val_t2) \
                    = divide_set(val_x, val_y, val_t, col, value)

                (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                    = divide_set(train_x, train_y, train_t, col, value)

                # ----------------------------------------------------------------
                # Regular objective
                # ----------------------------------------------------------------
                tb_eval, tb_trigger, tb_mse = self._eval(train_y1, train_t1, val_y1, val_t1)
                fb_eval, fb_trigger, fb_mse = self._eval(train_y2, train_t2, val_y2, val_t2)

                (est_x1, est_x2, est_y1, est_y2, est_t1, est_t2) \
                    = divide_set(est_x, est_y, est_t, col, value)
                _, _, est_check1 = min_size_value_bool(self.min_size, est_t1, trigger=tb_trigger)
                _, _, est_check2 = min_size_value_bool(self.min_size, est_t2, trigger=fb_trigger)
                if est_check1 or est_check2:
                    continue

                # ----------------------------------------------------------------
                # Honest penalty
                # ----------------------------------------------------------------
                var_treat1, var_control1 = variance_trigger(train_y1, train_t1, trigger=tb_trigger)
                var_treat2, var_control2 = variance_trigger(train_y2, train_t2, trigger=fb_trigger)
                tb_nt, tb_nc = get_treat_size(train_t1, tb_trigger)
                fb_nt, fb_nc = get_treat_size(train_t2, fb_trigger)
                tb_treated_share = tb_nt / train_x.shape[0] if tb_nt > 0 else 1.0
                tb_control_share = 1 - tb_treated_share if tb_treated_share < 1 else 1.0
                fb_treated_share = fb_nt / train_x.shape[0] if fb_nt > 0 else 1.0
                fb_control_share = 1 - fb_treated_share if fb_treated_share < 1 else 1.0
                tb_var = (1 + self.train_to_est_ratio) * (
                        (var_treat1 / tb_treated_share) + (var_control1 / tb_control_share))
                fb_var = (1 + self.train_to_est_ratio) * (
                        (var_treat2 / fb_treated_share) + (var_control2 / fb_control_share))

                # combine honest and our objective
                split_eval = (tb_eval + fb_eval) - (tb_var + fb_var)
                # print(node.obj - node.var, split_eval)
                gain = -(node.obj - node.var) + split_eval

                if gain > best_gain:
                    best_gain = gain
                    best_attributes = [col, value]
                    best_tb_obj, best_fb_obj = (tb_eval, fb_eval)
                    best_tb_var, best_fb_var = (tb_var, fb_var)
                    best_tb_trigger, best_fb_trigger = (tb_trigger, fb_trigger)

        if best_gain > 0:
            node.col = best_attributes[0]
            node.value = best_attributes[1]

            (train_x1, train_x2, train_y1, train_y2, train_t1, train_t2) \
                = divide_set(train_x, train_y, train_t, node.col, node.value)

            (val_x1, val_x2, val_y1, val_y2, val_t1, val_t2) \
                = divide_set(val_x, val_y, val_t, node.col, node.value)

            (est_x1, est_x2, est_y1, est_y2, est_t1, est_t2) \
                = divide_set(est_x, est_y, est_t, node.col, node.value)

            best_tb_effect = ace_trigger(est_y1, est_t1, best_tb_trigger)
            best_fb_effect = ace_trigger(est_y2, est_t2, best_fb_trigger)
            tb_p_val = get_pval_trigger(est_y1, est_t1, best_tb_trigger)
            fb_p_val = get_pval_trigger(est_y2, est_t2, best_fb_trigger)

            self.obj = self.obj - (node.obj - node.var) + (best_tb_obj + best_fb_obj -
                                                           best_tb_var - best_fb_var)
            # ----------------------------------------------------------------
            # Ignore "mse" here, come back to it later?
            # ----------------------------------------------------------------

            tb = TriggerHonestNode(obj=best_tb_obj, effect=best_tb_effect, p_val=tb_p_val,
                                   node_depth=node.node_depth + 1, var=best_tb_var,
                                   num_samples=est_y1.shape[0], trigger=best_tb_trigger)
            fb = TriggerHonestNode(obj=best_fb_obj, effect=best_fb_effect, p_val=fb_p_val,
                                   node_depth=node.node_depth + 1, var=best_fb_var,
                                   num_samples=est_y2.shape[0], trigger=best_fb_trigger)

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
