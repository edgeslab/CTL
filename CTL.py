from sklearn.model_selection import train_test_split
import subprocess
from CTL.ct_util import *
import numpy as np


# TODO: need to get better objective value of heterogeneity rather than w/e objective I use
# TODO: should I weight the objective?
# TODO: remove the magnitude?
# TODO: triggers, NaN
# TODO: better way to set best gains

# Class that defines the causal tree nodes
class CausalTree:
    def __init__(self, cont=False, max_depth=-1, min_size=2, weight=0.5, seed=None, split_size=0.5, honest=False,
                 val_honest=False, variables=None, weight_obj=False, base_obj=True, quartile=False):
        self.root = None
        self.max = -np.inf
        self.min = np.inf
        self.num_leaves = 0
        self.max_depth = max_depth
        self.min_size = min_size
        self.weight = weight
        self.seed = seed
        if cont:
            self.eval_func = self.objective_cont
        else:
            self.eval_func = self.objective
        self.quartile = quartile

        self.weight_obj = weight_obj
        self.base_obj = base_obj

        self.honest = honest
        self.val_honest = val_honest

        self.split_size = split_size
        self.obj = 0.0
        self.tree_depth = 0
        self.variables = variables
        self.mse = 0.0

    class Node:
        def __init__(self, col=-1, value=None, true_branch=None, false_branch=None, effect=0.0,
                     treat_split=None, leaf=False, leaf_num=None, current_obj=0.0, p_val=-1, samples=0, node_var=0.0,
                     node_depth=0, node_mse=0.0):
            self.col = col  # the column of the feature used for splitting
            self.value = value  # the value that splits the data

            self.current_obj = current_obj
            self.effect = effect
            self.p_val = p_val
            self.treat_split = treat_split  # treatment splitting location (for continuous values)
            self.variance = node_var

            self.true_branch = true_branch  # pointer to node for true branch
            self.false_branch = false_branch  # pointer to node for false branch
            self.leaf = leaf  # true/false if leaf or not
            self.leaf_num = leaf_num  # the leaf label

            self.samples = samples

            self.feature_name = None
            self.feature_split = None

            self.node_depth = node_depth
            self.node_mse = node_mse

    def fit(self, rows, labels, treatment):
        if rows.shape[0] == 0:
            return self.Node()

        if self.seed is not None:
            np.random.seed(self.seed)

        curr_split = None
        current_var = 0.0
        if self.eval_func == self.objective:
            if self.val_honest or self.honest:
                if self.val_honest:
                    train_rows, est_rows, train_outcome, est_labels, train_treat, est_treatment = \
                        train_test_split(rows, labels, treatment, shuffle=True, test_size=self.split_size)
                else:
                    train_rows, est_rows, train_outcome, est_labels, train_treat, est_treatment = \
                        train_test_split(rows, labels, treatment, shuffle=True, test_size=0.5)

                _, effect = tau_squared(est_labels, est_treatment)
                p_val = get_pval(est_labels, est_treatment)

                train_to_est_ratio = est_rows.shape[0] / train_rows.shape[0]
                current_var_treat, current_var_control = variance(train_outcome, train_treat)
                num_cont, num_treat = get_num_treat(train_treat)[1:]
                current_var = (1 + train_to_est_ratio) * (
                        (current_var_treat / num_treat) + (current_var_control / num_cont))
            else:
                _, effect = tau_squared(labels, treatment)
                p_val = get_pval(labels, treatment)
        elif self.eval_func == self.objective_cont:
            if self.val_honest or self.honest:
                if self.val_honest:
                    train_rows, est_rows, train_outcome, est_labels, train_treat, est_treatment = \
                        train_test_split(rows, labels, treatment, shuffle=True, test_size=self.split_size)
                else:
                    train_rows, est_rows, train_outcome, est_labels, train_treat, est_treatment = \
                        train_test_split(rows, labels, treatment, shuffle=True, test_size=0.5)

                _, _, curr_split = tau_squared_cont(train_rows, train_treat)
                _, effect, _ = tau_squared(est_rows, est_treatment, treat_split=curr_split)
                p_val = get_pval(est_rows, est_treatment, treat_split=curr_split)

                train_to_est_ratio = est_rows.shape[0] / train_rows.shape[0]
                current_var_treat, current_var_control = variance(train_outcome, train_treat)
                num_cont, num_treat = get_num_treat(train_treat, self.min_size)[1:]
                current_var = (1 + train_to_est_ratio) * (
                        (current_var_treat / num_treat) + (current_var_control / num_cont))
            else:
                _, effect, curr_split = tau_squared_cont(labels, treatment, self.min_size)
                p_val = get_pval(labels, treatment, curr_split)
        else:
            # otherwise something is wrong, assume binary learn
            _, effect = tau_squared(labels, treatment)
            p_val = get_pval(labels, treatment)

        if self.seed is not None:
            np.random.seed(self.seed)

        if self.honest:
            rows, est_rows, labels, est_labels, treatment, est_treatment = \
                train_test_split(rows, labels, treatment, shuffle=True, test_size=0.5)
            self.root = self.Node(col=-1, value=None, current_obj=0.0, effect=effect,
                                  p_val=p_val, treat_split=curr_split, node_var=current_var, node_depth=0)
            self.root = self.fit_r(rows, labels, treatment, curr_depth=0, node=self.root,
                                   est_rows=est_rows, est_labels=est_labels, est_treatment=est_treatment)
        else:
            self.root = self.Node(col=-1, value=None, current_obj=0.0, effect=effect,
                                  p_val=p_val, treat_split=curr_split, node_depth=0)
            self.root = self.fit_r(rows, labels, treatment, curr_depth=0, node=self.root)

    def fit_r(self, rows, labels, treatment, curr_depth=0, node=None,
              est_rows=None, est_labels=None, est_treatment=None):

        if rows.shape[0] == 0:
            return node

        train_rows, val_rows, train_outcome, \
        val_outcome, train_treat, val_treat = train_test_split(rows, labels, treatment, shuffle=True,
                                                               test_size=self.split_size)
        if self.val_honest:
            train_to_est_ratio = val_rows.shape[0] / train_rows.shape[0]
            node.samples = val_rows.shape[0]
        elif self.honest:
            train_to_est_ratio = est_rows.shape[0] / rows.shape[0]
            node.samples = est_rows.shape[0]
        else:
            train_to_est_ratio = -1
            node.samples = train_rows.shape[0]

        if curr_depth > self.tree_depth:
            self.tree_depth = curr_depth

        # I'm thinking to initialize the nodes before doing the recursive call
        if self.max_depth == curr_depth:
            # node leaf number
            self.num_leaves += 1
            # add node leaf number to node class
            node.leaf_num = self.num_leaves
            node.leaf = True
            return node

        best_gain = 0.0
        best_attribute = None
        best_sets = {}

        best_tb_obj = 0.0
        best_fb_obj = 0.0

        best_tb_split = 0.0
        best_fb_split = 0.0

        best_tb_var = 0.0
        best_fb_var = 0.0

        tb_var = 0.0
        fb_var = 0.0

        best_tb_mse = 0.0
        best_fb_mse = 0.0

        curr_depth += 1

        column_count = rows.shape[1]
        for col in range(0, column_count):
            # unique values
            unique_vals = np.unique(rows[:, col])

            for value in unique_vals:
                # binary treatment splitting
                if self.eval_func == self.objective:

                    # (set1, set2, y1, y2, treat1, treat2) = divide_set(rows, labels, treatment, col, value)

                    if size_check_fail(train_rows, train_outcome, train_treat, col, value, self.min_size):
                        continue
                    if size_check_fail(val_rows, val_outcome, val_treat, col, value, 2):
                        continue

                    if not self.val_honest and self.honest:
                        if size_check_fail(est_rows, est_labels, est_treatment, col, value, 2):
                            continue
                        (est_set1, est_set2, est_y1, est_y2, est_treat1, est_treat2) \
                            = divide_set(est_rows, est_labels, est_treatment, col, value)
                    else:
                        est_set1, est_set2, est_y1, est_y2, est_treat1, est_treat2 = [0] * 6

                    (set1, set2, y1, y2, treat1, treat2) = divide_set(rows, labels, treatment, col, value)

                    (train_set1, train_set2, train_y1, train_y2, train_treat1, train_treat2) \
                        = divide_set(train_rows, train_outcome, train_treat, col, value)

                    (val_set1, val_set2, val_y1, val_y2, val_treat1, val_treat2) \
                        = divide_set(val_rows, val_outcome, val_treat, col, value)

                    if self.honest:
                        _, tb_num_cont, tb_num_treat = get_num_treat(train_treat1, self.min_size)
                        _, fb_num_cont, fb_num_treat = get_num_treat(train_treat2, self.min_size)
                        var1_treat, var1_control = variance(train_y1, train_treat1)
                        var2_treat, var2_control = variance(train_y2, train_treat2)
                        tb_var = (1 + train_to_est_ratio) * (
                                (var1_treat / (tb_num_treat + 1)) + (var1_control / (tb_num_cont + 1)))
                        fb_var = (1 + train_to_est_ratio) * (
                                (var2_treat / (fb_num_treat + 1)) + (var2_control / (fb_num_cont + 1)))

                    tb_eval, tb_mse = self.eval_func(train_y1, train_treat1, val_y1, val_treat1)
                    fb_eval, fb_mse = self.eval_func(train_y2, train_treat2, val_y2, val_treat2)

                    split_eval = (tb_eval + fb_eval) - (tb_var + fb_var)
                    gain = -(node.current_obj - node.variance) + split_eval

                    if gain > best_gain:
                        best_gain = gain
                        best_tb_obj = tb_eval
                        best_fb_obj = fb_eval
                        best_attribute = (col, value)
                        best_sets['set1'] = set1
                        best_sets['y1'] = y1
                        best_sets['treat1'] = treat1
                        best_sets['set2'] = set2
                        best_sets['y2'] = y2
                        best_sets['treat2'] = treat2
                        best_sets['est_set1'] = est_set1
                        best_sets['est_set2'] = est_set2
                        best_sets['est_y1'] = est_y1
                        best_sets['est_y2'] = est_y2
                        best_sets['est_treat1'] = est_treat1
                        best_sets['est_treat2'] = est_treat2
                        best_sets['train_y1'] = train_y1
                        best_sets['train_y2'] = train_y2
                        best_sets['train_treat1'] = train_treat1
                        best_sets['train_treat2'] = train_treat2
                        best_sets['val_y1'] = val_y1
                        best_sets['val_y2'] = val_y2
                        best_sets['val_treat1'] = val_treat1
                        best_sets['val_treat2'] = val_treat2
                        best_tb_var = tb_var
                        best_fb_var = fb_var
                        best_tb_mse = tb_mse
                        best_fb_mse = fb_mse
                        # if self.use_mse:
                        #     best_tb_mse = self.eval_func(train_y1, train_treat1, val_y1, val_treat1, return_mse=True)
                        #     best_fb_mse = self.eval_func(train_y2, train_treat2, val_y2, val_treat2, return_mse=True)

                # continuous treatment splitting
                if self.eval_func == self.objective_cont:
                    (set1, set2, y1, y2, treat1, treat2) = divide_set(rows, labels, treatment, col, value)

                    if np.unique(y1).shape[0] <= 1 or np.unique(y2).shape[0] <= 1:
                        continue

                    (train_set1, train_set2, train_y1, train_y2, train_treat1, train_treat2) \
                        = divide_set(train_rows, train_outcome, train_treat, col, value)

                    (val_set1, val_set2, val_y1, val_y2, val_treat1, val_treat2) \
                        = divide_set(val_rows, val_outcome, val_treat, col, value)

                    if np.unique(val_y1).shape[0] <= 1 or np.unique(val_y2).shape[0] <= 1 or \
                            np.unique(train_y1).shape[0] <= 1 or np.unique(train_y2).shape[0] <= 1:
                        continue

                    if not self.val_honest and self.honest:
                        (est_set1, est_set2, est_y1, est_y2, est_treat1, est_treat2) \
                            = divide_set(est_rows, est_labels, est_treatment, col, value)
                        if np.unique(est_y1).shape[0] <= 1 or np.unique(est_y1).shape[0] <= 1:
                            continue
                    else:
                        est_set1, est_set2, est_y1, est_y2, est_treat1, est_treat2 = [0] * 6

                    tb_eval, tb_split, tb_mse = self.eval_func(train_y1, train_treat1, val_y1, val_treat1)
                    fb_eval, fb_split, fb_mse = self.eval_func(train_y2, train_treat2, val_y2, val_treat2)
                    if self.honest:
                        _, tb_num_cont, tb_num_treat = get_num_treat(train_treat1, self.min_size, treat_split=tb_split)
                        _, fb_num_cont, fb_num_treat = get_num_treat(train_treat2, self.min_size, treat_split=fb_split)
                        var1_treat, var1_control = variance(train_y1, train_treat1, treat_split=tb_split)
                        var2_treat, var2_control = variance(train_y2, train_treat2, treat_split=fb_split)
                        tb_var = (1 + train_to_est_ratio) * (
                                (var1_treat / (tb_num_treat + 1)) + (var1_control / (tb_num_cont + 1)))
                        fb_var = (1 + train_to_est_ratio) * (
                                (var2_treat / (fb_num_treat + 1)) + (var2_control / (fb_num_cont + 1)))

                    split_eval = (tb_eval + fb_eval)
                    gain = -node.current_obj + split_eval

                    if gain > best_gain:
                        best_gain = gain
                        best_tb_obj = tb_eval
                        best_fb_obj = fb_eval
                        best_attribute = (col, value)
                        best_sets['set1'] = set1
                        best_sets['y1'] = y1
                        best_sets['treat1'] = treat1
                        best_sets['set2'] = set2
                        best_sets['y2'] = y2
                        best_sets['treat2'] = treat2
                        best_sets['est_set1'] = est_set1
                        best_sets['est_set2'] = est_set2
                        best_sets['est_y1'] = est_y1
                        best_sets['est_y2'] = est_y2
                        best_sets['est_treat1'] = est_treat1
                        best_sets['est_treat2'] = est_treat2
                        best_sets['train_y1'] = train_y1
                        best_sets['train_y2'] = train_y2
                        best_sets['train_treat1'] = train_treat1
                        best_sets['train_treat2'] = train_treat2
                        best_sets['val_y1'] = val_y1
                        best_sets['val_y2'] = val_y2
                        best_sets['val_treat1'] = val_treat1
                        best_sets['val_treat2'] = val_treat2
                        best_tb_var = tb_var
                        best_fb_var = fb_var
                        best_tb_split = tb_split
                        best_fb_split = fb_split
                        best_tb_mse = tb_mse
                        best_fb_mse = fb_mse
                        # if self.use_mse:
                        #     best_tb_mse = self.eval_func(train_y1, train_treat1, val_y1, val_treat1, return_mse=True)
                        #     best_fb_mse = self.eval_func(train_y2, train_treat2, val_y2, val_treat2, return_mse=True)

        if self.eval_func == self.objective:
            if best_gain > 0:
                node.col = best_attribute[0]
                node.value = best_attribute[1]

                if self.honest:
                    if self.val_honest:
                        best_tb_effect = self.effect(best_sets['val_y1'], best_sets['val_treat1'])
                        best_fb_effect = self.effect(best_sets['val_y2'], best_sets['val_treat2'])
                        tb_p_val = get_pval(best_sets['val_y1'], best_sets['val_treat1'])
                        fb_p_val = get_pval(best_sets['val_y2'], best_sets['val_treat2'])
                    else:
                        best_tb_effect = self.effect(best_sets['est_y1'], best_sets['est_treat1'])
                        best_fb_effect = self.effect(best_sets['est_y2'], best_sets['est_treat2'])
                        tb_p_val = get_pval(best_sets['est_y1'], best_sets['est_treat1'])
                        fb_p_val = get_pval(best_sets['est_y2'], best_sets['est_treat2'])
                else:
                    best_tb_effect = self.effect(best_sets['train_y1'], best_sets['train_treat1'])
                    best_fb_effect = self.effect(best_sets['train_y2'], best_sets['train_treat2'])
                    tb_p_val = get_pval(best_sets['train_y1'], best_sets['train_treat1'])
                    fb_p_val = get_pval(best_sets['train_y2'], best_sets['train_treat2'])

                self.obj = self.obj - (node.current_obj - node.variance) + (best_tb_obj + best_fb_obj -
                                                                            best_tb_var - best_fb_var)

                self.mse = self.mse - node.node_mse + best_tb_mse + best_fb_mse
                # if self.use_mse:
                #     self.obj = self.obj - node.current_obj + best_tb_mse + best_fb_mse

                tb = self.Node(current_obj=best_tb_obj, effect=best_tb_effect, p_val=tb_p_val, node_var=best_tb_var,
                               node_depth=curr_depth, node_mse=best_tb_mse)
                fb = self.Node(current_obj=best_fb_obj, effect=best_fb_effect, p_val=fb_p_val, node_var=best_fb_var,
                               node_depth=curr_depth, node_mse=best_fb_mse)

                node.true_branch = self.fit_r(best_sets['set1'], best_sets['y1'], best_sets['treat1'],
                                              curr_depth=curr_depth, node=tb,
                                              est_rows=best_sets['est_set1'],
                                              est_labels=best_sets['est_y1'],
                                              est_treatment=best_sets['est_treat1'])
                node.false_branch = self.fit_r(best_sets['set2'], best_sets['y2'], best_sets['treat2'],
                                               curr_depth=curr_depth, node=fb,
                                               est_rows=best_sets['est_set2'],
                                               est_labels=best_sets['est_y2'],
                                               est_treatment=best_sets['est_treat2'])

                if node.effect > self.max:
                    self.max = node.effect
                if node.effect < self.min:
                    self.min = node.effect

                return node
                # min and max
            else:
                if node.effect > self.max:
                    self.max = node.effect
                if node.effect < self.min:
                    self.min = node.effect

                # node leaf number
                self.num_leaves += 1
                # add node leaf number to node class
                node.leaf_num = self.num_leaves
                node.leaf = True
                return node

        elif self.eval_func == self.objective_cont:
            if best_gain > 0:

                node.col = best_attribute[0]
                node.value = best_attribute[1]

                if self.honest:
                    if self.val_honest:
                        best_tb_effect = self.effect(best_sets['val_y1'], best_sets['val_treat1'],
                                                     treat_split=best_tb_split)
                        best_fb_effect = self.effect(best_sets['val_y2'], best_sets['val_treat2'],
                                                     treat_split=best_fb_split)
                        tb_p_val = get_pval(best_sets['val_y1'], best_sets['val_treat1'], treat_split=best_tb_split)
                        fb_p_val = get_pval(best_sets['val_y2'], best_sets['val_treat2'], treat_split=best_fb_split)
                    else:
                        best_tb_effect = self.effect(best_sets['est_y1'], best_sets['est_treat1'],
                                                     treat_split=best_tb_split)
                        best_fb_effect = self.effect(best_sets['est_y2'], best_sets['est_treat2'],
                                                     treat_split=best_fb_split)
                        tb_p_val = get_pval(best_sets['est_y1'], best_sets['est_treat1'], treat_split=best_tb_split)
                        fb_p_val = get_pval(best_sets['est_y2'], best_sets['est_treat2'], treat_split=best_fb_split)
                else:
                    best_tb_effect = self.effect(best_sets['train_y1'], best_sets['train_treat1'],
                                                 treat_split=best_tb_split)
                    best_fb_effect = self.effect(best_sets['train_y2'], best_sets['train_treat2'],
                                                 treat_split=best_fb_split)
                    tb_p_val = get_pval(best_sets['train_y1'], best_sets['train_treat1'], treat_split=best_tb_split)
                    fb_p_val = get_pval(best_sets['train_y2'], best_sets['train_treat2'], treat_split=best_fb_split)

                self.obj = self.obj - (node.current_obj - node.variance) + (best_tb_obj + best_fb_obj -
                                                                            best_tb_var - best_fb_var)

                self.mse = self.mse - node.node_mse + best_tb_mse + best_fb_mse
                # if self.use_mse:
                #     self.obj = self.obj - node.current_obj + best_tb_mse + best_fb_mse

                tb = self.Node(current_obj=best_tb_obj, effect=best_tb_effect, p_val=tb_p_val,
                               treat_split=best_tb_split, node_mse=best_tb_mse)
                fb = self.Node(current_obj=best_fb_obj, effect=best_fb_effect, p_val=fb_p_val,
                               treat_split=best_fb_split, node_mse=best_fb_mse)

                node.true_branch = self.fit_r(best_sets['set1'], best_sets['y1'], best_sets['treat1'],
                                              curr_depth=curr_depth, node=tb)
                node.false_branch = self.fit_r(best_sets['set2'], best_sets['y2'], best_sets['treat2'],
                                               curr_depth=curr_depth, node=fb)

                if node.effect > self.max:
                    self.max = node.effect
                if node.effect < self.min:
                    self.min = node.effect

                return node
            else:
                if node.effect > self.max:
                    self.max = node.effect
                if node.effect < self.min:
                    self.min = node.effect

                # node leaf number
                self.num_leaves += 1
                # add node leaf number to node class
                node.leaf_num = self.num_leaves
                node.leaf = True
                return node

    def objective(self, train_outcome, train_treatment, val_outcome, val_treatment):
        """Calculates the objective value

        outcome: the observed outcome vector
        treatment: the treatment vector
        """
        total_train = train_outcome.shape[0]
        total_val = val_outcome.shape[0]

        return_val = (-np.inf, -np.inf)

        if total_train == 0 or total_val == 0:
            return return_val

        if isinstance(train_outcome[0], str):
            train_outcome = string_to_int(train_outcome)
            val_outcome = string_to_int(val_outcome)

        train_effect = ace(train_outcome, train_treatment)
        val_effect = ace(val_outcome, val_treatment)

        # train_mse = total_val * (train_effect ** 2)
        # val_mse = total_val * (val_effect ** 2)
        # train_mse = (1-weight) * total_train * (train_effect ** 2)
        train_mse = (1 - self.weight) * total_train * (train_effect ** 2)
        cost = self.weight * total_val * np.abs(train_effect - val_effect)

        if self.base_obj:
            obj = (train_mse - cost) / (np.abs(total_train - total_val) + 1)
            mse = total_train * (train_effect ** 2)
            if self.weight_obj:
                obj = total_train * obj
        else:
            train_mse = (1 - self.weight) * train_effect ** 2
            cost = self.weight * np.abs(train_effect - val_effect)
            obj = (train_mse - cost)
            mse = total_train * (train_effect ** 2)
            if self.weight_obj:
                obj = total_train * obj

        return obj, mse

    def objective_cont(self, train_outcome, train_treatment, val_outcome, val_treatment):
        """Continuous case"""
        total_train = train_outcome.shape[0]
        total_val = val_outcome.shape[0]

        return_val = (-np.inf, -np.inf, -np.inf)

        if total_train == 0 or total_val == 0:
            return return_val

        unique_treatment = np.unique(train_treatment)

        if unique_treatment.shape[0] == 1:
            return return_val

        unique_treatment = (unique_treatment[1:] + unique_treatment[:-1]) / 2

        if self.quartile:
            first_quartile = int(np.floor(unique_treatment.shape[0] / 4))
            third_quartile = int(np.ceil(3 * unique_treatment.shape[0] / 4))

            unique_treatment = unique_treatment[first_quartile:third_quartile]

        if isinstance(train_outcome[0], str):
            train_outcome = string_to_int(train_outcome)
            val_outcome = string_to_int(val_outcome)

        yyt = np.tile(train_outcome, (unique_treatment.shape[0], 1))
        ttt = np.tile(train_treatment, (unique_treatment.shape[0], 1))
        yyv = np.tile(val_outcome, (unique_treatment.shape[0], 1))
        ttv = np.tile(val_treatment, (unique_treatment.shape[0], 1))

        xt = np.transpose(np.transpose(ttt) > unique_treatment)
        ttt[xt] = 1
        ttt[np.logical_not(xt)] = 0
        xv = np.transpose(np.transpose(ttv) > unique_treatment)
        ttv[xv] = 1
        ttv[np.logical_not(xv)] = 0

        # do the min_size check on validation set for now
        treat_num = np.sum(ttv == 1, axis=1)
        cont_num = np.sum(ttv == 0, axis=1)
        min_size_idx = np.where(np.logical_and(treat_num >= self.min_size, cont_num >= self.min_size))

        unique_treatment = unique_treatment[min_size_idx]
        ttt = ttt[min_size_idx]
        yyt = yyt[min_size_idx]
        ttv = ttv[min_size_idx]
        yyv = yyv[min_size_idx]

        if ttv.shape[0] == 0:
            return return_val
        if ttt.shape[0] == 0:
            return return_val

        y_t_m_t = np.sum((yyt * (ttt == 1)), axis=1) / np.sum(ttt == 1, axis=1)
        y_c_m_t = np.sum((yyt * (ttt == 0)), axis=1) / np.sum(ttt == 0, axis=1)

        y_t_m_v = np.sum((yyv * (ttv == 1)), axis=1) / np.sum(ttv == 1, axis=1)
        y_c_m_v = np.sum((yyv * (ttv == 0)), axis=1) / np.sum(ttv == 0, axis=1)

        train_effect = y_t_m_t - y_c_m_t
        train_err = train_effect ** 2

        val_effect = y_t_m_v - y_c_m_v
        # val_err = val_effect ** 2

        if self.base_obj:
            train_mse = (1 - self.weight) * (total_train * train_err)
            cost = self.weight * total_val * np.abs(train_effect - val_effect)
            obj = (train_mse - cost) / (np.abs(total_train - total_val) + 1)
            if self.weight_obj:
                obj = total_train * obj
        else:
            train_mse = (1 - self.weight) * train_err
            cost = self.weight * np.abs(train_effect - val_effect)
            obj = (train_mse - cost)
            if self.weight_obj:
                obj = total_train * obj

        argmax_obj = np.argmax(obj)
        best_obj = obj[argmax_obj]
        best_split = unique_treatment[argmax_obj]
        mse = train_err[argmax_obj]

        return best_obj, best_split, mse

        # max_err = np.argmax(train_err)
        # train_best_err = train_err[max_err]
        # # val_best_err = val_err[max_err]
        #
        # train_best_effect = train_effect[max_err]
        # val_best_effect = val_effect[max_err]
        #
        # best_split = unique_treatment[max_err]
        #
        # best_effect = train_effect[max_err]
        #
        # mse = train_best_err
        #
        # if self.base_obj:
        #     train_mse = (1 - self.weight) * (total_train * train_best_err)
        #     cost = self.weight * total_val * np.abs(train_best_effect - val_best_effect)
        #     obj = (train_mse - cost) / (np.abs(total_train - total_val) + 1)
        #     if self.weight_obj:
        #         obj = total_train * obj
        # else:
        #     train_mse = (1 - self.weight) * train_best_err
        #     cost = self.weight * np.abs(train_best_effect - val_best_effect)
        #     obj = (train_mse - cost)
        #     if self.weight_obj:
        #         obj = total_train * obj
        #
        # return obj, best_split, best_effect, mse

    def tree_to_dot(self, tree, feat_names, filename='tree', alpha=0.05, show_pval=True):
        filename = filename + '.dot'
        feat_names = col_dict(feat_names)
        with open(filename, 'w') as f:
            f.write('digraph Tree {\n')
            f.write('node [shape=box, fontsize=32] ;\n')
            f.write('edge [fontsize=24] ;\n')
            self.tree_to_dot_r(tree, feat_names, f, counter=0, alpha=alpha, show_pval=show_pval)
            f.write("}")

    @staticmethod
    def dot_to_png(dot_filename="tree", output_file=None, extension="png"):

        if output_file is None:
            command = ["dot", "-T" + extension, "-Gdpi=500", dot_filename + '.dot', "-o",
                       dot_filename + "." + extension]
        else:
            command = ["dot", "-T" + extension, "-Gdpi=500", dot_filename + '.dot', "-o", output_file + "." + extension]
        try:
            if os.name == 'nt':
                subprocess.check_call(command, shell=True)
            else:
                subprocess.check_call(command)
        except subprocess.CalledProcessError:
            exit("Could not run dot, ie graphviz, to "
                 "produce visualization")

    @staticmethod
    def effect(outcome, treatment, treat_split=None):
        total = outcome.shape[0]

        return_val = (-np.inf, -np.inf)

        if total == 0:
            return return_val

        if isinstance(outcome[0], str):
            outcome = string_to_int(outcome)

        treat_vect = treatment

        if treat_split is not None:
            treat_vect = np.copy(treatment)
            treat = treat_vect > treat_split
            control = treat_vect <= treat_split
            treat_vect[treat] = 1
            treat_vect[control] = 0

        effect = ace(outcome, treat_vect)

        return effect

    def plot_tree(self, feat_names, file="tree", alpha=0.05, show_pval=True, create_png=True, extension="png"):

        name_split = file.split('/')
        if len(name_split) > 1:
            img_folder = name_split[0:-1]
            file_name = name_split[-1]

            img_folder = '/'.join(img_folder)

            dot_folder = img_folder + '/dot_folder/'

            check_dir(img_folder + '/')
            check_dir(dot_folder)

            # if not os.path.exists(img_folder):
            #     os.makedirs(img_folder)
            #
            # if not os.path.exists(dot_folder):
            #     os.makedirs(dot_folder)

            dot_file_name = dot_folder + file_name
            img_file_name = file
            self.tree_to_dot(self.root, feat_names, dot_file_name, alpha=alpha, show_pval=show_pval)
            if create_png:
                self.dot_to_png(dot_file_name, img_file_name, extension=extension)
        else:
            self.tree_to_dot(self.root, feat_names, file, alpha=alpha, show_pval=show_pval)
            if create_png:
                self.dot_to_png(file, extension=extension)

    def tree_to_dot_r(self, node, feat_names, f, counter, alpha=0.05, show_pval=True):
        curr_node = counter
        f.write(str(counter) + ' ')
        f.write('[')
        node_str = list(['label=\"'])

        # number of samples
        node_str.append('samples = ')
        node_str.append(str(node.samples))

        # add entropy/ATE here
        node_str.append('\\neffect = ')
        ace_str = '%.3f' % node.effect
        node_str.append(ace_str)

        # p_values
        if show_pval:
            node_str.append('\\np = ')
            p_val_str = '%.3f' % node.p_val
            node_str.append(p_val_str)

        if node.treat_split is not None:
            if curr_node == 0:
                node_str.append('\\nTreatment split: ')
                node_str.append('treatment > ')
            else:
                node_str.append('\\ntreatment > ')
            treat_str = '%s' % node.treat_split
            node_str.append(treat_str)

        if not node.leaf:
            sz_col = 'Column %s' % node.col
            if feat_names and sz_col in feat_names:
                sz_col = feat_names[sz_col]
            if isinstance(node.value, int):
                decision = '%s >= %s' % (sz_col, node.value)
                # opp_decision = '%s < %s' % (sz_col, tree.value)
            elif isinstance(node.value, float):
                decision = '%s >= %.3f' % (sz_col, node.value)
                # opp_decision = '%s < %.3f' % (sz_col, tree.value)
            else:
                decision = '%s == %s' % (sz_col, node.value)
                # opp_decision = '%s =/=' % (sz_col, tree.value)
            node.feature_split = decision

            # if curr_node == 0:
            #     node_str.append('Splitting feature: ')
            node_str.append('\\n' + decision + '\\n')

        node_str.append('\"')

        node_str.append(", style=filled")
        effect_range = np.linspace(self.min, self.max, 10)
        effect = node.effect
        color = '\"#ffffff\"'
        color_idx = 0
        for idx, effect_r in enumerate(effect_range[:-1]):
            if effect_range[idx] <= effect <= effect_range[idx + 1]:
                color = "\"/blues9/%i\"" % (idx + 1)
                color_idx = idx
                break

        color_str = ", fillcolor=" + color
        node_str.append(color_str)
        # node_str.append("style=filled, color=\"/blues3/2\"")

        if color_idx >= 7:
            font_color = ", fontcolor=white"
            node_str.append(font_color)

        if node.p_val <= alpha:
            # node_str.append(", shape=box")
            # node_str.append(", sides=4")
            # node_str.append(", peripheries=3")
            node_str.append(", color=red")
            node_str.append(", penwidth=3.0")

        node_str.append('] ;\n')
        f.write(''.join(node_str))

        # start doing the branches
        counter = counter + 1
        if node.true_branch is not None:
            if curr_node == 0:
                f.write(str(curr_node) + ' -> ' + str(counter) +
                        ' [labeldistance=2.5, labelangle=45, headlabel=\"True\"] ;\n')
            else:
                f.write(str(curr_node) + ' -> ' + str(counter) + ' ;\n')
            # f.write(str(curr_node) + ' -> ' + str(counter) +
            #         ' [labeldistance=2.5, labelangle=45, headlabel=' + decision + '];\n')
            counter = self.tree_to_dot_r(node.true_branch, feat_names, f, counter, alpha=alpha, show_pval=show_pval)
        if node.false_branch is not None:
            if curr_node == 0:
                f.write(str(curr_node) + ' -> ' + str(counter) +
                        ' [labeldistance=2.5, labelangle=-45, headlabel=\"False\"] ;\n')
            else:
                f.write(str(curr_node) + ' -> ' + str(counter) + ' ;\n')
            # f.write(str(curr_node) + ' -> ' + str(counter) +
            #         ' [labeldistance=2.5, labelangle=45, headlabel=' + opp_decision + '];\n')
            counter = self.tree_to_dot_r(node.false_branch, feat_names, f, counter, alpha=alpha, show_pval=show_pval)

        return counter

    def prune(self, alpha=.05):
        """Prunes the obtained tree according to the statistical significance gain. """

        def prune_r(node):

            if node.true_branch is None or node.false_branch is None:
                return

            # recursive call for each branch
            if not node.true_branch.leaf:
                prune_r(node.true_branch)
            if not node.false_branch.leaf:
                prune_r(node.false_branch)

            # merge leaves (potentially)
            if node.true_branch.leaf and node.false_branch.leaf:

                # Get branch labels
                tb = node.true_branch
                fb = node.false_branch

                tb_pval = tb.p_val
                fb_pval = fb.p_val

                if tb_pval > alpha and fb_pval > alpha:
                    node.leaf_num = node.true_branch.leaf_num
                    node.true_branch, node.false_branch = None, None
                    node.leaf = True
                    self.num_leaves = self.num_leaves - 1
                    self.obj = self.obj - (tb.current_obj + fb.current_obj - tb.variance - fb.variance) + \
                               node.current_obj - node.variance
                    self.mse = self.mse - (tb.node_mse + fb.node_mse) + node.node_mse
                    if tb.node_depth == self.tree_depth:
                        self.tree_depth = self.tree_depth - 1

        prune_r(self.root)

    def predict(self, test_data, return_features=False, variables=None):

        if return_features:
            if self.root.feature_name is None:
                if variables is not None:
                    self.feature_split_labels(variables)
                else:
                    print("You need variable names")
                    return_features = False

        def classify_r(node, observation, features=None):
            if node.leaf:
                if features is not None:
                    return node.leaf_num, node.treat_split, node.effect, features
                else:
                    return node.leaf_num, node.treat_split, node.effect
            else:
                v = observation[node.col]
                if isinstance(v, int) or isinstance(v, float):
                    if v >= node.value:
                        branch = node.true_branch
                        if isinstance(v, int):
                            decision_str = "%s >= %d" % (node.feature_name, v)
                        else:
                            decision_str = "%s >= %.3f" % (node.feature_name, v)
                    else:
                        branch = node.false_branch
                        if isinstance(v, int):
                            decision_str = "%s < %d" % (node.feature_name, v)
                        else:
                            decision_str = "%s < %.3f" % (node.feature_name, v)
                else:
                    if v == node.value:
                        branch = node.true_branch
                        decision_str = "%s == %s" % (node.feature_name, v)
                    else:
                        branch = node.false_branch
                        decision_str = "%s != %s" % (node.feature_name, v)

            if features is not None:
                features.append(decision_str)
            return classify_r(branch, observation, features=features)

        if len(test_data.shape) == 1:
            leaf_results = classify_r(self.root, test_data)
            return leaf_results

        num_test = test_data.shape[0]

        leaf_results = np.zeros(num_test)
        leaf_treat_split = np.zeros(num_test)
        predict = np.zeros(num_test)

        test_feature_lists = []
        for i in range(num_test):
            test_example = test_data[i, :]
            if return_features:
                features_list = []
                leaf_results[i], leaf_treat_split[i], predict[i], features_list = classify_r(self.root, test_example,
                                                                                             features=features_list)
                test_feature_lists.append(features_list)
            else:
                leaf_results[i], leaf_treat_split[i], predict[i] = classify_r(self.root, test_example)

        if return_features:
            return predict, leaf_results, leaf_treat_split, test_feature_lists
        else:
            return predict, leaf_results, leaf_treat_split

    def feature_split_labels(self, variable_names):

        variable_names = col_dict(variable_names)

        def feature_split_labels_r(node, feat_names):

            if not node.leaf:
                sz_col = 'Column %s' % node.col
                if feat_names and sz_col in feat_names:
                    sz_col = feat_names[sz_col]
                decision = '%s' % sz_col
                node.feature_name = decision

                sz_col = 'Column %s' % node.col
                if feat_names and sz_col in feat_names:
                    sz_col = feat_names[sz_col]
                if isinstance(node.value, int):
                    decision = '%s >= %s' % (sz_col, node.value)
                    # opp_decision = '%s < %s' % (sz_col, tree.value)
                elif isinstance(node.value, float):
                    decision = '%s >= %.3f' % (sz_col, node.value)
                    # opp_decision = '%s < %.3f' % (sz_col, tree.value)
                else:
                    decision = '%s == %s' % (sz_col, node.value)
                    # opp_decision = '%s =/=' % (sz_col, tree.value)
                node.feature_split = decision

            # start doing the branches
            if node.true_branch is not None:
                feature_split_labels_r(node.true_branch, feat_names)
            if node.false_branch is not None:
                feature_split_labels_r(node.false_branch, feat_names)

        feature_split_labels_r(self.root, variable_names)

    def get_variables_used(self, variable_names=None, cat=False):

        if self.root.feature_name is None:
            if variable_names is not None:
                self.feature_split_labels(variable_names)

        def get_variables_r(node, list_vars):

            if node.leaf:
                return list_vars
            else:
                if cat:
                    if '==' in node.feature_split:
                        # list_fs = node.feature_split.replace(" ", "").split("==")
                        list_fs = node.feature_split.split("==")
                        list_fs = [i.strip() for i in list_fs]
                        to_append = "_".join(list_fs)
                        # list_vars.append(to_append)
                        if to_append not in list_vars:
                            list_vars.append(to_append)
                    else:
                        # list_vars.append(node.feature_name)
                        if node.feature_name not in list_vars:
                            list_vars.append(node.feature_name)
                else:
                    # list_vars.append(node.feature_name)
                    if node.feature_name not in list_vars:
                        list_vars.append(node.feature_name)
                list_vars = get_variables_r(node.true_branch, list_vars)
                list_vars = get_variables_r(node.false_branch, list_vars)

                return list_vars

        list_of_vars = []
        list_of_vars = get_variables_r(self.root, list_of_vars)

        return list_of_vars
