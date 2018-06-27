import numpy as np
import os
import subprocess
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances as pdist
from scipy.stats import kruskal

total_mse = 0.0
leaf_label = -1
total_obj = 0.0


# Class that defines the causal tree nodes
class CausalTree:
    def __init__(self, col=-1, value=None, true_branch=None, false_branch=None, summary=None,
                 treat_split=None, leaf=False, max_effect=None, min_effect=None, leaf_num=None, eval_func=None):
        self.col = col  # the column of the feature used for splitting
        self.value = value  # the value in the column
        self.true_branch = true_branch  # pointer to node for true branch
        self.false_branch = false_branch  # pointer to node for false branch
        # self.results = results  # None for nodes, not None for leaves
        self.summary = summary  # summary of results (ACE, number of samples)
        self.treat_split = treat_split  # treatment splitting location (for continuous values)
        self.leaf = leaf  # False for non-leaf nodes, True for leaf nodes
        self.max = max_effect
        self.min = min_effect
        self.leaf_num = leaf_num
        self.eval_func = eval_func


# Wrapper for growing causal trees
def grow_causal_tree(rows, labels, treatment, cont=False,
                     max_depth=-1, min_size=2, weight=1., seed=None, split_size=.5):

    global total_mse
    global leaf_label
    global total_obj

    total_mse = 0.0
    leaf_label = -1
    total_obj = 0.0

    if cont:
        eval_func = objective_cont
    else:
        eval_func = objective

    if seed is not None:
        np.random.seed(seed)

    ct = grow_causal_tree_r(rows, labels, treatment, eval_func,
                            max_depth=max_depth, min_size=min_size, weight=weight,
                            split_size=split_size)

    total_mse = total_mse / labels.shape[0]

    return ct


def grow_causal_tree_r(rows, labels, treatment, eval_func,
                       min_size=1, max_depth=-1, curr_depth=0,
                       current_obj=0.0, effect=0.0, p_val=0.0,
                       curr_split=0.0, current_mse=0.0, weight=1., split_size=.5):
    if rows.shape[0] == 0:
        return CausalTree()

    global total_mse
    global leaf_label
    global total_obj

    # Calculate values for root node
    if curr_depth == 0:
        if eval_func == objective:
            current_mse, effect = tau_squared(labels, treatment)
            p_val = get_pval(labels, treatment)
            current_mse = current_mse
            total_mse = current_mse
        elif eval_func == objective_cont:
            current_mse, effect, curr_split = tau_squared_cont(labels, treatment, min_size)
            p_val = get_pval(labels, treatment, curr_split)
            current_mse = current_mse
            total_mse = current_mse

    train_rows, val_rows, train_outcome, \
        val_outcome, train_treat, val_treat = \
        train_test_split(rows, labels, treatment, shuffle=True, test_size=split_size)

    dcy = {}
    if eval_func == objective:
        dcy = {'ACE': effect, 'samples': '%d' % rows.shape[0], 'p_val': p_val}
    elif eval_func == objective_cont:
        dcy = {'ACE': effect, 'samples': '%d' % rows.shape[0],
               'treat_split': '%d' % np.floor(curr_split), 'p_val': p_val}

    if max_depth == curr_depth:
        leaf_label = leaf_label + 1
        return CausalTree(summary=dcy, leaf=True,
                          max_effect=(np.max(labels) - np.min(labels)),
                          min_effect=(np.min(labels) - np.max(labels)),
                          leaf_num=leaf_label, eval_func=eval_func)

    best_gain = 0.0
    best_attribute = None
    best_sets = {}

    best_tb_obj = 0.0
    best_fb_obj = 0.0

    best_tb_split = -np.inf
    best_fb_split = -np.inf

    # cv_num = 2

    column_count = rows.shape[1]
    for col in range(0, column_count):

        # unique values
        unique_vals = np.unique(rows[:, col])

        for value in unique_vals:
            ########################################################################################
            #  binary adaptive splitting ###########################################################
            ########################################################################################
            if eval_func == objective:
                (set1, set2, y1, y2, treat1, treat2) = divide_set(rows, labels, treatment, col, value)

                if np.unique(y1).shape[0] <= 1 or np.unique(y2).shape[0] <= 1:
                    continue

                tb_size_check = get_num_treat(treat1, min_size)[0]
                fb_size_check = get_num_treat(treat2, min_size)[0]

                if not tb_size_check or not fb_size_check:
                    continue

                # kfold = KFold(n_splits=cv_num, shuffle=True)
                # tb_eval = 0.0
                # for train_idx, val_idx in kfold.split(y1):
                #     train_y = y1[train_idx]
                #     val_y = y1[val_idx]
                #     train_treat = treat1[train_idx]
                #     val_treat = treat1[val_idx]
                #     tb_eval = tb_eval + eval_func(train_y, train_treat, val_y, val_treat, weight=weight)
                #
                # fb_eval = 0.0
                # for train_idx, val_idx in kfold.split(y2):
                #     train_y = y2[train_idx]
                #     val_y = y2[val_idx]
                #     train_treat = treat2[train_idx]
                #     val_treat = treat2[val_idx]
                #     fb_eval = fb_eval + eval_func(train_y, train_treat, val_y, val_treat, weight=weight)
                #
                # tb_eval = tb_eval / cv_num
                # fb_eval = fb_eval / cv_num

                (train_set1, train_set2, train_y1, train_y2, train_treat1, train_treat2) \
                    = divide_set(train_rows, train_outcome, train_treat, col, value)

                (val_set1, val_set2, val_y1, val_y2, val_treat1, val_treat2) \
                    = divide_set(val_rows, val_outcome, val_treat, col, value)

                if np.unique(val_y1).shape[0] <= 1 or np.unique(val_y2).shape[0] <= 1 or \
                        np.unique(train_y1).shape[0] <= 1 or np.unique(train_y2).shape[0] <= 1:
                    continue

                tb_size_check = get_num_treat(val_treat1, min_size=2)[0]
                fb_size_check = get_num_treat(val_treat2, min_size=2)[0]

                if not tb_size_check or not fb_size_check:
                    continue

                tb_size_check = get_num_treat(train_treat1, min_size)[0]
                fb_size_check = get_num_treat(train_treat2, min_size)[0]

                if not tb_size_check or not fb_size_check:
                    continue

                tb_eval = eval_func(train_y1, train_treat1, val_y1, val_treat1, weight=weight)
                fb_eval = eval_func(train_y2, train_treat2, val_y2, val_treat2, weight=weight)

                split_eval = (tb_eval + fb_eval)
                gain = total_obj - current_obj + split_eval

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

            ###########################################################################################
            # continuous adaptive splitting ###########################################################
            ###########################################################################################
            elif eval_func == objective_cont:
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

                tb_eval, tb_split, tb_effect = eval_func(train_y1, train_treat1, val_y1, val_treat1, min_size=min_size,
                                                         weight=weight)
                fb_eval, fb_split, fb_effect = eval_func(train_y2, train_treat2, val_y2, val_treat2, min_size=min_size,
                                                         weight=weight)

                split_eval = (tb_eval + fb_eval)
                gain = total_obj - current_obj + split_eval

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
                    best_tb_split = tb_split
                    best_fb_split = fb_split

    ################################################################################################
    #  binary adaptive recursive call ##############################################################
    ################################################################################################
    if eval_func == objective:
        if best_gain > 0:

            best_tb_mse, best_tb_effect = tau_squared(best_sets['y1'], best_sets['treat1'])
            best_fb_mse, best_fb_effect = tau_squared(best_sets['y2'], best_sets['treat2'])

            total_mse = total_mse - current_mse + (best_tb_mse + best_fb_mse)

            tb_p_val = get_pval(best_sets['y1'], best_sets['treat1'])
            fb_p_val = get_pval(best_sets['y2'], best_sets['treat2'])

            true_branch = grow_causal_tree_r(best_sets['set1'], best_sets['y1'], best_sets['treat1'], eval_func,
                                             min_size=min_size, current_mse=best_tb_mse, weight=weight,
                                             max_depth=max_depth, curr_depth=curr_depth + 1,
                                             effect=best_tb_effect, p_val=tb_p_val, current_obj=best_tb_obj,
                                             split_size=split_size)

            false_branch = grow_causal_tree_r(best_sets['set2'], best_sets['y2'], best_sets['treat2'], eval_func,
                                              min_size=min_size, current_mse=best_fb_mse, weight=weight,
                                              max_depth=max_depth, curr_depth=curr_depth + 1,
                                              effect=best_fb_effect, p_val=fb_p_val, current_obj=best_fb_obj,
                                              split_size=split_size)

            return CausalTree(col=best_attribute[0], value=best_attribute[1], true_branch=true_branch,
                              false_branch=false_branch, summary=dcy,
                              max_effect=(np.max(labels) - np.min(labels)),
                              min_effect=(np.min(labels) - np.max(labels)),
                              eval_func=eval_func
                              )
        else:
            leaf_label = leaf_label + 1
            return CausalTree(summary=dcy, leaf=True,
                              max_effect=(np.max(labels) - np.min(labels)),
                              min_effect=(np.min(labels) - np.max(labels)),
                              leaf_num=leaf_label, eval_func=eval_func)

    ################################################################################################
    # continuous adaptive recursive call ###########################################################
    ################################################################################################
    elif eval_func == objective_cont:
        if best_gain > 0:

            best_tb_mse, best_tb_effect = tau_squared(best_sets['y1'], best_sets['treat1'], best_tb_split)
            best_fb_mse, best_fb_effect = tau_squared(best_sets['y2'], best_sets['treat2'], best_fb_split)

            total_mse = total_mse - current_mse + (best_tb_mse + best_fb_mse)

            tb_labels = np.copy(best_sets['y1'])
            tb_treatment = np.copy(best_sets['treat1'])
            tb_p_val = get_pval(tb_labels, tb_treatment, best_tb_split)

            fb_labels = np.copy(best_sets['y2'])
            fb_treatment = np.copy(best_sets['treat2'])
            fb_p_val = get_pval(fb_labels, fb_treatment, best_fb_split)

            true_branch = grow_causal_tree_r(best_sets['set1'], best_sets['y1'], best_sets['treat1'], eval_func,
                                             min_size=min_size, current_mse=best_tb_mse, weight=weight,
                                             max_depth=max_depth, curr_depth=curr_depth + 1,
                                             effect=best_tb_effect, p_val=tb_p_val, current_obj=best_tb_obj,
                                             curr_split=best_tb_split,
                                             split_size=split_size)

            false_branch = grow_causal_tree_r(best_sets['set2'], best_sets['y2'], best_sets['treat2'], eval_func,
                                              min_size=min_size, current_mse=best_fb_mse, weight=weight,
                                              max_depth=max_depth, curr_depth=curr_depth + 1,
                                              effect=best_fb_effect, p_val=fb_p_val, current_obj=best_fb_obj,
                                              curr_split=best_fb_split,
                                              split_size=split_size)

            return CausalTree(col=best_attribute[0], value=best_attribute[1], true_branch=true_branch,
                              false_branch=false_branch, summary=dcy, treat_split=curr_split,
                              max_effect=(np.max(labels) - np.min(labels)),
                              min_effect=(np.min(labels) - np.max(labels)),
                              eval_func=eval_func)
        else:
            leaf_label = leaf_label + 1
            return CausalTree(summary=dcy, leaf=True,
                              max_effect=(np.max(labels) - np.min(labels)),
                              min_effect=(np.min(labels) - np.max(labels)),
                              leaf_num=leaf_label, eval_func=eval_func, treat_split=curr_split)


# gets the adaptive estimate
def objective(train_outcome, train_treatment, val_outcome, val_treatment, weight=1.):
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
    train_mse = (1-weight) * total_train * (train_effect ** 2)
    cost = weight * total_val * np.abs(train_effect - val_effect)

    obj = (train_mse - cost) / (np.abs(total_train - total_val) + 1)

    return obj


def objective_cont(train_outcome, train_treatment, val_outcome, val_treatment, min_size=1, weight=1.):
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
    min_size_idx = np.where(np.logical_and(treat_num >= min_size, cont_num >= min_size))

    unique_treatment = unique_treatment[min_size_idx]
    ttt = ttt[min_size_idx]
    yyt = yyt[min_size_idx]
    ttv = ttv[min_size_idx]
    yyv = yyv[min_size_idx]

    if ttv.shape[0] == 0:
        return return_val

    y_t_m_t = np.sum((yyt * (ttt == 1)), axis=1) / np.sum(ttt == 1, axis=1)
    y_c_m_t = np.sum((yyt * (ttt == 0)), axis=1) / np.sum(ttt == 0, axis=1)

    y_t_m_v = np.sum((yyv * (ttv == 1)), axis=1) / np.sum(ttv == 1, axis=1)
    y_c_m_v = np.sum((yyv * (ttv == 0)), axis=1) / np.sum(ttv == 0, axis=1)

    train_effect = y_t_m_t - y_c_m_t
    train_err = train_effect ** 2

    val_effect = y_t_m_v - y_c_m_v
    val_err = val_effect ** 2

    max_err = np.argmax(train_err)
    train_best_err = train_err[max_err]
    val_best_err = val_err[max_err]

    train_best_effect = train_effect[max_err]
    val_best_effect = val_effect[max_err]

    best_split = unique_treatment[max_err]

    best_effect = train_effect[max_err]

    train_mse = (1-weight) * (total_train * train_best_err)
    cost = weight * total_val * np.abs(train_best_effect - val_best_effect)
    obj = (train_mse - cost) / (np.abs(total_train - total_val) + 1)

    return obj, best_split, best_effect


def tau_squared(outcome, treatment, treat_split=None):

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
    err = (effect ** 2) * total

    return err, effect


def tau_squared_cont(outcome, treatment, min_size=1):
    """Continuous case"""
    total = outcome.shape[0]

    return_val = (-np.inf, -np.inf, -np.inf)

    if total == 0:
        return return_val

    unique_treatment = np.unique(treatment)

    if unique_treatment.shape[0] == 1:
        return return_val

    unique_treatment = (unique_treatment[1:] + unique_treatment[:-1]) / 2

    first_quartile = int(np.floor(unique_treatment.shape[0] / 4))
    third_quartile = int(np.ceil(3 * unique_treatment.shape[0] / 4))

    unique_treatment = unique_treatment[first_quartile:third_quartile]

    if isinstance(outcome[0], str):
        outcome = string_to_int(outcome)

    yy = np.tile(outcome, (unique_treatment.shape[0], 1))
    tt = np.tile(treatment, (unique_treatment.shape[0], 1))

    x = np.transpose(np.transpose(tt) > unique_treatment)

    tt[x] = 1
    tt[np.logical_not(x)] = 0

    treat_num = np.sum(tt == 1, axis=1)
    cont_num = np.sum(tt == 0, axis=1)
    min_size_idx = np.where(np.logical_and(treat_num >= min_size, cont_num >= min_size))

    tt = tt[min_size_idx]
    yy = yy[min_size_idx]

    if tt.shape[0] == 0:
        return return_val

    y_t_m = np.sum((yy * (tt == 1)), axis=1) / np.sum(tt == 1, axis=1)
    y_c_m = np.sum((yy * (tt == 0)), axis=1) / np.sum(tt == 0, axis=1)

    effect = y_t_m - y_c_m
    err = effect ** 2

    max_err = np.argmax(err)

    best_effect = effect[max_err]
    best_err = err[max_err]
    best_split = unique_treatment[max_err]

    best_err = total * best_err

    return best_err, best_effect, best_split


# divides the rows based on an outcome value
def divide_set(rows, y, treatment, column, value):
    if isinstance(value, int) or isinstance(value, float):  # for int and float values
        idx1 = rows[:, column] >= value
        idx2 = ~idx1
    else:  # for strings
        idx1 = rows[:, column] == value
        idx2 = ~idx1
    # split features
    list1 = rows[idx1]
    list2 = rows[idx2]
    # split outcome
    y1 = y[idx1]
    y2 = y[idx2]
    # split treatment
    treat1 = treatment[idx1]
    treat2 = treatment[idx2]
    return list1, list2, y1, y2, treat1, treat2


# gets unique values of the outcome
def unique_counts(vect):
    vals, counts = np.unique(vect, return_counts=True)
    results = {}
    for i, val in enumerate(vals):
        results[val] = counts[i]
    return results


def ace(outcome, treatment):
    """
    Computes the average treatment effect
    :param outcome: the outcome vector
    :param treatment: the treatment vector (in binary format)
    :return: the average treatment effect
    """
    treat = treatment == 1
    control = treatment == 0

    yt = outcome[treat]
    yc = outcome[control]

    mu1 = 0.0
    mu0 = 0.0
    if yt.shape[0] != 0:
        mu1 = np.mean(yt)
    if yc.shape[0] != 0:
        mu0 = np.mean(yc)

    return mu1 - mu0


def get_pval(outcome, treatment, treat_split=None):
    """
    Computes the t-test for treatments
    :param outcome: outcome vector
    :param treatment: treatment vector
    :param treat_split: splitting value for treatment vector
    :return: p_value from t-test of treatment vs control
    """

    treat_vect = np.copy(treatment)

    if treat_split is not None:
        treat = treat_vect > treat_split
        control = treat_vect <= treat_split
        treat_vect[treat] = 1
        treat_vect[control] = 0

    outcome_cont = outcome[treat_vect == 0]
    outcome_trt = outcome[treat_vect == 1]

    p_val = ttest_ind(outcome_cont, outcome_trt, equal_var=False)[1]

    return p_val


def get_num_treat(treatment, min_size=1, treat_split=None):

    if treat_split is not None:
        treat_vect = np.copy(treatment)
        treat = treat_vect > treat_split
        control = treat_vect <= treat_split
        treat_vect[treat] = 1
        treat_vect[control] = 0
    else:
        treat_vect = treatment

    num_treatment = np.sum(treat_vect == 1)
    num_control = np.sum(treat_vect == 0)

    if num_treatment >= min_size and num_control >= min_size:
        min_size_check = True
    else:
        min_size_check = False

    return min_size_check, num_control, num_treatment


def string_to_int(classes):
    unique_classes = np.unique(classes)
    class_list = {}
    j = 0
    for i in unique_classes:
        class_list[i] = j
        j += 1
    return np.array([class_list[i] for i in classes])


def tree_to_dot_r(tree, feat_names, f, counter, show_classes=False, max_effect=1.0, min_effect=-1.0,
                  alpha=0.05, show_pval=True):
    curr_node = counter
    f.write(str(counter) + ' ')
    f.write('[')
    node_str = ['label=\"']

    # num samples
    node_str.append('samples = ')
    node_str.append(tree.summary['samples'])

    # num of each sample
    if show_classes:
        node_str.append('\\nvalue = ')
        results_list = list()
        for i in tree.results:
            results_list.append(tree.results[i])
        results_str = str(results_list)
        node_str.append(results_str)

    # add entropy/ATE here
    node_str.append('\\nACE = ')
    ace_str = '%.6f' % tree.summary['ACE']
    node_str.append(ace_str)

    # p_values
    if show_pval:
        node_str.append('\\np = ')
        p_val_str = '%.3f' % tree.summary['p_val']
        node_str.append(p_val_str)

    if 'treat_split' in tree.summary:
        # if curr_node == 0:
        #     node_str.append('Treatment split: ')
        node_str.append('\\ntreatment > ')
        treat_str = '%s' % tree.summary['treat_split']
        node_str.append(treat_str)

    if not tree.leaf:
        sz_col = 'Column %s' % tree.col
        if feat_names and sz_col in feat_names:
            sz_col = feat_names[sz_col]
        if isinstance(tree.value, int):
            decision = '%s >= %s' % (sz_col, tree.value)
        elif isinstance(tree.value, float):
            decision = '%s >= %.3f' % (sz_col, tree.value)
        else:
            decision = '%s == %s' % (sz_col, tree.value)

        # if curr_node == 0:
        #     node_str.append('Splitting feature: ')
        node_str.append('\\n' + decision + '\\n')

    node_str.append('\"')

    node_str.append(", style=filled")
    effect_range = np.linspace(min_effect, max_effect, 10)
    effect = tree.summary['ACE']
    color = '#ffffff'
    color_idx = 0
    for idx, effect_r in enumerate(effect_range[:-1]):
        if effect_range[idx] <= effect <= effect_range[idx+1]:
            color = "\"/blues9/%i\"" % (idx + 1)
            color_idx = idx
            break

    color_str = ", fillcolor=" + color
    node_str.append(color_str)
    # node_str.append("style=filled, color=\"/blues3/2\"")

    if color_idx >= 7:
        font_color = ", fontcolor=white"
        node_str.append(font_color)

    if tree.summary['p_val'] <= alpha:
        # node_str.append(", shape=box")
        # node_str.append(", sides=4")
        # node_str.append(", peripheries=3")
        node_str.append(", color=red")
        node_str.append(", penwidth=3.0")

    node_str.append('] ;\n')
    f.write(''.join(node_str))

    # start doing the branches
    counter = counter + 1
    if tree.true_branch is not None:
        if curr_node == 0:
            f.write(str(curr_node) + ' -> ' + str(counter) +
                    ' [labeldistance=2.5, labelangle=45, headlabel=\"True\"] ;\n')
        else:
            f.write(str(curr_node) + ' -> ' + str(counter) + ' ;\n')
        counter = tree_to_dot_r(tree.true_branch, feat_names, f, counter,
                                max_effect=max_effect, min_effect=min_effect, alpha=alpha, show_pval=show_pval)
    if tree.false_branch is not None:
        if curr_node == 0:
            f.write(str(curr_node) + ' -> ' + str(counter) +
                    ' [labeldistance=2.5, labelangle=-45, headlabel=\"False\"] ;\n')
        else:
            f.write(str(curr_node) + ' -> ' + str(counter) + ' ;\n')
        counter = tree_to_dot_r(tree.false_branch, feat_names, f, counter,
                                max_effect=max_effect, min_effect=min_effect, alpha=alpha, show_pval=show_pval)

    return counter


def tree_to_dot(tree, feat_names, filename='tree', alpha=0.05, show_pval=True, show_legend=False):
    filename = filename + '.dot'
    feat_names = col_dict(feat_names)

    max_effect = tree.max
    min_effect = tree.min
    with open(filename, 'w') as f:
        f.write('digraph Tree {\n')
        f.write('node [shape=box, fontsize=32] ;\n')
        f.write('edge [fontsize=24] ;\n')
        if show_legend:
            if 'treat_split' in tree.summary:
                f.write('-1[label=\"1: Sample size (samples)\n'
                        '2: Average causal effect (ACE)\n'
                        '3: t-test p-value (p)\n'
                        '4: ACE treatment threshold\n'
                        '5: Splitting feature\"] ;\n')
            else:
                f.write('-1[label=\"1: Sample size (samples)\n'
                        '2: Average causal effect (ACE)\n'
                        '3: t-test p-value (p)\n'
                        '4: Splitting feature\"] ;\n')
        tree_to_dot_r(tree, feat_names, f, counter=0, max_effect=max_effect, min_effect=min_effect,
                      alpha=alpha, show_pval=show_pval)
        f.write("}")


def dot_to_png(dot_filename="tree", output_file=None, extension="pdf"):

    if output_file is None:
        command = ["dot", "-T" + extension, "-Gdpi=500", dot_filename + '.dot', "-o", dot_filename + "." + extension]
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


def plot_tree(tree, feat_names, file="tree", alpha=0.05, show_pval=True, show_legend=False):

    name_split = file.split('/')
    if len(name_split) > 1:
        img_folder = name_split[0:-1]
        file_name = name_split[-1]

        img_folder = '/'.join(img_folder)

        dot_folder = img_folder + '/dot_folder/'

        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        if not os.path.exists(dot_folder):
            os.makedirs(dot_folder)

        dot_file_name = dot_folder + file_name
        img_file_name = file
        tree_to_dot(tree, feat_names, dot_file_name, alpha=alpha, show_pval=show_pval, show_legend=show_legend)
        dot_to_png(dot_file_name, img_file_name)
    else:
        tree_to_dot(tree, feat_names, file, alpha=alpha, show_pval=show_pval, show_legend=show_legend)
        dot_to_png(file)


def col_dict(names):
    feat_names = {}
    for i, name in enumerate(names):
        column = "Column %s" % i
        feat_names[column] = name
    return feat_names


# prune method needs fixing
def prune(tree, alpha=.05, notify=False):
    """Prunes the obtained tree according to the statistical significance gain. """
    global leaf_label

    if tree.true_branch is None or tree.false_branch is None:
        return

    # recursive call for each branch
    if not tree.true_branch.leaf:
        prune(tree.true_branch, alpha, notify)
    if not tree.false_branch.leaf:
        prune(tree.false_branch, alpha, notify)

    # merge leaves (potentially)
    if tree.true_branch.leaf and tree.false_branch.leaf:

        # Get branch labels
        tb = tree.true_branch
        fb = tree.false_branch

        tb_pval = tb.summary['p_val']
        fb_pval = fb.summary['p_val']

        if tb_pval > alpha and fb_pval > alpha:
            tree.leaf_num = tree.true_branch.leaf_num
            tree.true_branch, tree.false_branch = None, None
            tree.leaf = True
            leaf_label = leaf_label - 1


def classify(test_data, tree, alpha=1.):

    if len(test_data.shape) == 1:
        leaf_results = classify_r(test_data, tree, alpha=alpha)
        return leaf_results

    num_test = test_data.shape[0]

    leaf_results = np.zeros(num_test)
    leaf_treat_split = np.zeros(num_test)
    predict = np.zeros(num_test)

    for i in range(num_test):
        test_example = test_data[i, :]
        leaf_results[i], leaf_treat_split[i], predict[i] = classify_r(test_example, tree, alpha=alpha)

    return leaf_results, leaf_treat_split, predict


def classify_r(observation, tree, alpha=1.):
    if tree.leaf:
        if tree.summary['p_val'] <= alpha:
            return tree.leaf_num, tree.treat_split, tree.summary['ACE']
        else:
            return np.nan, np.nan, np.nan
    else:
        v = observation[tree.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.true_branch
            else:
                branch = tree.false_branch
        else:
            if v == tree.value:
                branch = tree.true_branch
            else:
                branch = tree.false_branch
    return classify_r(observation, branch, alpha=alpha)


def compare_mse(test_data, outcome, treatment, tree):
    global total_mse

    test_mse = get_test_mse(test_data, outcome, treatment, tree)

    mse_error = np.abs(total_mse - test_mse)
    mse_error = mse_error/max(total_mse, test_mse)

    return mse_error


def get_new_total(data, outcome, treatment, tree):

    global total_mse

    total_mse = get_test_mse(data, outcome, treatment, tree)


def get_test_ace_sym(test_data, outcome, treatment, tree, alpha=1.):

    leaf_results, leaf_treat_split, predict = classify(test_data, tree, alpha=alpha)

    unique_leaf = np.unique(leaf_results)

    test_acee = 0

    if np.isnan(leaf_treat_split[0]):
        cont = False
    else:
        cont = True

    counter = 0
    for i in unique_leaf:
        if np.isnan(i):
            continue
        y_i = outcome[leaf_results == i]
        p = predict[leaf_results == i]
        if cont:
            treat_split = leaf_treat_split[leaf_results == i][0]
            treat_vect = np.array(treatment[leaf_results == i])
            trt = treat_split > i
            ctrl = treat_split <= i
            treat_vect[trt] = 1
            treat_vect[ctrl] = 0
            yi_mse, yi_effect = tau_squared(y_i, treat_vect)
        else:
            treat_vect = treatment[leaf_results == i]
            yi_mse, yi_effect = tau_squared(y_i, treat_vect)

        numerator = np.abs(yi_effect - p[0])
        denominator = (np.abs(yi_effect) + np.abs(p[0]))/2

        test_acee = test_acee + numerator/denominator
        counter = counter + 1

    if counter < 1:
        counter = 1
    test_acee = test_acee / counter

    return test_acee


def get_test_ace(test_data, outcome, treatment, tree, alpha=1.):

    leaf_results, leaf_treat_split, predict = classify(test_data, tree, alpha=alpha)

    unique_leaf = np.unique(leaf_results)

    test_acee = 0

    if np.isnan(leaf_treat_split[0]):
        cont = False
    else:
        cont = True

    counter = 0
    for i in unique_leaf:
        if np.isnan(i):
            continue
        y_i = outcome[leaf_results == i]
        p = predict[leaf_results == i]
        if cont:
            treat_split = leaf_treat_split[leaf_results == i][0]
            treat_vect = np.array(treatment[leaf_results == i])
            trt = treat_split > i
            ctrl = treat_split <= i
            treat_vect[trt] = 1
            treat_vect[ctrl] = 0
            yi_mse, yi_effect = tau_squared(y_i, treat_vect)
        else:
            treat_vect = treatment[leaf_results == i]
            yi_mse, yi_effect = tau_squared(y_i, treat_vect)

        numerator = np.abs(yi_effect - p[0])
        denominator = (np.abs(yi_effect) + np.abs(p[0]))

        test_acee = test_acee + numerator/denominator
        counter = counter + 1

    if counter < 1:
        counter = 1
    test_acee = test_acee / counter

    return test_acee


def get_test_mse(test_data, outcome, treatment, tree):

    leaf_results, leaf_treat_split, predict = classify(test_data, tree)

    unique_leaf = np.unique(leaf_results)

    mse = 0.0

    if np.isnan(leaf_treat_split[0]):
        cont = False
    else:
        cont = True

    for i in unique_leaf:
        if np.isnan(i):
            continue
        y_i = outcome[leaf_results == i]
        if cont:
            treat_split = leaf_treat_split[leaf_results == i][0]
            treat_vect = np.array(treatment[leaf_results == i])
            trt = treat_split > i
            ctrl = treat_split <= i
            treat_vect[trt] = 1
            treat_vect[ctrl] = 0
            yi_mse = tau_squared(y_i, treat_vect)[0]
        else:
            treat_vect = treatment[leaf_results == i]
            yi_mse = tau_squared(y_i, treat_vect)[0]

        mse = mse + yi_mse

    mse = (1/test_data.shape[0]) * mse

    return mse


def get_test_effect(test_data, tree):

    leaf_results, leaf_treat_split, predict = classify(test_data, tree)

    return predict


def get_test_leaf(test_data, tree, alpha=1.):

    leaf_results, leaf_treat_split, predict = classify(test_data, tree, alpha=alpha)

    return leaf_results


def get_total_mse():
    global total_mse
    return total_mse


def percent_error(y_actual, y_predict, leaf_results=None):
    if leaf_results is not None:
        y_actual = y_actual[~np.isnan(leaf_results)]
        y_predict = y_predict[~np.isnan(leaf_results)]

    if y_actual.shape[0] < 1:
        return 1.0

    numerator = np.abs(y_actual - y_predict)
    denom = (np.abs(y_actual) + np.abs(y_predict))/2

    smape = np.sum(numerator / denom)

    smape = smape/y_actual.shape[0]

    return smape


# changed to SMAPE
def nrmse(y_actual, y_predict, leaf_results=None):
    if leaf_results is not None:
        y_actual = y_actual[~np.isnan(leaf_results)]
        y_predict = y_predict[~np.isnan(leaf_results)]

    if y_actual.shape[0] < 1:
        return 1.0

    numerator = np.abs(y_actual - y_predict)
    denom = np.abs(y_actual) + np.abs(y_predict)

    smape = np.sum(numerator / denom)

    smape = smape / y_actual.shape[0]

    # numerator = np.sqrt(np.sum((y_actual - y_predict) ** 2))
    # denom = np.sqrt(y_actual.shape[0])
    # rmse = numerator / denom
    #
    # max_actual = np.max(y_actual)
    # min_actual = np.min(y_actual)
    # max_predict = np.max(y_predict)
    # min_predict = np.min(y_predict)
    #
    # denom_n = max_actual - min_actual
    #
    # nrmse = rmse / denom_n

    return smape


def matching(test, outcome, treatment, treat_split=None):

    treat_vect = treatment

    if treat_split is not None:
        treat_vect = np.copy(treatment)
        treat = treat_vect > treat_split
        cont = treat_vect <= treat_split
        treat_vect[treat] = 1
        treat_vect[cont] = 0

    treat = treat_vect == 1
    cont = treat_vect == 0

    treat_feats = test[treat, :]
    cont_feats = test[cont, :]

    p = pdist(treat_feats, cont_feats)
    idx = np.argmin(p, axis=1)

    treat_outcome = outcome[treat]
    cont_outcome = outcome[cont]

    ice = treat_outcome - cont_outcome[idx]

    return ice


def matching_cate(test, treatment, tree, treat_split=None):
    treat_vect = treatment

    if treat_split is not None:
        treat_vect = np.copy(treatment)
        treat = treat_vect > treat_split
        cont = treat_vect <= treat_split
        treat_vect[treat] = 1
        treat_vect[cont] = 0

    treat = treat_vect == 1
    cont = treat_vect == 0

    treat_feats = test[treat, :]
    cont_feats = test[cont, :]

    p = pdist(treat_feats, cont_feats)
    idx = np.argmin(p, axis=1)

    treat_predict = get_test_effect(treat_feats, tree)
    cont_predict = get_test_effect(cont_feats, tree)

    cate = (1/2) * (treat_predict + cont_predict[idx])

    return cate


def get_num_leaves():
    global leaf_label
    return leaf_label + 1


def nrmse_ttest(y_actual, y_predict1, y_predict2, y_predict3, leaf_results1=None, leaf_results2=None, leaf_results3=None):

    vect1 = get_nrmse(y_actual, y_predict1, leaf_results1)
    vect2 = get_nrmse(y_actual, y_predict2, leaf_results2)
    vect3 = get_nrmse(y_actual, y_predict3, leaf_results3)

    test = kruskal(vect1, vect2, vect3)[1]

    return test


def get_nrmse(y_actual, y_predict, leaf_results=None):
    if leaf_results is not None:
        y_actual = y_actual[~np.isnan(leaf_results)]
        y_predict = y_predict[~np.isnan(leaf_results)]

    if y_actual.shape[0] < 1:
        return np.array([1.0])

    numerator = np.abs(y_actual - y_predict)
    denom = np.abs(y_actual) + np.abs(y_predict)

    vect = numerator / denom

    return vect


def nrmse_std(y_actual, y_predict, leaf_results=None):
    if leaf_results is not None:
        y_actual = y_actual[~np.isnan(leaf_results)]
        y_predict = y_predict[~np.isnan(leaf_results)]

    if y_actual.shape[0] < 1:
        return np.array([1.0])

    numerator = np.abs(y_actual - y_predict)
    denom = np.abs(y_actual) + np.abs(y_predict)

    vect = numerator / denom

    return np.std(vect)


def percent_error_ttest(y_actual, y_predict1, y_predict2, y_predict3, leaf_results1=None, leaf_results2=None, leaf_results3=None):
    vect1 = get_nrmse(y_actual, y_predict1, leaf_results1)
    vect2 = get_nrmse(y_actual, y_predict2, leaf_results2)
    vect3 = get_nrmse(y_actual, y_predict3, leaf_results3)

    test = kruskal(vect1, vect2, vect3)[1]

    return test


def get_percent_err(y_actual, y_predict, leaf_results=None):
    if leaf_results is not None:
        y_actual = y_actual[~np.isnan(leaf_results)]
        y_predict = y_predict[~np.isnan(leaf_results)]

    if y_actual.shape[0] < 1:
        return np.array([1.0])

    numerator = np.abs(y_actual - y_predict)
    denom = (np.abs(y_actual) + np.abs(y_predict))/2

    vect = numerator / denom

    return vect


def get_test_ace_std(test_data, outcome, treatment, tree, alpha=1.):

    leaf_results, leaf_treat_split, predict = classify(test_data, tree, alpha=alpha)

    unique_leaf = np.unique(leaf_results)

    test_acee = 0

    if np.isnan(leaf_treat_split[0]):
        cont = False
    else:
        cont = True

    test_acee_vect = list()

    counter = 0
    for i in unique_leaf:
        if np.isnan(i):
            continue
        y_i = outcome[leaf_results == i]
        p = predict[leaf_results == i]
        if cont:
            treat_split = leaf_treat_split[leaf_results == i][0]
            treat_vect = np.array(treatment[leaf_results == i])
            trt = treat_split > i
            ctrl = treat_split <= i
            treat_vect[trt] = 1
            treat_vect[ctrl] = 0
            yi_mse, yi_effect = tau_squared(y_i, treat_vect)
        else:
            treat_vect = treatment[leaf_results == i]
            yi_mse, yi_effect = tau_squared(y_i, treat_vect)

        numerator = np.abs(yi_effect - p[0])
        denominator = (np.abs(yi_effect) + np.abs(p[0]))

        test_acee = test_acee + numerator / denominator
        test_acee_vect.append(numerator/denominator)
        counter = counter + 1

    if counter < 1:
        counter = 1
    test_acee = test_acee / counter

    test_acee_vect = np.array(test_acee_vect)

    test_acee_std = np.std(test_acee_vect)

    return test_acee_std