import os
import errno
import numpy as np
from scipy.stats import ttest_ind
import subprocess
# from CTL.CTL import *
import time


def check_dir(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def unique_counts(vect):
    vals, counts = np.unique(vect, return_counts=True)
    results = {}
    for i, val in enumerate(vals):
        results[val] = counts[i]
    return results


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


def string_to_int(classes):
    unique_classes = np.unique(classes)
    class_list = {}
    j = 0
    for i in unique_classes:
        class_list[i] = j
        j += 1
    return np.array([class_list[i] for i in classes])


def col_dict(names):
    feat_names = {}
    for i, name in enumerate(names):
        column = "Column %s" % i
        feat_names[column] = name
    return feat_names


def tau_squared_cont(outcome, treatment, min_size=1, quartile=False):
    """Continuous case"""
    total = outcome.shape[0]

    return_val = (-np.inf, -np.inf, -np.inf)

    if total == 0:
        return return_val

    unique_treatment = np.unique(treatment)

    if unique_treatment.shape[0] == 1:
        return return_val

    unique_treatment = (unique_treatment[1:] + unique_treatment[:-1]) / 2

    yy = np.tile(outcome, (unique_treatment.shape[0], 1))
    tt = np.tile(treatment, (unique_treatment.shape[0], 1))

    x = np.transpose(np.transpose(tt) > unique_treatment)

    tt[x] = 1
    tt[np.logical_not(x)] = 0

    treat_num = np.sum(tt == 1, axis=1)
    cont_num = np.sum(tt == 0, axis=1)
    min_size_idx = np.where(np.logical_and(
        treat_num >= min_size, cont_num >= min_size))

    unique_treatment = unique_treatment[min_size_idx]
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


def tau_squared(outcome, treatment, treat_split=None):
    total = outcome.shape[0]

    return_val = (-np.inf, -np.inf)

    if total == 0:
        return return_val

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

    p_val = ttest_ind(outcome_cont, outcome_trt)[1]

    if np.isnan(p_val):
        return 0.000

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


def smape(y_actual, y_predict, leaf_results=None):
    if leaf_results is not None:
        y_actual = y_actual[~np.isnan(leaf_results)]
        y_predict = y_predict[~np.isnan(leaf_results)]

    if y_actual.shape[0] < 1:
        return np.array([1.0])

    numerator = np.abs(y_actual - y_predict)
    denom = np.abs(y_actual) + np.abs(y_predict)

    vect = numerator / denom

    return vect


def size_check_fail(rows, labels, treatment, col, value, min_size):
    (set1, set2, y1, y2, treat1, treat2) = divide_set(
        rows, labels, treatment, col, value)

    if np.unique(y1).shape[0] <= 1 or np.unique(y2).shape[0] <= 1:
        return True

    tb_size_check = get_num_treat(treat1, min_size)[0]
    fb_size_check = get_num_treat(treat2, min_size)[0]

    if not tb_size_check or not fb_size_check:
        return True

    return False


def variance(y, treatment, treat_split=None):
    treat_vect = np.copy(treatment)

    if treat_split is not None:
        trt = treat_vect > treat_split
        cont = treat_vect <= treat_split
        treat_vect[trt] = 1
        treat_vect[cont] = 0

    treat = treat_vect == 1
    control = treat_vect == 0

    if y.shape[0] == 0:
        return np.array([np.inf, np.inf])

    yt = y[treat]
    yc = y[control]

    if yt.shape[0] == 0:
        var_t = np.inf
    else:
        var_t = np.var(yt)

    if yc.shape[0] == 0:
        var_c = np.inf
    else:
        var_c = np.var(yc)

    return var_t, var_c


def dot_png(folder, extension='png', dpi=200):
    items = os.listdir(folder + 'dot_folder/')
    items_dir = [folder + 'dot_folder/' + i for i in items]

    for i, item in enumerate(items_dir):
        command = ["dot", "-T" + extension, "-Gdpi=" +
                   str(dpi), item, "-o", folder + items[i][:-3] + extension]
        # print(command)
        try:
            if os.name == 'nt':
                subprocess.check_call(command, shell=True)
            else:
                subprocess.check_call(command)
        except subprocess.CalledProcessError:
            exit("Could not run dot, ie graphviz, to "
                 "produce visualization")


def get_treat_size(t, treat_split=0.5):

    num_treatment = t[t > treat_split].shape[0]
    num_control = t[t <= treat_split].shape[0]

    return num_treatment, num_control


def check_min_size(min_size, t, treat_split=0.5):
    nt, nc = get_treat_size(t, treat_split)

    return nt < min_size or nc < min_size


def get_test_mse(test_data, outcome, treatment, tree):
    # leaf_results, leaf_treat_split, predict = predict(test_data, tree)
    #
    # unique_leaf = np.unique(leaf_results)
    #
    # mse = 0.0
    #
    # if np.isnan(leaf_treat_split[0]):
    #     cont = False
    # else:
    #     cont = True
    #
    # for i in unique_leaf:
    #     if np.isnan(i):
    #         continue
    #     y_i = outcome[leaf_results == i]
    #     if cont:
    #         treat_split = leaf_treat_split[leaf_results == i][0]
    #         treat_vect = np.array(treatment[leaf_results == i])
    #         trt = treat_split > i
    #         ctrl = treat_split <= i
    #         treat_vect[trt] = 1
    #         treat_vect[ctrl] = 0
    #         yi_mse = tau_squared(y_i, treat_vect)[0]
    #     else:
    #         treat_vect = treatment[leaf_results == i]
    #         yi_mse = tau_squared(y_i, treat_vect)[0]
    #
    #     mse = mse + yi_mse
    #
    # mse = (1/test_data.shape[0]) * mse
    #
    # return mse

    pass


def get_test_effect(test_data, tree):
    pass

    # leaf_results, leaf_treat_split, predict = predict(test_data, tree)
    #
    # return predict
