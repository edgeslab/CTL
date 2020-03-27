import os
import errno
import numpy as np
from scipy.stats import ttest_ind
import subprocess
import time


def check_dir(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def divide_set(x, y, t, col, value):
    idx1 = x[:, col] >= value
    idx2 = ~idx1

    x1 = x[idx1]
    x2 = x[idx2]

    y1 = y[idx1]
    y2 = y[idx2]

    t1 = t[idx1]
    t2 = t[idx2]

    return x1, x2, y1, y2, t1, t2


def tau_squared(y, t):
    total = y.shape[0]

    return_val = (-np.inf, -np.inf)

    if total == 0:
        return return_val

    treat_vect = t

    effect = ace(y, treat_vect)
    err = (effect ** 2) * total

    return err, effect


def ace(y, t):
    treat = t == 1
    control = t == 0

    yt = y[treat]
    yc = y[control]

    mu1 = 0.0
    mu0 = 0.0
    if yt.shape[0] != 0:
        mu1 = np.mean(yt)
    if yc.shape[0] != 0:
        mu0 = np.mean(yc)

    return mu1 - mu0


def get_pval(y, t):
    treat_vect = t

    outcome_cont = y[treat_vect == 0]
    outcome_trt = y[treat_vect == 1]

    p_val = ttest_ind(outcome_cont, outcome_trt)[1]

    if np.isnan(p_val):
        return 0.000

    return p_val


def min_size_value_bool(min_size, t, treat_split=0.5):
    nt, nc = get_treat_size(t, treat_split=treat_split)

    return nt, nc, nt < min_size or nc < min_size


def check_min_size(min_size, t, treat_split=0.5):
    nt, nc = get_treat_size(t, treat_split)

    return nt < min_size or nc < min_size


def get_treat_size(t, treat_split=0.5):
    num_treatment = t[t > treat_split].shape[0]
    num_control = t[t <= treat_split].shape[0]

    return num_treatment, num_control


def variance(y, t):
    treat_vect = t

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
