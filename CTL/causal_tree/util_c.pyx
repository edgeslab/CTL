import os
import errno
import numpy as np
from scipy.stats import ttest_ind
import subprocess
import time

cimport cython
# from libcpp cimport bool
from cpython cimport bool
cimport numpy as np

# TODO: Category types

# ----------------------------------------------------------------
# General helper functions
# ----------------------------------------------------------------

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def check_dir(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def divide_set(x, y, t, col, value):
    idx1 = x[:, col] >= value
    idx2 = x[:, col] < value

    x1 = x[idx1]
    x2 = x[idx2]

    y1 = y[idx1]
    y2 = y[idx2]

    t1 = t[idx1]
    t2 = t[idx2]

    return x1, x2, y1, y2, t1, t2

def yield_divide(x, y, t, col, unique_vals):
    for value in unique_vals:
        idx1 = x[:, col] >= value
        idx2 = ~idx1

        yield x[idx1], x[idx2], y[idx1], y[idx2], t[idx1], t[idx2]

def col_dict(names):
    feat_names = {}
    for i, name in enumerate(names):
        column = "Column %s" % i
        feat_names[column] = name
    return feat_names

cpdef min_size_value_bool(min_size, t, trigger=0.5):
    cdef int nt
    cdef int nc
    cdef bool treat_check
    cdef bool control_check
    cdef check
    nt, nc = get_treat_size(t, trigger)

    treat_check = nt < min_size
    control_check = nc < min_size

    check = treat_check or control_check

    return nt, nc, check

cpdef check_min_size(int min_size, np.ndarray[np.float_t, ndim=1] t, trigger=0.5):
    cdef int nt
    cdef int nc
    cdef bool treat_check
    cdef bool control_check
    nt, nc = get_treat_size(t, trigger)

    treat_check = nt < min_size
    control_check = nc < min_size
    return nt < min_size or nc < min_size

cpdef get_treat_size(np.ndarray[np.float_t, ndim=1] t, float trigger=0.5):
    cdef int num_treatment = 0
    cdef int num_control = 0
    for i in range(len(t)):
        if t[i] >= trigger:
            num_treatment += 1
        else:
            num_control += 1

    return num_treatment, num_control

# ----------------------------------------------------------------
# Binary functions
# ----------------------------------------------------------------

cpdef variance(np.ndarray[np.float_t, ndim=1] y, np.ndarray[np.float_t, ndim=1] t):
    cdef int tmax = len(t)

    cdef float mu1 = 0.0
    cdef float mu0 = 0.0
    cdef int mu1_denom = 0
    cdef int mu0_denom = 0

    cdef float var_t = 0
    cdef float var_c = 0

    for i in range(tmax):
        if t[i] <= 0.5:
            mu0 += y[i]
            mu0_denom += 1
        else:
            mu1 += y[i]
            mu1_denom += 1

    if mu0_denom == 0:
        mu0 = 0
    else:
        mu0 = mu0 / mu0_denom

    if mu1_denom == 0:
        mu1 = 0
    else:
        mu1 = mu1 / mu1_denom

    for i in range(tmax):
        if t[i] <= 0.5:
            var_c += (y[i] - mu0)**2
        else:
            var_t += (y[i] - mu1)**2

    if mu0_denom == 0:
        var_c = 0
    else:
        var_c = var_c / mu0_denom

    if mu1_denom == 0:
        var_t = 0
    else:
        var_t = var_t / mu1_denom

    return var_t, var_c

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef ace(np.ndarray[np.float_t, ndim=1] y, np.ndarray[np.float_t, ndim=1] t):
    cdef int tmax = len(t)

    cdef float mu1 = 0.0
    cdef float mu0 = 0.0
    cdef int mu1_denom = 0
    cdef int mu0_denom = 0

    for i in range(tmax):
        if t[i] <= 0.5:
            mu0 += y[i]
            mu0_denom += 1
        else:
            mu1 += y[i]
            mu1_denom += 1

    if mu0_denom == 0:
        mu0 = 0
    else:
        mu0 = mu0 / mu0_denom

    if mu1_denom == 0:
        mu1 = 0
    else:
        mu1 = mu1 / mu1_denom

    return mu1 - mu0

cpdef tau_squared(y, t):
    cdef int total = y.shape[0]
    cdef float[2] return_val = [-np.inf, -np.inf]

    if total == 0:
        return return_val

    cdef float effect = ace(y, t)
    cdef float err = (effect ** 2) * total

    return effect

cpdef get_pval(y, t):
    treat = t <= 0.5
    # control = t == 0
    control = ~treat

    outcome_cont = y[treat]
    outcome_trt = y[control]

    p_val = ttest_ind(outcome_cont, outcome_trt)[1]

    if np.isnan(p_val):
        return 0.000

    return p_val

# ----------------------------------------------------------------
# Trigger functions
# ----------------------------------------------------------------
cpdef ace_trigger(np.ndarray[np.float_t, ndim=1] y, np.ndarray[np.float_t, ndim=1] t, float trigger):
    cdef int tmax = len(t)

    cdef float mu1 = 0.0
    cdef float mu0 = 0.0
    cdef int mu1_denom = 0
    cdef int mu0_denom = 0

    for i in range(tmax):
        if t[i] < trigger:
            mu0 += y[i]
            mu0_denom += 1
        else:
            mu1 += y[i]
            mu1_denom += 1

    if mu0_denom == 0:
        mu0 = 0
    else:
        mu0 = mu0 / mu0_denom

    if mu1_denom == 0:
        mu1 = 0
    else:
        mu1 = mu1 / mu1_denom

    return mu1 - mu0

cpdef variance_trigger(np.ndarray[np.float_t, ndim=1] y, np.ndarray[np.float_t, ndim=1] t, float trigger):
    cdef int tmax = len(t)

    cdef float mu1 = 0.0
    cdef float mu0 = 0.0
    cdef int mu1_denom = 0
    cdef int mu0_denom = 0

    cdef float var_t = 0
    cdef float var_c = 0

    for i in range(tmax):
        if t[i] < trigger:
            mu0 += y[i]
            mu0_denom += 1
        else:
            mu1 += y[i]
            mu1_denom += 1

    if mu0_denom == 0:
        mu0 = 0
    else:
        mu0 = mu0 / mu0_denom

    if mu1_denom == 0:
        mu1 = 0
    else:
        mu1 = mu1 / mu1_denom

    for i in range(tmax):
        if t[i] < trigger:
            var_c += y[i] - mu0
        else:
            var_t += y[i] - mu1

    if mu0_denom == 0:
        var_c = 0
    else:
        var_c = var_c / mu0_denom

    if mu1_denom == 0:
        var_t = 0
    else:
        var_t = var_t / mu1_denom

    return var_t, var_c

cpdef get_pval_trigger(y, t, trigger):
    treat = t >= trigger
    control = ~treat

    outcome_cont = y[treat]
    outcome_trt = y[control]

    p_val = ttest_ind(outcome_cont, outcome_trt)[1]

    if np.isnan(p_val):
        return 0.000

    return p_val

# cpdef tau_squared_trigger(np.ndarray[np.float_t, ndim=1] y, np.ndarray[np.float_t, ndim=1] t, int min_size,
#                           bool quartile):
#     cdef int total = y.shape[0]
#     cdef int first_quartile
#     cdef int third_quartile
#     cdef np.ndarray treated
#     cdef np.ndarray control
#     cdef np.ndarray[np.float_t, ndim=1] yt
#     cdef np.ndarray[np.float_t, ndim=1] yc
#     cdef float yt_mean
#     cdef float yc_mean
#     cdef float effect
#     cdef float err
#
#     cdef np.ndarray[np.float_t, ndim=1] unique_treatment = np.unique(t)
#     cdef float best_err = 0.0
#     cdef float best_effect = 0.0
#     cdef float best_trigger = unique_treatment[0]
#
#     return_val = (-np.inf, -np.inf, -np.inf)
#
#     if total == 0:
#         return return_val
#
#     if unique_treatment.shape[0] == 1:
#         return return_val
#
#     unique_treatment = (unique_treatment[1:] + unique_treatment[:-1]) / 2
#     unique_treatment = unique_treatment[1:-1]
#
#     if quartile:
#         first_quartile = int(np.floor(unique_treatment.shape[0] / 4))
#         third_quartile = int(np.ceil(3 * unique_treatment.shape[0] / 4))
#
#         unique_treatment = unique_treatment[first_quartile:third_quartile]
#
#     for trigger in unique_treatment:
#         treated = t >= trigger
#         control = ~treated
#
#         yt = y[treated]
#         yc = y[control]
#
#         yt_mean = np.mean(yt)
#         yc_mean = np.mean(yc)
#
#         effect = yt_mean - yc_mean
#         err = (effect ** 2) * total
#         if err > best_err:
#             best_effect = effect
#             best_err = err
#             best_trigger = trigger
#
#     return best_effect, best_trigger

cpdef tau_squared_trigger(np.ndarray[np.float_t, ndim=1] y, np.ndarray[np.float_t, ndim=1] t, int min_size,
                          bool quartile):
    cdef int total = y.shape[0]
    cdef int first_quartile
    cdef int third_quartile
    cdef np.ndarray[np.float_t, ndim=2] yy
    cdef np.ndarray[np.float_t, ndim=2] tt
    cdef np.ndarray x

    cdef np.ndarray[np.long_t, ndim=1] treat_num
    cdef np.ndarray[np.long_t, ndim=1] cont_num
    cdef np.ndarray[np.long_t, ndim=1] min_size_idx

    cdef np.ndarray[np.float_t, ndim=1] y_t_m
    cdef np.ndarray[np.float_t, ndim=1] y_c_m
    cdef np.ndarray[np.float_t, ndim=1] effect
    cdef np.ndarray[np.float_t, ndim=1] err
    cdef int max_err

    cdef np.ndarray[np.float_t, ndim=1] unique_treatment = np.unique(t)
    cdef float best_err = 0.0
    cdef float best_effect = 0.0
    cdef float best_trigger = unique_treatment[0]

    return_val = (-np.inf, -np.inf)

    if total == 0:
        return return_val

    if unique_treatment.shape[0] == 1:
        return return_val

    unique_treatment = (unique_treatment[1:] + unique_treatment[:-1]) / 2
    unique_treatment = unique_treatment[1:-1]

    if quartile:
        first_quartile = int(np.floor(unique_treatment.shape[0] / 4))
        third_quartile = int(np.ceil(3 * unique_treatment.shape[0] / 4))

        unique_treatment = unique_treatment[first_quartile:third_quartile]

    yy = np.tile(y, (unique_treatment.shape[0], 1))
    tt = np.tile(t, (unique_treatment.shape[0], 1))

    x = np.transpose(np.transpose(tt) >= unique_treatment)

    tt[x] = 1
    tt[np.logical_not(x)] = 0

    treat_num = np.sum(tt == 1, axis=1)
    cont_num = np.sum(tt == 0, axis=1)
    min_size_idx = np.where(np.logical_and(treat_num >= min_size, cont_num >= min_size))[0]

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

    return best_effect, best_split