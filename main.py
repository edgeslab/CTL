import pandas as pd
import time
from causal_tree_learn import *

if __name__ == '__main__':

    survey = pd.read_csv('data/processed_data.csv')

    survey_s = survey[(survey['CF-3'] > 3) | (survey['CF-3'] < 3)]

    s_labels = survey_s['CF-3'].as_matrix()
    treatment = survey_s['date_diff'].as_matrix()
    treatment_binary = survey_s['date_diff'].copy().as_matrix()

    treat_split = 180

    treat = treatment_binary > treat_split
    control = treatment_binary <= treat_split

    treatment_binary[treat] = 1
    treatment_binary[control] = 0

    g3 = s_labels > 3
    l3 = s_labels < 3

    s_labels[g3] = -1
    s_labels[l3] = 1

    survey_s_dat = survey_s.drop(['CF-3'], axis=1)
    survey_s_dat = survey_s_dat.drop(['CF-2'], axis=1)  # proxy for sensitvity
    survey_s_dat = survey_s_dat.drop(['CF-4'], axis=1)
    survey_s_dat = survey_s_dat.drop(['date_diff'], axis=1)

    survey_s_dat_f = survey_s_dat.copy()

    s_data = survey_s_dat_f.as_matrix()

    x = s_data
    y = s_labels

    var = survey_s_dat_f.columns

    max_depth = -1

    start = time.time()
    causal_tree = grow_causal_tree(x, y, treatment_binary, cont=False, min_size=10, max_depth=max_depth, seed=0)
    prune(causal_tree, alpha=.01)
    plot_tree(causal_tree, var, "binary")
    end = time.time()
    print(end - start)