import pandas as pd
import numpy as np
from causal_tree_learn import *
from sklearn.model_selection import train_test_split

asthma = pd.read_csv('data/asthma.txt', delimiter=' ', index_col=None)

asthma.columns = ['physician', 'age', 'sex', 'education', 'insurance','drug coverage', 'severity',
                  'comorbidity', 'physical comorbidity', 'mental comorbidity', 'satisfaction']

y = asthma['satisfaction'].as_matrix()
treatment = asthma['physician'].as_matrix()

x = asthma.drop(['satisfaction', 'physician'], axis=1).as_matrix()

columns = asthma.drop(['satisfaction', 'physician'], axis=1).columns

y[y == 0] = -1

treatment[treatment==1] = 0
treatment[treatment==2] = 1

# x_train, x_test, y_train, y_test, treat_train, treat_test = train_test_split(x, y, treatment,
#                                                                                  test_size=0.5, random_state=42)

x_train, y_train, treat_train = x, y, treatment

causal_tree = grow_causal_tree(x_train, y_train, treat_train, cont=False, min_size=5, max_depth=4)
plot_tree(causal_tree, columns, "asthma")