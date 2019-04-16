import pandas as pd
import numpy as np
from CTL import CausalTree
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

x_train, x_test, y_train, y_test, treat_train, treat_test = train_test_split(x, y, treatment,
                                                                             test_size=0.5, random_state=42)

ct = CausalTree()
ct.fit(x_train, y_train, treat_train)
effect_prediction, leaf_results, trigger_results = ct.predict(x_test)
