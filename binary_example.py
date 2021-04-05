import pandas as pd
from CTL.causal_tree_learn import CausalTree
from sklearn.model_selection import train_test_split
import numpy as np

asthma = pd.read_csv('data/asthma.txt', delimiter=' ', index_col=None)

asthma.columns = ['physician', 'age', 'sex', 'education', 'insurance', 'drug coverage', 'severity',
                  'comorbidity', 'physical comorbidity', 'mental comorbidity', 'satisfaction']

y = asthma['satisfaction'].values
treatment = asthma['physician'].values

x = asthma.drop(['satisfaction', 'physician'], axis=1).values

columns = asthma.drop(['satisfaction', 'physician'], axis=1).columns

y[y == 0] = -1

treatment[treatment == 1] = 0
treatment[treatment == 2] = 1

np.random.seed(0)


x_train, x_test, y_train, y_test, treat_train, treat_test = train_test_split(x, y, treatment,
                                                                             test_size=0.5, random_state=42)

# regular CTL
ctl = CausalTree(magnitude=False)
ctl.fit(x_train, y_train, treat_train)
ctl.prune()
ctl_predict = ctl.predict(x_test)

# honest CTL (CT-HL)
cthl = CausalTree(honest=True)
cthl.fit(x_train, y_train, treat_train)
cthl.prune()
cthl_predict = cthl.predict(x_test)

# val honest CTL (CT-HV)
cthv = CausalTree(val_honest=True)
cthv.fit(x_train, y_train, treat_train)
cthv.prune()
cthv_predict = cthv.predict(x_test)

# adaptive CT (Athey and Imbens, PNAS 2016)
ct_adaptive = CausalTree(weight=0.0, split_size=0.0)
ct_adaptive.fit(x_train, y_train, treat_train)
ct_adaptive.prune()
ct_adaptive_predict = cthv.predict(x_test)

# honest CT (Athey and Imbens, PNAS 2016)
ct_honest = CausalTree(honest=True, weight=0.0, split_size=0.0)
ct_honest.fit(x_train, y_train, treat_train)
ct_honest.prune()
ct_honest_predict = ct_honest.predict(x_test)

ct_adaptive.plot_tree(features=columns, filename="output/bin_tree_adaptive", show_effect=True)
ct_honest.plot_tree(features=columns, filename="output/bin_tree_honest", show_effect=True)
ctl.plot_tree(features=columns, filename="output/bin_tree", show_effect=True)
cthl.plot_tree(features=columns, filename="output/bin_tree_honest_learn", show_effect=True)
cthv.plot_tree(features=columns, filename="output/bin_tree_honest_validation", show_effect=True)