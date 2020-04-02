from CTL.causal_tree_learn import CausalTree
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(0)

x = np.random.randn(100, 10)
y = np.random.randn(100)
treatment = np.random.randn(100)

x_train, x_test, y_train, y_test, treat_train, treat_test = train_test_split(x, y, treatment,
                                                                             test_size=0.5, random_state=42)

variable_names = []
for i in range(x.shape[1]):
    variable_names.append(f"Column {i}")

# regular CTL
ctl = CausalTree(cont=True)
ctl.fit(x_train, y_train, treat_train)
ctl_predict = ctl.predict(x_test)

# honest CTL
cth = CausalTree(cont=True, honest=True)
cth.fit(x_train, y_train, treat_train)
cth_predict = cth.predict(x_test)

# val honest CTL
cthv = CausalTree(cont=True, val_honest=True)
cthv.fit(x_train, y_train, treat_train)
cthv_predict = cthv.predict(x_test)

# adaptive CT
ct_adaptive = CausalTree(weight=0.0, split_size=0.0, cont=True)
ct_adaptive.fit(x_train, y_train, treat_train)
ct_adaptive_predict = cthv.predict(x_test)

# to get which examples are in which leaf
groups = cthv.get_groups(x_test)

# to get triggers
triggers = cthv.get_triggers(x_test)
print(triggers)

# to get features used, input the columns
features_used = cthv.get_variables_used(variable_names)
print(features_used)

# to get the decision for every example
features = cthv.get_features(x)
print(features)

# if you want to plot a tree
cthv.plot_tree(filename="output/trigger_tree")


