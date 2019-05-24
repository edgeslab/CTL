import pandas as pd
from CTL.CTL import CausalTree
from sklearn.model_selection import train_test_split
import numpy as np

x = np.random.randn(100, 10)
y = np.random.randn(100)
treatment = np.random.randn(100)

x_train, x_test, y_train, y_test, treat_train, treat_test = train_test_split(x, y, treatment,
                                                                             test_size=0.5, random_state=42)

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

# if you want to plot a tree
# ctl.plot_tree(training_data=x_train)

# if you have variable names
# ctl.plot_tree(feat_names=variable_names)
