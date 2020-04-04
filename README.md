# CTL

Christopher Tran, Elena Zheleva, ["Learning Triggers for Heterogeneous Treatment Effects", AAAI 2019.](https://arxiv.org/pdf/1902.00087.pdf)

Our method is based on and adapted from: https://github.com/susanathey/causalTree


## Requirements
* Python 3
* sklearn
* scipy
* graphviz (if you want to plot the tree)

## Installation

through pip

```bash
pip install causal_tree_learn
```

or clone the repository
```bash
python setup.py build_ext --inplace
```

## Demo Code

Two demo codes are available to run.

```bash
python binary_example.py
```
Runs the tree on a binary example (asthma.txt)

```bash
python trigger_example.py
```
Runs a tree on a trigger problem where the treatment is continuous (note for now the example is made up and treatment does not affect outcome, this is only to show example code)
