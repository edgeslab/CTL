# CTL

Christopher Tran, Elena Zheleva, ["Learning Triggers for Heterogeneous Treatment Effects", AAAI 2019.](https://www.cs.uic.edu/~ctran/docs/learning_triggers_HTE_aaai19.pdf)

Our method is based on and adapted from: https://github.com/susanathey/causalTree


## Requirements
* Python 3
* sklearn
* scipy
* graphviz (if you want to plot the tree)

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
