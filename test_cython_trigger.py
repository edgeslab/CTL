import numpy as np
import timeit

setup = '''
from CTL.causal_tree_learn import CausalTree
import numpy as np

x = np.random.random(size=(1000, 10))
y = np.random.random(size=(1000,))
t = np.random.random(size=(1000,))

ctl = CausalTree(cont=True, old=False)
'''

run_time = timeit.repeat("ctl.fit(x, y, t)", setup=setup, repeat=10, number=1)
print(f"Cython speed: {np.mean(run_time)}")

setup = '''
from CTL.causal_tree_learn import CausalTree
import numpy as np

x = np.random.random(size=(1000, 10))
y = np.random.random(size=(1000,))
t = np.random.random(size=(1000,))

ctl = CausalTree(cont=True, old=True)
'''

run_time = timeit.repeat("ctl.fit(x, y, t)", setup=setup, repeat=10, number=1)
print(f"Python speed: {np.mean(run_time)}")
