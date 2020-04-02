import numpy as np
import timeit

# print(t)

setup = '''
from CTL.causal_tree.util_c import ace, variance, get_pval, get_treat_size, check_min_size, min_size_value_bool
from CTL.causal_tree.util_c import divide_set, ace_trigger, variance_trigger, tau_squared_trigger
import numpy as np

x = np.random.random(size=(1000, 10))
y = np.random.random(size=(1000,))
t = np.random.random(size=(1000,))
# t = np.random.choice([0.0, 1.0], size=1000)
'''

# cy = timeit.repeat("ace(y, t)", setup=setup, repeat=1, number=10000)
# run_time = timeit.repeat("variance(y, t)", setup=setup, repeat=1, number=10000)
# run_time = timeit.repeat("get_pval(y, t)", setup=setup, repeat=5, number=10000)
# run_time = timeit.repeat("get_treat_size(t)", setup=setup, repeat=10, number=50000)
# run_time = timeit.repeat("check_min_size(5, t)", setup=setup, repeat=10, number=50000)
# run_time = timeit.repeat("min_size_value_bool(5, t)", setup=setup, repeat=10, number=50000)
# run_time = timeit.repeat("divide_set(x, y, t, 0, 0)", setup=setup, repeat=10, number=10000)

# run_time = timeit.repeat("variance_trigger(y, t, 0.0)", setup=setup, repeat=10, number=10000)
run_time = timeit.repeat("tau_squared_trigger(y, t, 5, True)", setup=setup, repeat=10, number=100)
print(f"Cython speed: {np.mean(run_time)}")

setup = '''
from CTL.causal_tree.util import ace, variance, get_pval, get_treat_size, check_min_size, min_size_value_bool
from CTL.causal_tree.util import divide_set, ace_trigger, variance_trigger, tau_squared_trigger
import numpy as np

x = np.random.random(size=(1000, 10))
y = np.random.random(size=(1000,))
t = np.random.random(size=(1000,))
# t = np.random.choice([0.0, 1.0], size=1000)
'''

# run_time = timeit.repeat("ace(y, t)", setup=setup, repeat=1, number=10000)
# run_time = timeit.repeat("variance(y, t)", setup=setup, repeat=1, number=10000)
# run_time = timeit.repeat("get_pval(y, t)", setup=setup, repeat=5, number=10000)
# run_time = timeit.repeat("get_treat_size(t)", setup=setup, repeat=10, number=50000)
# run_time = timeit.repeat("check_min_size(5, t)", setup=setup, repeat=10, number=50000)
# run_time = timeit.repeat("min_size_value_bool(5, t)", setup=setup, repeat=10, number=50000)
# run_time = timeit.repeat("divide_set(x, y, t, 0, 0)", setup=setup, repeat=10, number=10000)

# run_time = timeit.repeat("variance_trigger(y, t, 0.0)", setup=setup, repeat=10, number=10000)
run_time = timeit.repeat("tau_squared_trigger(y, t, 5, True)", setup=setup, repeat=10, number=100)
print(f"Python speed: {np.mean(run_time)}")
