"""

"""

import numpy as np

def filter_priv_covars_to_add_covars(vars):
    assert (not all(x>=0 for x in vars)), "Privilege level variances must all be positive! Aborting."
    n = len(vars)
    assert (1/(n-1) * sum(vars) > x for x in vars), "Condition not met for privilege level variances! Requires that: 1/(n-1) * (v_1+...+v_n) > v_i for all 0<=i<n. Aborting."
    A = np.array([i*[1] + [0] + (n-i-1)*[1] for i in range(n)])
    return np.linalg.solve(A, vars)
