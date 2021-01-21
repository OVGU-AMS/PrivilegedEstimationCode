"""

"""

import numpy as np

def priv_vars_to_vars_to_remove(variances):
    n = len(variances)
    assert all(x>=0 for x in variances), "Privilege variances must all be positive! Aborting."
    assert all(1/(n-1) * sum(variances) > x for x in variances), "Conditions not met for privilege variances! Requires that: 1/(n-1) * (v_1+...+v_n) > v_i for all 0<=i<n! Aborting."
    A = np.array([i*[1] + [0] + (n-i-1)*[1] for i in range(n)])
    return np.linalg.solve(A, variances)

def priv_covars_to_covars_to_remove(covars):
    n = len(covars)
    assert all(is_positive_semidefinite(P) for P in covars), "Privilege covariances must all be positive! Aborting."
    assert all(is_positive_semidefinite(1/(n-1) * sum(covars) - P) for P in covars), "Conditions not met for privilege covariance! Requires that: 1/(n-1) * (P_1+...+P_n) > P_i for all 0<=i<n! Aborting."
    A = np.array([i*[1] + [0] + (n-i-1)*[1] for i in range(n)])
    Ainv = np.linalg.inv(A)
    noise_covars = []
    for row in Ainv:
        noise_covars.append(sum(row[i]*covars[i] for i in range(n)))
    return noise_covars

def is_positive_semidefinite(M):
    return all(e>=0 for e in np.linalg.eigvals(M))
