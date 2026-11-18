import numpy as np
from lplearn._legacy.utils import numba_njit, matmul


# @numba_njit
def compute_U(K, aggregated_states):
    U = np.zeros(((K+1), len(aggregated_states)))
    for cluster in range(len(aggregated_states)):
        cluster_states = aggregated_states[cluster]
        for state in cluster_states:
            U[state, cluster] = 1 / len(cluster_states)
    return U


# @numba_njit
def compute_V(K, aggregated_states):
    V = np.zeros((len(aggregated_states), (K+1)))
    for cluster in range(len(aggregated_states)):
        cluster_states = aggregated_states[cluster]
        for state in cluster_states:
            V[cluster, state] = 1
    return V


# @numba_njit
def reshape_state_space(
        var: np.ndarray,
        map: np.ndarray,
    ) -> np.ndarray:
    if len(var.shape) == 1 and len(map.shape) == 2:
        var_new = matmul(map, var)
    elif len(var.shape) == 2 and len(map.shape) == 2:
        var_new = np.zeros([map.shape[0], var.shape[1]])
        for t in range(var.shape[1]):
            var_new[:, t] = matmul(map, var[:, t])
    elif len(var.shape) == 4 and len(map.shape) == 2:
        var_new = np.zeros([map.shape[0], *var.shape[1:]])
        for a in range(var.shape[1]):
            for b in range(var.shape[2]):
                for t in range(var.shape[3]):
                    var_new[:, a, b, t] = matmul(map, var[:, a, b, t])
    else:
        var_new = np.array([])
    return var_new


# @numba_njit
def true_vs_expanded(true, expanded, U, V):
    return np.array_equal(matmul(matmul(U, V), true), matmul(matmul(U, V), expanded))
