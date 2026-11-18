'''
This module contains the storage cost function.
'''


import numpy as np
from numba.typed import List
from lplearn._legacy.utils import numba_njit



@numba_njit
def storage_cost(
        cluster: List,
        K: int,
        cost_exp: int = 1,
        normalize: bool = False,
    ) -> float:

    '''
    The storage cost to be used in the objective function\n
    `cluster`: States that are aggregated in a cluster.\n
    `K`: Maximum number of jobs in the queues.\n
    `cost_exp`: The cost exponent. default is 1. which means the cost is linear.\n
    '''

    cost, jobs_in_cluster, max_jobs_in_cluster = 0, 0, K

    if isinstance(cluster, int):
        jobs_in_cluster = cluster

    elif isinstance(cluster, List):
        if normalize:
            max_jobs_in_cluster = K * len(cluster)
            for s in cluster:
                jobs_in_cluster += s
        else:
            for s in cluster:
                cost += s / K

    if cost == 0:
        cost = jobs_in_cluster / max_jobs_in_cluster
    cost = cost ** cost_exp

    return cost
