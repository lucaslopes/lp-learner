'''
This module contains the processing (or energy) cost function.
'''


from lplearn._legacy.utils import numba_njit


@numba_njit
def processing_cost(
        processing_prob: float,
        cost_exp: int = 1,
    ) -> float:

    '''
    The processing cost to be used in the objective function\n
    `processing_prob`: The processing probability.\n
    `cost_exp`: The cost exponent. default is 1. which means the cost is linear.\n
    '''

    cost = (2 * (1 + processing_prob)) ** cost_exp

    return cost
