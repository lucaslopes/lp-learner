'''
This module contains functions related to the binomial distribution.\n
'''


from lplearn._legacy.utils.decorators import numba_njit
from lplearn._legacy.utils.calculations import factorial


@numba_njit
def binom_coeff(
        n: int,
        k: int,
    ) -> int:

    '''
    Calculate binomial coefficient manually for Numba compatibility.\n
    `n`: Total number of trials.\n
    `k`: Number of successes.\n
    '''

    result = 0

    if k <= n:
        result = factorial(n) // (factorial(k) * factorial(n - k))

    return result


@numba_njit
def binom_pmf(
        k: int,
        n: int,
        p: float,
    ) -> float:

    '''
    Binomial PMF implemented manually for Numba compatibility.\n
    `k`: Number of successes.\n
    `n`: Total number of trials.\n
    `p`: Probability of success.\n
    '''

    result = 0.0

    if k <= n:
        result = binom_coeff(n, k) * (p ** k) * ((1 - p) ** (n - k))

    return result


@numba_njit
def binom_cdf(
        k: int,
        n: int,
        p: float,
    ) -> float:

    '''
    Binomial CDF implemented manually for Numba compatibility.\n
    `k`: Number of successes.\n
    `n`: Total number of trials.\n
    `p`: Probability of success.\n
    '''

    cdf = 0.0

    for i in range(k + 1):
        cdf += binom_pmf(i, n, p)

    return cdf
