'''
This module contain the function to compute the instantaneous transition probability.
'''


from lplearn._legacy.utils import numba_njit
from lplearn._legacy.utils import binom_pmf, binom_cdf


@numba_njit
def instantaneous_probability(
        case: int,
        s: int,
        s_: int,
        a: int,
        q: float,
        tau: int,
    ) -> float:

    '''
    Instantaneous transition probability.\n
    `case`: Transition probability case:\n
        - 1 = We consider each job can be processed at any time instant in the interval τ with probability b.We have s + a − s′ such jobs and so consider a binomial distribution over them;\n
        - 2 = We consider only one job can be processed at a time instant, so that reduces to s + a − s′ number of successes in time interval τ with success probability at each instance b.\n
    `s`: Current state of the system.\n
    `s_`: Next state.\n
    `a`: Load balancing action (0 to reject) or (1 to accept).\n
    `q`: Processing probability.\n
    `tau`: Time interval.\n
    '''

    if case == 1:  # CJS
        p_failure = (1 - q) ** tau
        jobs_left = s_
        n_trials = s + a
        pmf = binom_pmf(jobs_left, n_trials, p_failure)

    elif case == 2:  # SJS
        p_success = q
        n_trials = tau
        if s_ == 0:
            n_successes = s + a
            pmf = 1 - binom_cdf(n_successes, n_trials, p_success) + binom_pmf(n_successes, n_trials, p_success)
        else:
            n_successes = s + a - s_
            pmf = binom_pmf(n_successes, n_trials, p_success)

    else:
        raise ValueError(f'Invalid case number {case} for processing probability {q} in function `instantaneous_probability()`')

    return pmf
