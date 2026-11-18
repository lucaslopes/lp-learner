'''
This module contains functions to compute the smoothness constant kappa.
'''


import numpy as np
from lplearn._legacy.utils.decorators import numba_njit


@numba_njit
def kappa(
        K: int,
        T: int,
        A: int = 2,
        B: int = 2,
    ) -> float:

    '''
    Computes the smoothness constant kappa based on the provided lemma.\n
    It is useful to known the minimum value of the regularization to convergence of the algorithm.\n

    Parameters:\n
    - `K` *(int)*: Buffer Size that limits the number of jobs in a queue |𝒮| = K+1.\n
    - `T` *(int)*: Time horizon T.\n
    - `A` *(int)*: Size of the set of actions |𝒜|.\n
    - `B` *(int)*: Size of the set of actions |ℬ|.\n

    Returns:\n
    - float: The computed kappa value\n
    '''

    S_size = K + 1

    term1 = 18 * (S_size ** 2) * A * B * T
    term2 = 2 * (A * B + S_size * A * B) + S_size * (A + B) + 1
    max_term = max(term1, term2)
    kappa_value = np.sqrt(max_term)

    return kappa_value


def main():
    # Example input values
    K = 5  # Size of the set of states
    T = 10 # Time horizon

    # Compute kappa
    kappa_value = kappa(K, T)
    print(f'The computed kappa value is: {kappa_value}')
    return True


__name__ == '__main__' and main()
