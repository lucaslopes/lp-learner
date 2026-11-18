'''
This module contains functions to calculate the probability of having full queues.
'''


import numpy as np
from lplearn._legacy.utils import numba_njit


@numba_njit
def fraction_of_full_queues_numpy(
    y: np.ndarray,
    to_round: bool = False
  ) -> float:

  '''
  Fraction of full queues.\n
  `y`: Policy matrix.\n
  `to_round`: Whether to round the result to 4 decimal places or not.\n
  return $pi_F(t) = y_{K,0,0} + y_{K,0,1}$.\n
  '''

  T = y.shape[-1]
  K = y.shape[0] - 1
  sum = y[K,0,:,:].sum()
  prob = sum / T
  prob = round(prob, 4) if to_round else prob

  return prob


@numba_njit
def fraction_of_full_queues_loop(
    y: np.ndarray,
    to_round: bool = False
  ) -> float:

  '''
  Fraction of full queues.\n
  `y`: Policy matrix.\n
  `to_round`: Whether to round the result to 4 decimal places or not.\n
  return $pi_F(t) = y_{K,0,0} + y_{K,0,1}$.\n
  '''

  T = y.shape[-1]
  K = y.shape[0] - 1
  sum = 0
  for t in range(T):
    sum += y[K,0,0,t] + y[K,0,1,t]
  prob = sum / T
  prob = round(prob, 4) if to_round else prob

  return prob
