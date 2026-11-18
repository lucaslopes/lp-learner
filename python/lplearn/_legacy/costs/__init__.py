'''
This module contains the cost function to be used in the objective function.
'''


import numpy as np
from lplearn._legacy.utils import numba_njit
from .process import processing_cost
from .storage import storage_cost


@numba_njit
def instantaneous_cost(
        y: np.ndarray,
        t: int,
        s: int,
        b_low: float,
        b_high: float,
        gamma: float = 0,
        gamma_bar: float = 0,
        Gamma_y: float = 0,
        gradient: bool = False,
        allow_accept_when_full: bool = True,
    ) -> tuple[float, np.ndarray]:

    '''
    Instantaneous cost for the parallel queues with service rate selection problem.\n
    `y`: Policy.\n
    `t`: Current time interval.\n
    `s`: Current state of the system.\n
    `b_low`: processing probability when action b=0.\n
    `b_high`: processing probability when action b=1.\n
    `gamma`: penalty for missed allocation of jobs.\n
    `gamma_bar`: penalty for accepting when queue is full.\n
    `Gamma_y`: L2 regularization term for the policy matrix.\n
    `gradient`: Whether to return the gradient of the cost function or not.\n
    `allow_accept_when_full`: Allow the system to accept a job when the queue is full.\n
    '''

    K = y.shape[0] - 1

    cost = 0.0
    cost += (storage_cost(s, K) + processing_cost(b_low)) * (y[s,0,0,t] + y[s,1,0,t])  # storage and proceesing cost when b=0
    cost += (storage_cost(s, K) + processing_cost(b_high)) * (y[s,0,1,t] + y[s,1,1,t])  # storage and processing cost when b=1
    cost += gamma * (y[s,0,0,t] + y[s,0,1,t])  # missed allocation cost when a=0
    if allow_accept_when_full and s == K:
        cost += gamma_bar * (y[K,1,0,t] + y[K,1,1,t])  #penalty for accepting when queue is full
    cost += Gamma_y * np.sum(y[s, :, :, t]**2)  # regularization term

    grad = np.zeros_like(y[s, :, :, t])
    if gradient:
        grad[0, 0] = storage_cost(s, K) + processing_cost(b_low) + gamma
        grad[0, 1] = storage_cost(s, K) + processing_cost(b_high) + gamma
        grad[1, 0] = storage_cost(s, K) + processing_cost(b_low)
        grad[1, 1] = storage_cost(s, K) + processing_cost(b_high)
        grad += 2 * Gamma_y * y[s, :, :, t]
        if allow_accept_when_full and s == K:
            grad[1, 0] += gamma_bar
            grad[1, 1] += gamma_bar

    return cost, grad


@numba_njit
def calculate_cost(
        y: np.ndarray,
        b_low: float,
        b_high: float,
        gamma: float,
        gamma_bar: float = 0,
        Gamma_y: float = 0,
        gradient: bool = False,
        allow_accept_when_full: bool = True,
    ) -> float:

    '''
    The processing cost to be used in the objective function\n
    `y`: Policy.\n
    `b_low`: processing probability when action b=0.\n
    `b_high`: processing probability when action b=1.\n
    `T`: Number of time intervals (time horizon).\n
    `K`: Maximum number of jobs in the queues.\n
    `gamma`: penalty for missed allocation of jobs.\n
    `gamma_bar`: penalty for accepting when queue is full.\n
    `Gamma_y`: L2 regularization term for the policy matrix.\n
    `gradient`: Whether to return the gradient of the cost function or not.\n
    `allow_accept_when_full`: Allow the system to accept a job when the queue is full.\n
    '''

    cost = 0
    K = y.shape[0] - 1
    T = y.shape[-1]

    for t in range(T):
        for s in range(K+1):
            inst_cost, _ = instantaneous_cost(y, t, s, b_low, b_high, gamma, gamma_bar, Gamma_y, gradient, allow_accept_when_full)
            cost += inst_cost

    return cost
