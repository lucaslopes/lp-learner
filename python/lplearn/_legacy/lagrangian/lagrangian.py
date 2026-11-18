'''
This module contains the function to compute the L of the optimization problem.
'''


import numpy as np
from numba.typed import List
from lplearn._legacy.utils import numba_njit
from lplearn._legacy.costs import storage_cost, processing_cost


@numba_njit
def initial_constraints(
        y: np.ndarray,
        mu_1: np.ndarray,
        m_s_0: np.ndarray,
    ) -> float:
    val = 0.0
    K = y.shape[0] - 1
    for s in range(K+1):
        val == mu_1[s] * (np.sum(y[s,:,:,0]) - m_s_0[s])
    return val


@numba_njit
def cost_value(
        y: np.ndarray,
        t: int,
        cost_storage: List,
        cost_low: float,
        cost_high: float,
        gamma: float,
        gamma_bar: float,
        allow_accept_when_full: bool,
    ) -> float:
    val = 0.0
    K = y.shape[0] - 1
    for s in range(K+1):
        val += (cost_storage[s] + cost_low) * np.sum(y[s,:,0,t])  # storage and proceesing cost when b=0
        val += (cost_storage[s] + cost_high) * np.sum(y[s,:,1,t])  # storage and processing cost when b=1
        val += gamma * np.sum(y[s,0,:,t])  # missed allocation cost when a=0
        if allow_accept_when_full and s == K:
            val += gamma_bar * np.sum(y[K,1,:,t])  # penalty for accepting when queue is full
    return val


# @numba_njit
def dynamic_constraints(
        y: np.ndarray,
        t: int,
        P: np.ndarray,
        mu_2: np.ndarray,
        mu_3: np.ndarray = None,
    ) -> float:
    val = 0.0
    if t >= 1:
        K = y.shape[0] - 1
        for s in range(K+1):
            probs = np.sum(P[s,:,:,:] * y[:,:,:,t-1])
            m_s_t = np.sum(y[s,:,:,t])
            val += mu_2[s,t] * (m_s_t - probs)
            if mu_3 is not None and s == K:
                val += np.sum(mu_3[:,t] * y[K,1,:,t])
    return val


@numba_njit
def budget_constraints(
        y: np.ndarray,
        t: int,
        alpha: float,
        beta: float,
        lambda_1: np.ndarray,
        lambda_2: np.ndarray,
    ) -> float:
    val = 0.0
    val += lambda_1[t] * (np.sum(y[:,1,:,t]) - alpha)
    val += lambda_2[t] * (np.sum(y[:,:,1,t]) - beta)
    return val


@numba_njit
def positivity_constraints(
        y: np.ndarray,
        t: int,
        lambda_3: np.ndarray,
    ) -> float:
    return np.sum(lambda_3[:,:,:,t] * y[:,:,:,t])


@numba_njit
def regularization_policy(
        y: np.ndarray,
        Gamma_y: float,
    ) -> float:
    return Gamma_y * np.sum(y**2)


# @numba_njit
def regularization_dual(
        Gamma_lambda: float,
        lambda_1: np.ndarray,
        lambda_2: np.ndarray,
        lambda_3: np.ndarray,
        Gamma_mu: float,
        mu_1: np.ndarray,
        mu_2: np.ndarray,
        mu_3: np.ndarray = None,
    ) -> float:
    val = 0.0
    # regularization on the lambda
    val += Gamma_lambda * np.sum(lambda_1**2)
    val += Gamma_lambda * np.sum(lambda_2**2)
    val += Gamma_lambda * np.sum(lambda_3**2)
    # regularization on the mu
    val += Gamma_mu * np.sum(mu_1**2)
    val += Gamma_mu * np.sum(mu_2[:,1:]**2)  # TODO: we're regularizating only [:,1:], val should reflect this?
    if mu_3 is not None:
        val += Gamma_mu * np.sum(mu_3**2)
    return val


@numba_njit
def compute_lagrangian_value(
        P: np.ndarray,
        T: int,
        m_s_0: np.ndarray,
        b_low: float,
        b_high: float,
        alpha: float,
        beta: float,
        gamma: float,
        gamma_bar: float,
        Gamma_y: float,
        Gamma_lambda: float,
        Gamma_mu: float,
        y: np.ndarray,
        lambda_1: np.ndarray,
        lambda_2: np.ndarray,
        lambda_3: np.ndarray,
        mu_1: np.ndarray,
        mu_2: np.ndarray,
        mu_3: np.ndarray,
        regularize_duals: bool = True,
        allow_accept_when_full: bool = True,
    ) -> tuple[float]:

    '''Compute the lagrangian of the optimization problem with gradients.\n
    `P`: Transition matrix of the system.
    `T`: Time horizon that defines the number of arrivals.
    `m_s_0`: Initial mass of the system.
    `b_low`: processing probability when action b=0.
    `b_high`: processing probability when action b=1.
    `alpha`: Budget for action a (load balancing).
    `beta`: Budget for action b (service rate).
    `gamma`: Cost for missed allocation job.
    `gamma_bar`: Cost for accepting when the queue is full.
    `Gamma_y`: L2 regularization term for the policy matrix.
    `Gamma_lambda`: L2 regularization term for the lambda (dual variables).
    `Gamma_mu`: L2 regularization term for the mu (dual variables).
    `y`: Policy matrix.
    `lambda_1`, `lambda_2`, `lambda_3`: Lagrange multipliers (lambda).
    `mu_1`, `mu_2`, `mu_3`: Lagrange multipliers (mu).
    `regularize_duals`: Apply regularization on the dual variables (Lagrange multipliers).
    `allow_accept_when_full`: Allow the system to accept a job when the queue is full.\n
    '''

    # Initialize the Lagrangian
    L = 0  # Lagrangian
    K = P.shape[0] - 1  # number of queues in the system (buffer size).
    cost_low = processing_cost(b_low)
    cost_high = processing_cost(b_high)
    cost_storage = [storage_cost(s, K) for s in range(K + 1)]

    # Compute the Lagrangian
    L += initial_constraints(y, mu_1, m_s_0)  # Initial condition constraints
    for t in range(T):  # Constraints for each time step
        L += cost_value(y, t, cost_storage, cost_low, cost_high, gamma, gamma_bar, allow_accept_when_full)
        L += dynamic_constraints(y, t, P, mu_2, mu_3)
        L += budget_constraints(y, t, alpha, beta, lambda_1, lambda_2)
        L -= positivity_constraints(y, t, lambda_3)
    L += regularization_policy(y, Gamma_y)  # Regularization on the policy matrix
    L -= regularization_dual(Gamma_lambda, lambda_1, lambda_2, lambda_3, Gamma_mu, mu_1, mu_2, mu_3) if regularize_duals else 0  # Regularization on the dual variables

    return L
