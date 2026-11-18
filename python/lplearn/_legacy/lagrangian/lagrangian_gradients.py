'''
This module contains the function to compute the L of the optimization problem.
'''


import numpy as np
from numba.typed import List
from lplearn._legacy.utils import numba_njit, verify_aggregation
from lplearn._legacy.costs import processing_cost, storage_cost


@numba_njit
def compute_lagrangian_and_gradients(
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
        aggregated_states: List = List([0]),
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
    `gamma`: Cost for missed allocation job.m
    `gamma_bar`: Cost for accepting when the queue is full.
    `Gamma_y`: L2 regularization term for the policy matrix.
    `Gamma_lambda`: L2 regularization term for the lambda (dual variables).
    `Gamma_mu`: L2 regularization term for the mu (dual variables).
    `y`: Policy matrix.
    `lambda_1`, `lambda_2`, `lambda_3`: Lagrange multipliers (lambda).
    `mu_1`, `mu_2`, `mu_3`: Lagrange multipliers (mu).
    `regularize_duals`: Apply regularization on the dual variables (Lagrange multipliers).
    `aggregated_states`: List of states aggregated.
    `allow_accept_when_full`: Allow the system to accept a job when the queue is full.\n
    '''

    L = 0
    K = P.shape[0] - 1  # number of queues in the system (buffer size).
    valid_aggregation = verify_aggregation(K, aggregated_states)

    # Initialize gradients
    y_grad = np.zeros_like(y)
    lambda_1_grad = np.zeros_like(lambda_1)
    lambda_2_grad = np.zeros_like(lambda_2)
    lambda_3_grad = np.zeros_like(lambda_3)
    mu_1_grad = np.zeros_like(mu_1)
    mu_2_grad = np.zeros_like(mu_2)
    if not allow_accept_when_full:
        mu_3_grad = np.zeros_like(mu_3)

    # initial condition constraints
    y_0_0 = np.sum(y[0,:,:,0])
    L += mu_1[0] * (y_0_0 - m_s_0[0])
    mu_1_grad[0] += (y_0_0 - m_s_0[0])
    y_grad[0,0,0,0] += mu_1[0]
    y_grad[0,0,1,0] += mu_1[0]
    y_grad[0,1,0,0] += mu_1[0]
    y_grad[0,1,1,0] += mu_1[0]

    for s in range(1, K + 1):
        y_s_0 = np.sum(y[s,:,:,0])
        L += mu_1[s] * (y_s_0 - m_s_0[s])
        mu_1_grad[s] += (y_s_0 - m_s_0[s])
        y_grad[s,0,0,0] += mu_1[s]
        y_grad[s,0,1,0] += mu_1[s]
        y_grad[s,1,0,0] += mu_1[s]
        y_grad[s,1,1,0] += mu_1[s]

    for t in range(T):

        # cost function
        for s in range(K + 1):
            cost_storage = storage_cost(aggregated_states[s], K) if valid_aggregation else storage_cost(s, K)
            cost_low = processing_cost(b_low)
            cost_high = processing_cost(b_high)
            L += (cost_storage + cost_low) * (y[s, 0, 0, t] + y[s, 1, 0, t])  # storage and proceesing cost when b=0
            L += (cost_storage + cost_high) * (y[s, 0, 1, t] + y[s, 1, 1, t])  # storage and processing cost when b=1
            L += gamma * (y[s, 0, 0, t] + y[s, 0, 1, t])  # missed allocation cost when a=0
            y_grad[s, 0, 0, t] += cost_storage + cost_low + gamma
            y_grad[s, 0, 1, t] += cost_storage + cost_high + gamma
            y_grad[s, 1, 0, t] += cost_storage + cost_low
            y_grad[s, 1, 1, t] += cost_storage + cost_high
            if allow_accept_when_full and s == K:  # TODO: check identation
                L += gamma_bar * (y[K, 1, 0, t] + y[K, 1, 1, t])  # penalty for accepting when queue is full
                y_grad[K, 1, 0, t] += gamma_bar
                y_grad[K, 1, 1, t] += gamma_bar

        # other constraints
        for s in range(K + 1):
            if t >= 1:
                temp = 0
                for s_ in range(K + 1):
                    temp += P[s, s_, 0, 0] * y[s_, 0, 0, t - 1]
                    temp += P[s, s_, 0, 1] * y[s_, 0, 1, t - 1]
                    temp += P[s, s_, 1, 0] * y[s_, 1, 0, t - 1]
                    temp += P[s, s_, 1, 1] * y[s_, 1, 1, t - 1]
                    y_grad[s_, 0, 0, t - 1] -= P[s, s_, 0, 0] * mu_2[s, t]
                    y_grad[s_, 0, 1, t - 1] -= P[s, s_, 0, 1] * mu_2[s, t]
                    y_grad[s_, 1, 0, t - 1] -= P[s, s_, 1, 0] * mu_2[s, t]
                    y_grad[s_, 1, 1, t - 1] -= P[s, s_, 1, 1] * mu_2[s, t]
                m_s_t = y[s, 0, 0, t] + y[s, 0, 1, t] + y[s, 1, 0, t] + y[s, 1, 1, t]
                L += mu_2[s,t] * (m_s_t - temp)
                mu_2_grad[s, t] += (m_s_t - temp)
                y_grad[s, 0, 0, t] += mu_2[s, t]
                y_grad[s, 0, 1, t] += mu_2[s, t]
                y_grad[s, 1, 0, t] += mu_2[s, t]
                y_grad[s, 1, 1, t] += mu_2[s, t]

        if not allow_accept_when_full:
            L += mu_3[0, t] * y[K, 1, 0, t] + mu_3[1, t] * y[K, 1, 1, t]
            mu_3_grad[0, t] += y[K, 1, 0, t]
            mu_3_grad[1, t] += y[K, 1, 1, t]
            y_grad[K, 1, 0, t] += mu_3[0, t]
            y_grad[K, 1, 1, t] += mu_3[1, t]

        # budget constraints
        temp_a, temp_b = 0, 0
        for s in range(K + 1):
            temp_a += y[s, 1, 0, t] + y[s, 1, 1, t]  # a = 1
            temp_b += y[s, 0, 1, t] + y[s, 1, 1, t]  # b = 1
        L += lambda_1[t] * (temp_a - alpha)
        L += lambda_2[t] * (temp_b - beta)
        lambda_1_grad[t] += (temp_a - alpha)
        lambda_2_grad[t] += (temp_b - beta)
        y_grad[:, 1, 0, t] += lambda_1[t]
        y_grad[:, 1, 1, t] += lambda_1[t]
        y_grad[:, 0, 1, t] += lambda_2[t]
        y_grad[:, 1, 1, t] += lambda_2[t]

        # positivity constraints
        for s in range(K + 1):
            for a in range(2):
                for b in range(2):
                    L -= lambda_3[s,a,b,t] * y[s,a,b,t]
                    lambda_3_grad[s, a, b, t] -= y[s, a, b, t]
                    y_grad[s, a, b, t] -= lambda_3[s, a, b, t]

    # regularization on the policy
    L += Gamma_y * np.sum(y**2)
    y_grad += 2 * Gamma_y * y

    # regularization on the lagrange multipliers
    if regularize_duals:
        # regularization on the lambda
        L -= Gamma_lambda * np.sum(lambda_1**2)
        lambda_1_grad -= 2 * Gamma_lambda * lambda_1
        L -= Gamma_lambda * np.sum(lambda_2**2)
        lambda_2_grad -= 2 * Gamma_lambda * lambda_2
        L -= Gamma_lambda * np.sum(lambda_3**2)
        lambda_3_grad -= 2 * Gamma_lambda * lambda_3

        # regularization on the mu
        L -= Gamma_mu * np.sum(mu_1**2)
        mu_1_grad -= 2 * Gamma_mu * mu_1
        L -= Gamma_mu * np.sum(mu_2[:,1:]**2)
        mu_2_grad[:,1:] -= 2 * Gamma_mu * mu_2[:,1:]
        if not allow_accept_when_full:
            L -= Gamma_mu * np.sum(mu_3**2)
            mu_3_grad -= 2 * Gamma_mu * mu_3

    return L, y_grad, lambda_1_grad, lambda_2_grad, lambda_3_grad, mu_1_grad, mu_2_grad, mu_3_grad if not allow_accept_when_full else np.zeros((2,T))
