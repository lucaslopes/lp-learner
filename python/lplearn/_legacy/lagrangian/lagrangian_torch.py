'''
This module contains the function to compute the L of the optimization problem.
'''

import torch
from numba.typed import List
from lplearn._legacy.costs import processing_cost, storage_cost


def compute_lagrangian_torch(
        P: torch.Tensor,
        T: int,
        m_s_0: torch.Tensor,
        b_low: float,
        b_high: float,
        alpha: float,
        beta: float,
        gamma: float,
        gamma_bar: float,
        Gamma_y: float,
        Gamma_lambda: float,
        Gamma_mu: float,
        y: torch.Tensor,
        lambda_1: torch.Tensor,
        lambda_2: torch.Tensor,
        lambda_3: torch.Tensor,
        mu_1: torch.Tensor,
        mu_2: torch.Tensor,
        mu_3: torch.Tensor,
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

    # initial condition constraints
    y_0_0 = torch.sum(y[0,:,:,0])
    L += mu_1[0] * (y_0_0 - m_s_0[0])

    for s in range(1, K + 1):
        y_s_0 = torch.sum(y[s,:,:,0])
        L += mu_1[s] * (y_s_0 - m_s_0[s])

    for t in range(T):

        # cost function
        for s in range(K + 1):
            cost_storage = storage_cost(s, K)
            cost_low = processing_cost(b_low)
            cost_high = processing_cost(b_high)
            L += (cost_storage + cost_low) * (y[s, 0, 0, t] + y[s, 1, 0, t])  # storage and proceesing cost when b=0
            L += (cost_storage + cost_high) * (y[s, 0, 1, t] + y[s, 1, 1, t])  # storage and processing cost when b=1
            L += gamma * (y[s, 0, 0, t] + y[s, 0, 1, t])  # missed allocation cost when a=0
            if allow_accept_when_full and s == K:  # TODO: check identation
                L += gamma_bar * (y[K, 1, 0, t] + y[K, 1, 1, t])  # penalty for accepting when queue is full

        # other constraints
        for s in range(K + 1):
            if t >= 1:
                temp = 0
                for s_ in range(K + 1):
                    temp += P[s, s_, 0, 0] * y[s_, 0, 0, t - 1]
                    temp += P[s, s_, 0, 1] * y[s_, 0, 1, t - 1]
                    temp += P[s, s_, 1, 0] * y[s_, 1, 0, t - 1]
                    temp += P[s, s_, 1, 1] * y[s_, 1, 1, t - 1]
                m_s_t = y[s, 0, 0, t] + y[s, 0, 1, t] + y[s, 1, 0, t] + y[s, 1, 1, t]
                L += mu_2[s,t] * (m_s_t - temp)

        if not allow_accept_when_full:
            L += mu_3[0, t] * y[K, 1, 0, t] + mu_3[1, t] * y[K, 1, 1, t]

        # budget constraints
        temp_a, temp_b = 0, 0
        for s in range(K + 1):
            temp_a += y[s, 1, 0, t] + y[s, 1, 1, t]  # a = 1
            temp_b += y[s, 0, 1, t] + y[s, 1, 1, t]  # b = 1
        L += lambda_1[t] * (temp_a - alpha)
        L += lambda_2[t] * (temp_b - beta)

        # positivity constraints
        for s in range(K + 1):
            for a in range(2):
                for b in range(2):
                    L -= lambda_3[s,a,b,t] * y[s,a,b,t]

    # regularization on the policy
    L += Gamma_y * torch.sum(y**2)

    # regularization on the lagrange multipliers
    if regularize_duals:
        # regularization on the lambda
        L -= Gamma_lambda * torch.sum(lambda_1**2)
        L -= Gamma_lambda * torch.sum(lambda_2**2)
        L -= Gamma_lambda * torch.sum(lambda_3**2)

        # regularization on the mu
        L -= Gamma_mu * torch.sum(mu_1**2)
        L -= Gamma_mu * torch.sum(mu_2[:,1:]**2)
        if not allow_accept_when_full:
            L -= Gamma_mu * torch.sum(mu_3**2)

    return L
