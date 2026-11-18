'''
This module contains the function to compute the L of the optimization problem.
'''


import numpy as np
from lplearn._legacy.utils import numba_njit
from lplearn._legacy.costs import storage_cost, processing_cost


@numba_njit
def get_grad_y_sabt(
        y: np.ndarray,
        s: int,
        a: int,
        b: int,
        t: int,
        process: float,
        probs: np.ndarray,
        gamma: float,
        gamma_bar: float,
        Gamma_y: float,
        l1: np.ndarray,
        l2: np.ndarray,
        l3: np.ndarray,
        mu1: np.ndarray,
        mu2: np.ndarray,
        mu3: np.ndarray,
        regularize_duals=True,
        allow_accept_when_full=True,
    ) -> np.array:
    K, T = y.shape[0] - 1, y.shape[-1]
    grad_y_sabt = storage_cost(s, K) + processing_cost(process)
    grad_y_sabt += gamma * int(a==0)
    grad_y_sabt += gamma_bar * int(s==K and a==1)
    grad_y_sabt += mu1[s] * int(t==0)
    grad_y_sabt += mu2[s,t] * int(t>=1)
    if t < T - 1:
        for s_ in range(K+1):
            grad_y_sabt -= mu2[s_,t+1] * probs[s_,s,a,b]
    grad_y_sabt += l1[t] * int(a==1)
    grad_y_sabt += l2[t] * int(b==1)
    grad_y_sabt -= l3[s,a,b,t]
    grad_y_sabt += 2 * Gamma_y * y[s,a,b,t]
    return grad_y_sabt


@numba_njit
def compute_grad_y(
        y: np.ndarray,
        probs: np.ndarray,
        b_low: float,
        b_high: float,
        gamma: float,
        gamma_bar: float,
        Gamma_y: float,
        l1: np.ndarray,
        l2: np.ndarray,
        l3: np.ndarray,
        mu1: np.ndarray,
        mu2: np.ndarray,
        mu3: np.ndarray,
        regularize_duals=True,
        allow_accept_when_full=True,
    ) -> np.array:
    K, T = y.shape[0] - 1, y.shape[-1]
    grad_y = np.zeros_like(y)
    for t in range(T):
        for s in range(K+1):
            for a in range(2):
                for b in range(2):
                    process = b_low if b == 0 else b_high
                    grad_y[s,a,b,t] = get_grad_y_sabt(y, s, a, b, t, process, probs, gamma, gamma_bar, Gamma_y, l1, l2, l3, mu1, mu2, mu3)
    if mu3 is not None:
        grad_y[K,1,0,t] += mu3[0,t]
        grad_y[K,1,1,t] += mu3[1,t]
    return grad_y


@numba_njit
def compute_grad_mu(
        y: np.ndarray,
        m_s_0: np.ndarray,
        probs: np.ndarray,
        Gamma_mu: float,
        mu1: np.ndarray,
        mu2: np.ndarray,
        mu3: np.ndarray,
        regularize_duals: bool = True,
        allow_accept_when_full: bool = True
    ) -> np.array:
    K, T = y.shape[0] - 1, y.shape[-1]

    grad_mu1 = np.zeros_like(mu1)
    grad_mu2 = np.zeros_like(mu2)
    if mu3 is not None:
        grad_mu3 = np.zeros_like(mu3)
        grad_mu3 -= 2 * Gamma_mu * mu3

    for s in range(K+1):
        grad_mu1[s] = np.sum(y[s,:,:,0]) - m_s_0[s] - 2 * Gamma_mu * mu1[s]
        for t in range(1, T):
            temp = 0
            for s_ in range(K+1):
                for a in range(2):
                    for b in range(2):
                        temp += y[s_,a,b,t-1] * probs[s,s_,a,b]
            y_s_t = np.sum(y[s,:,:,t])
            grad_mu2[s,t] = y_s_t - temp - 2 * Gamma_mu * mu2[s,t]
            if mu3 is not None:
                grad_mu3[0, t] += y[K, 1, 0, t]
                grad_mu3[1, t] += y[K, 1, 1, t]

    return grad_mu1, grad_mu2, grad_mu3


@numba_njit
def compute_grad_lambda(
        y: np.ndarray,
        alpha: float,
        beta: float,
        Gamma_lambda: float,
        l1: np.ndarray,
        l2: np.ndarray,
        l3: np.ndarray,
        regularize_duals: bool = True,
    ) -> np.array:
    grad_l1 = np.zeros_like(l1)
    grad_l2 = np.zeros_like(l2)
    grad_l3 = - y - 2 * Gamma_lambda * l3

    for t in range(y.shape[-1]):
        grad_l1[t] = np.sum(y[:,1,:,t]) - alpha - 2 * Gamma_lambda * l1[t]
        grad_l2[t] = np.sum(y[:,:,1,t]) - beta - 2 * Gamma_lambda * l2[t]

    return grad_l1, grad_l2, grad_l3
