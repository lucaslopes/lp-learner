from .lagrangian import compute_lagrangian_value
from .lagrangian_gradients import compute_lagrangian_and_gradients as compute_lagrangian
try:  # PyTorch is optional; only the NumPy/PyTorch comparison path needs it.
    from .lagrangian_torch import compute_lagrangian_torch
except ImportError:  # pragma: no cover - depends on the local environment
    compute_lagrangian_torch = None
from .gradients import compute_grad_y, compute_grad_mu, compute_grad_lambda
import time
import numpy as np


# TODO: the new implementation is not matching with the old one
def compute_lagrangian_compare(
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

    '''
    Compute the lagrangian of the optimization problem with gradients.\n
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

    duration_a = time.time()
    lagrangian_data = compute_lagrangian(P, T, m_s_0, b_low, b_high, alpha, beta, gamma, gamma_bar, Gamma_y, Gamma_lambda, Gamma_mu, y, lambda_1, lambda_2, lambda_3, mu_1, mu_2, mu_3, regularize_duals, allow_accept_when_full)
    duration_a = time.time() - duration_a

    lgrn_grads_a = {
        'L': lagrangian_data[0],
        'y': lagrangian_data[1],
        'lambda_1': lagrangian_data[2],
        'lambda_2': lagrangian_data[3],
        'lambda_3': lagrangian_data[4],
        'mu_1': lagrangian_data[5],
        'mu_2': lagrangian_data[6],
        'mu_3': lagrangian_data[7]}

    # return lgrn_grads_a

    duration_b = time.time()
    L = compute_lagrangian_value(P, T, m_s_0, b_low, b_high, alpha, beta, gamma, gamma_bar, Gamma_y, Gamma_lambda, Gamma_mu, y, lambda_1, lambda_2, lambda_3, mu_1, mu_2, mu_3, regularize_duals, allow_accept_when_full)
    grad_mus = compute_grad_mu(y, m_s_0, P, Gamma_mu, mu_1, mu_2, mu_3, regularize_duals, allow_accept_when_full)
    grad_lambdas = compute_grad_lambda(y, alpha, beta, Gamma_lambda, lambda_1, lambda_2, lambda_3, regularize_duals, allow_accept_when_full)
    grad_y = compute_grad_y(y, P, b_low, b_high, gamma, gamma_bar, Gamma_y, lambda_1, lambda_2, lambda_3, mu_1, mu_2, mu_3, regularize_duals, allow_accept_when_full)
    duration_b = time.time() - duration_b

    lgrn_grads_b = {
        'L': L,
        'y': grad_y}
    for i in range(len(grad_lambdas)):
        lgrn_grads_b[f'lambda_{i+1}'] = grad_lambdas[i]
    for i in range(len(grad_mus)):
        lgrn_grads_b[f'mu_{i+1}'] = grad_mus[i]

    # if not np.allclose(lgrn_grads_a['L'], lgrn_grads_b['L']):
    #     print('Lagrangian value mismatch')
    #     print(f'Value A: {lgrn_grads_a["L"]}')
    #     print(f'Value B: {lgrn_grads_b["L"]}')
    # if not np.allclose(lgrn_grads_a['y'], lgrn_grads_b['y']):
    #     print('Gradient y mismatch')
    #     print(f'Value A: {lgrn_grads_a["y"]}')
    #     print(f'Value B: {lgrn_grads_b["y"]}')
    # if not np.allclose(lgrn_grads_a['lambda_1'], lgrn_grads_b['lambda_1']):
    #     print('Gradient lambda_1 mismatch')
    #     print(f'Value A: {lgrn_grads_a["lambda_1"]}')
    #     print(f'Value B: {lgrn_grads_b["lambda_1"]}')
    # if not np.allclose(lgrn_grads_a['lambda_2'], lgrn_grads_b['lambda_2']):
    #     print('Gradient lambda_2 mismatch')
    #     print(f'Value A: {lgrn_grads_a["lambda_2"]}')
    #     print(f'Value B: {lgrn_grads_b["lambda_2"]}')
    # if not np.allclose(lgrn_grads_a['lambda_3'], lgrn_grads_b['lambda_3']):
    #     print('Gradient lambda_3 mismatch')
    #     print(f'Value A: {lgrn_grads_a["lambda_3"]}')
    #     print(f'Value B: {lgrn_grads_b["lambda_3"]}')
    # if not np.allclose(lgrn_grads_a['mu_1'], lgrn_grads_b['mu_1']):
    #     print('Gradient mu_1 mismatch')
    #     print(f'Value A: {lgrn_grads_a["mu_1"]}')
    #     print(f'Value B: {lgrn_grads_b["mu_1"]}')
    # if not np.allclose(lgrn_grads_a['mu_2'], lgrn_grads_b['mu_2']):
    #     print('Gradient mu_2 mismatch')
    #     print(f'Value A: {lgrn_grads_a["mu_2"]}')
    #     print(f'Value B: {lgrn_grads_b["mu_2"]}')
    # if not np.allclose(lgrn_grads_a['mu_3'], lgrn_grads_b['mu_3']):
    #     print('Gradient mu_3 mismatch')
    #     print(f'Value A: {lgrn_grads_a["mu_3"]}')
    #     print(f'Value B: {lgrn_grads_b["mu_3"]}')

    print(f'Duration A: {duration_a}')
    print(f'Duration B: {duration_b}')
    print(f'New implementation is faster? {duration_b < duration_a}')

    return lgrn_grads_a
