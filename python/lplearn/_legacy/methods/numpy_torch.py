import torch
import numpy as np
from tqdm import tqdm
from numba.typed import List
from lplearn._legacy import compute_lagrangian
from lplearn._legacy.costs import calculate_cost
from lplearn._legacy.lagrangian import compute_lagrangian, compute_lagrangian_torch


def numpy_vs_torch(
        P: np.ndarray,
        K: int,
        T: int,
        initial_mass: np.ndarray,
        b_low: float,
        b_high: float,
        alpha: float,
        beta: float,
        gamma: float,
        gamma_bar: float,
        Gamma_y: float,
        Gamma_lambda: float,
        Gamma_mu: float,
        epochs: int = 1_000,
        lr_descent: float = 1e-1,
        lr_ascent: float = 1e-2,
        seed: int = 42,
        regularize_duals: bool = True,
        allow_accept_when_full: bool = True,
    ) -> bool:

    # Initialize the variables numpy
    torch.manual_seed(seed)
    y_torch = torch.rand((K+1, 2, 2, T), requires_grad=True, dtype=torch.float64)
    l1_torch = torch.rand((T), requires_grad=True, dtype=torch.float64)
    l2_torch = torch.rand((T), requires_grad=True, dtype=torch.float64)
    l3_torch = torch.rand((K+1, 2, 2, T), requires_grad=True, dtype=torch.float64)
    mu1_torch = torch.rand((K+1), requires_grad=True, dtype=torch.float64)
    mu2_torch = torch.rand((K+1, T), requires_grad=True, dtype=torch.float64)
    mu3_torch = torch.zeros((2,T), requires_grad=True, dtype=torch.float64) if allow_accept_when_full else torch.rand((2,T), requires_grad=True, dtype=torch.float64)

    # Initialize the variables torch
    y_np = y_torch.detach().numpy()
    l1_np = l1_torch.detach().numpy()
    l2_np = l2_torch.detach().numpy()
    l3_np = l3_torch.detach().numpy()
    mu1_np = mu1_torch.detach().numpy()
    mu2_np = mu2_torch.detach().numpy()
    mu3_np = mu3_torch.detach().numpy()

    # Run the algorithm
    for epoch in tqdm(range(epochs)):

        if y_torch.grad is not None:
            y_torch.zero_grad()
        if l1_torch.grad is not None:
            l1_torch.zero_grad()
        if l2_torch.grad is not None:
            l2_torch.zero_grad()
        if l3_torch.grad is not None:
            l3_torch.zero_grad()
        if mu1_torch.grad is not None:
            mu1_torch.zero_grad()
        if mu2_torch.grad is not None:
            mu2_torch.zero_grad()

        # Compute the lagrangian and the gradients
        lagrangian_data = compute_lagrangian(
            P, T, initial_mass, b_low, b_high,
            alpha, beta, gamma, gamma_bar,
            Gamma_y, Gamma_lambda, Gamma_mu,
            y_np, l1_np, l2_np, l3_np, mu1_np, mu2_np, mu3_np,
            regularize_duals, List([0]), allow_accept_when_full)

        lagrangian_torch = compute_lagrangian_torch(
            P, T, initial_mass, b_low, b_high,
            alpha, beta, gamma, gamma_bar,
            Gamma_y, Gamma_lambda, Gamma_mu,
            y_torch, l1_torch, l2_torch, l3_torch, mu1_torch, mu2_torch, mu3_torch,
            regularize_duals, allow_accept_when_full)

        # Check the lagrangian
        lagrangian = lagrangian_data[0]
        if lagrangian < -1e+10 or lagrangian > 1e+10:
            raise ValueError(f"Lagrangian is too large. Exiting at Epoch: {epoch} | Lagrangian: {lagrangian} | Cost: {cost}")
        gradients = {
            'y': (lagrangian_data[1]),
            'l1': lagrangian_data[2],
            'l2': lagrangian_data[3],
            'l3': lagrangian_data[4],
            'mu1': lagrangian_data[5],
            'mu2': lagrangian_data[6],
        }
        if not allow_accept_when_full:
            gradients['mu3'] = lagrangian_data[7]

        lagrangian_torch.backward() # compute the gradient

        abs_tol = 1e-10 # 1e-1
        compare_lagrangians = np.allclose(lagrangian, lagrangian_torch.item(), atol=abs_tol)
        compare_ys = np.allclose(y_np, y_torch.detach().numpy(), atol=abs_tol)
        compare_mu1 = np.allclose(mu1_np, mu1_torch.detach().numpy(), atol=abs_tol)
        compare_mu2 = np.allclose(mu2_np, mu2_torch.detach().numpy(), atol=abs_tol)
        if not allow_accept_when_full:
            compare_mu3 = np.allclose(mu3_np, mu3_torch.detach().numpy(), atol=abs_tol)
            compare_grad_mu3 = np.allclose(gradients['mu3'], mu3_torch.grad.detach().numpy(), atol=abs_tol)
        compare_l1 = np.allclose(l1_np, l1_torch.detach().numpy(), atol=abs_tol)
        compare_l2 = np.allclose(l2_np, l2_torch.detach().numpy(), atol=abs_tol)
        compare_l3 = np.allclose(l3_np, l3_torch.detach().numpy(), atol=abs_tol)
        compare_grad_y = np.allclose(gradients['y'], y_torch.grad.detach().numpy(), atol=abs_tol)
        compare_grad_mu1 = np.allclose(gradients['mu1'], mu1_torch.grad.detach().numpy(), atol=abs_tol)
        compare_grad_mu2 = np.allclose(gradients['mu2'], mu2_torch.grad.detach().numpy(), atol=abs_tol)
        compare_grad_l1 = np.allclose(gradients['l1'], l1_torch.grad.detach().numpy(), atol=abs_tol)
        compare_grad_l2 = np.allclose(gradients['l2'], l2_torch.grad.detach().numpy(), atol=abs_tol)
        compare_grad_l3 = np.allclose(gradients['l3'], l3_torch.grad.detach().numpy(), atol=abs_tol)
        cost_np = calculate_cost(y_np, b_low, b_high, gamma, gamma_bar, Gamma_y)
        cost_torch = calculate_cost(y_torch.detach().numpy(), b_low, b_high, gamma, gamma_bar, Gamma_y)
        compare_cost = np.allclose(cost_np, cost_torch, atol=abs_tol)

        if not (compare_lagrangians and compare_cost and compare_ys and compare_mu1 and compare_mu2 and compare_l1 and compare_l2 and compare_l3 and compare_grad_y and compare_grad_mu1 and compare_grad_mu2 and compare_grad_l1 and compare_grad_l2 and compare_grad_l3):
            print("Failed!!!")
            print("Epoch:", epoch)
            print("Lagrangian:", compare_lagrangians, lagrangian.item(), lagrangian_torch.item())
            print("cost:", compare_cost, cost_np, cost_torch)
            print("y:", compare_ys)
            print("mu1:", compare_mu1)
            print("mu2:", compare_mu2)
            if not allow_accept_when_full:
                print("mu3:", compare_mu3)
            print("l1:", compare_l1)
            print("l2:", compare_l2)
            print("l3:", compare_l3)
            print("y grad:", compare_grad_y)
            print("mu1 grad:", compare_grad_mu1)
            print("mu2 grad:", compare_grad_mu2)
            if not allow_accept_when_full:
                print("mu3 grad:", compare_grad_mu3)
            print("l1 grad:", compare_grad_l1)
            print("l2 grad:", compare_grad_l2)
            print("l3 grad:", compare_grad_l3)
            break

        # Update the variables
        y_np = (y_np - lr_descent * gradients['y'])
        l1_np = np.clip(l1_np + lr_ascent * gradients['l1'], a_min=0, a_max=None)
        l2_np = np.clip(l2_np + lr_ascent * gradients['l2'], a_min=0, a_max=None)
        l3_np = np.clip(l3_np + lr_ascent * gradients['l3'], a_min=0, a_max=None)
        mu1_np = (mu1_np + lr_ascent * gradients['mu1'])
        mu2_np = (mu2_np + lr_ascent * gradients['mu2'])
        if not allow_accept_when_full:
            mu3_np = (mu3_np + lr_ascent * gradients['mu3'])

        with torch.no_grad():
            y_torch = (y_torch - lr_descent * y_torch.grad).requires_grad_(True)
            l1_torch = torch.clamp(l1_torch + lr_ascent * l1_torch.grad, min=0).requires_grad_(True) # projection
            l2_torch = torch.clamp(l2_torch + lr_ascent * l2_torch.grad, min=0).requires_grad_(True) # projection
            l3_torch = torch.clamp(l3_torch + lr_ascent * l3_torch.grad, min=0).requires_grad_(True) # projection
            mu1_torch = (mu1_torch + lr_ascent * mu1_torch.grad).requires_grad_(True)
            mu2_torch = (mu2_torch + lr_ascent * mu2_torch.grad).requires_grad_(True)
            if not allow_accept_when_full:
                mu3_torch = (mu3_torch + lr_ascent * mu3_torch.grad).requires_grad_(True)

    return True
