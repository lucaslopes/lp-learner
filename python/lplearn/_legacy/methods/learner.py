'''
This module contains the implementation of the stochastic gradient descent algorithm for the Lagrangian relaxation of the queueing problem.
'''


import numpy as np
from time import time
from tqdm import tqdm
from numba.typed import List
from lplearn._legacy import compute_lagrangian
from lplearn._legacy.utils import to_numba_list
from lplearn._legacy.costs import calculate_cost
from lplearn._legacy.probabilities import estimate_P
from lplearn._legacy.aggregation import reshape_state_space


def get_result_dict(initial_mass, aggregated_states, allow_accept_when_full):
    result = {
        'status': 'failed',
        'duration': time(),
        'initial_mass': initial_mass,
        'aggregated_states': aggregated_states if aggregated_states else None,
        'epoch': list(),
        'lr_descent': list(),
        'lr_ascent': list(),
        'lagrangian': list(),
        'solution': list(),
        'y': list(),
        'lambda_1': list(),
        'lambda_2': list(),
        'lambda_3': list(),
        'mu_1': list(),
        'mu_2': list(),
        'grad_y': list(),
        'grad_lambda_1': list(),
        'grad_lambda_2': list(),
        'grad_lambda_3': list(),
        'grad_mu_1': list(),
        'grad_mu_2': list()}
    if not allow_accept_when_full:
        result['mu_3'] = list()
        result['grad_mu_3'] = list()
    return result


def learner(
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
        aggregated_states: List | bool = False,
        U: np.ndarray = np.array([]),
        V: np.ndarray = np.array([]),
        estimate_probs: bool = False,
        epochs: int = 10_000,
        lr_descent: float = 1e-1,
        lr_ascent: float = 1e-2,
        n_decays: int = 0,
        window: int = None,
        seed: int = 42,
        max_norm_tolerance: float = 1e-2,
        result_cvxpy: dict | None = None,
        n_simulations: int = 10,
        regularize_duals: bool = True,
        allow_accept_when_full: bool = True,
        agg_on_all_epochs: bool = False,
        detail_first_epochs: bool = False,
    ) -> tuple[dict, dict]:

    '''
    This function runs the stochastic gradient descent algorithm for the Lagrangian relaxation of the queueing problem.\n
    `P`: np.transition probabilities P(s'|s,a,b) = P[s',s,a,b].
    `K`: number of queues in the system (buffer size).
    `T`: time horizon.
    `initial_mass`: Initial mass of the system.
    `b_low`: processing probability when action b=0.
    `b_high`: processing probability when action b=1.
    `alpha`: budget for load balancing.
    `beta`: budget for high service rate.
    `gamma`: Cost for missed allocation job.
    `gamma_bar`: Cost for accepting when the queue is full.
    `Gamma_y`: L2 regularization term for the policy matrix.
    `Gamma_lambda`: L2 regularization term for the lambda (dual variables).
    `Gamma_mu`: L2 regularization term for the mu (dual variables).
    `aggregated_states`: List of states aggregated.
    `U`: Map from the aggregated space to the original space.
    `V`: Map from original space to the aggregated space.
    `estimate_probs`: whether to estimate the transition probabilities or not.
    `epochs`: number of epochs to run the algorithm.
    `lr_descent`: learning rate for gradient descent.
    `lr_ascent`: learning rate for gradient ascent.
    `n_decays`: number of decays for the learning rate `window = int(1/(n_decays+1) * epochs)`.
    `window`: window size for the learning rate decay, number of last y's to store, and calculate the convergence.
    `seed`: random seed for reproducibility.
    `max_norm_tolerance`: This is the threshold for the maximum acceptable difference (norm) between the optimal and learned values. If the algorithm reduces the norm of the Cost and Policy to a value less than or equal to this threshold, it is considered that the algorithm has converged and will therefore stop.
    `result_cvxpy`: the result of the solver function.
    `n_simulations`: number of simulations to estimate the transition probabilities.
    `regularize_duals`: Apply regularization on the dual variables (Lagrange multipliers).
    `allow_accept_when_full`: Allow the system to accept a job when the queue is full.\n
    `agg_on_all_epochs`: Aggregate the state space on all epochs.\n
    `detail_first_epochs`: Store the results of the first `window` epochs.\n
    '''

    # Initialize the variables
    rng = np.random.default_rng(seed)
    y = rng.random((K+1, 2, 2, T)) if agg_on_all_epochs else rng.random((P.shape[0], 2, 2, T))
    lambda_1 = rng.random(T)
    lambda_2 = rng.random(T)
    lambda_3 = rng.random((K+1, 2, 2, T)) if agg_on_all_epochs else rng.random((P.shape[0], 2, 2, T))
    mu_1 = rng.random(K+1) if agg_on_all_epochs else rng.random(P.shape[0])
    mu_2 = rng.random((K+1, T)) if agg_on_all_epochs else rng.random((P.shape[0], T))
    mu_3 = np.zeros((2,T)) if allow_accept_when_full else rng.random((2, T))
    # if result_cvxpy:
    #     noise_level = .00001
    #     y = np.copy(result_cvxpy['y']) + rng.normal(0, noise_level, result_cvxpy['y'].shape)
    #     lambda_1 = np.copy(result_cvxpy['lambda_1']) + rng.normal(0, noise_level, result_cvxpy['lambda_1'].shape)
    #     lambda_2 = np.copy(result_cvxpy['lambda_2']) + rng.normal(0, noise_level, result_cvxpy['lambda_2'].shape)
    #     lambda_3 = np.copy(result_cvxpy['lambda_3']) + rng.normal(0, noise_level, result_cvxpy['lambda_3'].shape)
    #     mu_1 = np.copy(result_cvxpy['mu_1']) + rng.normal(0, noise_level, result_cvxpy['mu_1'].shape)
    #     mu_2 = np.copy(result_cvxpy['mu_2']) + rng.normal(0, noise_level, result_cvxpy['mu_2'].shape)
    #     mu_3 = np.zeros((2,T)) if allow_accept_when_full else np.copy(result_cvxpy['mu_3']) + rng.normal(0, noise_level, result_cvxpy['mu_3'].shape)

    # Set basic parameters
    # window_lr = int(1/(n_decays+1) * epochs)
    # convergence = [0,0]  # [epoch, occurences]
    if V.shape[0] == P.shape[0] and V.shape[1] == K+1:
        initial_mass = reshape_state_space(initial_mass, V)
    result = get_result_dict(initial_mass, aggregated_states, allow_accept_when_full)
    agg_states = to_numba_list(aggregated_states) if aggregated_states else List([0])
    ys_hist_size = 100_000
    last_ys = np.zeros((ys_hist_size,) + y.shape)
    idx = 0

    # Run the learning algorithm
    with tqdm(total=epochs, desc='Learning') as pbar:

        for epoch in range(epochs):

            # Compute the lagrangian and the gradients
            lagrangian_data = compute_lagrangian(
                (estimate_P(P, n_simulations, rng) if estimate_probs else P),
                T, initial_mass, b_low, b_high,
                alpha, beta, gamma, gamma_bar,
                Gamma_y, Gamma_lambda, Gamma_mu,
                (reshape_state_space(y, V) if agg_on_all_epochs else y),
                lambda_1, lambda_2,
                (reshape_state_space(lambda_3, V) if agg_on_all_epochs else lambda_3),
                (reshape_state_space(mu_1, V) if agg_on_all_epochs else mu_1),
                (reshape_state_space(mu_2, V) if agg_on_all_epochs else mu_2),
                mu_3, regularize_duals, agg_states, allow_accept_when_full)

            # lagrangian = lagrangian_data['L']
            # gradients = {k:v for k, v in lagrangian_data.items() if k != 'L'}

            # Check the lagrangian
            lagrangian = lagrangian_data[0]
            if lagrangian < -(2**63) or lagrangian > (2**63 - 1):
                # floor = np.floor(np.array(epoch / window_lr))
                # lr_descent /= (1. + (floor + 1.) * np.log(floor + 1.))
                # lr_ascent /= (floor + 1.)
                # print(f"Decayed the learning rates at Epoch: {epoch} | lr_descent: {lr_descent} | lr_ascent: {lr_ascent}")
                raise ValueError(f"Lagrangian is too large. Exiting at Epoch: {epoch} | Lagrangian: {lagrangian} | Cost: {cost}")

            # Get the gradients
            gradients = dict()
            gradients['y'] = (reshape_state_space(lagrangian_data[1], U) if agg_on_all_epochs else lagrangian_data[1])
            gradients['lambda_1'] = lagrangian_data[2]
            gradients['lambda_2'] = lagrangian_data[3]
            gradients['lambda_3'] = (reshape_state_space(lagrangian_data[4], U) if agg_on_all_epochs else lagrangian_data[4])
            gradients['mu_1'] = (reshape_state_space(lagrangian_data[5], U) if agg_on_all_epochs else lagrangian_data[5])
            gradients['mu_2'] = (reshape_state_space(lagrangian_data[6], U) if agg_on_all_epochs else lagrangian_data[6])
            if not allow_accept_when_full:
                gradients['mu_3'] = lagrangian_data[7]
            ####################################
            # gradients = {'y': compute_grad_y_np(y, lambda_1, lambda_2, lambda_3, mu_1, mu_2, probs)}
            # gradients['lambda_1'], gradients['lambda_2'], gradients['lambda_3'] = compute_grad_l_np(y, lambda_1, lambda_2, lambda_3)
            # gradients['mu_1'], gradients['mu_2'] = compute_grad_mu_np(y, mu_1, mu_2, probs)

            # Update the learning rates
            # v1, v2 = 1e-4, 1e-3
            # lr_descent = rng.uniform(v1, v2)
            # lr_ascent = rng.uniform(v1, v2)
            # if n_decays > 0 and epoch % window_lr == 0:
            #     print(f"Decaying the learning rates at Epoch: {epoch} | lr_descent: {lr_descent} | lr_ascent: {lr_ascent}")
            #     floor = np.floor(np.array(epoch / window_lr))
            #     lr_descent /= (1. + (floor + 1.) * np.log(floor + 1.))
            #     lr_ascent /= (floor + 1.)
            #     print(f"Decayed the learning rates at Epoch: {epoch} | lr_descent: {lr_descent} | lr_ascent: {lr_ascent}")

            # Update the cumulative sum of last y's to compute the mean without storing all y's
            if estimate_probs:
                last_ys[idx % ys_hist_size] = y
                idx += 1

            # Check the convergence
            # if not result_cvxpy and regularize_duals and convergence[1] > window:
            #     result['status'] = 'converged'
            # else:
            #     grad_y_sum = np.abs(np.round(np.sum(gradients['y']), 4))
            #     if np.abs(grad_y_sum) < max_norm_tolerance:  # TODO: actually the gradients can stagnate in any value, not only zero, so need to adapt this condition
            #         if epoch == convergence[0]:
            #             convergence[0] = epoch + 1
            #             convergence[1] += 1
            #         else:
            #             convergence = [epoch+1, 1]  # reset the convergence counter

            # Store the results
            if result['status'] == 'converged' or (epoch < window and detail_first_epochs) or epoch % window == 0 or epoch == epochs - 1:
                y_avg = np.mean(np.array(last_ys), axis=0) if estimate_probs else None
                y_compare = y_avg if estimate_probs and y_avg is not None and epoch > window else y
                cost = calculate_cost(y_compare, b_low, b_high, gamma, gamma_bar, Gamma_y, allow_accept_when_full=allow_accept_when_full)
                result['epoch'].append(epoch)
                result['lagrangian'].append(lagrangian)
                result['lr_descent'].append(lr_descent)
                result['lr_ascent'].append(lr_ascent)
                result['solution'].append(cost)
                result['y'].append(y_compare)
                result['lambda_1'].append(lambda_1)
                result['lambda_2'].append(lambda_2)
                result['lambda_3'].append(lambda_3)
                result['mu_1'].append(mu_1)
                result['mu_2'].append(mu_2)
                result['grad_y'].append(gradients['y'])
                result['grad_lambda_1'].append(gradients['lambda_1'])
                result['grad_lambda_2'].append(gradients['lambda_2'])
                result['grad_lambda_3'].append(gradients['lambda_3'])
                result['grad_mu_1'].append(gradients['mu_1'])
                result['grad_mu_2'].append(gradients['mu_2'])
                if not allow_accept_when_full:
                    result['mu_3'].append(mu_3)
                    result['grad_mu_3'].append(gradients['mu_3'])
                norm_y = None
                if result_cvxpy:
                    y_learner = y if result_cvxpy['y'].shape == y_compare.shape else reshape_state_space(y, V) if result_cvxpy['y'].shape[0] < y_compare.shape[0] else reshape_state_space(y, U)
                    lambda_1_learner = lambda_1
                    lambda_2_learner = lambda_2
                    lambda_3_learner = lambda_3 if result_cvxpy['lambda_3'].shape == lambda_3.shape else reshape_state_space(lambda_3, V) if result_cvxpy['lambda_3'].shape[0] < lambda_3.shape[0] else reshape_state_space(lambda_3, U)
                    mu_1_learner = mu_1 if result_cvxpy['mu_1'].shape == mu_1.shape else reshape_state_space(mu_1, V) if result_cvxpy['mu_1'].shape[0] < mu_1.shape[0] else reshape_state_space(mu_1, U)
                    mu_2_learner = mu_2 if result_cvxpy['mu_2'].shape == mu_2.shape else reshape_state_space(mu_2, V) if result_cvxpy['mu_2'].shape[0] < mu_2.shape[0] else reshape_state_space(mu_2, U)
                    cost_learner = calculate_cost(y_learner, b_low, b_high, gamma, gamma_bar, Gamma_y, allow_accept_when_full=allow_accept_when_full)
                    norm_cost = np.linalg.norm(cost_learner - result_cvxpy['solution'])
                    norm_y = np.linalg.norm(y_learner - result_cvxpy['y'])
                    lambda_1_norm = np.linalg.norm(lambda_1_learner - result_cvxpy['lambda_1'])
                    lambda_2_norm = np.linalg.norm(lambda_2_learner - result_cvxpy['lambda_2'])
                    lambda_3_norm = np.linalg.norm(lambda_3_learner - result_cvxpy['lambda_3'])
                    mu_1_norm = np.linalg.norm(mu_1_learner - result_cvxpy['mu_1'])
                    mu_2_norm = np.linalg.norm(mu_2_learner[:,1:] - result_cvxpy['mu_2'][:,1:])
                    mu_3_norm = np.linalg.norm(mu_3 - result_cvxpy['mu_3']) if not allow_accept_when_full else None
                    if not (epoch > 0 and epoch < window):
                        pbar.set_postfix(Cost_err=norm_cost, Y_err=norm_y, Y_norm=np.linalg.norm(y_compare), Y_grad=np.linalg.norm(y_compare), _Lambda_1_err=lambda_1_norm, _Lambda_2_err=lambda_2_norm, _Lambda_3_err=lambda_3_norm, _Mu_1_err=mu_1_norm, _Mu_2_err=mu_2_norm, _Lambda_1_grad=np.linalg.norm(gradients['lambda_1']), _Lambda_2_grad=np.linalg.norm(gradients['lambda_2']), _Lambda_3_grad=np.linalg.norm(gradients['lambda_3']), _Mu_1_grad=np.linalg.norm(gradients['mu_1']), _Mu_2_grad=np.linalg.norm(gradients['mu_2']), _Lambda_1_norm=np.linalg.norm(lambda_1), _Lambda_2_norm=np.linalg.norm(lambda_2), _Lambda_3_norm=np.linalg.norm(lambda_3), _Mu_1_norm=np.linalg.norm(mu_1), _Mu_2_norm=np.linalg.norm(mu_2), _Mu_3_err=mu_3_norm if mu_3_norm is not None else 0, _Mu_3_grad=np.linalg.norm(gradients['mu_3']) if 'mu_3' in gradients else 0, _Mu_3_norm=np.linalg.norm(mu_3) if mu_3_norm is not None else 0)
                if result['status'] == 'converged' or (norm_y is not None and norm_y <= max_norm_tolerance):
                    print(f"\n\nAlgorithm converged in epoch {epoch}!\n\n")
                    result['status'] = 'converged'
                    break

            # Update the variables
            y = (y - lr_descent * gradients['y'])
            lambda_1 = np.clip(lambda_1 + lr_ascent * gradients['lambda_1'], a_min=0, a_max=None)
            lambda_2 = np.clip(lambda_2 + lr_ascent * gradients['lambda_2'], a_min=0, a_max=None)
            lambda_3 = np.clip(lambda_3 + lr_ascent * gradients['lambda_3'], a_min=0, a_max=None)
            mu_1 = (mu_1 + lr_ascent * gradients['mu_1'])
            mu_2 = (mu_2 + lr_ascent * gradients['mu_2'])
            if not allow_accept_when_full:
                mu_3 = (mu_3 + lr_ascent * gradients['mu_3'])

            pbar.update(1)

    if not agg_on_all_epochs and len(U.shape) == 2:
        result['y'].append(reshape_state_space(result['y'][-1], U))
        result['lambda_1'].append(result['lambda_1'][-1])
        result['lambda_2'].append(result['lambda_2'][-1])
        result['lambda_3'].append(reshape_state_space(result['lambda_3'][-1], U))
        result['mu_1'].append(reshape_state_space(result['mu_1'][-1], U))
        result['mu_2'].append(reshape_state_space(result['mu_2'][-1], U))
        if not allow_accept_when_full:
            result['mu_3'].append(result['mu_3'][-1])

    result['status'] = 'finished' if result['status'] == 'failed' else result['status']
    result['duration'] = time() - result['duration']

    return result
