'''
This module contains the implementation of the Linear Programming solver for the problem using the CVXPY library.
'''


import numpy as np
import cvxpy as cp
from lplearn._legacy.costs import processing_cost, storage_cost


def solver(
        P: np.ndarray,
        T: int,
        b_low: float,
        b_high: float,
        alpha: float,
        beta: float,
        gamma: float,
        gamma_bar: float,
        Gamma_y: float = 0,
        regularize_policy: bool = False,
        initial_mass: np.ndarray = None,
        allow_accept_when_full: bool = True,
    ) -> dict:

    '''This function solves the LP problem using the CVXPY library.
    `P`: Transition matrix of the system.
    `T`: Time horizon that defines the number of arrivals.
    `b_low`: processing probability when action b=0.
    `b_high`: processing probability when action b=1.
    `alpha`: Budget for action a (load balancing).
    `beta`: Budget for action b (service rate).
    `gamma`: Cost for missed allocation job.
    `gamma_bar`: Cost for accepting when the queue is full.
    `Gamma_y`: L2 regularization term for the policy matrix.
    `regularize_policy`: Apply regularization on the policy.
    `initial_mass`: Initial mass of the system.
    `allow_accept_when_full`: Allow the system to accept a job when the queue is full.\n
    '''

    K = P.shape[0] - 1  # number of queues in the system (buffer size).
    if initial_mass is None:
        initial_mass = np.zeros(K+1)
        initial_mass[0] = 1.0  # All queues start empty

    # Creating the Lagrange multipliers #
    l1 = np.zeros((T))
    l2 = np.zeros((T))
    l3 = np.zeros((K+1, 2, 2, T))
    mu1 = np.zeros((K+1))
    mu2 = np.zeros((K+1, T))
    if not allow_accept_when_full:
        mu3 = np.zeros((2, T))

    # Creating LP variables #
    y_00 = cp.Variable((K+1, T), name='y_00')  # y00_s(t) = fraction of queues in state s at time t with action a=0 and b=0
    y_01 = cp.Variable((K+1, T), name='y_01')  # y01_s(t) = fraction of queues in state s at time t with action a=0 and b=1
    y_10 = cp.Variable((K+1, T), name='y_10')  # y10_s(t) = fraction of queues in state s at time t with action a=1 and b=0
    y_11 = cp.Variable((K+1, T), name='y_11')  # y11_s(t) = fraction of queues in state s at time t with action a=1 and b=1

    # Cost function #
    cost = 0.0
    for t in range(T):
        for s in range(K+1):
            cost += (storage_cost(s, K) + processing_cost(b_low)) * (y_00[s,t] + y_10[s,t])  # storage and proceesing cost when b=0
            cost += (storage_cost(s, K) + processing_cost(b_high)) * (y_01[s,t] + y_11[s,t])  # storage and processing cost when b=1
            cost += gamma * (y_00[s,t] + y_01[s,t])  # missed allocation cost when a=0
            if allow_accept_when_full and s == K:  # TODO: check identation
                cost += gamma_bar * (y_10[K,t] + y_11[K,t])  # penalty for accepting when queue is full
            if regularize_policy:  # Regularization term on Y
                cost += Gamma_y * (y_00[s,t]**2 + y_01[s,t]**2 + y_10[s,t]**2 + y_11[s,t]**2)  # L2 norm

    # Constraints #
    constr, constr_info = [], []
    for t in range(T):

        # Constraints on all time instants #
        budget_a, budget_b = 0, 0
        for s in range(K+1):
            budget_a += y_10[s,t] + y_11[s,t]  # sum_{s,b} y_{s,1,b}(t) <= alpha
            budget_b += y_01[s,t] + y_11[s,t]  # sum_{s,a} y_{s,a,1}(t) <= beta
        constr.append(budget_a <= alpha)  # lambda_1(t)
        constr_info.append(('lambda', 1, (t)))
        constr.append(budget_b <= beta)  # lambda_2(t)
        constr_info.append(('lambda', 2, (t)))

        for s in range(K+1):
            # Constraints on all states at time instant t #

            if t == 0: # Intital condition
                # always start with empty queues
                constr.append(y_00[0,0] + y_01[0,0] + y_10[0,0] + y_11[0,0] == initial_mass[0])  # mu_1^0
                constr_info.append(('mu', 1, (0)))
                for s_ in range(1, K+1):
                    constr.append((y_00[s_,0] + y_01[s_,0] + y_10[s_,0] + y_11[s_,0]) == initial_mass[s_])  # mu_1^s_ where s_ other than 0
                    constr_info.append(('mu', 1, (s_,)))
            else:
                z = 0
                for s_ in range(K+1):
                    z = z + P[s, s_, 0, 0] * y_00[s_,t-1]  # a=0 and b=0
                    z = z + P[s, s_, 0, 1] * y_01[s_,t-1]  # a=0 and b=1
                    z = z + P[s, s_, 1, 0] * y_10[s_,t-1]  # a=1 and b=0
                    z = z + P[s, s_, 1, 1] * y_11[s_,t-1]  # a=1 and b=1
                constr.append((y_00[s,t] + y_01[s,t] + y_10[s,t] + y_11[s,t]) == z)  # mu_2^s(t)
                constr_info.append(('mu', 2, (s,t)))

            # y_{s,a,b}(t) >= 0
            constr.append(y_00[s,t]>=0)  # lambda_3^(s,0,0)(t)
            constr_info.append(('lambda', 3, (s,0,0,t)))
            constr.append(y_01[s,t]>=0)  # lambda_3^(s,0,1)(t)
            constr_info.append(('lambda', 3, (s,0,1,t)))
            constr.append(y_10[s,t]>=0)  # lambda_3^(s,1,0)(t)
            constr_info.append(('lambda', 3, (s,1,0,t)))
            constr.append(y_11[s,t]>=0)  # lambda_3^(s,1,1)(t)
            constr_info.append(('lambda', 3, (s,1,1,t)))

        # allocation action not available when the queue is full
        if not allow_accept_when_full:   # TODO: check identation
            constr.append(y_10[K, t] == 0)  # mu_3^0(t)
            constr_info.append(('mu', 3, (0, t)))
            constr.append(y_11[K, t] == 0)  # mu_3^1(t)
            constr_info.append(('mu', 3, (1, t)))

    # Solve the LP #
    problem = cp.Problem(cp.Minimize(cost), constr)
    try:
        solution = problem.solve(verbose=False)
    except cp.SolverError:
        solution = np.nan
        print('SolverError: The solver failed to solve the problem.')

    # Save the results #
    result = {
        'status': problem.status,
        'duration': problem.solver_stats.solve_time if problem.solver_stats else np.nan,
        'solution': solution,
        'y': np.nan}
    if solution is not np.nan:
        result['y'] = np.zeros((K+1, 2, 2, T))
        for s in range(K+1):
            for t in range(T):
                result['y'][s,0,0,t] = y_00.value[s,t]
                result['y'][s,0,1,t] = y_01.value[s,t]
                result['y'][s,1,0,t] = y_10.value[s,t]
                result['y'][s,1,1,t] = y_11.value[s,t]
        lambdas = [l1, l2, l3]
        mus = [mu1, mu2] if allow_accept_when_full else [mu1, mu2, mu3]
        dual_values = [c.dual_value for c in problem.constraints]
        for label, dual_value in zip(constr_info, dual_values):
            value = np.array(dual_value)
            if label[0] == 'mu':
                mus[label[1]-1][label[2]] = value
            elif label[0] == 'lambda':
                lambdas[label[1]-1][label[2]] = value
        result['lambda_1'], result['lambda_2'], result['lambda_3'] = lambdas[0], lambdas[1], lambdas[2]
        result['mu_1'], result['mu_2'] = mus[0], mus[1]
        if not allow_accept_when_full:
            result['mu_3'] = mus[2]

    return result
