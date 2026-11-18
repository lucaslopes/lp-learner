'''
This module contains the functions to compute the normalized transition probabilities.
'''


import numpy as np
from lplearn._legacy.utils import numba_njit
from lplearn._legacy.probabilities.simulation import simulate_next_state
from lplearn._legacy.probabilities.mdp import instantaneous_markov_decision_process as inst_MDP


@numba_njit
def normalize_probabilities(
        K: int,
        probabilities: np.ndarray,
    ) -> np.ndarray:

    '''
    Normalize
    '''

    P = probabilities.copy()
    for s in range(K+1):
        for a in range(2):
            for b in range(2):
                if not (s == K and a == 1):
                    P[:,s,a,b] = P[:,s,a,b] / (np.sum(P[:,s,a,b]) + 1e-6)

    return P


@numba_njit
def verify_normalized_probabilities(
        K: int,
        P: np.ndarray,
    ) -> bool:

    '''
    Verifying the normalized transition probabilities
    '''

    for s in range(K+1):
        for a in range(2):
            for b in range(2):
                if not (s == K and a == 1):
                    p_sum = np.sum(P[:,s,a,b])
                    if not (0.95 <= p_sum <= 1.05) and not (-0.05 <= p_sum <= 0.05):
                        print(f"Not a valid probability for P(s'|s={s},a={a},b={b}) = {p_sum}")
                        return False

    return True


@numba_njit
def compute_transition_probabilities(
        case: int,
        K: int,
        p: float,
        b_low: float,
        b_high: float,
        allow_accept_when_full: bool = True,
    ) -> np.ndarray:

    '''
    Compute the normalized transition probabilities.\n
    `case`: Transition probability case.\n
        - CJS (case 1): to process all jobs in the queue at each time instant;\n
        - SJS (case 2): to process only one job in the queue at each time instant.\n
    `K`: Maximum number of jobs in the queue.\n
    `p`: Arrival probability of the alpha N jobs.\n
    `b_low`: processing probability when action b=0.\n
    `b_high`: processing probability when action b=1.\n
    `allow_accept_when_full`: Allow the system to accept a job when the queue is full.\n
    '''

    # P(s'|s,a,b) = P[s',s,a,b]
    P = np.zeros((K+1, K+1, 2, 2))
    for s_ in range(K+1):
        for s in range(K+1):
            for a in range(2):
                for b in range(2):
                    q = b_low if b == 0 else b_high if b == 1 else None
                    P[s_,s,a,b] = inst_MDP(case, K, p, s, s_, a, q, allow_accept_when_full)
    P = normalize_probabilities(K, P)

    if verify_normalized_probabilities(K, P):
        return P


def compute_aggregated_transition_probabilities(
        P: np.ndarray,
        aggregated_states: np.ndarray,
    ) -> np.ndarray:

    '''
    Compute the aggregated transition probabilities.
    `P`: Transition probabilities.
    `aggregated_states`: Aggregated states.
    '''

    Q = np.zeros((len(aggregated_states), len(aggregated_states), 2, 2))

    for i in range(len(aggregated_states)):
        for j in range(len(aggregated_states)):
            for a in range(2):
                for b in range(2):
                    for s in aggregated_states[i]:
                        for s_ in aggregated_states[j]:
                            Q[j,i,a,b] += (P[s_,s,a,b] / len(aggregated_states[i]))

    return Q


def verify_agg_probs_via_sim(P, P_agg, K, agg_states, num_p_samples = 10_000):

    n_simulations = np.zeros((K+1, 2, 2))
    p_hat = np.zeros((K+1, K+1, 2, 2))

    n_simulations_agg = np.zeros((len(agg_states), 2, 2))
    p_hat_agg = np.zeros((len(agg_states), len(agg_states), 2, 2))

    for n in range(num_p_samples):
        for s in range(K+1):
            # Finding the aggregated state for s
            agg_s = None
            for i, p in enumerate(agg_states):
                if s in p:
                    agg_s = i
                    break
            for a in range(2):
                for b in range(2):
                    s_ = simulate_next_state(P, s, a, b)
                    p_hat[s_,s,a,b] += 1
                    n_simulations[s][a][b] += 1
                    # Finding the aggregated state for s_
                    agg_s_ = None
                    for i, p in enumerate(agg_states):
                        if s_ in p:
                            agg_s_ = i
                            break
                    p_hat_agg[agg_s_,agg_s,a,b] += 1
                    n_simulations_agg[agg_s][a][b] += 1

    for s_ in range(K+1):
        for s in range(K+1):
            for a in range(2):
                for b in range(2):
                    p_hat[s_,s,a,b] /= n_simulations[s][a][b]

    for s_ in range(len(agg_states)):
        for s in range(len(agg_states)):
            for a in range(2):
                for b in range(2):
                    p_hat_agg[s_,s,a,b] /= n_simulations_agg[s][a][b]

    norm_diff_p = np.linalg.norm(P - p_hat)
    norm_diff_q = np.linalg.norm(P_agg - p_hat_agg)

    return norm_diff_p, norm_diff_q


def compute_overline_P(
        P: np.ndarray,
        Q: np.ndarray,
        aggregation: list,
    ) -> np.ndarray:

    K = P.shape[0] - 1
    P_ = np.zeros((P.shape[0], P.shape[1], 2, 2))

    clusters = dict()
    for i, cluster in enumerate(aggregation):
        for state in cluster:
            clusters[state] = i

    for departure in range(K+1):
        cluster_departure = clusters[departure]
        for arrival in range(K+1):
            cluster_arrival = clusters[arrival]
            for a in range(2):
                for b in range(2):
                    q = Q[cluster_departure,cluster_arrival,a,b]
                    q /= len(aggregation[cluster_arrival])
                    P_[departure,arrival,a,b] = q

    return P_


def main():
    K = 4
    P = compute_transition_probabilities(
        case = 1,
        K = K,
        p = 0.6,
        b_low = 0.2,
        b_high = 0.5)
    agg_states = [[0,1,2], [3,4]]
    P_agg = compute_aggregated_transition_probabilities(P, agg_states)
    norm_diff_p, norm_diff_q = verify_agg_probs_via_sim(P, P_agg, K, agg_states)
    print("Norm Difference of P and P_hat:", norm_diff_p)
    print("Norm difference of Q and Q_hat:", norm_diff_q)


__name__ == '__main__' and main()
