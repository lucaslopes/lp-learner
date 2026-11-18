'''
This module contains the implementation of the MDP for the parallel queues with load balance and service rate selection problem.
'''


from lplearn._legacy.utils import numba_njit
from lplearn._legacy.probabilities.instant_probability import instantaneous_probability as inst_prob


@numba_njit
def instantaneous_markov_decision_process(
        case: int,
        K: int,
        p: float,
        s: int,
        s_: int,
        a: int,
        q: float,
        allow_accept_when_full: bool = True,
    ) -> float:

    '''
    Instantaneous MDP for the parallel queues with service rate selection problem.\n
    `case`: Transition probability case.\n
        - CJS (case 1): to process all jobs in the queue at each time instant (we have s + a − s′ such jobs and so consider a binomial distribution over them);\n
        - SJS (case 2): to process only one job in the queue at each time instant (we have s + a − s′ number of successes in time interval τ with success probability at each instance b).\n
    `K`: Maximum number of jobs in the queues.\n
    `p`: Arrival probability of the alpha N jobs.\n
    `s`: Current state of the system.\n
    `s_`: Next state.\n
    `a`: Load balancing action (0 to reject) or (1 to accept).\n
    `q`: Processing probability.\n
    `allow_accept_when_full`: Allow the system to accept a job when the queue is full.\n
    '''

    probability = 0.0

    valid_next_state = s_ <= min(a + s, K)
    valid_queue_size = s <= K if allow_accept_when_full else s + a <= K
    if valid_next_state and valid_queue_size:
        tau, delta = 1, 1.0  # we consider the time interval τ = 1,...,∞
        while tau <= 20 or delta > 1e-12:  # 20! is the maximum factorial that Numba can calculate; OR if the delta probability is too small, we stop the summation of probabilities
            delta = p * ((1 - p) ** (tau - 1))  # probability of the job arriving at time tau
            delta *= inst_prob(case, s, s_, (a if s != K else 0), q, tau)
            probability += delta
            tau += 1

    return probability
