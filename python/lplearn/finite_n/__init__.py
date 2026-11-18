'''
Finite-N policies and simulation for the queueing WC-MDP of Section 7 of the manuscript.

The LP occupation measures of Section 3 (detailed, or aggregate and lifted) are turned into feasible
controls for N arms, which are simulated with budgets enforced on every sample path.

Controller protocol
-------------------
A controller is an object with

``reset(N: int, rng: numpy.random.Generator) -> None``
    Called once per replication before stage 0. ``N`` is the number of arms and ``rng`` is the
    controller's generator for this replication.

``act(t: int, states: numpy.ndarray, rng: numpy.random.Generator) -> (a_req, b_req)``
    Stage ``t`` (0-based). ``states`` is an integer array of shape (N,) holding the queue lengths in
    0..K. The return value is a pair of int8 arrays of shape (N,) with the REQUESTED admission and
    high-rate decisions. Requests become realized actions only through a mechanism.

Optionally a controller has a ``counters`` dict. The simulator reads ``'repairs'`` (arms whose requested
action was changed by a feasibility repair) and ``'fallbacks'`` (stages in which an LP re-solve failed)
after each ``act`` call and records their per-stage increments.

Mechanism protocol
------------------
``name: str`` and ``apply(t, states, a_req, b_req, budgets_int, rng) -> (a, b, info)``. ``budgets_int`` is
``(floor(alpha N + 1e-9), floor(beta N + 1e-9))``, ``rng`` is the mechanism's generator, and ``info`` is a dict
with int entries ``'denied_adm'`` and ``'denied_high'``. Mechanisms are in ``lplearn.finite_n.mechanisms``:
``Unenforced`` (realized = requested), ``StrictBudget`` (realized = requested, raises on overload) and
``RandomRationing`` (uniform random rationing).

Common random numbers
---------------------
``simulate(model, make_controller, N, n_reps, seed, mechanism, rep_offset=0)`` uses, for replication
r = rep_offset + i, the streams ``[seed, r, 0]`` (transitions), ``[seed, r, 1]`` (controller),
``[seed, r, 2]`` (mechanism) and ``[seed, r, 3]`` (initial states). Replications with the same seed and index
see the same transition uniforms under every controller and mechanism.

Controllers
-----------
- ``LPProjectionController(pi, budgets)``: open-loop rounding and repair of a conditional policy ``pi[t, s, u]``.
- ``LPUpdateController(model, partition=None, Gamma_y=0.0)``: closed-loop LP re-solve at every stage.
- ``IndependentRandomizedController(pi)``: decentralized sampling with no coordination.

The construction, the cited sources and the deviations from them are in ``reports/finite_n/README.md``.
'''

from .mechanisms import MECHANISMS, RandomRationing, StrictBudget, Unenforced, integer_budgets
from .policies import (
    IndependentRandomizedController, LPProjectionController, LPUpdateController, assign_arms, normalize_policy,
    repair_counts, systematic_round,
)
from .simulate import simulate


__all__ = [
    'IndependentRandomizedController', 'LPProjectionController', 'LPUpdateController', 'MECHANISMS',
    'RandomRationing', 'StrictBudget', 'Unenforced', 'assign_arms', 'integer_budgets', 'normalize_policy',
    'repair_counts', 'simulate', 'systematic_round',
]
