'''
No-coordinator comparison: queues that optimise their own cost, against the LP-coordinated policy.

- ``selfish``: the private objective (backward induction on the single-arm model, no budgets), the price-taking
  variant with LP shadow prices, the tie-break rule and a fluid best-response check.
- ``mechanisms``: what happens when requests exceed the shared budgets (M0 unenforced, M1 rationing, M2 congestion),
  and the N -> infinity (fluid) evaluation under each mechanism.
- ``experiment``: the finite-N simulation with common random numbers, the populations and the job runner used by the
  ``commons`` stage (``configs/commons.toml``).

The modelling assumptions (mechanisms M0 to M2, the tie-break rule, the shadow-price accounting) are documented in the
module docstrings; the stage's results are summarised by ``lplearn.experiments.stages.commons``.
'''

from lplearn.commons.mechanisms import MECHANISM_NAMES, fluid_mechanism, make_mechanism
from lplearn.commons.selfish import PrivateSolution, private_dp, priced_solution, selfish_solution

__all__ = [
    'MECHANISM_NAMES', 'PrivateSolution', 'fluid_mechanism', 'make_mechanism', 'priced_solution', 'private_dp',
    'selfish_solution',
]
