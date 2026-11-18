'''
Matrix-form core used by the numerical evaluation of the manuscript.\n
It re-implements the load-balancing / auto-scaling WC-MDP, its finite-horizon LP relaxation,
state aggregation, the regularized saddle point and GDA/SGDA directly in the manuscript's
notation, independently of the legacy `Model` API (which tests cross-check it against).
'''


from .instance import ACTIONS, ADMIT, HIGH, CASES, QueueParams, WCMDP, queue_wcmdp, transition_kernel, storage_cost, processing_cost
from .aggregation import PARTITIONS_K5, PARTITION_K11, aggregate, as_partition, block_sizes, deltas, lift_y, aggregate_y, lifted_kernel, maps, singleton_partition
from .lp import LPStructure, build_lp, solve_lp
from .saddle import (
    CONVENTIONS, Regularization, SaddleMaps, G_step, best_spectral_bound, block_projector, eta_grid, kappa, lemma1_contraction, lifted_G, mismatch,
    regularization_vector, regularized_saddle_point, saddle_maps, signed_gradient, skew_operator, spectral_contraction,
)
from .gda import run_gda, w_from_result
from .policy import conditional_policy, fluid_evaluation
