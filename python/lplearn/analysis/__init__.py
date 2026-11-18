'''
Quantitative comparison metrics for the numerical evaluation of the manuscript (methods (A)-(K)).

The pure functions live in `lplearn.analysis.metrics`; the experiment stages `lp_solutions` and
`learners` compute the metrics in their `summarize` steps. Definitions: reports/metrics/README.md.
'''

from lplearn.analysis.metrics import (
    Problem,
    Reference,
    MethodRuns,
    comparison_latex,
    comparison_rows,
    feasibility,
    fluid_evaluate,
    frobenius,
    legacy_to_detailed,
    matrix_to_csv,
    mean_sd,
    pairwise_frobenius,
    pairwise_matrices,
    policy_distance,
    price_distance,
    reference_metrics,
    rows_to_csv,
    run_to_run,
    solution_metrics,
    summarize_runs,
    to_jsonable,
)

__all__ = [
    'Problem',
    'Reference',
    'MethodRuns',
    'comparison_latex',
    'comparison_rows',
    'feasibility',
    'fluid_evaluate',
    'frobenius',
    'legacy_to_detailed',
    'matrix_to_csv',
    'mean_sd',
    'pairwise_frobenius',
    'pairwise_matrices',
    'policy_distance',
    'price_distance',
    'reference_metrics',
    'rows_to_csv',
    'run_to_run',
    'solution_metrics',
    'summarize_runs',
    'to_jsonable',
]
