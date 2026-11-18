try:
    import numba
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False

from importlib import import_module


# The legacy public API is resolved lazily so that `import lplearn._legacy.core` (used by the
# experiment workers) does not pay for importing CVXPY, matplotlib and the legacy modules.
_LAZY = {
    'calculate_cost': ('.costs', 'calculate_cost'),
    'instantaneous_cost': ('.costs', 'instantaneous_cost'),
    'processing_cost': ('.costs', 'processing_cost'),
    'storage_cost': ('.costs', 'storage_cost'),
    'compute_lagrangian': ('.lagrangian', 'compute_lagrangian'),
    'solver': ('.methods', 'solver'),
    'learner': ('.methods', 'learner'),
    'Model': ('.model.model', 'Model'),
    'compute_U': ('.aggregation', 'compute_U'),
    'compute_V': ('.aggregation', 'compute_V'),
    'estimate_P': ('.probabilities', 'estimate_P'),
    'compute_P': ('.probabilities', 'compute_P'),
    'compute_Q': ('.probabilities', 'compute_Q'),
    'inst_prob': ('.probabilities', 'inst_prob'),
    'inst_MDP': ('.probabilities', 'inst_MDP'),
    'simulate_next_state': ('.probabilities', 'simulate_next_state'),
    'acceptance': ('.probabilities.status', 'acceptance'),
    'highness': ('.probabilities.status', 'highness'),
    'fullness': ('.probabilities.status', 'fullness'),
    'kappa': ('.utils', 'kappa'),
    'get_updated_params': ('.utils', 'get_updated_params'),
    'verify_aggregation': ('.utils', 'verify_aggregation'),
    'compare_results': ('.utils', 'compare_results'),
    'METHODS': ('.utils', 'METHODS'),
    'PARAMETERS_DESCRIPTION': ('.utils', 'PARAMETERS_DESCRIPTION'),
    'policy_pie_plot': ('.visualizations', 'policy_pie_plot'),
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        module, attr = _LAZY[name]
        value = getattr(import_module(module, __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
