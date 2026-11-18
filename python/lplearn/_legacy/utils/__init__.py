from .calculations import factorial, area_from_radius, radius_from_area, kappa, matmul
from .calculations.binomial import binom_pmf, binom_cdf
from .dataframes import compare_results
from .decorators import numba_njit
from .inspectors import get_updated_params, get_obj_attr_by_func_params, verify_aggregation, to_numba_list
from .variables import palettes, METHODS, PARAMETERS_DESCRIPTION
