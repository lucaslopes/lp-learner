import inspect
import numpy as np
from deepdiff import DeepDiff
from numba.typed import List
from lplearn._legacy.utils import numba_njit


def get_hist_var(
        attr: str,
    ) -> str:

    '''
    Given an attribute, returns the __hist_{attr}__.
    '''

    return f'__hist_{attr}__'


def get_obj_attr_by_func_params(
        obj: object,
        f: callable,
    ) -> dict:

    '''
    Given an object and a function, returns a dictionary with the parameters of the function as keys and the object attributes as values.
    '''

    p = inspect.signature(f).parameters  # get the parameters of the function
    d = {key: getattr(obj, key) for key in p if hasattr(obj, key)}  # get the object attributes

    return d


def get_updated_params(
        obj: object,
        attr: str,
        f: callable,
    ) -> tuple[dict | None, str]:

    '''
    Given an object, an attribute and a function, returns a dictionary with the parameters of the function as keys and the object attributes as values if this dictionary is different from the object's attribute hist_{attr}. Otherwise, returns None.
    '''

    current = get_obj_attr_by_func_params(obj, f)
    hist_var = get_hist_var(attr)

    if hasattr(obj, hist_var):
        hist = getattr(obj, hist_var)
        if not DeepDiff(current, hist):
            current = None

    return current, hist_var


def to_numba_list(
        input_list: list,
    ) -> List:

    '''
    Convert a list to a numba typed list.\n
    `input_list`: List to be converted.\n
    '''

    numba_list = List()

    if isinstance(input_list, List):
        numba_list = input_list
    elif all(isinstance(i, list) for i in input_list):
        for sublist in input_list:
            numba_sublist = List()
            for item in sublist:
                numba_sublist.append(item)
            numba_list.append(numba_sublist)
    elif isinstance(input_list, list):
        for item in input_list:
            numba_list.append(item)

    return numba_list


@numba_njit
def verify_aggregation(
        buffer_size: int,
        aggregated_states: List,
    ) -> bool:

    '''
    Verify if the aggregated states are valid.\n
    `buffer_size`: Maximum number of jobs in the queue.\n
    `aggregated_states`: List of states aggregated.\n
    '''

    pos = 0
    if isinstance(aggregated_states, List) and len(aggregated_states) > 0:
        for cluster in aggregated_states:
            if isinstance(cluster, List) and len(cluster) > 0:
                for state in cluster:
                    if state != pos:
                        return False
                    pos += 1
            else:
                return False
    else:
        return False

    if buffer_size + 1 != pos:
        return False

    return True
