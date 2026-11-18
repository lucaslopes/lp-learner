from lplearn._legacy import USE_NUMBA


def numba_njit(func):
    if USE_NUMBA:
        return __import__('numba').njit(func)
    else:
        return func
