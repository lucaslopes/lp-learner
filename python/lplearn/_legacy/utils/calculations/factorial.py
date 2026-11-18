'''
This module contains functions for calculations that are not available in Numba.
Ranking of methods by their mean duration:
classic: 5.803108215332031e-06
recursive: 9.169578552246094e-06
recursive_numba: 0.00019863605499267578
classic_numba: 0.0019834136962890627
'''


import time
from lplearn._legacy.utils.decorators import numba_njit


def factorial_range(n: int) -> int:
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


@numba_njit
def factorial_range_numba(n: int) -> int:
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def factorial_recursive(n: int) -> int:
    if n == 0:
        return 1
    else:
        return n * factorial_recursive(n - 1)


@numba_njit
def factorial_recursive_numba(n: int) -> int:
    if n == 0:
        return 1
    else:
        return n * factorial_recursive_numba(n - 1)


############################


def main():

    duration_classic = list()
    for _ in range(100):
        duration = time.time()
        for i in range(21):
            r = factorial_range(i)
        duration = time.time() - duration
        duration_classic.append(duration)

    duration_classic_numba = list()
    for _ in range(100):
        duration = time.time()
        for i in range(21):
            r = factorial_range_numba(i)
        duration = time.time() - duration
        duration_classic_numba.append(duration)

    duration_recursive = list()
    for _ in range(100):
        duration = time.time()
        for i in range(21):
            r = factorial_recursive(i)
        duration = time.time() - duration
        duration_recursive.append(duration)

    duration_recursive_numba = list()
    for _ in range(100):
        duration = time.time()
        for i in range(21):
            r = factorial_recursive_numba(i)
        duration = time.time() - duration
        duration_recursive_numba.append(duration)

    mean_classic = sum(duration_classic) / len(duration_classic)
    mean_recursive = sum(duration_recursive) / len(duration_recursive)
    mean_classic_numba = sum(duration_classic_numba) / len(duration_classic_numba)
    mean_recursive_numba = sum(duration_recursive_numba) / len(duration_recursive_numba)

    methods = {
        'classic': mean_classic,
        'classic_numba': mean_classic_numba,
        'recursive': mean_recursive,
        'recursive_numba': mean_recursive_numba
    }

    sorted_methods = sorted(methods.items(), key=lambda x: x[1])

    print("Ranking of methods by their mean duration:")
    for method, mean in sorted_methods:
        print(f'{method}: {mean}')


__name__ == '__main__' and main()
