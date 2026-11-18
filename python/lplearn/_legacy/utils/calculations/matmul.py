import numpy as np
from lplearn._legacy.utils.decorators import numba_njit


@numba_njit
def matmul_numba(
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:

    '''
    Matrix multiplication using numba (similar to `numpy.matmul`) but only for 1D x 2D operations.
    `a`: Matrix A.
    `b`: Matrix B.\n
    '''

    var_1d = a if a.ndim == 1 else b
    var_2d = a if a.ndim == 2 else b
    # U.shape[0] > U.shape[1] # expand
    # V.shape[0] < V.shape[1] # aggregate

    result = np.zeros(var_2d.shape[0])

    for i in range(var_2d.shape[0]):
        for j in range(var_1d.shape[0]):
            result[i] += var_1d[j] * var_2d[i, j]

    return result

    # if a_was_1d and b_was_1d:
    #     return result[0, 0]  # Return a scalar
    # elif a_was_1d:
    #     return result[0, :]  # Return a 1D array
    # elif b_was_1d:
    #     return result[:, 0]  # Return a 1D array
    # else:
    #     return result        # Return a 2D array

a = np.random.rand(*(5,))
def main():
    shapes = [
        ((6,), (6, 3)),
        ((3,), (3, 6)),
        ((6, 3), (6,)),
        ((3, 6), (3,)),
    ]

    for shape_a, shape_b in shapes:
        a = np.random.rand(*shape_a)
        b = np.random.rand(*shape_b)

        result_numba = matmul_numba(a, b)
        result_numpy = np.matmul(b, a) if len(shape_a) == 2 and len(shape_b) == 1 else np.matmul(a, b)

        assert np.allclose(result_numba, result_numpy), f"Mismatch found for shapes {shape_a} and {shape_b}"

    print("All tests passed!")


__name__ == '__main__' and main()


# if var_1d == a and var_2d == b:
# var_1d @ var_2d OR var_1d.T @ var_2d OR var_2d.T @ var_1d

# if var_1d == b and var_2d == a:
# var_1d @ var_2d OR var_1d.T @ var_2d OR var_2d.T @ var_1d


# shapes[0]  # (3,)
# a = np.random.rand(*shapes[0][0]) # 1D
# b = np.random.rand(*shapes[0][1]) # 2D
# a @ b
# a.T @ b
# b.T @ a
# b @ a # error

# # shapes[1]  # (6,)
# a = np.random.rand(*shapes[1][0]) # 1D
# b = np.random.rand(*shapes[1][1]) # 2D
# a @ b
# a.T @ b
# b.T @ a
# b @ a # error

# # shapes[2]  # (3,)
# a = np.random.rand(*shapes[2][0]) # 2D
# b = np.random.rand(*shapes[2][1]) # 1D
# a.T @ b
# b.T @ a
# b @ a
# a @ b # error

# # shapes[3]  # (6,)
# a = np.random.rand(*shapes[3][0]) # 2D
# b = np.random.rand(*shapes[3][1]) # 1D
# a.T @ b
# b.T @ a
# b @ a
# a @ b # error


# (n,k),(k,m)->(n,m)


# Map
# a or b can be:
# - (C,S) or (S,C)

# Var
# a or b can be:
# - (S,) or (C,)

# Combinations
# - (S,) x (S,C) -> (C,)
# - (C,) x (C,S) -> (S,)
# - (S,C) x (S,) -> (C,)
# - (C,S) x (C,) -> (S,)
