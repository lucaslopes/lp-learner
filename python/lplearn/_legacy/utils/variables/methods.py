METHODS = {
    'all': {'cvxpy_unregul', 'cvxpy', 'gda', 'sgda'},
    'cvxpys': {'cvxpy_unregul', 'cvxpy'},
    'cvxpy': {'cvxpy'},  # Regularized CVXPY
    'cvxpy_unregul': {'cvxpy_unregul'},  # Original CVXPY without regularization nor aggregation
    'learners': {'gda', 'sgda'},
    'gda': {'gda'},
    'sgda': {'sgda'},
}
