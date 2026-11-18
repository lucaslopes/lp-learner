probabilities = {
    'case': 'System case (CJS=1, SJS=2)',
    'p': 'Probability of arrival',
    'b_low': 'Low service rate',
    'b_high': 'High service rate',
    'allow_accept_when_full': 'Whether to allow acceptance when the system is full',
    'aggregated_states': 'List of states aggregated',
}


constraints = {
    'K': 'Number of queues in the system (buffer size)',
    'T': 'Time horizon (number of arrivals)',
    'alpha': 'Batch size ratio',
    'beta': 'High service rate budget ratio',
    'gamma': 'Penalty for rejection',
    'gamma_bar': 'Penalty for accepting when full',
}


regularization = {
    'Gamma_y': 'Regularization parameter for y',
    'Gamma_lambda': 'Regularization parameter for lambda',
    'Gamma_mu': 'Regularization parameter for mu',
    'regularize_duals': 'Whether to regularize duals or not',
}


learning = {
    'epochs': 'Number of epochs to run the algorithm',
    'lr_descent': 'Learning rate for gradient descent',
    'lr_ascent': 'Learning rate for gradient ascent',
    'n_decays': 'Number of decays for the learning rate',
    'window': 'Window size for the learning rate decay',
    'max_norm_tolerance': 'Max norm tolerance for convergence',
    'seed': 'Random seed for reproducibility',
}


PARAMETERS_DESCRIPTION = {
    'Probabilities': probabilities,
    'Costs and Constraints': constraints,
    'Regularization': regularization,
    'Learning': learning
}
