import numpy as np
import lplearn._legacy as lpl
from lplearn._legacy import utils as _utils  # Initialize the module used through lpl.utils below.


class Model:


    def __init__(
            self: object,
            case: int = 1,
            p: float = 0.7,
            b_low: float = 0.01,
            b_high: float = 0.1,
            initial_mass: list[float] = None,
            aggregated_states: str | float | int | list[int] | bool = False,
            allow_accept_when_full: bool = True,
            K: int = 3,
            T: int = 7,
            alpha: float = 0.75,
            beta: float = 0.25,
            gamma: float = 100,
            gamma_bar: float = 100,
            Gamma_y: float = 0.5,
            Gamma_lambda: float = 10e-6,
            Gamma_mu: float = 10e-6,
            regularize_duals: bool = False,
            epochs: int = 1_000_000,
            lr_descent: float = 0.1,
            lr_ascent: float = 0.01,
            n_decays: int = 0,
            window: int = 10_000,
            agg_on_all_epochs: bool = False,
            max_norm_tolerance: float = 3e-2,
            seed: int = 42,
        ) -> None:

        '''
        Initialize the model with the given parameters:\n
        - `case` *(int)*: CJS=1, SJS=2.\n
        - `p` *(float)*: Probability of arrival.\n
        - `b_low` *(float)*: Low service rate.\n
        - `b_high` *(float)*: High service rate.\n
        - `initial_mass` *(list[float])*: Initial mass of the system.\n
        - `aggregated_states` *(str | float | int | list[int] | bool)*: Aggregated states.\n
        - `allow_accept_when_full` *(bool)*: Whether to allow acceptance when the system is full.\n
        - `K` *(int)*: Number of queues in the system (buffer size).\n
        - `T` *(int)*: Time horizon (number of arrivals).\n
        - `alpha` *(float)*: Batch size ratio.\n
        - `beta` *(float)*: High service rate budget ratio.\n
        - `gamma` *(float)*: Penalty for rejection.\n
        - `gamma_bar` *(float)*: Penalty for accepting when full.\n
        - `Gamma_y` *(float)*: Regularization parameter.\n
        - `Gamma_lambda` *(float)*: Regularization parameter.\n
        - `Gamma_mu` *(float)*: Regularization parameter.\n
        - `regularize_duals` *(bool)*: Whether to regularize Lagrange multipliers or not.\n
        - `epochs` *(int)*: Number of epochs to run the algorithm.\n
        - `lr_descent` *(float)*: Learning rate for gradient descent.\n
        - `lr_ascent` *(float)*: Learning rate for gradient ascent.\n
        - `n_decays` *(int)*: Whether to decay the learning rate or not.\n
        - `window` *(int)*: Window size for the learning rate decay.\n
        - `agg_on_all_epochs` *(bool)*: Aggregate the states on all epochs.\n
        - `max_norm_tolerance` *(float)*: Maximum norm tolerance.\n
        - `seed` *(int)*: Random seed for reproducibility.\n
        '''

        # Probabilities
        self.case = case
        self.p = p
        self.b_low = b_low
        self.b_high = b_high
        self.initial_mass = initial_mass
        self.allow_accept_when_full = allow_accept_when_full
        self.aggregated_states = aggregated_states

        # Costs and Constraints
        self.K = K
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.gamma_bar = gamma_bar

        # Regularization
        self.Gamma_y = Gamma_y
        self.Gamma_lambda = Gamma_lambda
        self.Gamma_mu = Gamma_mu
        self.regularize_duals = regularize_duals

        # Learning
        self.epochs = epochs
        self.lr_descent = lr_descent
        self.lr_ascent = lr_ascent
        self.n_decays = n_decays
        self.window = window
        self.agg_on_all_epochs = agg_on_all_epochs
        self.max_norm_tolerance = max_norm_tolerance
        self.seed = seed

        # Initialize the results
        self.P = np.array([])  # Original transition probabilities
        self.Q = np.array([])  # Aggregated transition probabilities
        self.U = np.array([0])  # Used to convert from the aggregated space to the original space
        self.V = np.array([0])  # Used to convert from original space to the aggregated space
        self.result_cvxpy = None
        self.result_cvxpy_unregul = None
        self.result_gda = None
        self.result_sgda = None
        self.result_gda_agg = None
        self.result_sgda_agg = None


    def __str__(
            self: object,
        ) -> str:

        return self.show_parameters(print_params=False)


    def __compute_inital_mass__(
            self: object,
        ) -> bool:

        if self.initial_mass is None or len(self.initial_mass) != self.K+1:
            init_mass = np.zeros(self.K+1)
            init_mass[0] = 1.0  # All queues start empty
        else:
            init_mass = np.array(self.initial_mass)
            init_mass /= np.sum(init_mass)  # Normalize the initial mass

        setattr(self, 'initial_mass', init_mass)

        return True


    def __compute_aggregated_states__(
            self: object,
        ) -> bool:

        success = False
        if self.aggregated_states:
            if type(self.aggregated_states) == str:  # Predefined aggregation
                if self.aggregated_states in {'empty_rest_full', 'empty_full_rest'}:  # 3 states: Empty, Rest and Full
                    self.aggregated_states = [[0], [i for i in range(1, self.K)], [self.K]] if self.K > 1 else [[0], [1]]
                    success = True
                elif self.aggregated_states in {'state_as_cluster', 'singleton'}:  # 3 states: Empty, Rest and Full
                    self.aggregated_states = [[i] for i in range(self.K+1)]  # Each state is a cluster (no aggregation)
                    success = True
                elif self.aggregated_states in {'half_k', 'half_K', 'half'}:
                    n_clusters = int((self.K + 1) / 2)
                    self.aggregated_states = [[] for _ in range(n_clusters)]
                    states_per_cluster = int((self.K+1) / n_clusters)
                    for i in range(n_clusters):
                        self.aggregated_states[i] = [j for j in range(i*states_per_cluster, (i+1)*states_per_cluster)]
                    success = True
                elif self.aggregated_states in {'third_k', 'third_K', 'third'}:
                    n_clusters = int((self.K + 1) / 3)
                    self.aggregated_states = [[] for _ in range(n_clusters)]
                    states_per_cluster = int((self.K+1) / n_clusters)
                    for i in range(n_clusters):
                        self.aggregated_states[i] = [j for j in range(i*states_per_cluster, (i+1)*states_per_cluster)]
                    success = True
            elif type(self.aggregated_states) == float and 0 <= self.aggregated_states <= 1:  # Threshold
                state_split = int((self.K + 1) * self.aggregated_states)
                self.aggregated_states = [i for i in range(state_split+1)] + [i for i in range(state_split+1, self.K+1)]
                success = True
            elif type(self.aggregated_states) == int and self.K+1 >= self.aggregated_states:  # Uniform aggregation over n clusters
                n_clusters = self.aggregated_states
                self.aggregated_states = [[] for _ in range(n_clusters)]
                states_per_cluster = int((self.K+1) / n_clusters)
                for i in range(n_clusters):
                    self.aggregated_states[i] = [j for j in range(i*states_per_cluster, (i+1)*states_per_cluster)]
                success = True
            elif type(self.aggregated_states) == list:  # Explicit define an aggregation
                aggregation = lpl.utils.to_numba_list(self.aggregated_states)
                if lpl.utils.verify_aggregation(self.K, aggregation):
                    success = True
            elif lpl.utils.verify_aggregation(self.K, self.aggregated_states):
                success = True

        if success:
            self.__compute_U__()
            self.__compute_V__()
            print(f"Aggregating states as {self.aggregated_states}")
            setattr(self, 'aggregated_states', lpl.utils.to_numba_list(self.aggregated_states))
        else:
            setattr(self, 'aggregated_states', False)
            setattr(self, 'agg_on_all_epochs', False)

        return success


    def __compute_V__(
            self: object,
            attr_name: str = 'V',
        ) -> bool:

        # params, hist_var = lpl.get_updated_params(self, attr_name, lpl.compute_V)
        # if params is not None:
        #     setattr(self, hist_var, params)
        #     setattr(self, attr_name, lpl.compute_V(**params))
        params = lpl.utils.get_obj_attr_by_func_params(self, lpl.compute_V)
        setattr(self, attr_name, lpl.compute_V(**params))

        return True


    def __compute_U__(
            self: object,
            attr_name: str = 'U',
        ) -> bool:

        # params, hist_var = lpl.get_updated_params(self, attr_name, lpl.compute_U)
        # if params is not None:
        #     setattr(self, hist_var, params)
        #     setattr(self, attr_name, lpl.compute_U(**params))
        params = lpl.utils.get_obj_attr_by_func_params(self, lpl.compute_U)
        setattr(self, attr_name, lpl.compute_U(**params))

        return True


    def __compute_kappa__(
            self: object,
            attr_name: str = 'kappa',
        ) -> bool:

        # params, hist_var = lpl.get_updated_params(self, attr_name, lpl.kappa)
        # if params is not None:
        #     setattr(self, hist_var, params)
        #     setattr(self, attr_name, lpl.kappa(**params))
        params = lpl.utils.get_obj_attr_by_func_params(self, lpl.kappa)
        setattr(self, attr_name, lpl.kappa(**params))

        return True


    def __compute_P__(
            self: object,
            attr_name: str = 'P',
        ) -> bool:

        # params, hist_var = lpl.get_updated_params(self, attr_name, lpl.compute_P)
        # if params is not None:
        #     setattr(self, hist_var, params)
        #     setattr(self, attr_name, lpl.compute_P(**params))
        params = lpl.utils.get_obj_attr_by_func_params(self, lpl.compute_P)
        setattr(self, attr_name, lpl.compute_P(**params))

        return True


    def __compute_Q__(
            self: object,
            attr_name: str = 'Q',
        ) -> bool:

        sucess = False

        if self.aggregated_states:
            # params, hist_var = lpl.get_updated_params(self, attr_name, lpl.compute_Q)
            # if params is not None:
            #     setattr(self, hist_var, params)
            #     setattr(self, attr_name, lpl.compute_Q(**params))
            params = lpl.utils.get_obj_attr_by_func_params(self, lpl.compute_Q)
            setattr(self, attr_name, lpl.compute_Q(**params))
            sucess = True

            for a in range(2):
                for b in range(2):
                    P = (self.P[:,:,a,b]).T
                    Q = (self.Q[:,:,a,b]).T
                    agg_P = (self.U.T) @ (P) @ (self.V.T)
                    if not np.allclose(agg_P, Q):  # Verify if the aggregated transition probabilities are correct
                        sucess = False
                        break

        return sucess


    def __solve_cvxpy__(
            self: object,
            regularize_policy: bool = False,
            attr_name: str = 'result_cvxpy',
        ) -> bool:

        attr_name += '_unregul' if not regularize_policy else ''
        setattr(self, 'regularize_policy', regularize_policy)

        current_P = getattr(self, 'P')
        if len(self.Q) > 0:
            setattr(self, 'P', self.Q)

        # params, hist_var = lpl.get_updated_params(self, attr_name, lpl.solver)
        # if params is not None:
        #     setattr(self, hist_var, params)
        #     setattr(self, attr_name, lpl.solver(**params))
        params = lpl.utils.get_obj_attr_by_func_params(self, lpl.solver)
        setattr(self, attr_name, lpl.solver(**params))

        setattr(self, 'P', current_P)
        if hasattr(self, 'regularize_policy'):
            delattr(self, 'regularize_policy')

        return True


    def __solve_learner__(
            self: object,
            estimate_probs: bool = False,
            attr_name: str = 'gda',
        ) -> dict:

        attr_name = 's' + attr_name if estimate_probs else attr_name
        attr_name = 'result_' + attr_name
        attr_name += '_agg' if self.aggregated_states else ''

        current_P = getattr(self, 'P')
        if len(self.Q) > 0:
            setattr(self, 'P', self.Q)
        setattr(self, 'estimate_probs', estimate_probs)

        # params, hist_var = lpl.get_updated_params(self, attr_name, lpl.learner)
        # if params is not None:
        #     setattr(self, hist_var, params)
        #     setattr(self, attr_name, lpl.learner(**params))
        params = lpl.utils.get_obj_attr_by_func_params(self, lpl.learner)
        setattr(self, attr_name, lpl.learner(**params))

        setattr(self, 'P', current_P)
        if hasattr(self, 'estimate_probs'):
            delattr(self, 'estimate_probs')


    def update_params(
            self: object,
            **kwargs: dict,
        ) -> None:

        '''
        Update the parameters of the model.
        This allows updating one or more parameters without resetting all.
        '''

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter '{key}' is not a valid model parameter.")


    def show_parameters(
            self: object,
            print_params: bool = True,
        ) -> str:

        '''
        Display the current parameters and their descriptions.
        '''

        string = '\n\nPARAMETERS DESCRIPTION\n\n'

        for category, params in lpl.PARAMETERS_DESCRIPTION.items():
            string += f"\n{category} Parameters:\n"
            for param, description in params.items():
                value = getattr(self, param)
                string += f"{param}:\t{value}\t# {description}\n"
            string += '\n'

        if print_params:
            print(string)

        return string


    def export(
            self: object,
        ) -> None:

        '''
        Export the model parameters as a dictonary.
        '''

        d = {
            'parameters': {category: {param: getattr(self, param) for param in params} for category, params in lpl.PARAMETERS_DESCRIPTION.items()},
            'probabilities': {'P': self.P if hasattr(self, 'P') else None, 'Q': self.Q if hasattr(self, 'Q') else None, 'U': self.U, 'V': self.V},
            'solutions': {},
        }

        for method in ['cvxpy_unregul', 'cvxpy', 'gda', 'sgda', 'gda_agg', 'sgda_agg']:
            if hasattr(self, f'result_{method}'):
                d['solutions'][method] = getattr(self, f'result_{method}')

        return d


    def solve(
            self: object,
            methods: set | str = 'cvxpy_unregul',
            verbose: bool = True,
            P: np.ndarray = None,
        ) -> object:

        '''
        Solve the LP problem and learn the policy.
        `methods`: Methods to use for solving the LP problem and learning the policy.
        '''

        self.__compute_inital_mass__()  # Initial mass of the system
        self.__compute_aggregated_states__()  # Aggregate the states (if applicable)
        self.__compute_kappa__()  # Compute the condition number
        if P is None:
            self.__compute_P__()  # Original transition probabilities
        else:
            setattr(self, 'P', P)
        self.__compute_Q__()  # Aggregated transition probabilities (if applicable)

        methods = lpl.METHODS[methods] if type(methods) == str else set(methods)

        # Solve the LP problem and learn the policy
        if 'cvxpy_unregul' in methods:  # Solve the LP problem (without regularization)
            if verbose:
                print('Solving with CVXPY (without any regularization)...')
            self.__solve_cvxpy__()
            if verbose:
                print(f'Done in {self.result_cvxpy_unregul['duration']} seconds with CVXPY (without any regularization).')
        if 'cvxpy' in methods:  # Solve the LP problem (with regularization)
            if verbose:
                print('Solving with CVXPY (regularized)...')
            self.__solve_cvxpy__(regularize_policy=True)
            if verbose:
                print(f'Done in {self.result_cvxpy['duration']} seconds with CVXPY (regularized).')
        if 'gda' in methods:  # Learn the policy using the GDA algorithm
            if verbose:
                print('Solving with GDA...')
            self.__solve_cvxpy__(regularize_policy=True)
            self.__solve_learner__()
            if verbose:
                attr_name = 'result_gda'
                attr_name += '_agg' if lpl.utils.verify_aggregation(self.K, self.aggregated_states) else ''
                print(f'Done in {getattr(self, attr_name)['duration']} seconds with GDA.')
        if 'sgda' in methods:  # Learn the policy using the SGDA algorithm
            if verbose:
                print('Solving with SGDA...')
            self.__solve_cvxpy__(regularize_policy=True)
            self.__solve_learner__(estimate_probs=True)
            if verbose:
                attr_name = 'result_sgda'
                attr_name += '_agg' if lpl.utils.verify_aggregation(self.K, self.aggregated_states) else ''
                print(f'Done in {getattr(self, attr_name)['duration']} seconds with GDA.')

        return self.export()
