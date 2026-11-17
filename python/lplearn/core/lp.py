'''
Matrix form of the finite-horizon relaxed LP (eq. general-lp) and interior-point solves.\n
Variable ordering:\n
- y is stored as y[t, s, u] and flattened in C order;\n
- equality rows (multiplier mu): initial rows s = 0..S-1, then flow rows (t, s) for t = 1..T-1;\n
- inequality rows (multiplier lambda >= 0): resource rows (l, t), then nonnegativity rows (t, s, u)
  written as -y <= 0.\n
The saddle variable w = (y, lambda, mu) is the concatenation of those blocks in that order.
'''


from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from .instance import WCMDP


@dataclass(frozen=True, eq=False)
class LPStructure:
    model: WCMDP
    A: sp.csr_matrix   # equality constraints A y = b
    b: np.ndarray
    D: sp.csr_matrix   # inequality constraints D y <= r (resources, then -y <= 0)
    r: np.ndarray
    c: np.ndarray      # flattened cost vector

    @property
    def T(self) -> int:
        return self.model.T

    @property
    def S(self) -> int:
        return self.model.S

    @property
    def U(self) -> int:
        return self.model.U

    @property
    def L(self) -> int:
        return self.model.L

    @property
    def n_y(self) -> int:
        return self.T * self.S * self.U

    @property
    def n_lam(self) -> int:
        return self.D.shape[0]

    @property
    def n_mu(self) -> int:
        return self.A.shape[0]

    @property
    def n_w(self) -> int:
        return self.n_y + self.n_lam + self.n_mu

    def split(self, w: np.ndarray) -> dict[str, np.ndarray]:
        '''Split a flat saddle vector into named, shaped blocks (views).'''
        T, S, U, L = self.T, self.S, self.U, self.L
        i = 0
        out = {}
        for name, shape in (
            ('y', (T, S, U)),
            ('lam_res', (L, T)),
            ('lam_nn', (T, S, U)),
            ('mu0', (S,)),
            ('muf', (T - 1, S)),
        ):
            n = int(np.prod(shape))
            out[name] = w[i:i + n].reshape(shape)
            i += n
        return out

    def join(self, y, lam_res, lam_nn, mu0, muf) -> np.ndarray:
        return np.concatenate([np.ravel(y), np.ravel(lam_res), np.ravel(lam_nn), np.ravel(mu0), np.ravel(muf)])

    def coordinate_blocks(self) -> dict[str, slice]:
        sizes = [('y', self.n_y), ('lam_res', self.L * self.T), ('lam_nn', self.n_y), ('mu0', self.S), ('muf', (self.T - 1) * self.S)]
        out, i = {}, 0
        for name, n in sizes:
            out[name] = slice(i, i + n)
            i += n
        return out

    def objective(self, y: np.ndarray) -> float:
        '''Unregularized LP objective c^T y for y[t, s, u].'''
        return float(self.c @ np.ravel(y))


def build_lp(model: WCMDP) -> LPStructure:
    T, S, U, L = model.T, model.S, model.U, model.L
    idx = np.arange(T * S * U).reshape(T, S, U)

    rows, cols, vals = [], [], []
    b = np.zeros(S + (T - 1) * S)
    for s in range(S):  # initial condition: sum_u y[0, s, u] = m0[s]
        rows += [s] * U
        cols += list(idx[0, s])
        vals += [1.0] * U
        b[s] = model.m0[s]
    for t in range(1, T):  # flow: sum_u y[t, s, u] - sum_{s', u} y[t-1, s', u] P[u, s', s] = 0
        for s in range(S):
            row = S + (t - 1) * S + s
            rows += [row] * U
            cols += list(idx[t, s])
            vals += [1.0] * U
            for s_prev in range(S):
                for u in range(U):
                    prob = model.P[u, s_prev, s]
                    if prob != 0.0:
                        rows.append(row)
                        cols.append(idx[t - 1, s_prev, u])
                        vals.append(-prob)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(len(b), T * S * U))
    A.sum_duplicates()

    rows, cols, vals = [], [], []
    r = np.zeros(L * T + T * S * U)
    for l in range(L):
        for t in range(T):
            row = l * T + t
            for s in range(S):
                for u in range(U):
                    if model.d[l, s, u] != 0.0:
                        rows.append(row)
                        cols.append(idx[t, s, u])
                        vals.append(model.d[l, s, u])
            r[row] = model.budgets[l]
    n_res = L * T
    rows += list(range(n_res, n_res + T * S * U))
    cols += list(range(T * S * U))
    vals += [-1.0] * (T * S * U)
    D = sp.csr_matrix((vals, (rows, cols)), shape=(len(r), T * S * U))

    c = np.broadcast_to(model.c, (T, S, U)).ravel().copy()
    return LPStructure(model=model, A=A, b=b, D=D, r=r, c=c)


def solve_lp(
        struct: LPStructure,
        Gamma_y: float | np.ndarray = 0.0,
        solver: str = 'CLARABEL',
        **solver_options,
    ) -> dict:
    '''
    Solve the relaxed LP, optionally with the primal regularization Gamma_y/2 ||y||^2
    (manuscript convention), with an interior-point solver (Clarabel by default).\n
    Returns y[t, s, u], the multipliers in the saddle convention of eq. (lagrangian-compact)
    (lambda >= 0 for D y <= r, mu for A y = b, so that c + Gamma_y y + D^T lambda + A^T mu = 0),
    the unregularized objective c^T y, the KKT stationarity residual and solver metadata.
    '''
    n_res = struct.L * struct.T
    y = cp.Variable(struct.n_y)
    eq = struct.A @ y == struct.b
    res = struct.D[:n_res] @ y <= struct.r[:n_res]
    nonneg = y >= 0
    gamma = np.broadcast_to(np.asarray(Gamma_y, dtype=float), (struct.n_y,))  # scalar or per-coordinate weights
    objective = struct.c @ y
    if np.any(gamma > 0):
        objective = objective + 0.5 * cp.sum(cp.multiply(gamma, cp.square(y)))
    problem = cp.Problem(cp.Minimize(objective), [eq, res, nonneg])
    start = perf_counter()
    problem.solve(solver=solver, **solver_options)
    wall = perf_counter() - start
    if problem.status not in ('optimal', 'optimal_inaccurate'):
        raise RuntimeError(f'LP solve failed with status {problem.status}')

    y_val = np.asarray(y.value)
    lam = np.concatenate([np.asarray(res.dual_value), np.asarray(nonneg.dual_value)])
    lam = np.maximum(lam, 0.0)
    mu = np.asarray(eq.dual_value)
    stationarity = struct.c + gamma * y_val + struct.D.T @ lam + struct.A.T @ mu
    if np.linalg.norm(stationarity) > 1e-4 * max(1.0, np.linalg.norm(struct.c)):
        mu = -mu  # guard against a sign convention change in CVXPY
        stationarity = struct.c + gamma * y_val + struct.D.T @ lam + struct.A.T @ mu
    w = struct.join(y_val, lam[:n_res], lam[n_res:], mu[: struct.S], mu[struct.S:])
    parts = struct.split(w)
    return {
        'y': parts['y'].copy(),
        'lam_res': parts['lam_res'].copy(),
        'lam_nn': parts['lam_nn'].copy(),
        'mu0': parts['mu0'].copy(),
        'muf': parts['muf'].copy(),
        'w': w,
        'objective': struct.objective(y_val),
        'reg_objective': float(problem.value),
        'Gamma_y': float(np.max(gamma)) if gamma.size else 0.0,
        'status': problem.status,
        'solver': problem.solver_stats.solver_name if problem.solver_stats else solver,
        'solve_time': float(problem.solver_stats.solve_time or np.nan) if problem.solver_stats else np.nan,
        'wall_time': wall,
        'kkt_stationarity': float(np.linalg.norm(stationarity)),
        'primal_residual_eq': float(np.linalg.norm(struct.A @ y_val - struct.b)),
        'primal_residual_ineq': float(np.maximum(struct.D @ y_val - struct.r, 0).max()),
    }
