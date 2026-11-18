'''
Regularized saddle point of the relaxed LP (Section 4), the projected descent-ascent operator G
of eq. (detailed-operator), its contraction certificates (Lemma 1), and the lifted aggregate
operator and one-step mismatch of Section 6.\n
Regularization follows the manuscript convention
L~ = L + Gamma_y/2 ||y||^2 - Gamma_lambda/2 ||lambda||^2 - Gamma_mu/2 ||mu||^2.
(The legacy `lp_learner` learner uses Gamma ||.||^2, i.e. legacy Gamma = manuscript Gamma / 2.)
'''


from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .aggregation import as_partition, block_sizes, maps
from .lp import LPStructure


@dataclass(frozen=True)
class Regularization:
    Gamma_y: float
    Gamma_lambda: float
    Gamma_mu: float

    @property
    def lower(self) -> float:
        return min(self.Gamma_y, self.Gamma_lambda, self.Gamma_mu)

    @property
    def upper(self) -> float:
        return max(self.Gamma_y, self.Gamma_lambda, self.Gamma_mu)

    def to_dict(self) -> dict:
        return {'Gamma_y': self.Gamma_y, 'Gamma_lambda': self.Gamma_lambda, 'Gamma_mu': self.Gamma_mu}


def regularization_vector(struct: LPStructure, reg: Regularization, block_weights: np.ndarray | None = None) -> np.ndarray:
    '''
    Diagonal of the regularization matrix R over the coordinates of w.\n
    With `block_weights` (one weight per aggregate state, e.g. block sizes) the state-indexed
    coordinates are reweighted: y by 1/weight and the state-indexed multipliers by weight.
    This is the metric induced on aggregate variables by the dual-consistent lifting.
    '''
    blocks = struct.coordinate_blocks()
    R = np.empty(struct.n_w)
    if block_weights is None:
        R[blocks['y']] = reg.Gamma_y
        R[blocks['lam_res']] = reg.Gamma_lambda
        R[blocks['lam_nn']] = reg.Gamma_lambda
        R[blocks['mu0']] = reg.Gamma_mu
        R[blocks['muf']] = reg.Gamma_mu
        return R
    wts = np.asarray(block_weights, dtype=float)
    T, S, U = struct.T, struct.S, struct.U
    w_tsu = np.broadcast_to(wts[None, :, None], (T, S, U)).ravel()
    R[blocks['y']] = reg.Gamma_y / w_tsu
    R[blocks['lam_res']] = reg.Gamma_lambda
    R[blocks['lam_nn']] = reg.Gamma_lambda * w_tsu
    R[blocks['mu0']] = reg.Gamma_mu * wts
    R[blocks['muf']] = reg.Gamma_mu * np.broadcast_to(wts[None, :], (T - 1, S)).ravel()
    return R


def skew_operator(struct: LPStructure) -> sp.csr_matrix:
    '''Linear part S of the signed saddle gradient F(w) = S w + (c, r, b) of the unregularized L.'''
    D, A = struct.D, struct.A
    n_y, n_l, n_m = struct.n_y, struct.n_lam, struct.n_mu
    Z = None
    return sp.bmat([[sp.csr_matrix((n_y, n_y)), D.T, A.T], [-D, sp.csr_matrix((n_l, n_l)), Z], [-A, Z, sp.csr_matrix((n_m, n_m))]], format='csr')


def kappa(struct: LPStructure) -> float:
    '''Exact Lipschitz constant of F: the spectral norm of the stacked constraint matrix [D; A].'''
    M = sp.vstack([struct.D, struct.A]).tocsc()
    if min(M.shape) <= 1500:
        return float(np.linalg.norm(M.toarray(), 2))
    return float(spla.svds(M, k=1, return_singular_vectors=False)[0])


def signed_gradient(struct: LPStructure, w: np.ndarray, R: np.ndarray) -> np.ndarray:
    '''F~(w) = (grad_y L~, -grad_lambda L~, -grad_mu L~) for the regularized Lagrangian.'''
    n_y, n_l = struct.n_y, struct.n_lam
    y, lam, mu = w[:n_y], w[n_y:n_y + n_l], w[n_y + n_l:]
    g_y = struct.c + struct.D.T @ lam + struct.A.T @ mu
    g_l = -(struct.D @ y - struct.r)
    g_m = -(struct.A @ y - struct.b)
    return np.concatenate([g_y, g_l, g_m]) + R * w


def G_step(struct: LPStructure, w: np.ndarray, R: np.ndarray, eta: float) -> np.ndarray:
    '''One projected descent-ascent step, eq. (detailed-operator), with a single step size eta.'''
    out = w - eta * signed_gradient(struct, w, R)
    n_y, n_l = struct.n_y, struct.n_lam
    out[n_y:n_y + n_l] = np.maximum(out[n_y:n_y + n_l], 0.0)
    return out


def regularized_saddle_point(
        struct: LPStructure,
        R: np.ndarray,
        y0: np.ndarray | None = None,
        tol: float = 1e-11,
        max_iter: int = 200,
    ) -> dict:
    '''
    Exact fixed point of G for R > 0: eliminate the multipliers,
    lambda = [D y - r]_+ / R_lambda and mu = (A y - b) / R_mu, and minimize the strongly convex
    piecewise-quadratic function
    phi(y) = c^T y + 1/2 y^T R_y y + 1/2 sum [D y - r]_+^2 / R_lambda + 1/2 sum (A y - b)^2 / R_mu
    with a damped semismooth Newton method (exact up to round-off once the active set settles).
    '''
    n_y, n_l = struct.n_y, struct.n_lam
    R_y, R_l, R_m = R[:n_y], R[n_y:n_y + n_l], R[n_y + n_l:]
    if np.any(R <= 0):
        raise ValueError('all regularization coefficients must be positive')
    D, A, r, b, c = struct.D, struct.A, struct.r, struct.b, struct.c
    AtA = (A.T @ sp.diags(1.0 / R_m) @ A).tocsc()
    Atb = A.T @ (b / R_m)

    def phi(y):
        viol = np.maximum(D @ y - r, 0.0)
        eq = A @ y - b
        return c @ y + 0.5 * y @ (R_y * y) + 0.5 * viol @ (viol / R_l) + 0.5 * eq @ (eq / R_m)

    def grad(y):
        viol = np.maximum(D @ y - r, 0.0)
        return c + R_y * y + D.T @ (viol / R_l) + AtA @ y - Atb

    y = np.zeros(n_y) if y0 is None else np.asarray(y0, dtype=float).ravel().copy()
    scale = max(1.0, np.linalg.norm(c))
    converged = False
    for it in range(max_iter):
        g = grad(y)
        if np.linalg.norm(g) <= tol * scale:
            converged = True
            break
        active = (D @ y - r) > 0
        Da = D[active]
        H = (sp.diags(R_y) + Da.T @ sp.diags(1.0 / R_l[active]) @ Da + AtA).tocsc()
        step = -spla.spsolve(H, g)
        f0, slope, t = phi(y), g @ step, 1.0
        while phi(y + t * step) > f0 + 1e-4 * t * slope and t > 1e-12:
            t *= 0.5
        y = y + t * step
    lam = np.maximum(D @ y - r, 0.0) / R_l
    mu = (A @ y - b) / R_m
    w = np.concatenate([y, lam, mu])
    eta = 1.0 / R.max()
    residual = np.linalg.norm(G_step(struct, w, R, eta) - w) / eta
    return {'w': w, 'iterations': it, 'converged': converged, 'grad_norm': float(np.linalg.norm(grad(y))), 'fixed_point_residual': float(residual)}


def lemma1_contraction(kappa_value: float, reg: Regularization, eta: float) -> float | None:
    '''c_G = 1 - eta (Gamma^- - kappa) when Lemma 1 applies (Gamma^- > kappa, 0 < eta <= 1/Gamma^+), else None.'''
    if reg.lower > kappa_value and 0 < eta <= 1.0 / reg.upper:
        return 1.0 - eta * (reg.lower - kappa_value)
    return None


def spectral_contraction(struct: LPStructure, R: np.ndarray, eta: float, skew: sp.csr_matrix | None = None) -> float:
    '''
    A valid Euclidean contraction constant of G: ||I - eta (diag(R) + S)||_2 (projection is
    nonexpansive). Unlike Lemma 1 it does not require Gamma^- > kappa.
    '''
    S = skew_operator(struct) if skew is None else skew
    M = sp.identity(struct.n_w, format='csr') - eta * (sp.diags(R) + S)
    if struct.n_w <= 2500:
        return float(np.linalg.norm(M.toarray(), 2))
    return float(spla.svds(M.tocsc(), k=1, return_singular_vectors=False, tol=1e-10)[0])


def eta_grid(kappa_value: float, R: np.ndarray, n: int = 25) -> np.ndarray:
    '''Log-spaced step sizes around eta_0 = Gamma^-/((Gamma^+)^2 + kappa^2), the optimal step for R = Gamma I.'''
    lo, hi = float(R.min()), float(R.max())
    eta0 = lo / (hi ** 2 + kappa_value ** 2)
    return np.unique(np.concatenate([eta0 * np.logspace(-1, 1, n), [min(1.0 / hi, 2.0 / (hi + kappa_value))]]))


def best_spectral_bound(struct: LPStructure, R: np.ndarray, residual_fn, etas: np.ndarray, skew: sp.csr_matrix | None = None) -> dict:
    '''
    Minimize the valid bound residual_fn(eta) / (1 - c(eta)) over `etas`, where c(eta) is the spectral
    contraction constant of G. Returns the best eta, c, residual and bound (inf if no eta contracts).
    '''
    S = skew_operator(struct) if skew is None else skew
    best = {'eta': np.nan, 'c': np.nan, 'residual': np.nan, 'bound': np.inf}
    for eta in etas:
        c = spectral_contraction(struct, R, float(eta), skew=S)
        if c >= 1:
            continue
        res = residual_fn(float(eta))
        bound = res / (1 - c)
        if bound < best['bound']:
            best = {'eta': float(eta), 'c': float(c), 'residual': float(res), 'bound': float(bound)}
    return best


# ---------------------------------------------------------------------------
# Aggregation of saddle variables and the lifted aggregate operator
# ---------------------------------------------------------------------------

CONVENTIONS = ('paper', 'consistent')


@dataclass(frozen=True, eq=False)
class SaddleMaps:
    '''
    Aggregation (detailed -> aggregate) and lifting (aggregate -> detailed) of w = (y, lambda, mu).\n
    'paper': V and U act on every state-indexed coordinate (y, nonnegativity multipliers, mu),
    as written in Section 6; resource multipliers are left unchanged.\n
    'consistent': y is aggregated by V and lifted by U, while state-indexed multipliers are
    aggregated by block averaging U^T and lifted by replication V^T (the adjoint pairing that
    makes L_hat(y_hat, lam_hat, mu_hat) = L(U y_hat, V^T lam_hat, V^T mu_hat)).
    '''

    detailed: LPStructure
    aggregate: LPStructure
    partition: tuple
    convention: str

    def _maps(self):
        U, V = maps(self.partition, self.detailed.S)
        if self.convention == 'paper':
            return (V, U), (V, U)
        if self.convention == 'consistent':
            return (V, U), (U.T, V.T)
        raise ValueError(f'convention must be one of {CONVENTIONS}')

    def to_aggregate(self, w: np.ndarray) -> np.ndarray:
        (Vy, _), (Vd, _) = self._maps()
        p = self.detailed.split(w)
        return self.aggregate.join(
            np.einsum('js,tsu->tju', Vy, p['y']), p['lam_res'],
            np.einsum('js,tsu->tju', Vd, p['lam_nn']), Vd @ p['mu0'], p['muf'] @ Vd.T)

    def to_detailed(self, w_hat: np.ndarray) -> np.ndarray:
        (_, Uy), (_, Ud) = self._maps()
        p = self.aggregate.split(w_hat)
        return self.detailed.join(
            np.einsum('sj,tju->tsu', Uy, p['y']), p['lam_res'],
            np.einsum('sj,tju->tsu', Ud, p['lam_nn']), Ud @ p['mu0'], p['muf'] @ Ud.T)

    def aggregate_regularization(self, reg: Regularization) -> np.ndarray:
        if self.convention == 'paper':
            return regularization_vector(self.aggregate, reg)
        return regularization_vector(self.aggregate, reg, block_weights=block_sizes(self.partition))


def saddle_maps(detailed: LPStructure, aggregate: LPStructure, partition: Sequence[Sequence[int]], convention: str = 'paper') -> SaddleMaps:
    return SaddleMaps(detailed, aggregate, as_partition(partition, detailed.S), convention)


def block_projector(sm: SaddleMaps, w: np.ndarray) -> np.ndarray:
    '''Orthogonal projection Pi = U V onto block-constant state-indexed coordinates (lambda_res unchanged).'''
    U, V = maps(sm.partition, sm.detailed.S)
    Pi = U @ V
    p = sm.detailed.split(w)
    return sm.detailed.join(
        np.einsum('rs,tsu->tru', Pi, p['y']), p['lam_res'], np.einsum('rs,tsu->tru', Pi, p['lam_nn']),
        Pi @ p['mu0'], p['muf'] @ Pi.T)


def lifted_G(sm: SaddleMaps, w: np.ndarray, R_hat: np.ndarray, eta: float, R: np.ndarray | None = None) -> np.ndarray:
    '''
    Lifted aggregate operator, eq. (lifted-operator).\n
    'paper': G_bar(w) = U G_hat(V w) with the maps of Section 6 (R_hat = aggregate regularization).\n
    'consistent': G_bar(w) = Pi G(Pi w), the detailed step restricted to block-constant vectors
    (needs the detailed regularization R). Its fixed points are the lifts, by `to_detailed`, of the
    saddle points of the aggregate problem regularized with `aggregate_regularization`.
    '''
    if sm.convention == 'consistent':
        if R is None:
            raise ValueError("the 'consistent' lifted operator needs the detailed regularization R")
        return block_projector(sm, G_step(sm.detailed, block_projector(sm, w), R, eta))
    return sm.to_detailed(G_step(sm.aggregate, sm.to_aggregate(w), R_hat, eta))


def mismatch(sm: SaddleMaps, w: np.ndarray, R: np.ndarray, R_hat: np.ndarray, eta: float) -> float:
    '''One-step aggregation mismatch M(w) = ||G(w) - G_bar(w)||, eq. (mismatch).'''
    return float(np.linalg.norm(G_step(sm.detailed, w, R, eta) - lifted_G(sm, w, R_hat, eta, R=R)))
