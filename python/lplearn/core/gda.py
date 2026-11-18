'''
Gradient descent-ascent (GDA) and stochastic GDA (SGDA) on the regularized Lagrangian
(manuscript convention, Gamma/2 ||.||^2), as used by methods (C)-(K).\n
GDA uses the exact kernel P. SGDA replaces P, at every iteration, by an empirical kernel built
from `R` fresh next-state samples for every (u, s) pair (sampled from the kernel of the problem
being solved, i.e. the aggregate kernel for aggregate methods). All gradients are evaluated at
the current point and the update is simultaneous; multipliers of inequality constraints are
projected on the nonnegative orthant. Step sizes are constant: eta_y for y, eta_dual for the
multipliers. The returned "average" is the Polyak-Ruppert average of the iterates from
`avg_start` on (tail averaging).
'''


from __future__ import annotations

from time import perf_counter

import numpy as np
from numba import njit

from .instance import WCMDP
from .lp import LPStructure


@njit(cache=True)
def _gradients(P, c, d, budgets, m0, y, lam_res, lam_nn, mu0, muf, Gy, Gl, Gn, Gm, gy, glr, gln, gm0, gmf):
    T, S, U = y.shape
    L = d.shape[0]
    for t in range(T):
        for s in range(S):
            for u in range(U):
                g = c[s, u] + Gy[s] * y[t, s, u] - lam_nn[t, s, u]
                for l in range(L):
                    g += lam_res[l, t] * d[l, s, u]
                if t == 0:
                    g += mu0[s]
                else:
                    g += muf[t - 1, s]
                if t < T - 1:
                    acc = 0.0
                    for s2 in range(S):
                        acc += P[u, s, s2] * muf[t, s2]
                    g -= acc
                gy[t, s, u] = g
    for l in range(L):
        for t in range(T):
            acc = 0.0
            for s in range(S):
                for u in range(U):
                    acc += d[l, s, u] * y[t, s, u]
            glr[l, t] = acc - budgets[l] - Gl * lam_res[l, t]
    for t in range(T):
        for s in range(S):
            for u in range(U):
                gln[t, s, u] = -y[t, s, u] - Gn[s] * lam_nn[t, s, u]
    for s in range(S):
        acc = 0.0
        for u in range(U):
            acc += y[0, s, u]
        gm0[s] = acc - m0[s] - Gm[s] * mu0[s]
    for t in range(1, T):
        for s in range(S):
            acc = 0.0
            for u in range(U):
                acc += y[t, s, u]
            inflow = 0.0
            for s2 in range(S):
                for u in range(U):
                    inflow += y[t - 1, s2, u] * P[u, s2, s]
            gmf[t - 1, s] = acc - inflow - Gm[s] * muf[t - 1, s]


@njit(cache=True)
def _sample_kernel(cumP, R, out):
    U, S, _ = cumP.shape
    for u in range(U):
        for s in range(S):
            for s2 in range(S):
                out[u, s, s2] = 0.0
            for _ in range(R):
                x = np.random.random()
                k = 0
                while k < S - 1 and x >= cumP[u, s, k]:
                    k += 1
                out[u, s, k] += 1.0
            for s2 in range(S):
                out[u, s, s2] /= R


@njit(cache=True)
def _run(P, c, d, budgets, m0, Gy, Gl, Gn, Gm, eta_y, eta_d, eta_ys, eta_ds, n_iter, R, seed, avg_start, record_every,
         y, lam_res, lam_nn, mu0, muf):
    np.random.seed(seed)
    T, S, U = y.shape
    L = d.shape[0]
    cumP = np.empty_like(P)
    for u in range(U):
        for s in range(S):
            acc = 0.0
            for s2 in range(S):
                acc += P[u, s, s2]
                cumP[u, s, s2] = acc
    P_used = P.copy()
    gy = np.zeros_like(y)
    glr = np.zeros_like(lam_res)
    gln = np.zeros_like(lam_nn)
    gm0 = np.zeros_like(mu0)
    gmf = np.zeros_like(muf)
    y_avg = np.zeros_like(y)
    lr_avg = np.zeros_like(lam_res)
    ln_avg = np.zeros_like(lam_nn)
    m0_avg = np.zeros_like(mu0)
    mf_avg = np.zeros_like(muf)
    n_avg = 0
    n_rec = n_iter // record_every + 1
    rec_iter = np.zeros(n_rec, dtype=np.int64)
    rec_y = np.zeros((n_rec, T, S, U))
    rec_lam_res = np.zeros((n_rec, L, T))
    rec_mu_norm = np.zeros(n_rec)
    k = 0
    for it in range(n_iter):
        if R > 0:
            _sample_kernel(cumP, R, P_used)
        _gradients(P_used, c, d, budgets, m0, y, lam_res, lam_nn, mu0, muf, Gy, Gl, Gn, Gm, gy, glr, gln, gm0, gmf)
        if it % record_every == 0 and k < n_rec:
            rec_iter[k] = it
            rec_y[k] = y
            rec_lam_res[k] = lam_res
            rec_mu_norm[k] = np.sqrt(np.sum(mu0 ** 2) + np.sum(muf ** 2))
            k += 1
        for t in range(T):
            for s in range(S):
                for u in range(U):
                    y[t, s, u] -= eta_ys[s] * gy[t, s, u]
                    v = lam_nn[t, s, u] + eta_ds[s] * gln[t, s, u]
                    lam_nn[t, s, u] = v if v > 0.0 else 0.0
        for l in range(L):
            for t in range(T):
                v = lam_res[l, t] + eta_d * glr[l, t]
                lam_res[l, t] = v if v > 0.0 else 0.0
        for s in range(S):
            mu0[s] += eta_ds[s] * gm0[s]
        for t in range(T - 1):
            for s in range(S):
                muf[t, s] += eta_ds[s] * gmf[t, s]
        if it + 1 >= avg_start:
            n_avg += 1
            w = 1.0 / n_avg
            y_avg += w * (y - y_avg)
            lr_avg += w * (lam_res - lr_avg)
            ln_avg += w * (lam_nn - ln_avg)
            m0_avg += w * (mu0 - m0_avg)
            mf_avg += w * (muf - mf_avg)
    return (y, lam_res, lam_nn, mu0, muf, y_avg, lr_avg, ln_avg, m0_avg, mf_avg, n_avg,
            rec_iter[:k], rec_y[:k], rec_lam_res[:k], rec_mu_norm[:k])


def run_gda(
        model: WCMDP,
        Gamma_y: float,
        Gamma_lambda: float,
        Gamma_mu: float,
        eta_y: float,
        eta_dual: float,
        n_iter: int,
        samples_per_pair: int = 0,
        seed: int = 0,
        avg_window: int | None = None,
        record_every: int | None = None,
        init: str = 'uniform',
        state_weights: np.ndarray | None = None,
    ) -> dict:
    '''
    Run GDA (`samples_per_pair` = 0) or SGDA (`samples_per_pair` = R > 0) on `model`.\n
    `state_weights` (one positive weight per state, e.g. block sizes of an aggregate problem) reweights the
    state-indexed regularization as in `saddle.regularization_vector(..., block_weights=...)`: y by Gamma_y / w,
    nonnegativity multipliers by Gamma_lambda w, flow multipliers by Gamma_mu w (the regularization inherited from
    the detailed problem, Section 6); resource multipliers keep Gamma_lambda. The steps are scaled to match (eta_y w
    for y, eta_dual / w for state-indexed multipliers), which makes the iteration the detailed one restricted to
    block-constant vectors. Weights of 1 give the plain detailed iteration.\n
    `init='uniform'` draws every coordinate i.i.d. Uniform[0, 1) from numpy's default_rng(seed),
    matching the legacy initialization; `init='zeros'` starts from zero.\n
    Returns last iterate and tail average (iterations n_iter - avg_window + 1 .. n_iter) of
    every block, plus a trace of y and the resource multipliers every `record_every` iterations.
    '''
    T, S, U, L = model.T, model.S, model.U, model.L
    rng = np.random.default_rng(seed)
    shapes = [(T, S, U), (L, T), (T, S, U), (S,), (T - 1, S)]
    if init == 'uniform':
        y, lam_res, lam_nn, mu0, muf = (rng.random(sh) for sh in shapes)
    elif init == 'zeros':
        y, lam_res, lam_nn, mu0, muf = (np.zeros(sh) for sh in shapes)
    else:
        raise ValueError("init must be 'uniform' or 'zeros'")
    avg_window = n_iter if avg_window is None else min(avg_window, n_iter)
    record_every = max(1, n_iter // 200) if record_every is None else record_every
    w = np.ones(S) if state_weights is None else np.asarray(state_weights, dtype=np.float64)
    if w.shape != (S,) or np.any(w <= 0):
        raise ValueError('state_weights must be a positive vector with one entry per state')
    Gy, Gn, Gm = float(Gamma_y) / w, float(Gamma_lambda) * w, float(Gamma_mu) * w
    # Matching step sizes: eta_y w on block mass and eta_dual / w on block prices. With them, GDA on an aggregate
    # problem is exactly the detailed GDA restricted to block-constant vectors (the operator Pi G Pi of Section 6),
    # so it is stable whenever the detailed iteration is.
    eta_ys, eta_ds = float(eta_y) * w, float(eta_dual) / w
    start = perf_counter()
    out = _run(
        np.ascontiguousarray(model.P, dtype=np.float64), np.ascontiguousarray(model.c, dtype=np.float64),
        np.ascontiguousarray(model.d, dtype=np.float64), np.ascontiguousarray(model.budgets, dtype=np.float64),
        np.ascontiguousarray(model.m0, dtype=np.float64),
        Gy, float(Gamma_lambda), Gn, Gm, float(eta_y), float(eta_dual), eta_ys, eta_ds,
        int(n_iter), int(samples_per_pair), int(seed) % (2 ** 32), int(n_iter - avg_window + 1), int(record_every),
        y, lam_res, lam_nn, mu0, muf)
    wall = perf_counter() - start
    names = ['y', 'lam_res', 'lam_nn', 'mu0', 'muf']
    result = {f'{n}_last': out[i] for i, n in enumerate(names)}
    result.update({f'{n}_avg': out[5 + i] for i, n in enumerate(names)})
    result.update({
        'n_avg': int(out[10]), 'trace_iter': out[11], 'trace_y': out[12], 'trace_lam_res': out[13],
        'trace_mu_norm': out[14], 'wall_time': wall, 'seed': int(seed), 'n_iter': int(n_iter),
        'samples_per_pair': int(samples_per_pair), 'eta_y': float(eta_y), 'eta_dual': float(eta_dual),
        'state_weights': w,
    })
    return result


def w_from_result(struct: LPStructure, result: dict, which: str = 'avg') -> np.ndarray:
    '''Assemble a flat saddle vector from a `run_gda` result ('avg' or 'last').'''
    return struct.join(*(result[f'{n}_{which}'] for n in ('y', 'lam_res', 'lam_nn', 'mu0', 'muf')))
