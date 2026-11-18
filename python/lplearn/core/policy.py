"""Induce a Markov policy from occupation measures and evaluate its fluid flow.

Adapted from the pinned LP Learner Python reference; see NOTICE.
"""

from __future__ import annotations

import numpy as np

from .instance import WCMDP


def conditional_policy(y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Derive pi[t,s,u] from y[t,s,u], with safe fallbacks for zero-mass states."""
    y = np.asarray(y, dtype=float)
    if y.ndim != 3 or 0 in y.shape or not np.isfinite(y).all():
        raise ValueError("occupation must be a nonempty, finite [time, state, action] array")
    y = np.maximum(y, 0.0)
    T, S, U = y.shape
    mass = y.sum(axis=2)
    pi = np.zeros_like(y)
    for t in range(T):
        support = np.flatnonzero(mass[t] > eps)
        if support.size == 0:
            pi[t, :, 0] = 1.0
            continue
        for s in range(S):
            src = s if mass[t, s] > eps else support[
                np.argmin(np.abs(support - s) + 1e-3 * (support > s))
            ]
            pi[t, s] = y[t, src] / mass[t, src]
    return pi


def fluid_evaluation(model: WCMDP, pi: np.ndarray, m0: np.ndarray | None = None) -> dict:
    """Evaluate the conditional policy through the model's transition kernels."""
    pi = np.asarray(pi, dtype=float)
    if pi.ndim != 3 or not np.isfinite(pi).all():
        raise ValueError("policy must be a finite [time, state, action] array")
    T, S, U = pi.shape
    if (T, S, U) != (model.T, model.S, model.U):
        raise ValueError("policy shape does not match the model")
    if np.any(pi < 0) or not np.allclose(pi.sum(axis=2), 1.0, atol=1e-10, rtol=0):
        raise ValueError("each state policy must be a probability distribution")
    initial = np.asarray(model.m0 if m0 is None else m0, dtype=float)
    if (initial.shape != (S,) or not np.isfinite(initial).all()
            or np.any(initial < 0) or not np.isclose(initial.sum(), 1.0, atol=1e-10, rtol=0)):
        raise ValueError("initial distribution must have one finite probability per state and sum to one")
    mass = np.empty((T, S))
    mass[0] = initial
    occupations = np.empty_like(pi)
    for t in range(T):
        occupations[t] = mass[t][:, None] * pi[t]
        if t + 1 < T:
            mass[t + 1] = np.einsum("su,usr->r", occupations[t], model.P)
    cost_per_stage = np.einsum("tsu,su->t", occupations, model.c)
    usage = np.einsum("tsu,lsu->lt", occupations, model.d)
    return {
        "y": occupations,
        "mass": mass,
        "cost": float(cost_per_stage.sum()),
        "cost_per_stage": cost_per_stage,
        "usage": usage,
        "excess": np.maximum(usage - model.budgets[:, None], 0.0),
    }
