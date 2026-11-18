//! Structured saddle operator with contiguous contractions; no LP or KKT matrix.
use super::{Iterate, Settings};
use crate::model::{Problem, TransitionKernel};

/// Signed gradient (grad_y L, -grad_lambda L, -grad_mu L).
/// Every block reads the same immutable iterate. All storage belongs to the solver.
pub(crate) fn evaluate(
    m: &Problem,
    p: &TransitionKernel,
    q: &Settings,
    w: &Iterate,
    g: &mut [f64],
) {
    let d = m.dimensions();
    let (gy, rest) = g.split_at_mut(d.occupations());
    let (gr, rest) = rest.split_at_mut(d.resources() * d.horizon());
    let (gn, rest) = rest.split_at_mut(d.occupations());
    let (gi, gf) = rest.split_at_mut(d.states());
    primal(m, p, q, w, gy, gn);
    resources(m, q, w, gr);
    initial(m, q, w, gi);
    flow(m, p, q, w, gf);
}

/// A strict, ordered dot product. Slicing once per row removes repeated bounds
/// checks without changing the summation order or enabling floating reassociation.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0., |sum, (x, y)| sum + x * y)
}

/// Independent lanes permit vectorization while every lane keeps its original
/// accumulation order. Model and layout validation establish equal row lengths.
fn add_scaled(out: &mut [f64], scale: f64, row: &[f64]) {
    debug_assert_eq!(out.len(), row.len());
    for (value, coefficient) in out.iter_mut().zip(row) {
        *value += scale * coefficient;
    }
}

fn primal(
    m: &Problem,
    p: &TransitionKernel,
    q: &Settings,
    w: &Iterate,
    gy: &mut [f64],
    gn: &mut [f64],
) {
    let d = m.dimensions();
    let (s, u, tmax) = (d.states(), d.actions(), d.horizon());
    let stage = s * u;
    let y = w.occupations();
    let ln = w.nonnegative_dual();
    let lr = w.resource_dual();
    let mf = w.flow_dual();
    for t in 0..tmax {
        let start = t * stage;
        let out = &mut gy[start..start + stage];
        let ny = &mut gn[start..start + stage];
        let yt = &y[start..start + stage];
        let nt = &ln[start..start + stage];
        for state in 0..s {
            let begin = state * u;
            let costs = &m.costs()[begin..begin + u];
            let py = q.primal_reg[state];
            let pn = q.nonnegative_reg[state];
            for ((((g, n), &a), &b), &c) in out[begin..begin + u]
                .iter_mut()
                .zip(&mut ny[begin..begin + u])
                .zip(&yt[begin..begin + u])
                .zip(&nt[begin..begin + u])
                .zip(costs)
            {
                *g = c + py * a - b;
                *n = a + pn * b;
            }
        }
        // Keep resource accumulation in l order, but traverse a contiguous stage.
        for (l, row) in m.resources().chunks_exact(stage).enumerate() {
            add_scaled(out, lr[l * tmax + t], row);
        }
        let current = if t == 0 {
            w.initial_dual()
        } else {
            &mf[(t - 1) * s..t * s]
        };
        for (state, row) in out.chunks_exact_mut(u).enumerate() {
            for value in row {
                *value += current[state];
            }
        }
        if t + 1 < tmax {
            let next = &mf[t * s..(t + 1) * s];
            for (state, row) in out.chunks_exact_mut(u).enumerate() {
                for (action, value) in row.iter_mut().enumerate() {
                    let begin = (action * s + state) * s;
                    *value -= dot(&p.values()[begin..begin + s], next);
                }
            }
        }
    }
}
fn resources(m: &Problem, q: &Settings, w: &Iterate, out: &mut [f64]) {
    let d = m.dimensions();
    let stage = d.states() * d.actions();
    let tmax = d.horizon();
    for (l, resource) in m.resources().chunks_exact(stage).enumerate() {
        for (t, y) in w.occupations().chunks_exact(stage).enumerate() {
            let index = l * tmax + t;
            out[index] = m.budgets()[l] - dot(resource, y)
                + q.regularization.inequality * w.resource_dual()[index];
        }
    }
}
fn initial(m: &Problem, q: &Settings, w: &Iterate, out: &mut [f64]) {
    let d = m.dimensions();
    for (state, y) in w.occupations()[..d.states() * d.actions()]
        .chunks_exact(d.actions())
        .enumerate()
    {
        out[state] = m.initial()[state] - y.iter().sum::<f64>()
            + q.equality_reg[state] * w.initial_dual()[state];
    }
}
fn flow(m: &Problem, p: &TransitionKernel, q: &Settings, w: &Iterate, out: &mut [f64]) {
    let d = m.dimensions();
    let (s, u) = (d.states(), d.actions());
    let stage = s * u;
    let y = w.occupations();
    for (t, inflow) in out.chunks_exact_mut(s).enumerate() {
        inflow.fill(0.);
        // Accumulate prev/action in the reference order. The innermost loop now
        // follows contiguous next-state rows, rather than striding across actions.
        let previous = &y[t * stage..(t + 1) * stage];
        for (prev, row) in previous.chunks_exact(u).enumerate() {
            for (action, &mass) in row.iter().enumerate() {
                let begin = (action * s + prev) * s;
                add_scaled(inflow, mass, &p.values()[begin..begin + s]);
            }
        }
        let current = &y[(t + 1) * stage..(t + 2) * stage];
        let mu = &w.flow_dual()[t * s..(t + 1) * s];
        for (((value, row), &price), &reg) in inflow
            .iter_mut()
            .zip(current.chunks_exact(u))
            .zip(mu)
            .zip(&q.equality_reg)
        {
            *value = *value - row.iter().sum::<f64>() + reg * price;
        }
    }
}
