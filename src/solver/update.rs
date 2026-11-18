//! Blockwise candidate construction. The engine alone commits state and counters.
use super::{Iterate, Settings};
use crate::{Error, Result};

/// Construct a complete candidate using scalar broadcasts or state-weighted rows.
/// Every raw value is checked before projection can conceal NaN or infinity.
/// An invalid buffer is never exposed or swapped into the live iterate.
pub(crate) fn candidate(
    state: &Iterate,
    gradient: &[f64],
    q: &Settings,
    out: &mut [f64],
) -> Result<()> {
    let l = &state.layout;
    let w = state.values();
    let n = l.dimensions.occupations();
    let (y, rest) = out.split_at_mut(n);
    let (lr, rest) = rest.split_at_mut(l.resource_dual.len());
    let (ln, mu) = rest.split_at_mut(n);
    let mut finite = broadcast::<true>(
        lr,
        &w[l.resource_dual.clone()],
        &gradient[l.resource_dual.clone()],
        q.resource_step,
    );
    if let Some(steps) = q.uniform_state_steps {
        finite &= broadcast::<false>(y, state.occupations(), &gradient[l.y.clone()], steps.primal);
        finite &= broadcast::<true>(
            ln,
            state.nonnegative_dual(),
            &gradient[l.nonnegative_dual.clone()],
            steps.dual,
        );
        finite &= broadcast::<false>(
            mu,
            &w[l.initial_dual.start..],
            &gradient[l.initial_dual.start..],
            steps.dual,
        );
    } else {
        finite &= state_rows::<false>(
            y,
            state.occupations(),
            &gradient[l.y.clone()],
            &q.primal_steps,
            l.dimensions.actions(),
        );
        finite &= state_rows::<true>(
            ln,
            state.nonnegative_dual(),
            &gradient[l.nonnegative_dual.clone()],
            &q.dual_steps,
            l.dimensions.actions(),
        );
        finite &= diagonal_rows(
            mu,
            &w[l.initial_dual.start..],
            &gradient[l.initial_dual.start..],
            &q.dual_steps,
        );
    }
    if finite {
        Ok(())
    } else {
        Err(Error::NumericalOverflow)
    }
}

/// Const projection selects two small loop kernels; no block dispatch or integer
/// division occurs per coordinate. Independent validity lanes admit SIMD too.
fn broadcast<const PROJECTED: bool>(
    out: &mut [f64],
    current: &[f64],
    gradient: &[f64],
    step: f64,
) -> bool {
    debug_assert_eq!(out.len(), current.len());
    debug_assert_eq!(out.len(), gradient.len());
    let mut finite = true;
    for ((next, &value), &g) in out.iter_mut().zip(current).zip(gradient) {
        let raw = value - step * g;
        finite &= raw.is_finite();
        *next = if PROJECTED { raw.max(0.) } else { raw };
    }
    finite
}
fn state_rows<const PROJECTED: bool>(
    out: &mut [f64],
    current: &[f64],
    gradient: &[f64],
    steps: &[f64],
    actions: usize,
) -> bool {
    let mut finite = true;
    for (((out, current), gradient), &step) in out
        .chunks_exact_mut(actions)
        .zip(current.chunks_exact(actions))
        .zip(gradient.chunks_exact(actions))
        .zip(steps.iter().cycle())
    {
        finite &= broadcast::<PROJECTED>(out, current, gradient, step);
    }
    finite
}
fn diagonal_rows(out: &mut [f64], current: &[f64], gradient: &[f64], steps: &[f64]) -> bool {
    let mut finite = true;
    for ((out, current), gradient) in out
        .chunks_exact_mut(steps.len())
        .zip(current.chunks_exact(steps.len()))
        .zip(gradient.chunks_exact(steps.len()))
    {
        for (((next, &value), &g), &step) in out.iter_mut().zip(current).zip(gradient).zip(steps) {
            let raw = value - step * g;
            finite &= raw.is_finite();
            *next = raw;
        }
    }
    finite
}
