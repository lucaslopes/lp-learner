//! Policy extraction is separate from the optimizer and application feasibility.
use crate::{error::length, model::Dimensions, Error, Result};

/// Conditional probabilities from clipped occupations, with nearest-state
/// fallback at each stage (ties toward the smaller state). A massless stage uses action 0.
#[derive(Debug, Clone)]
pub struct Policy {
    dimensions: Dimensions,
    probabilities: Vec<f64>,
}
impl Policy {
    pub fn from_occupations(d: Dimensions, y: &[f64], epsilon: f64) -> Result<Self> {
        length("occupations", y, d.occupations())?;
        check_epsilon(epsilon)?;
        let mut probabilities = vec![0.; y.len()];
        let s = d.states();
        let u = d.actions();
        for t in 0..d.horizon() {
            let mass: Vec<f64> = (0..s)
                .map(|x| {
                    y[(t * s + x) * u..(t * s + x + 1) * u]
                        .iter()
                        .map(|v| v.max(0.))
                        .sum()
                })
                .collect();
            if mass.iter().any(|x| !x.is_finite()) {
                return Err(Error::NumericalOverflow);
            }
            let support: Vec<usize> = (0..s).filter(|&x| mass[x] > epsilon).collect();
            for state in 0..s {
                if support.is_empty() {
                    probabilities[(t * s + state) * u] = 1.;
                    continue;
                }
                let src = if mass[state] > epsilon {
                    state
                } else {
                    *support
                        .iter()
                        .min_by_key(|&&x| (state.abs_diff(x), x))
                        .expect("nonempty support")
                };
                for a in 0..u {
                    probabilities[(t * s + state) * u + a] =
                        y[(t * s + src) * u + a].max(0.) / mass[src];
                }
            }
        }
        Ok(Self {
            dimensions: d,
            probabilities,
        })
    }
    pub fn values(&self) -> &[f64] {
        &self.probabilities
    }
    pub fn row(&self, stage: usize, state: usize) -> Result<&[f64]> {
        let start = self.dimensions.occupation_index(stage, state, 0)?;
        Ok(&self.probabilities[start..start + self.dimensions.actions()])
    }
}
fn check_epsilon(eps: f64) -> Result<()> {
    if !eps.is_finite() || eps < 0. {
        return Err(Error::InvalidParameter("policy epsilon"));
    }
    Ok(())
}
/// Explicit local action mask. Shared realized limits remain the caller's responsibility.
pub fn masked_probabilities(
    weights: &[f64],
    allowed: &[bool],
    fallback: usize,
    epsilon: f64,
) -> Result<Vec<f64>> {
    check_epsilon(epsilon)?;
    if weights.len() != allowed.len()
        || weights.is_empty()
        || fallback >= allowed.len()
        || !allowed[fallback]
    {
        return Err(Error::InvalidParameter("policy mask or fallback"));
    }
    if weights.iter().any(|x| !x.is_finite()) {
        return Err(Error::NonFinite("policy weights"));
    }
    let mut p: Vec<f64> = weights
        .iter()
        .zip(allowed)
        .map(|(x, ok)| if *ok { x.max(0.) } else { 0. })
        .collect();
    let sum: f64 = p.iter().sum();
    if !sum.is_finite() {
        return Err(Error::NumericalOverflow);
    }
    if sum <= epsilon {
        p.fill(0.);
        p[fallback] = 1.;
    } else {
        for x in &mut p {
            *x /= sum;
        }
    }
    Ok(p)
}
