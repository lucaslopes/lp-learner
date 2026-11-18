//! Real observations and optimizer samples have separate types and counters.
//!
//! [`EmpiricalTransitions`] blends an explicit prior row with an empirical row.
//! For a row `(s, u)` with effective count `n` and effective cell counts `c`:
//!
//! ```text
//! p(s' | s, u) = w * prior(s' | s, u) + (1 - w) * c(s') / n,   w = strength / (strength + n)
//! ```
//!
//! and `p = prior` while `n == 0`. The [`EstimatorMode`] decides only what `n`
//! and `c` are: every observation ever seen (the default, unchanged from the
//! original estimator), the last `size` observations of the row, or
//! exponentially decayed counts. All three are per `(s, u)` row and use the
//! row's own observation clock, so the memory never grows with time and the
//! estimator never reads a wall clock (the library path stays wasm32-friendly).
use crate::{
    model::{Dimensions, TransitionKernel},
    Error, Result,
};

/// How much of the observation history the empirical part of the estimate
/// remembers. Windows and half-lives are counted in observations of the same
/// `(state, action)` row, not in global time; see `docs/ONLINE_QUEUE_ADAPTER.md`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)
)]
pub enum EstimatorMode {
    /// All observations, never forgotten (the original estimator; default).
    #[default]
    Cumulative,
    /// Exact counts over the last `size` observations of each `(s, u)` row,
    /// held in a fixed-capacity ring buffer per row.
    SlidingWindow { size: usize },
    /// Counts decayed by `beta = 2^(-1 / half_life)` at every new observation
    /// of the row: an observation `half_life` row-observations old has weight 1/2.
    Exponential { half_life: f64 },
}

impl EstimatorMode {
    pub fn validate(self) -> Result<()> {
        match self {
            Self::Cumulative => Ok(()),
            Self::SlidingWindow { size } if size > 0 => Ok(()),
            Self::SlidingWindow { .. } => Err(Error::InvalidParameter("window size")),
            Self::Exponential { half_life } if half_life.is_finite() && half_life > 0. => Ok(()),
            Self::Exponential { .. } => Err(Error::InvalidParameter("half-life")),
        }
    }
}

/// Fixed-capacity per-row history of next states for the sliding window.
#[derive(Debug, Clone)]
struct Ring {
    size: usize,
    /// slots[row * size + k]: next state of a remembered observation.
    slots: Vec<u32>,
    /// Next slot to overwrite in each row (the oldest once the row is full).
    head: Vec<usize>,
}

#[derive(Debug, Clone)]
enum Memory {
    /// Integer counts; `ring` is `Some` for the sliding window.
    Integer {
        counts: Vec<u64>,
        row_totals: Vec<u64>,
        ring: Option<Ring>,
    },
    /// Decayed real-valued counts with per-row decay factor `beta`.
    Decayed {
        beta: f64,
        counts: Vec<f64>,
        row_totals: Vec<f64>,
    },
}

/// Prior-weighted empirical model. Row totals make observation and point lookup
/// O(1) for the integer modes; the exponential mode costs O(states) per observation.
#[derive(Debug, Clone)]
pub struct EmpiricalTransitions {
    prior: TransitionKernel,
    strength: f64,
    mode: EstimatorMode,
    memory: Memory,
    observations: u64,
}

#[inline]
fn blend(strength: f64, prior: f64, count: f64, n: f64) -> f64 {
    // Convex-combination form avoids overflowing strength * prior + count.
    let weight = strength / (strength + n);
    weight * prior + (1. - weight) * (count / n)
}

impl EmpiricalTransitions {
    /// The original cumulative estimator.
    pub fn new(prior: TransitionKernel, strength: f64) -> Result<Self> {
        Self::with_mode(prior, strength, EstimatorMode::Cumulative)
    }
    pub fn with_mode(prior: TransitionKernel, strength: f64, mode: EstimatorMode) -> Result<Self> {
        if !strength.is_finite() || strength <= 0. {
            return Err(Error::InvalidParameter("prior strength"));
        }
        mode.validate()?;
        let d = prior.dimensions();
        let rows = d.actions() * d.states();
        let memory = match mode {
            EstimatorMode::Cumulative => Memory::Integer {
                counts: vec![0; d.kernel_len()],
                row_totals: vec![0; rows],
                ring: None,
            },
            EstimatorMode::SlidingWindow { size } => {
                if u32::try_from(d.states() - 1).is_err() {
                    return Err(Error::InvalidDimensions);
                }
                let len = rows.checked_mul(size).ok_or(Error::InvalidDimensions)?;
                Memory::Integer {
                    counts: vec![0; d.kernel_len()],
                    row_totals: vec![0; rows],
                    ring: Some(Ring {
                        size,
                        slots: vec![0; len],
                        head: vec![0; rows],
                    }),
                }
            }
            EstimatorMode::Exponential { half_life } => Memory::Decayed {
                beta: decay_factor(half_life),
                counts: vec![0.; d.kernel_len()],
                row_totals: vec![0.; rows],
            },
        };
        Ok(Self {
            prior,
            strength,
            mode,
            memory,
            observations: 0,
        })
    }
    pub fn dimensions(&self) -> Dimensions {
        self.prior.dimensions()
    }
    pub fn mode(&self) -> EstimatorMode {
        self.mode
    }
    /// The per-observation decay factor of the exponential mode.
    pub fn decay_factor(&self) -> Option<f64> {
        match self.memory {
            Memory::Decayed { beta, .. } => Some(beta),
            Memory::Integer { .. } => None,
        }
    }
    /// Every real observation ever recorded, whatever the mode remembers.
    pub fn observation_count(&self) -> u64 {
        self.observations
    }
    /// The `n` of the blend rule for row `(state, action)`.
    pub fn effective_count(&self, state: usize, action: usize) -> Result<f64> {
        let d = self.dimensions();
        d.kernel_index(state, action, 0)?;
        let row = action * d.states() + state;
        Ok(match &self.memory {
            Memory::Integer { row_totals, .. } => row_totals[row] as f64,
            Memory::Decayed { row_totals, .. } => row_totals[row],
        })
    }
    /// The `c(next)` of the blend rule for row `(state, action)`.
    pub fn effective_cell_count(&self, state: usize, action: usize, next: usize) -> Result<f64> {
        let i = self.dimensions().kernel_index(state, action, next)?;
        Ok(match &self.memory {
            Memory::Integer { counts, .. } => counts[i] as f64,
            Memory::Decayed { counts, .. } => counts[i],
        })
    }
    pub fn observe(&mut self, state: usize, action: usize, next: usize) -> Result<()> {
        let d = self.dimensions();
        let s = d.states();
        let i = d.kernel_index(state, action, next)?;
        let row = action * s + state;
        // Every check precedes every write, so a rejected event changes nothing.
        let all = self
            .observations
            .checked_add(1)
            .ok_or(Error::CounterOverflow)?;
        match &mut self.memory {
            Memory::Integer {
                counts,
                row_totals,
                ring: None,
            } => {
                let cell = counts[i].checked_add(1).ok_or(Error::CounterOverflow)?;
                let total = row_totals[row]
                    .checked_add(1)
                    .ok_or(Error::CounterOverflow)?;
                counts[i] = cell;
                row_totals[row] = total;
            }
            Memory::Integer {
                counts,
                row_totals,
                ring: Some(ring),
            } => {
                // Counts are bounded by `size`, so they cannot overflow.
                let slot = row * ring.size + ring.head[row];
                if row_totals[row] as usize == ring.size {
                    let oldest = row * s + ring.slots[slot] as usize;
                    counts[oldest] -= 1;
                } else {
                    row_totals[row] += 1;
                }
                ring.slots[slot] = next as u32;
                ring.head[row] = (ring.head[row] + 1) % ring.size;
                counts[i] += 1;
            }
            Memory::Decayed {
                beta,
                counts,
                row_totals,
            } => {
                let cells = &mut counts[row * s..(row + 1) * s];
                for c in cells.iter_mut() {
                    *c *= *beta;
                }
                cells[next] += 1.;
                // Recomputing the total from the cells keeps sum(c) / n at one
                // up to rounding instead of letting two recursions drift apart.
                row_totals[row] = cells.iter().sum();
            }
        }
        self.observations = all;
        Ok(())
    }
    pub fn probability(&self, state: usize, action: usize, next: usize) -> Result<f64> {
        let d = self.dimensions();
        let i = d.kernel_index(state, action, next)?;
        let row = action * d.states() + state;
        let prior = self.prior.values()[i];
        Ok(match &self.memory {
            Memory::Integer {
                counts, row_totals, ..
            } => {
                let n = row_totals[row];
                if n == 0 {
                    prior
                } else {
                    blend(self.strength, prior, counts[i] as f64, n as f64)
                }
            }
            Memory::Decayed {
                counts, row_totals, ..
            } => {
                let n = row_totals[row];
                if n == 0. {
                    prior
                } else {
                    blend(self.strength, prior, counts[i], n)
                }
            }
        })
    }
    pub fn snapshot(&self) -> TransitionKernel {
        let mut out = self.prior.clone();
        self.write_snapshot(&mut out).expect("matching dimensions");
        out
    }
    pub fn write_snapshot(&self, out: &mut TransitionKernel) -> Result<()> {
        if out.dimensions() != self.dimensions() {
            return Err(Error::InvalidDimensions);
        }
        let s = self.dimensions().states();
        let prior = self.prior.values();
        match &self.memory {
            Memory::Integer {
                counts, row_totals, ..
            } => {
                for (i, x) in out.values.iter_mut().enumerate() {
                    let n = row_totals[i / s];
                    *x = if n == 0 {
                        prior[i]
                    } else {
                        blend(self.strength, prior[i], counts[i] as f64, n as f64)
                    };
                }
            }
            Memory::Decayed {
                counts, row_totals, ..
            } => {
                for (i, x) in out.values.iter_mut().enumerate() {
                    let n = row_totals[i / s];
                    *x = if n == 0. {
                        prior[i]
                    } else {
                        blend(self.strength, prior[i], counts[i], n)
                    };
                }
            }
        }
        Ok(())
    }
}

/// `2^(-1 / half_life)` from IEEE basic operations only (+, -, *, /), so native
/// and wasm32 builds produce the same bits; platform `powf`/`exp` may differ in
/// the last place. Argument halving keeps the Taylor series of `expm1` tiny and
/// the squaring step `e <- e * (2 + e)` (that is, `(1 + e)^2 - 1`) keeps the
/// relative error at a few ulps for half-lives of at least one observation.
pub fn decay_factor(half_life: f64) -> f64 {
    let x = -core::f64::consts::LN_2 / half_life;
    if x < -745. {
        return 0.; // exp(x) underflows below the smallest subnormal.
    }
    let mut y = x;
    let mut halvings = 0u32;
    while y < -1. / 1024. {
        y *= 0.5;
        halvings += 1;
    }
    let mut e = y
        * (1. + y * (1. / 2. + y * (1. / 6. + y * (1. / 24. + y * (1. / 120. + y * (1. / 720.))))));
    let mut squarings = 0;
    // Square in the e = beta - 1 form while beta >= 1/2 (that is, while
    // half_life >= 1 for the final result); there `1 + e` is exact enough.
    while squarings < halvings && e >= -0.5 {
        e *= 2. + e;
        squarings += 1;
    }
    // Strong decay (half_life < 1): square beta itself, since 1 + e would cancel.
    let mut beta = 1. + e;
    for _ in squarings..halvings {
        beta *= beta;
    }
    beta
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row_prior(states: usize) -> TransitionKernel {
        let d = Dimensions::new(states, 1, 1, 1).unwrap();
        TransitionKernel::new(d, vec![1. / states as f64; states * states]).unwrap()
    }

    #[test]
    fn decay_factor_is_accurate_for_a_wide_range_of_half_lives() {
        for h in [0.01, 0.1, 0.5, 1., 2., 7.5, 50., 100., 1e3, 1e6, 1e12] {
            let beta = decay_factor(h);
            let reference = 0.5f64.powf(1. / h);
            // A few ulps for practical half-lives (>= 1 observation); the
            // beta-squaring branch for h < 1 accumulates about 2^k ulps.
            let tolerance = if h >= 1. { 4. * f64::EPSILON } else { 1e-12 };
            assert!(
                (beta - reference).abs() <= tolerance * reference,
                "h={h}: {beta} vs {reference}"
            );
        }
        assert_eq!(decay_factor(1e-300), 0.);
        assert!(decay_factor(1e300) <= 1.);
    }

    #[test]
    fn invalid_modes_are_rejected() {
        for mode in [
            EstimatorMode::SlidingWindow { size: 0 },
            EstimatorMode::Exponential { half_life: 0. },
            EstimatorMode::Exponential { half_life: -1. },
            EstimatorMode::Exponential {
                half_life: f64::NAN,
            },
            EstimatorMode::Exponential {
                half_life: f64::INFINITY,
            },
        ] {
            assert!(EmpiricalTransitions::with_mode(row_prior(2), 1., mode).is_err());
        }
    }

    #[test]
    fn window_ring_wraps_and_counts_stay_exact() {
        let mut e = EmpiricalTransitions::with_mode(
            row_prior(3),
            1.,
            EstimatorMode::SlidingWindow { size: 3 },
        )
        .unwrap();
        let seq = [0, 1, 2, 2, 2, 1, 0, 0];
        for (k, &x) in seq.iter().enumerate() {
            e.observe(0, 0, x).unwrap();
            let window = &seq[(k + 1).saturating_sub(3)..=k];
            for next in 0..3 {
                let expected = window.iter().filter(|&&v| v == next).count() as f64;
                assert_eq!(
                    e.effective_cell_count(0, 0, next).unwrap(),
                    expected,
                    "step {k}"
                );
            }
            assert_eq!(e.effective_count(0, 0).unwrap(), window.len() as f64);
        }
        assert_eq!(e.observation_count(), seq.len() as u64);
        assert_eq!(e.effective_count(1, 0).unwrap(), 0.);
    }
}
