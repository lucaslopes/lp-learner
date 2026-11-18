use super::Dimensions;
use crate::{error::length, Error, Result};
pub const PROBABILITY_TOLERANCE: f64 = 1e-9;

/// Row-stochastic kernel stored as P[action,state,next]. Constructed once and immutable to clients.
#[derive(Debug, Clone, PartialEq)]
pub struct TransitionKernel {
    pub(crate) dimensions: Dimensions,
    pub(crate) values: Vec<f64>,
}
impl TransitionKernel {
    pub fn new(dimensions: Dimensions, values: Vec<f64>) -> Result<Self> {
        length("kernel", &values, dimensions.kernel_len())?;
        for row in values.chunks_exact(dimensions.states()) {
            validate_distribution("kernel row", row)?;
        }
        Ok(Self { dimensions, values })
    }
    pub fn dimensions(&self) -> Dimensions {
        self.dimensions
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    pub fn probability(&self, state: usize, action: usize, next: usize) -> Result<f64> {
        Ok(self.values[self.dimensions.kernel_index(state, action, next)?])
    }
}
pub(crate) fn validate_distribution(field: &'static str, row: &[f64]) -> Result<()> {
    if row.iter().any(|x| !x.is_finite() || *x < 0.)
        || (row.iter().sum::<f64>() - 1.).abs() > PROBABILITY_TOLERANCE
    {
        return Err(Error::InvalidProbability(field));
    }
    Ok(())
}
