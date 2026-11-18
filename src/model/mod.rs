//! Immutable, validated WC-MDP data. No application-specific action semantics.
mod dimensions;
mod kernel;
use crate::{error::length, Error, Result};
pub use dimensions::{Dimensions, Layout};
pub use kernel::{TransitionKernel, PROBABILITY_TOLERANCE};

/// Stage-independent costs c[state,action], resources d[resource,state,action],
/// per-stage budgets, and initial mass. Horizon stages are 0 through T-1.
#[derive(Debug, Clone)]
pub struct Problem {
    kernel: TransitionKernel,
    costs: Vec<f64>,
    resources: Vec<f64>,
    budgets: Vec<f64>,
    initial: Vec<f64>,
}
impl Problem {
    pub fn new(
        kernel: TransitionKernel,
        costs: Vec<f64>,
        resources: Vec<f64>,
        budgets: Vec<f64>,
        initial: Vec<f64>,
    ) -> Result<Self> {
        let d = kernel.dimensions();
        length("costs", &costs, d.states() * d.actions())?;
        length("resources", &resources, d.resource_len())?;
        length("budgets", &budgets, d.resources())?;
        length("initial", &initial, d.states())?;
        kernel::validate_distribution("initial", &initial)?;
        if budgets.iter().any(|x| *x < 0.) {
            return Err(Error::InvalidParameter("negative budget"));
        }
        Ok(Self {
            kernel,
            costs,
            resources,
            budgets,
            initial,
        })
    }
    pub fn dimensions(&self) -> Dimensions {
        self.kernel.dimensions()
    }
    pub fn kernel(&self) -> &TransitionKernel {
        &self.kernel
    }
    pub fn costs(&self) -> &[f64] {
        &self.costs
    }
    pub fn resources(&self) -> &[f64] {
        &self.resources
    }
    pub fn budgets(&self) -> &[f64] {
        &self.budgets
    }
    pub fn initial(&self) -> &[f64] {
        &self.initial
    }
}
