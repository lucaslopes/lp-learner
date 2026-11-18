use crate::{model::Dimensions, Error, Result};
/// Manuscript convention: Gamma/2 times the squared norm of each block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Regularization {
    pub primal: f64,
    pub inequality: f64,
    pub equality: f64,
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepSizes {
    pub primal: f64,
    pub dual: f64,
}
/// Validated per-state coefficients, compiled once when constructing a solver.
#[derive(Debug, Clone)]
pub struct Settings {
    pub(crate) regularization: Regularization,
    pub(crate) primal_reg: Vec<f64>,
    pub(crate) nonnegative_reg: Vec<f64>,
    pub(crate) equality_reg: Vec<f64>,
    pub(crate) primal_steps: Vec<f64>,
    pub(crate) dual_steps: Vec<f64>,
    pub(crate) resource_step: f64,
    pub(crate) uniform_state_steps: Option<StepSizes>,
}
impl Settings {
    pub fn new(
        d: Dimensions,
        r: Regularization,
        steps: StepSizes,
        weights: Option<&[f64]>,
    ) -> Result<Self> {
        if [r.primal, r.inequality, r.equality]
            .iter()
            .any(|x| !x.is_finite() || *x < 0.)
        {
            return Err(Error::InvalidParameter(
                "regularization must be finite and nonnegative",
            ));
        }
        if [steps.primal, steps.dual]
            .iter()
            .any(|x| !x.is_finite() || *x <= 0.)
        {
            return Err(Error::InvalidParameter("steps must be finite and positive"));
        }
        let w = weights.map_or_else(|| vec![1.; d.states()], <[f64]>::to_vec);
        if w.len() != d.states() || w.iter().any(|x| !x.is_finite() || *x <= 0.) {
            return Err(Error::InvalidParameter("state weights"));
        }
        let out = Self {
            regularization: r,
            primal_reg: w.iter().map(|x| r.primal / x).collect(),
            nonnegative_reg: w.iter().map(|x| r.inequality * x).collect(),
            equality_reg: w.iter().map(|x| r.equality * x).collect(),
            primal_steps: w.iter().map(|x| steps.primal * x).collect(),
            dual_steps: w.iter().map(|x| steps.dual / x).collect(),
            resource_step: steps.dual,
            uniform_state_steps: w.iter().all(|x| *x == w[0]).then_some(StepSizes {
                primal: steps.primal * w[0],
                dual: steps.dual / w[0],
            }),
        };
        if [
            &out.primal_reg,
            &out.nonnegative_reg,
            &out.equality_reg,
            &out.primal_steps,
            &out.dual_steps,
        ]
        .into_iter()
        .flatten()
        .any(|x| !x.is_finite())
        {
            return Err(Error::InvalidParameter("weighted coefficients overflow"));
        }
        if out
            .primal_steps
            .iter()
            .chain(&out.dual_steps)
            .any(|x| *x == 0.)
        {
            return Err(Error::InvalidParameter("weighted steps underflow"));
        }
        Ok(out)
    }
    pub fn regularization(&self) -> Regularization {
        self.regularization
    }
}
