//! Plug-in GDA over an observation-driven posterior-mean kernel.
//! Optimizer iterations do not produce observations or fresh simulator samples.
use crate::{
    model::{Problem, TransitionKernel},
    solver::{Diagnostics, ExactKernel, Iterate, Settings, Solver},
    transition::{EmpiricalTransitions, EstimatorMode},
    Error, Result,
};

/// Whether, and how, the online learner averages its primal-dual iterates
/// over optimizer steps. The average is read-only bookkeeping: the GDA
/// recursion, and so the last iterate, are the same in every mode.
///
/// The batch `Solver::run` averages a final window of iterations; online,
/// the estimated kernel changes between updates, so an average here spans
/// iterates computed against different kernels.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)
)]
pub enum IterateAverage {
    /// No average is kept (the default; `update` runs the original path).
    #[default]
    Off,
    /// Uniform (Polyak-Ruppert) average of every iterate produced by an
    /// optimizer step whose lifetime number exceeds `from_step`:
    /// `avg += (x - avg) / k` for the k-th averaged iterate, the update rule
    /// of `Solver::run`'s window average.
    Uniform { from_step: u64 },
    /// Exponential average, `avg += w (x - avg)` with
    /// `w = 1 - 2^(-1 / half_life_steps)` per optimizer step; the first
    /// averaged iterate initializes it.
    Exponential { half_life_steps: f64 },
}

impl IterateAverage {
    pub fn validate(self) -> Result<()> {
        if let IterateAverage::Exponential { half_life_steps } = self {
            if !(half_life_steps.is_finite() && half_life_steps > 0.) {
                return Err(Error::InvalidParameter("average half-life"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct OnlineLearner {
    solver: Solver,
    estimator: EmpiricalTransitions,
    kernel: TransitionKernel,
    snapshot_version: u64,
    average_mode: IterateAverage,
    average: Option<Iterate>,
    averaged_steps: u64,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpdateSummary {
    pub optimizer_steps: u64,
    pub observations: u64,
}
impl OnlineLearner {
    /// The original cumulative estimator (`EstimatorMode::Cumulative`).
    pub fn new(problem: Problem, settings: Settings, prior_strength: f64) -> Result<Self> {
        Self::with_estimator(problem, settings, prior_strength, EstimatorMode::Cumulative)
    }
    /// Choose how much observation history the plug-in kernel remembers.
    /// The primal-dual iterate persists across `update` calls in every mode.
    pub fn with_estimator(
        problem: Problem,
        settings: Settings,
        prior_strength: f64,
        mode: EstimatorMode,
    ) -> Result<Self> {
        let kernel = problem.kernel().clone();
        let estimator = EmpiricalTransitions::with_mode(kernel.clone(), prior_strength, mode)?;
        Ok(Self {
            solver: Solver::new(problem, settings)?,
            estimator,
            kernel,
            snapshot_version: 0,
            average_mode: IterateAverage::Off,
            average: None,
            averaged_steps: 0,
        })
    }
    /// Choose the iterate average; any average kept so far is discarded.
    pub fn set_average(&mut self, mode: IterateAverage) -> Result<()> {
        mode.validate()?;
        self.average_mode = mode;
        self.average = None;
        self.averaged_steps = 0;
        Ok(())
    }
    pub fn average_mode(&self) -> IterateAverage {
        self.average_mode
    }
    /// The averaged iterate, once at least one step has been averaged.
    pub fn average(&self) -> Option<&Iterate> {
        self.average.as_ref()
    }
    /// The number of iterates in the average.
    pub fn averaged_steps(&self) -> u64 {
        self.averaged_steps
    }
    fn accumulate(&mut self) -> Result<()> {
        let weight = match self.average_mode {
            IterateAverage::Off => return Ok(()),
            IterateAverage::Uniform { from_step } => {
                if self.solver.iterations() <= from_step {
                    return Ok(());
                }
                1. / (self.averaged_steps + 1) as f64
            }
            IterateAverage::Exponential { half_life_steps } => {
                if self.averaged_steps == 0 {
                    1.
                } else {
                    1. - 0.5f64.powf(1. / half_life_steps)
                }
            }
        };
        let x = self.solver.state();
        let avg = self
            .average
            .get_or_insert_with(|| Iterate::zeros(x.dimensions()));
        for (a, v) in avg.values.iter_mut().zip(&x.values) {
            *a += weight * (*v - *a);
            if !a.is_finite() {
                return Err(Error::NumericalOverflow);
            }
        }
        self.averaged_steps += 1;
        Ok(())
    }
    pub fn observe(&mut self, state: usize, action: usize, next: usize) -> Result<()> {
        self.estimator.observe(state, action, next)
    }
    pub fn observation_count(&self) -> u64 {
        self.estimator.observation_count()
    }
    pub fn estimator_mode(&self) -> EstimatorMode {
        self.estimator.mode()
    }
    pub fn estimator(&self) -> &EmpiricalTransitions {
        &self.estimator
    }
    pub fn iterations(&self) -> u64 {
        self.solver.iterations()
    }
    pub fn state(&self) -> &Iterate {
        self.solver.state()
    }
    pub fn problem(&self) -> &Problem {
        self.solver.problem()
    }
    pub fn transition_probability(&self, state: usize, action: usize, next: usize) -> Result<f64> {
        self.estimator.probability(state, action, next)
    }
    fn refresh(&mut self) -> Result<()> {
        if self.snapshot_version != self.observation_count() {
            self.estimator.write_snapshot(&mut self.kernel)?;
            self.snapshot_version = self.observation_count();
        }
        Ok(())
    }
    pub fn update(&mut self, steps: usize) -> Result<UpdateSummary> {
        self.refresh()?;
        if self.average_mode == IterateAverage::Off {
            self.solver.advance(&mut ExactKernel(&self.kernel), steps)?;
        } else {
            // `advance` with the exact kernel is this loop of `step`s, so
            // the iterates are the same; the average only reads them.
            for _ in 0..steps {
                self.solver.step(&self.kernel)?;
                self.accumulate()?;
            }
        }
        Ok(UpdateSummary {
            optimizer_steps: self.iterations(),
            observations: self.observation_count(),
        })
    }
    pub fn diagnostics(&mut self) -> Result<Diagnostics> {
        self.refresh()?;
        self.solver.diagnostics(&self.kernel)
    }
    pub fn restore(&mut self, state: Iterate) -> Result<()> {
        self.solver.set_state(state)
    }
}
