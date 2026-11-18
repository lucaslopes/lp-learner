use super::{operator, update, Iterate, KernelProvider, Settings};
use crate::{
    model::{Problem, TransitionKernel},
    Error, Result,
};

#[derive(Debug, Clone)]
pub struct Solver {
    problem: Problem,
    settings: Settings,
    state: Iterate,
    gradient: Vec<f64>,
    candidate: Vec<f64>,
    iterations: u64,
}
impl Solver {
    pub fn new(problem: Problem, settings: Settings) -> Result<Self> {
        let d = problem.dimensions();
        if settings.primal_reg.len() != d.states() {
            return Err(Error::InvalidDimensions);
        }
        Ok(Self {
            problem,
            settings,
            state: Iterate::zeros(d),
            gradient: vec![0.; d.coordinates()],
            candidate: vec![0.; d.coordinates()],
            iterations: 0,
        })
    }
    pub fn problem(&self) -> &Problem {
        &self.problem
    }
    pub fn state(&self) -> &Iterate {
        &self.state
    }
    pub fn iterations(&self) -> u64 {
        self.iterations
    }
    /// Restore values without resetting the lifetime optimizer-step counter.
    pub fn set_state(&mut self, state: Iterate) -> Result<()> {
        if state.dimensions() != self.problem.dimensions() {
            return Err(Error::InvalidDimensions);
        }
        self.state = state;
        Ok(())
    }
    pub fn signed_gradient(&mut self, kernel: &TransitionKernel) -> Result<&[f64]> {
        self.check_kernel(kernel)?;
        operator::evaluate(
            &self.problem,
            kernel,
            &self.settings,
            &self.state,
            &mut self.gradient,
        );
        if self.gradient.iter().any(|x| !x.is_finite()) {
            return Err(Error::NumericalOverflow);
        }
        Ok(&self.gradient)
    }
    fn check_kernel(&self, k: &TransitionKernel) -> Result<()> {
        if k.dimensions() != self.problem.dimensions() {
            return Err(Error::InvalidDimensions);
        }
        Ok(())
    }
    /// One simultaneous update. An error never commits partial numerical state.
    pub fn step(&mut self, kernel: &TransitionKernel) -> Result<()> {
        self.check_kernel(kernel)?;
        operator::evaluate(
            &self.problem,
            kernel,
            &self.settings,
            &self.state,
            &mut self.gradient,
        );
        self.commit_step()
    }
    fn commit_step(&mut self) -> Result<()> {
        let next = self
            .iterations
            .checked_add(1)
            .ok_or(Error::CounterOverflow)?;
        update::candidate(
            &self.state,
            &self.gradient,
            &self.settings,
            &mut self.candidate,
        )?;
        std::mem::swap(&mut self.state.values, &mut self.candidate);
        self.iterations = next;
        Ok(())
    }
    pub fn advance(&mut self, provider: &mut impl KernelProvider, steps: usize) -> Result<()> {
        for _ in 0..steps {
            self.step(provider.next_kernel()?)?;
        }
        Ok(())
    }
    pub fn advance_exact(&mut self, steps: usize) -> Result<()> {
        for _ in 0..steps {
            operator::evaluate(
                &self.problem,
                self.problem.kernel(),
                &self.settings,
                &self.state,
                &mut self.gradient,
            );
            self.commit_step()?;
        }
        Ok(())
    }
    pub fn run(&mut self, provider: &mut impl KernelProvider, plan: RunPlan) -> Result<RunResult> {
        if plan.record_every == 0 {
            return Err(Error::InvalidParameter("record_every must be positive"));
        }
        let d = self.problem.dimensions();
        let window = plan
            .average_window
            .unwrap_or(plan.iterations)
            .min(plan.iterations);
        let start = plan.iterations - window;
        let mut average = Iterate::zeros(d);
        let mut n_average = 0usize;
        let n_rec = if plan.iterations == 0 {
            0
        } else {
            (plan.iterations - 1) / plan.record_every + 1
        };
        let mut trace = Trace {
            iterations: Vec::with_capacity(n_rec),
            occupations: Vec::with_capacity(
                n_rec
                    .checked_mul(d.occupations())
                    .ok_or(Error::InvalidDimensions)?,
            ),
            resource_dual: Vec::with_capacity(
                n_rec
                    .checked_mul(d.resources())
                    .and_then(|n| n.checked_mul(d.horizon()))
                    .ok_or(Error::InvalidDimensions)?,
            ),
            mu_norm: Vec::with_capacity(n_rec),
        };
        for it in 0..plan.iterations {
            let kernel = provider.next_kernel()?;
            // The Python trace is recorded before an update, after drawing its kernel.
            if it % plan.record_every == 0 {
                trace.iterations.push(it as u64);
                trace
                    .occupations
                    .extend_from_slice(self.state.occupations());
                trace
                    .resource_dual
                    .extend_from_slice(self.state.resource_dual());
                trace.mu_norm.push(
                    self.state
                        .initial_dual()
                        .iter()
                        .chain(self.state.flow_dual())
                        .map(|x| x * x)
                        .sum::<f64>()
                        .sqrt(),
                );
                if !trace
                    .mu_norm
                    .last()
                    .expect("record just appended")
                    .is_finite()
                {
                    return Err(Error::NumericalOverflow);
                }
            }
            self.step(kernel)?;
            if it >= start {
                n_average += 1;
                let weight = 1. / n_average as f64;
                for (avg, x) in average.values.iter_mut().zip(&self.state.values) {
                    *avg += weight * (*x - *avg);
                    if !avg.is_finite() {
                        return Err(Error::NumericalOverflow);
                    }
                }
            }
        }
        Ok(RunResult {
            last: self.state.clone(),
            average,
            n_average,
            trace,
        })
    }
    pub fn diagnostics(&mut self, kernel: &TransitionKernel) -> Result<Diagnostics> {
        self.signed_gradient(kernel)?;
        let d = self.problem.dimensions();
        let l = &self.state.layout;
        let residual = self
            .state
            .values
            .iter()
            .zip(&self.gradient)
            .enumerate()
            .map(|(i, (x, g))| {
                let raw = x - g;
                let next = if i >= l.resource_dual.start && i < l.nonnegative_dual.end {
                    raw.max(0.)
                } else {
                    raw
                };
                (x - next).powi(2)
            })
            .sum::<f64>()
            .sqrt();
        let y = self.state.occupations();
        let mut mass = vec![0.; d.horizon()];
        let (mut resource_violation, mut flow_violation, mut nonnegative_violation) =
            (0f64, 0f64, 0f64);
        for t in 0..d.horizon() {
            for s in 0..d.states() {
                let here: f64 = (0..d.actions())
                    .map(|u| y[(t * d.states() + s) * d.actions() + u])
                    .sum();
                mass[t] += here;
                let expected = if t == 0 {
                    self.problem.initial()[s]
                } else {
                    let mut inflow = 0.;
                    for prev in 0..d.states() {
                        for u in 0..d.actions() {
                            inflow += y[((t - 1) * d.states() + prev) * d.actions() + u]
                                * kernel.values()[(u * d.states() + prev) * d.states() + s];
                        }
                    }
                    inflow
                };
                flow_violation = flow_violation.max((here - expected).abs());
            }
            for r in 0..d.resources() {
                let usage: f64 = (0..d.states())
                    .flat_map(|s| (0..d.actions()).map(move |u| (s, u)))
                    .map(|(s, u)| {
                        y[(t * d.states() + s) * d.actions() + u]
                            * self.problem.resources()[(r * d.states() + s) * d.actions() + u]
                    })
                    .sum();
                resource_violation = resource_violation.max(usage - self.problem.budgets()[r]);
            }
        }
        for x in y {
            nonnegative_violation = nonnegative_violation.max(-x);
        }
        if mass
            .iter()
            .chain(
                [
                    resource_violation,
                    flow_violation,
                    nonnegative_violation,
                    residual,
                ]
                .iter(),
            )
            .any(|x| !x.is_finite())
        {
            return Err(Error::NumericalOverflow);
        }
        Ok(Diagnostics {
            iterations: self.iterations,
            mass,
            resource_violation,
            flow_violation,
            nonnegative_violation,
            projected_residual: residual,
        })
    }
}
#[derive(Debug, Clone, Copy)]
pub struct RunPlan {
    pub iterations: usize,
    pub average_window: Option<usize>,
    pub record_every: usize,
}
#[derive(Debug, Clone, PartialEq)]
pub struct Trace {
    pub iterations: Vec<u64>,
    pub occupations: Vec<f64>,
    pub resource_dual: Vec<f64>,
    pub mu_norm: Vec<f64>,
}
#[derive(Debug, Clone)]
pub struct RunResult {
    pub last: Iterate,
    pub average: Iterate,
    pub n_average: usize,
    pub trace: Trace,
}
#[derive(Debug, Clone)]
pub struct Diagnostics {
    pub iterations: u64,
    pub mass: Vec<f64>,
    pub resource_violation: f64,
    pub flow_violation: f64,
    pub nonnegative_violation: f64,
    pub projected_residual: f64,
}
