const ACTIONS: usize = 4; // u = 2 * admit + use_tool
const RESOURCES: usize = 2;

/// The general resource-allocation decision. Each application assigns its
/// own meaning to `use_tool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Action {
    pub admit: bool,
    pub use_tool: bool,
}

impl Action {
    pub fn index(self) -> usize {
        2 * usize::from(self.admit) + usize::from(self.use_tool)
    }
    pub fn from_index(index: usize) -> Result<Self, &'static str> {
        if index >= ACTIONS {
            return Err("action index out of range");
        }
        Ok(Self {
            admit: index / 2 == 1,
            use_tool: index % 2 == 1,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub states: usize,
    pub horizon: usize,
    /// cost[s * 4 + u]
    pub cost: Vec<f64>,
    /// d[(resource * states + s) * 4 + u]
    pub resources: Vec<f64>,
    pub budgets: [f64; RESOURCES],
    pub initial: Vec<f64>,
    /// prior[(u * states + s) * states + next]; each row sums to one.
    pub prior: Vec<f64>,
    pub prior_strength: f64,
    pub gamma_y: f64,
    pub gamma_lambda: f64,
    pub gamma_mu: f64,
    pub eta_y: f64,
    pub eta_dual: f64,
}

impl Config {
    /// A small queue-length model; supplied values are model parameters, not
    /// calibrated empirical facts. Admission and tool-use budgets are fractions
    /// of queues per decision epoch. The prior permits any next queue length.
    pub fn queue(
        capacity: usize,
        horizon: usize,
        initial_state: usize,
    ) -> Result<Self, &'static str> {
        if capacity == 0 || initial_state > capacity || horizon == 0 {
            return Err("invalid queue dimensions");
        }
        let states = capacity.checked_add(1).ok_or("invalid queue dimensions")?;
        crate::model::Dimensions::new(states, ACTIONS, RESOURCES, horizon)
            .map_err(|_| "invalid queue dimensions")?;
        let mut cost = vec![0.0; states * ACTIONS];
        let mut resources = vec![0.0; RESOURCES * states * ACTIONS];
        for s in 0..states {
            for u in 0..ACTIONS {
                let admit = u / 2;
                let use_tool = u % 2;
                cost[s * ACTIONS + u] =
                    s as f64 + 2.0 * (1 - admit) as f64 + 0.15 * use_tool as f64;
                resources[s * ACTIONS + u] = admit as f64;
                resources[(states + s) * ACTIONS + u] = use_tool as f64;
            }
        }
        let mut initial = vec![0.0; states];
        initial[initial_state] = 1.0;
        Ok(Self {
            states,
            horizon,
            cost,
            resources,
            budgets: [2.0 / 3.0, 1.0 / 3.0],
            initial,
            prior: vec![1.0 / states as f64; ACTIONS * states * states],
            prior_strength: 1.0,
            gamma_y: 0.1,
            gamma_lambda: 0.01,
            gamma_mu: 0.01,
            eta_y: 0.04,
            eta_dual: 0.02,
        })
    }

    fn validate(&self) -> Result<(), &'static str> {
        let (s, t) = (self.states, self.horizon);
        crate::model::Dimensions::new(s, ACTIONS, RESOURCES, t)
            .map_err(|_| "invalid model dimensions")?;
        if s < 2
            || t == 0
            || self.cost.len() != s * ACTIONS
            || self.resources.len() != RESOURCES * s * ACTIONS
            || self.initial.len() != s
            || self.prior.len() != ACTIONS * s * s
        {
            return Err("invalid model dimensions");
        }
        if !self
            .cost
            .iter()
            .chain(&self.resources)
            .chain(self.budgets.iter())
            .chain(&self.initial)
            .chain(&self.prior)
            .chain(
                [
                    self.prior_strength,
                    self.gamma_y,
                    self.gamma_lambda,
                    self.gamma_mu,
                    self.eta_y,
                    self.eta_dual,
                ]
                .iter(),
            )
            .all(|x| x.is_finite())
        {
            return Err("model values must be finite");
        }
        if self.prior_strength <= 0.0
            || self.gamma_y < 0.0
            || self.gamma_lambda < 0.0
            || self.gamma_mu < 0.0
            || self.eta_y <= 0.0
            || self.eta_dual <= 0.0
            || self.budgets.iter().any(|x| *x < 0.0)
            || self.initial.iter().any(|x| *x < 0.0)
            || self.prior.iter().any(|x| *x < 0.0)
            || (self.initial.iter().sum::<f64>() - 1.0).abs() > 1e-9
        {
            return Err("invalid probabilities, budgets, or learning parameters");
        }
        for row in self.prior.chunks_exact(s) {
            if (row.iter().sum::<f64>() - 1.0).abs() > 1e-9 {
                return Err("prior rows must sum to one");
            }
        }
        Ok(())
    }
}

/// The primal-dual iterate, block by block (the layout of
/// `solver::Iterate`): `y[(t * states + s) * 4 + u]` occupations;
/// `lambda_resources[l * horizon + t]` the multiplier of resource `l`'s
/// budget at stage `t`; `lambda_nonnegative` one per occupation;
/// `mu_initial[s]` the multiplier of the initial-distribution constraint;
/// `mu_flow[(t - 1) * states + s]` that of the flow-balance constraint into
/// stage `t >= 1`.
#[derive(Debug, Clone)]
pub struct Variables {
    pub y: Vec<f64>,
    pub lambda_resources: Vec<f64>,
    pub lambda_nonnegative: Vec<f64>,
    pub mu_initial: Vec<f64>,
    pub mu_flow: Vec<f64>,
}

impl Variables {
    fn from_iterate(w: &crate::solver::Iterate) -> Self {
        Self {
            y: w.occupations().to_vec(),
            lambda_resources: w.resource_dual().to_vec(),
            lambda_nonnegative: w.nonnegative_dual().to_vec(),
            mu_initial: w.initial_dual().to_vec(),
            mu_flow: w.flow_dual().to_vec(),
        }
    }
}

/// Compatibility facade for the existing economy path dependency.
/// New clients should use `online::OnlineLearner` for validated state access.
#[derive(Debug, Clone)]
pub struct Learner {
    inner: crate::online::OnlineLearner,
    pub variables: Variables,
    pub updates: u64,
}
impl Learner {
    /// The original cumulative transition estimator.
    pub fn new(config: Config) -> Result<Self, &'static str> {
        Self::with_estimator(config, crate::EstimatorMode::Cumulative)
    }
    /// Same model and learning parameters, with a chosen transition-estimator
    /// memory (see `EstimatorMode`). `Config` is unchanged, so existing struct
    /// literals keep compiling; the mode is an explicit construction choice.
    pub fn with_estimator(
        config: Config,
        mode: crate::EstimatorMode,
    ) -> Result<Self, &'static str> {
        config.validate()?;
        let d = crate::model::Dimensions::new(config.states, ACTIONS, RESOURCES, config.horizon)
            .map_err(|_| "invalid model dimensions")?;
        let kernel = crate::model::TransitionKernel::new(d, config.prior)
            .map_err(|_| "invalid transition prior")?;
        let problem = crate::model::Problem::new(
            kernel,
            config.cost,
            config.resources,
            config.budgets.to_vec(),
            config.initial,
        )
        .map_err(|_| "invalid model")?;
        let settings = crate::solver::Settings::new(
            d,
            crate::solver::Regularization {
                primal: config.gamma_y,
                inequality: config.gamma_lambda,
                equality: config.gamma_mu,
            },
            crate::solver::StepSizes {
                primal: config.eta_y,
                dual: config.eta_dual,
            },
            None,
        )
        .map_err(|_| "invalid learning parameters")?;
        let inner = crate::online::OnlineLearner::with_estimator(
            problem,
            settings,
            config.prior_strength,
            mode,
        )
        .map_err(|e| match e {
            crate::Error::InvalidParameter("window size") => "invalid estimator window size",
            crate::Error::InvalidParameter("half-life") => "invalid estimator half-life",
            crate::Error::InvalidDimensions => "estimator window too large",
            _ => "invalid prior strength",
        })?;
        let variables = Variables {
            y: vec![0.; d.occupations()],
            lambda_resources: vec![0.; RESOURCES * d.horizon()],
            lambda_nonnegative: vec![0.; d.occupations()],
            mu_initial: vec![0.; d.states()],
            mu_flow: vec![0.; (d.horizon() - 1) * d.states()],
        };
        Ok(Self {
            inner,
            variables,
            updates: 0,
        })
    }
    pub fn observe(
        &mut self,
        state: usize,
        action: usize,
        next: usize,
    ) -> Result<(), &'static str> {
        self.inner
            .observe(state, action, next)
            .map_err(|e| match e {
                crate::Error::CounterOverflow => "observation counter overflow",
                _ => "observation out of range",
            })
    }
    pub fn observation_count(&self) -> u64 {
        self.inner.observation_count()
    }
    pub fn estimator_mode(&self) -> crate::EstimatorMode {
        self.inner.estimator_mode()
    }
    /// Effective observation count `n` of row `(state, action)` in the blend rule.
    pub fn effective_count(&self, state: usize, action: usize) -> Result<f64, &'static str> {
        self.inner
            .estimator()
            .effective_count(state, action)
            .map_err(|_| "transition index out of range")
    }
    pub fn transition(
        &self,
        state: usize,
        action: usize,
        next: usize,
    ) -> Result<f64, &'static str> {
        self.inner
            .transition_probability(state, action, next)
            .map_err(|_| "transition index out of range")
    }
    pub fn update(&mut self, iterations: usize) {
        // The historic interface exposes mutable blocks. Import once at the call
        // boundary; the numerical loop uses the contiguous validated state.
        let d = self.inner.state().dimensions();
        let flat = self
            .variables
            .y
            .iter()
            .chain(&self.variables.lambda_resources)
            .chain(&self.variables.lambda_nonnegative)
            .chain(&self.variables.mu_initial)
            .chain(&self.variables.mu_flow)
            .copied()
            .collect();
        self.inner
            .restore(crate::solver::Iterate::new(d, flat).expect("valid compatibility state"))
            .expect("matching compatibility state");
        self.inner
            .update(iterations)
            .expect("finite compatibility update");
        let w = self.inner.state();
        self.variables.y.copy_from_slice(w.occupations());
        self.variables
            .lambda_resources
            .copy_from_slice(w.resource_dual());
        self.variables
            .lambda_nonnegative
            .copy_from_slice(w.nonnegative_dual());
        self.variables.mu_initial.copy_from_slice(w.initial_dual());
        self.variables.mu_flow.copy_from_slice(w.flow_dual());
        self.updates = self.inner.iterations();
    }
    pub fn policy(&self, stage: usize, state: usize) -> Result<[f64; ACTIONS], &'static str> {
        self.policy_from_occupations(&self.variables.y, stage, state)
    }

    /// The policy row `policy` reads from the occupations `y` (any vector of
    /// the occupation layout, e.g. an averaged iterate): the same local action
    /// mask (no admission at the last state, no tool at state 0), clipping,
    /// normalization and fallback to action 0.
    pub fn policy_from_occupations(
        &self,
        y: &[f64],
        stage: usize,
        state: usize,
    ) -> Result<[f64; ACTIONS], &'static str> {
        let d = self.inner.state().dimensions();
        if y.len() != d.occupations() {
            return Err("occupation vector has the wrong length");
        }
        let start = d
            .occupation_index(stage, state, 0)
            .map_err(|_| "policy index out of range")?;
        let allowed = std::array::from_fn::<_, ACTIONS, _>(|u| {
            !((state + 1 == d.states() && u / 2 == 1) || (state == 0 && u == 1))
        });
        let weights =
            crate::policy::masked_probabilities(&y[start..start + ACTIONS], &allowed, 0, 1e-12)
                .map_err(|_| "invalid policy weights")?;
        Ok(weights.try_into().expect("four actions"))
    }

    /// Keep an average of the iterates over optimizer steps (see
    /// `IterateAverage`); `Off` (the default) keeps none. The last iterate
    /// (`variables`) is the same in every mode. Changing the mode discards
    /// the average kept so far. If a caller edits `variables` between
    /// updates, the average continues from the edited iterate.
    pub fn set_average(&mut self, mode: crate::IterateAverage) -> Result<(), &'static str> {
        self.inner
            .set_average(mode)
            .map_err(|_| "invalid iterate average")
    }

    pub fn average_mode(&self) -> crate::IterateAverage {
        self.inner.average_mode()
    }

    /// The number of optimizer iterates in the average.
    pub fn averaged_steps(&self) -> u64 {
        self.inner.averaged_steps()
    }

    /// The averaged iterate, once at least one step was averaged.
    pub fn averaged_variables(&self) -> Option<Variables> {
        self.inner.average().map(Variables::from_iterate)
    }

    /// The policy row of the averaged occupations.
    pub fn averaged_policy(
        &self,
        stage: usize,
        state: usize,
    ) -> Result<[f64; ACTIONS], &'static str> {
        let avg = self.inner.average().ok_or("no averaged iterate")?;
        self.policy_from_occupations(avg.occupations(), stage, state)
    }

    /// The budget multipliers `[admission, tool]` of stage `stage` in the
    /// last iterate (`lambda_resources[l * horizon + stage]`).
    pub fn resource_prices(&self, stage: usize) -> Result<[f64; RESOURCES], &'static str> {
        let horizon = self.inner.state().dimensions().horizon();
        if stage >= horizon {
            return Err("stage out of range");
        }
        Ok(std::array::from_fn(|l| {
            self.variables.lambda_resources[l * horizon + stage]
        }))
    }

    /// The equality multipliers of stage `stage` in the last iterate, one per
    /// state: `mu_initial` at stage 0, otherwise the flow-balance block into
    /// `stage`. At a KKT point of the unregularized LP, `-mu` is the
    /// Lagrangian cost-to-go on the support of the occupations; with the
    /// solver's regularization and online it is only an estimate of it.
    pub fn stage_multipliers(&self, stage: usize) -> Result<&[f64], &'static str> {
        let d = self.inner.state().dimensions();
        let s = d.states();
        match stage {
            0 => Ok(&self.variables.mu_initial),
            t if t < d.horizon() => Ok(&self.variables.mu_flow[(t - 1) * s..t * s]),
            _ => Err("stage out of range"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generic_actions_round_trip_and_reject_invalid_indices() {
        for index in 0..4 {
            let action = Action::from_index(index).unwrap();
            assert_eq!(action.index(), index);
        }
        assert_eq!(
            Action {
                admit: true,
                use_tool: false
            }
            .index(),
            2
        );
        assert!(Action::from_index(4).is_err());
    }

    #[test]
    fn observations_update_only_the_selected_transition_row() {
        let mut l = Learner::new(Config::queue(2, 3, 0).unwrap()).unwrap();
        assert!((l.transition(1, 2, 2).unwrap() - 1.0 / 3.0).abs() < 1e-12);
        l.observe(1, 2, 2).unwrap();
        assert!((l.transition(1, 2, 2).unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert!((l.transition(1, 2, 0).unwrap() - 1.0 / 6.0).abs() < 1e-12);
        assert!((l.transition(1, 3, 2).unwrap() - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(l.observation_count(), 1);
        assert!(l.observe(3, 2, 0).is_err());
    }
    #[test]
    fn updates_persist_and_policies_respect_local_action_masks() {
        let mut l = Learner::new(Config::queue(2, 3, 0).unwrap()).unwrap();
        l.update(200);
        let before = l.variables.y.clone();
        l.observe(0, 2, 1).unwrap();
        l.update(20);
        assert_eq!(l.updates, 220);
        assert_ne!(l.variables.y, before);
        let p = l.policy(0, 2).unwrap();
        assert_eq!(p[2], 0.0);
        assert_eq!(p[3], 0.0);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(l.policy(0, 0).unwrap()[1], 0.0);
    }
    #[test]
    fn malformed_models_are_rejected() {
        let mut c = Config::queue(2, 3, 0).unwrap();
        c.prior[0] = -1.0;
        assert!(Learner::new(c).is_err());
        assert!(Config::queue(0, 3, 0).is_err());
    }
}
