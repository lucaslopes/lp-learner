# Rust queue and online learning API

The Rust library API is available to native Rust consumers and to
`wasm32-unknown-unknown` builds. Its public API version is
`lp_learner::VERSION`, derived from the Cargo package version. The Python wheel
and CLI are distribution interfaces; they are not required by Rust or Wasm
consumers.

## Observation contract

Call `observe(state, action, next_state)` once for each transition that the
environment actually executed. A proposed action that a capacity limit,
admission guard, physical need floor, or other application rule rejected is
not an observation. The caller retains responsibility for action selection,
application constraints, simulation, and recording the realized transition.

`observation_count()` counts those realized transitions. `update(steps)` runs
optimizer iterations and reports their lifetime count separately. Optimizer
steps do not sample the environment and do not increment the observation
counter. Calling `update` after new observations refreshes the transition
estimate before advancing the iterate. This is the shared contract used by
the Island queue learner and Kantovian Stage 0; economic meanings and guards
remain in those consumer projects.

## Basic use

```rust
use lp_learner::{Action, Config, Learner};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut learner = Learner::new(Config::queue(4, 8, 0)?)?;

    // Record the state, action, and next state after execution.
    let action = Action { admit: true, use_tool: false };
    let realized_state = 0;
    let realized_next_state = 1;
    learner.observe(realized_state, action.index(), realized_next_state)?;

    // Optimizer work is tracked independently of environment observations.
    learner.update(250);
    assert_eq!(learner.observation_count(), 1);
    assert_eq!(learner.updates, 250);

    let probabilities = learner.policy(0, realized_state)?;
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    Ok(())
}
```

`Config::queue` is a small generic fixture, not an empirical calibration.
Production consumers can construct `model::Problem` and `solver::Settings`
from their own validated parameters, then use `online::OnlineLearner`
directly. `transition::EstimatorMode` selects cumulative, sliding-window,
or exponentially decayed observation history. `online::IterateAverage`
controls optional averaging over optimizer iterates; it does not change the
transition-observation clock.

## Native dependency

```toml
[dependencies]
lplearn = { git = "https://github.com/lucaslopes/lplearn", tag = "v1.0.0", default-features = false }
```

This plan distributes the Rust source through the versioned GitHub repository
tag; publishing the crate to crates.io is outside its scope. For reproducible
economy builds, pin the tag or its resolved commit and retain `Cargo.lock`.
The browser artifact should compile the same crate revision to Wasm. Python
does not need to be installed to link the Rust library.

The API does not promise that a learned policy is feasible for application
rules. Consumers must apply their own realized-action guards and record the
action that passed those guards. The online learner estimates transitions and
updates a primal-dual iterate; it is not a replacement for the execution
mechanism.
