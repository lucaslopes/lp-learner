use lp_learner::{Action, Config, Learner};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut learner = Learner::new(Config::queue(4, 8, 0)?)?;

    // Feed the learner the state/action/next-state tuple that actually ran.
    let executed_action = Action {
        admit: true,
        use_tool: false,
    };
    learner.observe(0, executed_action.index(), 1)?;

    // Solver updates never fabricate or count environment observations.
    learner.update(250);
    println!(
        "lplearn {}: {} observed transition(s), {} optimizer step(s)",
        lp_learner::VERSION,
        learner.observation_count(),
        learner.updates,
    );

    let policy = learner.policy(0, 0)?;
    println!("stage 0, state 0 action probabilities: {policy:?}");
    Ok(())
}
