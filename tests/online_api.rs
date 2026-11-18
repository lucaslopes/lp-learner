use lp_learner::{Action, Config, Learner};

#[test]
fn environment_observations_and_optimizer_steps_are_distinct() {
    let mut learner = Learner::new(Config::queue(3, 4, 0).expect("queue config")).expect("learner");

    let executed = Action {
        admit: true,
        use_tool: false,
    };
    learner
        .observe(0, executed.index(), 1)
        .expect("realized transition");
    learner.update(7);

    assert_eq!(learner.observation_count(), 1);
    assert_eq!(learner.updates, 7);
    let row = learner.policy(0, 0).expect("policy row");
    assert!(row.iter().all(|probability| probability.is_finite()));
    assert!(row.iter().all(|probability| *probability >= 0.0));
    assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn rejected_transition_indices_do_not_change_observation_count() {
    let mut learner = Learner::new(Config::queue(2, 3, 0).expect("queue config")).expect("learner");

    assert!(learner
        .observe(
            0,
            Action {
                admit: false,
                use_tool: false
            }
            .index(),
            3
        )
        .is_err());
    assert_eq!(learner.observation_count(), 0);
}
