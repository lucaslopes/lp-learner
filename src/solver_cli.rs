//! JSON boundary for one GDA or SGDA solve.
use lp_learner::{
    model::{Dimensions, Problem, TransitionKernel},
    solver::{
        ExactKernel, FreshSamples, Iterate, Mt19937, Regularization, RunPlan, RunResult, Settings,
        Solver, StepSizes,
    },
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    io::{self, BufRead},
    time::Instant,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelInput {
    states: usize,
    actions: usize,
    resources: usize,
    horizon: usize,
    p: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
    budgets: Vec<f64>,
    initial: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {
    gamma_y: f64,
    gamma_lambda: f64,
    gamma_mu: f64,
    eta_y: f64,
    eta_dual: f64,
    state_weights: Option<Vec<f64>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema: String,
    command: String,
    model: ModelInput,
    parameters: Parameters,
    initial_state: Vec<f64>,
    iterations: usize,
    samples_per_pair: usize,
    seed: u32,
    average_window: Option<usize>,
    record_every: usize,
}

fn execute(r: &Request) -> Result<RunResult, Box<dyn std::error::Error>> {
    if r.schema != "lpl-core/1" || r.command != "solve" {
        return Err("expected lpl-core/1 solve request".into());
    }
    let m = &r.model;
    let dimensions = Dimensions::new(m.states, m.actions, m.resources, m.horizon)?;
    let kernel = TransitionKernel::new(dimensions, m.p.clone())?;
    let problem = Problem::new(
        kernel.clone(),
        m.c.clone(),
        m.d.clone(),
        m.budgets.clone(),
        m.initial.clone(),
    )?;
    let p = &r.parameters;
    let settings = Settings::new(
        dimensions,
        Regularization {
            primal: p.gamma_y,
            inequality: p.gamma_lambda,
            equality: p.gamma_mu,
        },
        StepSizes {
            primal: p.eta_y,
            dual: p.eta_dual,
        },
        p.state_weights.as_deref(),
    )?;
    let mut solver = Solver::new(problem, settings)?;
    solver.set_state(Iterate::new(dimensions, r.initial_state.clone())?)?;
    let plan = RunPlan {
        iterations: r.iterations,
        average_window: r.average_window,
        record_every: r.record_every,
    };
    let result = if r.samples_per_pair == 0 {
        solver.run(&mut ExactKernel(&kernel), plan)?
    } else {
        let mut samples = FreshSamples::new(&kernel, r.samples_per_pair, Mt19937::new(r.seed))?;
        solver.run(&mut samples, plan)?
    };
    Ok(result)
}

fn main() {
    for line in io::stdin().lock().lines() {
        let run = || -> Result<Value, Box<dyn std::error::Error>> {
            let line = line?;
            if line.len() > 64 * 1024 * 1024 {
                return Err("request exceeds 64 MiB".into());
            }
            let request: Request = serde_json::from_str(&line)?;
            let start = Instant::now();
            let result = execute(&request)?;
            let execute_seconds = start.elapsed().as_secs_f64();
            Ok(json!({"schema":"lpl-core/1", "output":{
                "last":result.last.values(), "average":result.average.values(),
                "n_average":result.n_average, "trace_iter":result.trace.iterations,
                "trace_y":result.trace.occupations, "trace_resource_dual":result.trace.resource_dual,
                "trace_mu_norm":result.trace.mu_norm},
                "execute_seconds":execute_seconds,
                "backend_version":env!("CARGO_PKG_VERSION"),
                "debug_assertions":cfg!(debug_assertions)}))
        };
        match run() {
            Ok(value) => println!("{value}"),
            Err(error) => {
                eprintln!(
                    "{}",
                    json!({"schema":"lpl-core/1", "error":error.to_string()})
                );
                std::process::exit(2);
            }
        }
    }
}
