//! Persistent simultaneous GDA/SGDA with transactional steps and reusable buffers.
mod engine;
mod operator;
mod sampling;
mod settings;
mod state;
mod update;
pub use engine::{Diagnostics, RunPlan, RunResult, Solver, Trace};
pub use sampling::{
    ExactKernel, FreshSamples, KernelProvider, Mt19937, RandomTape, SplitMix64, UniformSource,
};
pub use settings::{Regularization, Settings, StepSizes};
pub use state::Iterate;
