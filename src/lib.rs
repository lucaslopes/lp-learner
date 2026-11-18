//! Validated model and observation-driven online GDA/SGDA solver.
#![forbid(unsafe_code)]

mod compat;
mod error;
pub mod model;
pub mod online;
pub mod policy;
pub mod solver;
pub mod transition;

pub use compat::{Action, Config, Learner, Variables};
pub use error::{Error, Result};
pub use online::IterateAverage;
pub use transition::EstimatorMode;

/// Version of the public Rust library API, matching the Cargo package version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
