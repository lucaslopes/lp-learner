//! Validated model and simultaneous GDA/SGDA solver.
#![forbid(unsafe_code)]

mod error;
pub mod model;
pub mod solver;
pub use error::{Error, Result};
