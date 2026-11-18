use std::fmt;

/// A rejected operation leaves the solver's iterate unchanged.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    InvalidDimensions,
    Length {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFinite(&'static str),
    InvalidProbability(&'static str),
    InvalidParameter(&'static str),
    IndexOutOfRange,
    CounterOverflow,
    TapeExhausted,
    NumericalOverflow,
}
pub type Result<T> = std::result::Result<T, Error>;
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length {
                field,
                expected,
                actual,
            } => write!(f, "{field}: expected {expected} entries, got {actual}"),
            Self::NonFinite(x) => write!(f, "{x} must be finite"),
            Self::InvalidProbability(x) => write!(f, "invalid probability distribution: {x}"),
            Self::InvalidParameter(x) => write!(f, "invalid parameter: {x}"),
            Self::InvalidDimensions => f.write_str("invalid or overflowing dimensions"),
            Self::IndexOutOfRange => f.write_str("index out of range"),
            Self::CounterOverflow => f.write_str("counter overflow"),
            Self::TapeExhausted => f.write_str("shared random tape exhausted"),
            Self::NumericalOverflow => f.write_str("nonfinite update; previous iterate preserved"),
        }
    }
}
impl std::error::Error for Error {}
pub(crate) fn length(field: &'static str, x: &[f64], expected: usize) -> Result<()> {
    if x.len() != expected {
        return Err(Error::Length {
            field,
            expected,
            actual: x.len(),
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFinite(field));
    }
    Ok(())
}
