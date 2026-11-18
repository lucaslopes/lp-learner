use crate::{Error, Result};
use std::ops::Range;

/// Validated tensor dimensions. All products and layout offsets fit in usize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dimensions {
    states: usize,
    actions: usize,
    resources: usize,
    horizon: usize,
    n_y: usize,
    n_w: usize,
}
impl Dimensions {
    pub fn new(states: usize, actions: usize, resources: usize, horizon: usize) -> Result<Self> {
        if [states, actions, resources, horizon].contains(&0) {
            return Err(Error::InvalidDimensions);
        }
        let n_y = horizon
            .checked_mul(states)
            .and_then(|n| n.checked_mul(actions))
            .ok_or(Error::InvalidDimensions)?;
        let n_w = n_y
            .checked_mul(2)
            .and_then(|n| n.checked_add(resources.checked_mul(horizon)?))
            .and_then(|n| n.checked_add(horizon.checked_mul(states)?))
            .ok_or(Error::InvalidDimensions)?;
        let kernel_len = states
            .checked_mul(states)
            .and_then(|n| n.checked_mul(actions))
            .ok_or(Error::InvalidDimensions)?;
        let resource_len = resources
            .checked_mul(states)
            .and_then(|n| n.checked_mul(actions))
            .ok_or(Error::InvalidDimensions)?;
        if [n_w, kernel_len, resource_len]
            .iter()
            .any(|&n| n > isize::MAX as usize / std::mem::size_of::<f64>())
        {
            return Err(Error::InvalidDimensions);
        }
        Ok(Self {
            states,
            actions,
            resources,
            horizon,
            n_y,
            n_w,
        })
    }
    pub fn states(self) -> usize {
        self.states
    }
    pub fn actions(self) -> usize {
        self.actions
    }
    pub fn resources(self) -> usize {
        self.resources
    }
    pub fn horizon(self) -> usize {
        self.horizon
    }
    pub fn occupations(self) -> usize {
        self.n_y
    }
    pub fn coordinates(self) -> usize {
        self.n_w
    }
    pub fn kernel_len(self) -> usize {
        self.actions * self.states * self.states
    }
    pub fn resource_len(self) -> usize {
        self.resources * self.states * self.actions
    }
    pub fn occupation_index(self, t: usize, s: usize, u: usize) -> Result<usize> {
        if t >= self.horizon || s >= self.states || u >= self.actions {
            return Err(Error::IndexOutOfRange);
        }
        Ok((t * self.states + s) * self.actions + u)
    }
    pub(crate) fn kernel_index(self, s: usize, u: usize, next: usize) -> Result<usize> {
        if s >= self.states || u >= self.actions || next >= self.states {
            return Err(Error::IndexOutOfRange);
        }
        Ok((u * self.states + s) * self.states + next)
    }
    pub fn layout(self) -> Layout {
        let a = self.n_y;
        let b = a + self.resources * self.horizon;
        let c = b + self.n_y;
        let d = c + self.states;
        Layout {
            dimensions: self,
            y: 0..a,
            resource_dual: a..b,
            nonnegative_dual: b..c,
            initial_dual: c..d,
            flow_dual: d..self.n_w,
        }
    }
}
/// Named ranges into a contiguous saddle vector, in the reference convention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    pub dimensions: Dimensions,
    pub y: Range<usize>,
    pub resource_dual: Range<usize>,
    pub nonnegative_dual: Range<usize>,
    pub initial_dual: Range<usize>,
    pub flow_dual: Range<usize>,
}
