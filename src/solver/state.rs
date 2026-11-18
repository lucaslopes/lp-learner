use crate::{
    error::length,
    model::{Dimensions, Layout},
    Error, Result,
};
/// One contiguous saddle vector. Its dimension and finite-value invariants are checked at construction.
#[derive(Debug, Clone, PartialEq)]
pub struct Iterate {
    pub(crate) layout: Layout,
    pub(crate) values: Vec<f64>,
}
impl Iterate {
    pub fn zeros(d: Dimensions) -> Self {
        Self {
            layout: d.layout(),
            values: vec![0.; d.coordinates()],
        }
    }
    pub fn new(d: Dimensions, values: Vec<f64>) -> Result<Self> {
        length("iterate", &values, d.coordinates())?;
        let layout = d.layout();
        if values[layout.resource_dual.start..layout.nonnegative_dual.end]
            .iter()
            .any(|x| *x < 0.)
        {
            return Err(Error::InvalidParameter(
                "inequality multipliers must be nonnegative",
            ));
        }
        Ok(Self { layout, values })
    }
    pub fn dimensions(&self) -> Dimensions {
        self.layout.dimensions
    }
    pub fn values(&self) -> &[f64] {
        &self.values
    }
    pub fn occupations(&self) -> &[f64] {
        &self.values[self.layout.y.clone()]
    }
    pub fn resource_dual(&self) -> &[f64] {
        &self.values[self.layout.resource_dual.clone()]
    }
    pub fn nonnegative_dual(&self) -> &[f64] {
        &self.values[self.layout.nonnegative_dual.clone()]
    }
    pub fn initial_dual(&self) -> &[f64] {
        &self.values[self.layout.initial_dual.clone()]
    }
    pub fn flow_dual(&self) -> &[f64] {
        &self.values[self.layout.flow_dual.clone()]
    }
}
