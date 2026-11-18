use crate::{model::TransitionKernel, Error, Result};

/// A kernel provider returns a validated kernel. Calls are optimizer steps, never observations.
pub trait KernelProvider {
    fn next_kernel(&mut self) -> Result<&TransitionKernel>;
}
#[derive(Debug)]
pub struct ExactKernel<'a>(pub &'a TransitionKernel);
impl KernelProvider for ExactKernel<'_> {
    fn next_kernel(&mut self) -> Result<&TransitionKernel> {
        Ok(self.0)
    }
}

pub trait UniformSource {
    fn next_uniform(&mut self) -> Result<f64>;
}
/// Versioned SplitMix64 stream, distinct from NumPy PCG64 and Numba MT19937.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}
impl UniformSource for SplitMix64 {
    fn next_uniform(&mut self) -> Result<f64> {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        Ok((z >> 11) as f64 / 9007199254740992.)
    }
}
/// Shared, pre-generated input, consumed in [iteration,action,state,sample] order.
#[derive(Debug)]
pub struct RandomTape<'a> {
    values: &'a [f64],
    cursor: usize,
}
impl<'a> RandomTape<'a> {
    pub fn new(values: &'a [f64]) -> Result<Self> {
        if values.iter().any(|x| !x.is_finite() || *x < 0. || *x >= 1.) {
            return Err(Error::InvalidParameter(
                "uniform tape values must be in [0,1)",
            ));
        }
        Ok(Self { values, cursor: 0 })
    }
    pub fn consumed(&self) -> usize {
        self.cursor
    }
}
impl UniformSource for RandomTape<'_> {
    fn next_uniform(&mut self) -> Result<f64> {
        let x = *self.values.get(self.cursor).ok_or(Error::TapeExhausted)?;
        self.cursor += 1;
        Ok(x)
    }
}
/// Fresh empirical kernel per iteration, R samples for every (action,state) pair.
/// CDF, sampled kernel and counters are allocated once. Sampling uses the problem
/// being solved, including an aggregate model supplied by an external client.
#[derive(Debug)]
pub struct FreshSamples<R> {
    cdf: Vec<f64>,
    kernel: TransitionKernel,
    source: R,
    samples: usize,
    draws: u64,
    kernels: u64,
}
impl<R: UniformSource> FreshSamples<R> {
    pub fn new(kernel: &TransitionKernel, samples: usize, source: R) -> Result<Self> {
        if samples == 0 {
            return Err(Error::InvalidParameter("samples per pair must be positive"));
        }
        let d = kernel.dimensions();
        let total = d
            .actions()
            .checked_mul(d.states())
            .and_then(|x| x.checked_mul(samples))
            .ok_or(Error::InvalidDimensions)?;
        u64::try_from(total).map_err(|_| Error::InvalidDimensions)?;
        let mut cdf = kernel.values().to_vec();
        for row in cdf.chunks_exact_mut(d.states()) {
            let mut sum = 0.;
            for v in row {
                sum += *v;
                *v = sum;
            }
        }
        Ok(Self {
            cdf,
            kernel: kernel.clone(),
            source,
            samples,
            draws: 0,
            kernels: 0,
        })
    }
    pub fn draws(&self) -> u64 {
        self.draws
    }
    pub fn kernels(&self) -> u64 {
        self.kernels
    }
    pub fn source(&self) -> &R {
        &self.source
    }
}
impl<R: UniformSource> KernelProvider for FreshSamples<R> {
    fn next_kernel(&mut self) -> Result<&TransitionKernel> {
        let s = self.kernel.dimensions().states();
        let count = (self.kernel.dimensions().actions() * s * self.samples) as u64;
        self.draws
            .checked_add(count)
            .ok_or(Error::CounterOverflow)?;
        let new_kernels = self.kernels.checked_add(1).ok_or(Error::CounterOverflow)?;
        self.kernel.values.fill(0.);
        for (cdf, row) in self
            .cdf
            .chunks_exact(s)
            .zip(self.kernel.values.chunks_exact_mut(s))
        {
            for _ in 0..self.samples {
                let x = self.source.next_uniform()?;
                self.draws += 1;
                if !x.is_finite() || !(0. ..1.).contains(&x) {
                    return Err(Error::InvalidParameter(
                        "uniform source returned value outside [0,1)",
                    ));
                }
                let mut i = 0;
                while i + 1 < s && x >= cdf[i] {
                    i += 1;
                }
                row[i] += 1.;
            }
            for v in row {
                *v /= self.samples as f64;
            }
        }
        self.kernels = new_kernels;
        Ok(&self.kernel)
    }
}

/// MT19937 with the legacy NumPy/Numba seed and 53-bit uniform conversion.
/// Shared-tape tests independently verify this stream against the pinned oracle.
#[derive(Clone)]
pub struct Mt19937 {
    state: [u32; 624],
    cursor: usize,
}
impl std::fmt::Debug for Mt19937 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mt19937")
            .field("cursor", &self.cursor)
            .finish_non_exhaustive()
    }
}
impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, cursor: 624 }
    }
    fn next_u32(&mut self) -> u32 {
        if self.cursor == 624 {
            for i in 0..624 {
                let x = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (x >> 1)
                    ^ if x & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.cursor = 0;
        }
        let mut y = self.state[self.cursor];
        self.cursor += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }
}
impl UniformSource for Mt19937 {
    fn next_uniform(&mut self) -> Result<f64> {
        let a = (self.next_u32() >> 5) as u64;
        let b = (self.next_u32() >> 6) as u64;
        Ok((a * 67108864 + b) as f64 / 9007199254740992.)
    }
}

#[cfg(test)]
mod tests {
    use super::{Mt19937, UniformSource};

    #[test]
    fn matches_numpy_random_state_uniforms() {
        // NumPy RandomState(7).random_sample(5), the seeded Numba stream.
        let expected = [
            0x3fb388f0a7465c98,
            0x3fe8f5184146a0a7,
            0x3fdc0ee597d36686,
            0x3fe726a071d284d9,
            0x3fef4bb0a93b67b2,
        ];
        let mut stream = Mt19937::new(7);
        for bits in expected {
            assert_eq!(stream.next_uniform().unwrap().to_bits(), bits);
        }
    }
}
