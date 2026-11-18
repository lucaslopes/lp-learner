# Online Learning of Weakly Coupled MDP Policies for Load Balancing and Auto Scaling

The `lplearn` Python package provides a Rust learning backend.
See [installation and usage](docs/python-rust-bridge.md).
Release `0.1.0` reports backend identity only; `0.2.0` adds single-configuration
CVXPY/Numba solves; `0.3.0` adds GDA/SGDA on Rust; and `0.4.0` adds policy
plots, feasibility checks and optional recovery; `0.5.0` adds reproducible,
recovery-aware policy plot benchmarks across Python and Rust backends; `0.6.0`
restores the legacy `Model` and top-level API exports; `0.7.0` adds the
resumable experiment runner and smoke stage; `0.8.0` adds LP-reference and
learning experiment stages; `0.9.0` adds aggregation bounds, `0.10.0` adds
finite-population experiments, `0.11.0` adds commons, and `0.12.0` adds
manuscript tables, figures, tutorial assets and web data export. Planned
`0.13.0` qualifies dependency and release support across the target matrix. See
[the single-job workflow and policy plots](docs/python-baseline.md).
For reproducible CLI timing and policy comparisons, see the
[benchmark protocol](docs/benchmark-protocol.md).
For programs using the earlier Python API, see
[legacy API compatibility](docs/legacy-api.md).
The resumable runner is described in [the experiment guide](docs/experiments.md).

## Authors
- S.R. Eshwar
- Lucas Lopes Felipe
- Alexandre Reiffers-Masson
- Daniel Sadoc Menasché
- Gugan Thoppe

## Abstract
Load balancing and auto scaling are at the core of scalable, contemporary systems, addressing dynamic resource allocation and service rate adjustments in response to workload changes. This paper introduces a novel model and algorithms for tuning load balancers coupled with auto scalers, considering bursty traffic arriving at finite queues. We begin by presenting the problem as a weakly coupled Markov Decision Processes (MDP), solvable via a linear program (LP). However, as the number of control variables of such LP grows combinatorially, we introduce a more tractable relaxed LP formulation, and extend it to tackle the problem of online parameter learning and policy optimization using a two-timescale algorithm based on the LP Lagrangian.

## Acknowledgements
Eshwar was supported by the Prime Minister’s Research Fellowship (PMRF). G. Thoppe’s research was supported in part by DST-SERB’s Core Research Grant CRG/2021/008330, Walmart Center for Tech Excellence, and the Pratiksha Trust Young Investigator Award. This work was partially supported by CAPES STIC AMSUD project RAMONaaS, CNPq and FAPERJ through grants 315110/2020-1, E-26/211.144/2019 and E-26/201.376/2021.

## Paper
For the most updated version of our paper, please click [here](https://arxiv.org/abs/2406.14141).
