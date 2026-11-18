# Online Learning of Weakly Coupled MDP Policies for Load Balancing and Auto Scaling

`lplearn` provides a Python API and CLI for single-configuration policy
solves with CVXPY, Numba, or a packaged Rust backend.
Version `0.5.0` adds reproducible comparisons of the full policy plot
workflow, with matched Numba and Rust learner settings and optional
feasibility recovery. See the
[bridge guide](https://github.com/lucaslopes/lplearn/blob/main/docs/python-rust-bridge.md),
[policy plot examples](https://github.com/lucaslopes/lplearn/blob/main/docs/python-baseline.md),
and [benchmark protocol](https://github.com/lucaslopes/lplearn/blob/main/docs/benchmark-protocol.md).

Version `0.6.0` restores the legacy `Model` and top-level API exports. See the
[legacy API guide](https://github.com/lucaslopes/lplearn/blob/main/docs/legacy-api.md).

Version `0.7.0` adds a resumable experiment runner and a smoke stage. See the
[experiment guide](https://github.com/lucaslopes/lplearn/blob/main/docs/experiments.md).

Version `0.8.0` adds CVXPY reference and Numba or Rust learning stages to
the experiment runner.

Version `0.9.0` adds the state-aggregation bound experiment.

Version `0.10.0` adds finite-population policy simulation.

Version `0.11.0` adds Commons comparison experiments.

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
