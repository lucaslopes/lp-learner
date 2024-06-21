# Online Learning of Weakly Coupled MDP Policies for Load Balancing and Auto Scaling

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