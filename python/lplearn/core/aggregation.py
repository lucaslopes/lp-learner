'''
State aggregation for the relaxed LP (Section 5 of the manuscript).\n
The aggregation map V (J x S) sums state-indexed mass inside a block and the lifting map
U (S x J) spreads block mass uniformly, so V U = I and U V is block averaging.
'''


from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .instance import WCMDP


Partition = tuple[tuple[int, ...], ...]


# Partitions of Table `tab:methods` (K = 5) and of the K = 11 instance.
PARTITIONS_K5 = {
    'E': ((0,), (1, 2, 3, 4), (5,)),
    'F': ((0, 1), (2, 3), (4, 5)),
    'G': ((0,), (1, 2, 3, 4, 5)),
    'H': ((0, 1), (2, 3, 4, 5)),
    'I': ((0, 1, 2), (3, 4, 5)),
    'J': ((0, 1, 2, 3), (4, 5)),
    'K': ((0, 1, 2, 3, 4), (5,)),
}
PARTITION_K11 = ((0,), (1, 2), (3, 4, 5), (6, 7, 8), (9, 10), (11,))


def as_partition(partition: Sequence[Sequence[int]], S: int) -> Partition:
    '''Validate that `partition` covers 0..S-1 exactly once and return it as nested tuples.'''
    blocks = tuple(tuple(int(s) for s in block) for block in partition)
    flat = sorted(s for block in blocks for s in block)
    if flat != list(range(S)) or any(len(block) == 0 for block in blocks):
        raise ValueError(f'{partition} is not a partition of 0..{S - 1}')
    return blocks


def singleton_partition(S: int) -> Partition:
    return tuple((s,) for s in range(S))


def maps(partition: Sequence[Sequence[int]], S: int) -> tuple[np.ndarray, np.ndarray]:
    '''Return the lifting map U (S x J) and the aggregation map V (J x S).'''
    blocks = as_partition(partition, S)
    U = np.zeros((S, len(blocks)))
    V = np.zeros((len(blocks), S))
    for j, block in enumerate(blocks):
        U[list(block), j] = 1.0 / len(block)
        V[j, list(block)] = 1.0
    return U, V


def block_sizes(partition: Sequence[Sequence[int]]) -> np.ndarray:
    return np.array([len(block) for block in partition], dtype=float)


def aggregate(model: WCMDP, partition: Sequence[Sequence[int]]) -> WCMDP:
    '''
    Aggregate WC-MDP of eqs. (agg-cost)-(agg-transition):
    block-average costs and consumptions, P_hat = D^{-1} V P V^T and m0_hat = V m0.
    '''
    U, V = maps(partition, model.S)
    W = U.T  # = D^{-1} V, block averaging of a state-indexed vector
    P_hat = np.einsum('js,usr,kr->ujk', W, model.P, V)
    c_hat = W @ model.c
    d_hat = np.einsum('js,lsu->lju', W, model.d)
    m0_hat = V @ model.m0
    return WCMDP(P=P_hat, c=c_hat, d=d_hat, budgets=model.budgets.copy(), m0=m0_hat, T=model.T)


def lift_y(y_hat: np.ndarray, U: np.ndarray) -> np.ndarray:
    '''Lift aggregate occupation measures y_hat[t, J, u] to the detailed grid, eq. (lift-y).'''
    return np.einsum('sj,tju->tsu', U, y_hat)


def aggregate_y(y: np.ndarray, V: np.ndarray) -> np.ndarray:
    '''Sum detailed occupation measures y[t, s, u] inside each block.'''
    return np.einsum('js,tsu->tju', V, y)


def lifted_kernel(model: WCMDP, partition: Sequence[Sequence[int]]) -> np.ndarray:
    '''
    Lifted aggregate kernel P_bar[u] = V^T P_hat[u] U^T (S x S): from a detailed state in block P,
    move to block P' with the aggregate probability and spread it uniformly inside P'.
    '''
    U, V = maps(partition, model.S)
    P_hat = aggregate(model, partition).P
    return np.einsum('js,ujk,rk->usr', V, P_hat, U)


def deltas(model: WCMDP, partition: Sequence[Sequence[int]]) -> dict[str, np.ndarray | float]:
    '''
    Within-block variation terms of Proposition 1 for every block and their maxima.\n
    `delta_P` uses the lifted row of `lifted_kernel`; `delta_P_block` compares only the
    probabilities of reaching each block, max_u max_{s in P} || V P^u_{s,.} - P_hat^u_{P,.} ||_1,
    which is the part that strong lumpability sets to zero.
    '''
    blocks = as_partition(partition, model.S)
    U, V = maps(blocks, model.S)
    P_bar = lifted_kernel(model, blocks)
    P_hat = aggregate(model, blocks).P
    out: dict[str, list[float]] = {k: [] for k in ('delta_c', 'delta_P', 'delta_P_block', 'delta_d', 'delta_0')}
    for j, block in enumerate(blocks):
        idx = list(block)
        out['delta_c'].append(float(np.max(model.c[idx].max(axis=0) - model.c[idx].min(axis=0))))
        rows = np.abs(model.P[:, idx, :] - P_bar[:, idx, :]).sum(axis=2)
        out['delta_P'].append(float(rows.max()))
        to_blocks = np.einsum('usr,jr->usj', model.P[:, idx, :], V)
        out['delta_P_block'].append(float(np.abs(to_blocks - P_hat[:, j][:, None, :]).sum(axis=2).max()))
        out['delta_d'].append(float(np.max(model.d[:, idx].max(axis=1) - model.d[:, idx].min(axis=1))))
        out['delta_0'].append(float(model.m0[idx].max() - model.m0[idx].min()))
    result: dict[str, np.ndarray | float] = {f'{k}_blocks': np.array(v) for k, v in out.items()}
    result.update({k: float(np.max(v)) for k, v in out.items()})
    return result
