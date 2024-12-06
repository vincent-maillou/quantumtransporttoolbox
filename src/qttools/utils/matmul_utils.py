# Copyright (c) 2024 ETH Zurich and the authors of the qttools package.

from qttools import xp
from qttools.datastructures import DSBSparse


def block_matmul_single(
    A: DSBSparse, B: DSBSparse, in_num_offdiag_blocks: int, i: int, j: int, stack
):
    tmp = xp.zeros((int(A.block_sizes[i]), int(B.block_sizes[j])), dtype=A.dtype)
    for k in range(
        max(i - in_num_offdiag_blocks, 0),
        min(i + in_num_offdiag_blocks + 1, A.num_blocks),
    ):
        if abs(k - j) <= in_num_offdiag_blocks:
            tmp += A.stack[stack].blocks[i, k] @ B.stack[stack].blocks[k, j]
    return tmp


def block_matmul(
    A: DSBSparse,
    B: DSBSparse,
    C: DSBSparse,
    in_num_offdiag_blocks: int,
    out_num_offdiag_blocks: int,
):
    # C = A @ B
    for stack in xp.ndindex(A.data.shape[:-1]):
        for i in range(C.num_blocks):
            for j in range(
                max(i - out_num_offdiag_blocks, 0),
                min(i + out_num_offdiag_blocks + 1, C.num_blocks),
            ):
                C.stack[stack].blocks[i, j] = block_matmul_single(
                    A, B, in_num_offdiag_blocks, i, j, stack
                )


def block_triple_matmul_single(
    A: DSBSparse,
    B: DSBSparse,
    C: DSBSparse,
    in_num_offdiag_blocks: int,
    i: int,
    j: int,
    stack,
):
    # tmp = A_{ik} @ B_{kl} @ C_{lj}
    tmp = xp.zeros((int(A.block_sizes[i]), int(C.block_sizes[j])), dtype=A.dtype)
    for k in range(
        max(i - in_num_offdiag_blocks, 0),
        min(i + in_num_offdiag_blocks + 1, A.num_blocks),
    ):
        for m in range(
            max(i - in_num_offdiag_blocks, 0),
            min(i + in_num_offdiag_blocks + 1, A.num_blocks),
        ):
            if (
                (abs(k - i) <= in_num_offdiag_blocks)
                and (abs(m - i) <= in_num_offdiag_blocks)
                and (abs(k - j) <= in_num_offdiag_blocks)
                and (abs(m - j) <= in_num_offdiag_blocks)
            ):
                tmp += (
                    A.stack[stack].blocks[i, k]
                    @ B.stack[stack].blocks[k, m]
                    @ C.stack[stack].blocks[m, j]
                )
    return tmp


def block_triple_matmul_symmetry_single(
    A: DSBSparse, B: DSBSparse, in_num_offdiag_blocks: int, i: int, j: int, stack
):
    # tmp = A_{ik} @ B_{kl} @ A^H_{lj}
    tmp = xp.zeros((int(A.block_sizes[i]), int(A.block_sizes[j])), dtype=A.dtype)
    for k in range(
        max(i - in_num_offdiag_blocks, 0),
        min(i + in_num_offdiag_blocks + 1, A.num_blocks),
    ):
        for m in range(
            max(i - in_num_offdiag_blocks, 0),
            min(i + in_num_offdiag_blocks + 1, A.num_blocks),
        ):
            if (
                (abs(k - i) <= in_num_offdiag_blocks)
                and (abs(m - i) <= in_num_offdiag_blocks)
                and (abs(k - j) <= in_num_offdiag_blocks)
                and (abs(m - j) <= in_num_offdiag_blocks)
            ):
                tmp += (
                    A.stack[stack].blocks[i, k]
                    @ B.stack[stack].blocks[k, m]
                    @ A.stack[stack].blocks[j, m].conj().T
                )
    return tmp


def block_triple_matmul(
    A: DSBSparse,
    B: DSBSparse,
    C: DSBSparse,
    D: DSBSparse,
    in_num_offdiag_blocks: int,
    out_num_offdiag_blocks: int,
):
    # D = A @ B @ C
    for stack in xp.ndindex(A.data.shape[:-1]):
        for i in range(D.num_blocks):
            for k in range(
                max(i - out_num_offdiag_blocks, 0),
                min(i + out_num_offdiag_blocks + 1, D.num_blocks),
            ):
                D.stack[stack].blocks[i, k] = xp.zeros(
                    (int(A.block_sizes[i]), int(C.block_sizes[k])), dtype=D.dtype
                )
            for j in range(
                max(i - in_num_offdiag_blocks, 0),
                min(i + in_num_offdiag_blocks + 1, A.num_blocks),
            ):
                tmp = block_matmul_single(A, B, in_num_offdiag_blocks, i, j, stack)
                for k in range(
                    max(i - out_num_offdiag_blocks, 0),
                    min(i + out_num_offdiag_blocks + 1, C.num_blocks),
                ):
                    D.stack[stack].blocks[i, k] += tmp @ C.stack[stack].blocks[j, k]


def block_triple_matmul_symmetry(
    A: DSBSparse,
    B: DSBSparse,
    D: DSBSparse,
    in_num_offdiag_blocks: int,
    out_num_offdiag_blocks: int,
):
    # D = A @ B @ A^H
    for stack in xp.ndindex(A.data.shape[:-1]):
        for i in range(D.num_blocks):
            for k in range(
                max(i - out_num_offdiag_blocks, 0),
                min(i + out_num_offdiag_blocks + 1, D.num_blocks),
            ):
                D.stack[stack].blocks[i, k] = xp.zeros(
                    (int(A.block_sizes[i]), int(D.block_sizes[k])), dtype=D.dtype
                )
            for j in range(
                max(i - in_num_offdiag_blocks, 0),
                min(i + in_num_offdiag_blocks + 1, A.num_blocks),
            ):
                tmp = block_matmul_single(A, B, in_num_offdiag_blocks, i, j, stack)
                for k in range(
                    max(i - in_num_offdiag_blocks, 0),
                    min(i + out_num_offdiag_blocks + 1, D.num_blocks),
                ):
                    D.stack[stack].blocks[i, k] += (
                        tmp @ A.stack[stack].blocks[k, j].conj().T
                    )
