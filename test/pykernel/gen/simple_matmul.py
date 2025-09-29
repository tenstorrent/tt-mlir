# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel._src.kernel_ast import *
from utils import assert_pcc
import torch


@pykernel_gen(**matmul_template)
def matmul(lhs, rhs, out):
    @compute()
    def mm(
        lhs_shard: Tensor,
        rhs_shard: Tensor,
        out_shard: Tensor,
    ):
        out = lhs_shard @ rhs_shard
        yield out

    return Program(mm)(lhs, rhs, out)


lhs = torch.randn(128, 128)
rhs = torch.randn(128, 128)
out = torch.zeros(128, 128)
matmul(lhs, rhs, out)

golden = lhs @ rhs
assert_pcc(golden, out)
