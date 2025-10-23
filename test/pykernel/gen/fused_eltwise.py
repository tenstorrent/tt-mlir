# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel._src.kernel_ast import *
from utils import assert_pcc
import torch


@pykernel_gen(**eltwise_fused_template(args=4))
def eltwise_fused(lhs, rhs, bias, out):
    @compute()
    def kernel(
        lhs_shard: Tensor,
        rhs_shard: Tensor,
        bias_shard: Tensor,
        out_shard: Tensor,
    ):
        out = lhs_shard * rhs_shard
        out = out + bias_shard

        yield out

    return Program(kernel)(lhs, rhs, bias, out)


lhs = torch.randn(128, 128)
rhs = torch.randn(128, 128)
bias = torch.randn(128, 128)
out = torch.zeros(128, 128)
eltwise_fused(lhs, rhs, bias, out)

golden = lhs * rhs + bias
assert_pcc(golden, out)
