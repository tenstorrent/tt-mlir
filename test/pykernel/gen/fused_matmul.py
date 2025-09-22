# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.kernel_ast import *
from utils import assert_pcc
import torch


@pykernel_gen(**matmul_fused_template(args=4), kernel_source_dir="tmp/", kernel_source_mode="store")
def fused_matmul(lhs, rhs, bias, out):
    @compute()
    def mm(
        lhs_shard: Tensor,
        rhs_shard: Tensor,
        bias_shard: Tensor,
        out_shard: Tensor,
    ):
        out = lhs_shard @ rhs_shard + out_shard

        out = out + bias_shard

        yield out

    return Program(mm)(lhs, rhs, bias, out)


lhs = torch.randn(128, 128)
rhs = torch.randn(128, 128)
bias = torch.randn(128, 128)
out = torch.zeros(128, 128)
fused_matmul(lhs, rhs, bias, out)

golden = lhs @ rhs
assert_pcc(golden, out)
