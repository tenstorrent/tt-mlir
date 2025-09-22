# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.kernel_ast import *
from utils import assert_pcc
import torch


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, scale=None
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))

    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    return attn_weight @ value


@pykernel_gen(**eltwise_fused_template(args=4))
def sdpa(query, key, value, out):
    @compute()
    def kernel(
        query_shard: Tensor,
        key_shard: Tensor,
        value_shard: Tensor,
        out_shard: Tensor,
    ):
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        out = attn_weight @ value

        yield out

    return Program(kernel)(query, key, value, out)


query = torch.randn(128, 128)
key = torch.randn(128, 128)
value = torch.randn(128, 128)
out = torch.zeros(128, 128)
eltwise_fused(query, key, value, out)

golden = torch.nn.functional.scaled_dot_product_attention(query, key, value)
assert_pcc(golden, out)
