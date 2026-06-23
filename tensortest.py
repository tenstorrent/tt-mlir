# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

torch.manual_seed(17617319950514170034)

shape = (32, 32)
dim_arg = 1
N = shape[0] if dim_arg == 0 else shape[1] if dim_arg == 1 else shape[0] * shape[1]
torch.set_printoptions(
    threshold=float("inf"), linewidth=10000, precision=4, sci_mode=False
)
x1 = torch.randn(shape, dtype=torch.bfloat16)
x2, golden = torch.max(x1, dim_arg)
x2 = x2.reshape(-1, 1)
x3 = torch.broadcast_to(x2, shape)
x4 = torch.eq(x1, x3)
x5 = torch.arange(N, 0, -1)
x5 = torch.broadcast_to(x5, shape)
x6 = torch.mul(x5, x4)
x7, _ = torch.max(x6, dim_arg)
x8 = torch.sub(N, x7)

# print(x2)
print(x8)
print(
    f"{'success' if torch.allclose(x8,golden) else f'fail, seed = {torch.initial_seed()}, golden = {golden}'}"
)
