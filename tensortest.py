# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

# # this seed has a duplicate max element
# torch.manual_seed(17617319950514170034)
torch.manual_seed(0)

shape = (3, 17)
dim_arg = 0
keep_dim = False
N = shape[0] if dim_arg == 0 else shape[1] if dim_arg == 1 else shape[0] * shape[1]
torch.set_printoptions(
    threshold=float("inf"), linewidth=10000, precision=4, sci_mode=False
)
x1 = torch.randn(shape, dtype=torch.bfloat16)
golden = torch.argmax(x1, dim_arg, keep_dim)
x2, _ = torch.max(x1, dim_arg, True)
x3 = torch.broadcast_to(x2, shape)
x4 = torch.eq(x1, x3).int()
x5 = torch.arange(N, 0, -1)
if dim_arg == 0:
    x5 = x5.view(-1, 1)
else:
    x5 = x5.view(1, -1)
x6 = torch.broadcast_to(x5, shape)
x7 = torch.mul(x4, x6)
x8, _ = torch.max(x7, dim_arg, keep_dim)
x9 = torch.sub(N, x8)
print(x9)

print(
    f"{'success' if torch.allclose(x9,golden) else f'fail, seed = {torch.initial_seed()}, golden = {golden}'}"
)
