# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch

torch.manual_seed(0)
shape = (32, 32)
torch.set_printoptions(
    threshold=float("inf"), linewidth=10000, precision=4, sci_mode=False
)
x1 = torch.randn(shape, dtype=torch.bfloat16)
x2, golden = torch.max(x1, 1)
x2 = x2.reshape(-1, 1)
x3 = torch.broadcast_to(x2, shape)
x4 = torch.eq(x1, x3).int()
x5 = torch.arange(0, shape[0], 1)
x5 = torch.broadcast_to(x5, shape)
x6 = torch.mul(x5, x4)
x7, _ = torch.max(x6, 1)
print(x1)
print(x7)
print(f"{'success' if torch.allclose(x7,golden) else 'fail'}")
