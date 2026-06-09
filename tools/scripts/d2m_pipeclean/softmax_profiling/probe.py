# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch, d2m_jit as d2m, sys

mode = sys.argv[1]
N = int(sys.argv[2])
S = N * 32
if mode == "softmax":

    @d2m.kernel
    def k(in_t, out_t, m, n):
        x = remote_load(in_t, [0, 0])
        mx = reduce_max(x, 1)
        e = exp(x - mx)
        sm = reduce_sum(e, 1)
        remote_store(out_t, [0, 0], e / sm)

    ref = lambda a: torch.softmax(a, dim=1)
elif mode == "exp":  # trivial single-op, few CBs

    @d2m.kernel
    def k(in_t, out_t, m, n):
        remote_store(out_t, [0, 0], exp(remote_load(in_t, [0, 0])))

    ref = lambda a: torch.exp(a)
elif mode == "ident":  # no compute op: pure load->store (tilize+DM+untilize)

    @d2m.kernel
    def k(in_t, out_t, m, n):
        remote_store(out_t, [0, 0], remote_load(in_t, [0, 0]))

    ref = lambda a: a
L = d2m.Layout(shape=(S, S), dtype=d2m.float32, block_shape=[N, N], grid_shape=[1, 1])
torch.manual_seed(0)
a = torch.randn(S, S)
o = d2m.zeros(L)
k(d2m.to_layout(a.contiguous(), L), o, 1, 1, grid=(1, 1))
res = o.to_host()
x = ref(a).flatten().double()
y = res.flatten().double()
print(
    "RESULT %s %dx%d PCC=%.6f"
    % (mode, N, N, torch.corrcoef(torch.stack([x, y]))[0, 1].item())
)
