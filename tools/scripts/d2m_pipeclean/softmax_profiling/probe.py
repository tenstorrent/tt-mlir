# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch, d2m_jit as d2m, sys

mode = sys.argv[1]
N = int(sys.argv[2])
# Optional 3rd arg = width in tiles (rows stays N tiles). Lets us build 1xK
# "wide" tensors (Ht=1) so ttnn softmax stays on a single core for an
# apples-to-apples single-core scaling sweep.
NC = int(sys.argv[3]) if len(sys.argv) > 3 else N
# Optional 4th arg = grid rows G: shard the N tile-rows across G cores (1 row-block
# each), reducing along the cols on each core. Lets us match ttnn's softmax core
# count (ttnn splits an NxN softmax across Ht=N cores) for an apples-to-apples
# multi-core comparison, and exercises d2m's programmable grid.
G = int(sys.argv[4]) if len(sys.argv) > 4 else 1
assert N % G == 0, "grid rows must divide tile rows"
S = N * 32
SC = NC * 32
if mode == "softmax":

    @d2m.kernel
    def k(in_t, out_t, m_blocks, n_blocks):
        # Per-core block offset (single-core: core_index==0, m_blocks==1 -> [0,0]).
        # Softmax reduces along the full width, so width stays ONE block (n=0);
        # only tile-rows are split across the grid (grid=(G,1)).
        m_off = core_index(0) * m_blocks
        for mm in range(m_blocks):
            x = remote_load(in_t, [m_off + mm, 0])
            mx = reduce_max(x, 1)
            e = exp(x - mx)
            sm = reduce_sum(e, 1)
            remote_store(out_t, [m_off + mm, 0], e / sm)

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
L = d2m.Layout(shape=(S, SC), dtype=d2m.float32, block_shape=[N // G, NC], grid_shape=[G, 1])
torch.manual_seed(0)
a = torch.randn(S, SC)
o = d2m.zeros(L)
k(d2m.to_layout(a.contiguous(), L), o, 1, 1, grid=(G, 1))
res = o.to_host()
x = ref(a).flatten().double()
y = res.flatten().double()
print(
    "RESULT %s %dx%d PCC=%.6f"
    % (mode, N, NC, torch.corrcoef(torch.stack([x, y]))[0, 1].item())
)
import os as _os

if _os.environ.get("DIAG") and mode == "softmax":
    rr = res.reshape(S, SC).double()
    rf = ref(a).reshape(S, SC).double()
    print("row-sum range:", rr.sum(-1).min().item(), rr.sum(-1).max().item())
    print("ref[0,:6] =", rf[0, :6].tolist())
    print("res[0,:6] =", rr[0, :6].tolist())
    # which rows are wrong? PCC per 32-row tile-row band
    for tr in range(N):
        band = slice(tr * 32, (tr + 1) * 32)
        p = torch.corrcoef(torch.stack([rf[band].flatten(), rr[band].flatten()]))[0, 1].item()
        print(f"  tile-row {tr}: PCC={p:.4f}")
