# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

_NONCONTIGUOUS_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="index_copy_ with a non-contiguous multi-element index does not honor "
    "the index values on device: the write lands in contiguous slots from 0 "
    "instead of at the indexed positions. A single-element index, or a "
    "contiguous index from 0, writes correctly.",
)


@pytest.mark.multichip
@pytest.mark.parametrize(
    "index",
    [
        pytest.param(torch.tensor([0, 1]), id="contiguous"),
        pytest.param(
            torch.tensor([1, 5]), id="noncontiguous", marks=_NONCONTIGUOUS_XFAIL
        ),
    ],
)
def test_index_copy_sharded_compile(tt_pg, index: torch.Tensor) -> None:
    """A sharded in-place `index_copy_` (the StaticCache write) on the compile
    path, checked vs CPU.

    Mirrors the cache layout `[batch, heads, seq, dim]`: shard the batch dim,
    write new tokens along the (unsharded) seq dim, so the write never crosses a
    shard boundary. bf16 in/out so the copy is bit-exact.
    """
    n = torch.tt.num_chips()
    mesh = torch.tt.init_device_mesh((n,), mesh_dim_names=("x",))
    base = torch.zeros(n * 2, 3, 8, 4, dtype=torch.bfloat16)
    src = torch.randn(n * 2, 3, index.numel(), 4, dtype=torch.bfloat16)

    def write(dst, source, idx):
        return dst.index_copy_(2, idx, source)

    d_base = distribute_tensor(base.to("tt"), mesh, [Shard(0)])
    d_src = distribute_tensor(src.to("tt"), mesh, [Shard(0)])
    d_idx = distribute_tensor(index.to("tt"), mesh, [Replicate()])

    out = torch.compile(write, backend="tt", dynamic=False)(d_base, d_src, d_idx)
    got = out.full_tensor().cpu()

    ref = base.clone().index_copy_(2, index, src)
    torch.testing.assert_close(got, ref, atol=0.0, rtol=0.0)
