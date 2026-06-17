# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Force ttnn.softmax onto a SINGLE core for a square KxK-tile tensor, so it is
apples-to-apples with the single-core d2m softmax (which runs KxK on 1 core).

ttnn's default softmax splits a KxK softmax across Ht=K tile-rows (=K cores). The
only way to pin it to a 1x1 grid is the sharded program factory, selected by
passing SoftmaxShardedMultiCoreProgramConfig with compute_with_storage_grid_size
= (1,1) and an L1-sharded input whose single shard = the whole tensor.

Run ONE K per process under the device profiler:
  TT_METAL_DEVICE_PROFILER=1 python3 ttnn_1core.py <K>
then read the device kernel duration from profile_log_device.csv (printed here).
"""
import os
import sys
import torch
import ttnn

K = int(sys.argv[1])
S = 32 * K

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    a = torch.randn(
        1, 1, S, S, dtype=torch.float32
    )  # rank-4: softmax_in_place needs dim=-1
    grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}
    )
    shard_spec = ttnn.ShardSpec(grid, [S, S], ttnn.ShardOrientation.ROW_MAJOR)
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec
    )
    # match the working examples: make it interleaved on device first, then reshard.
    t_il = ttnn.from_torch(a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    t = ttnn.to_memory_config(t_il, mem)
    # zero mask so scale_mask_softmax == plain softmax: input*1.0 + 0 = input.
    # (the sharded path requires scale, and scale requires a mask.)
    mask = ttnn.from_torch(
        torch.zeros(1, 1, S, S, dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
    )
    cfg = ttnn.SoftmaxShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(1, 1),
        subblock_w=K,
        block_h=K,
        block_w=K,
    )
    # warmup + the profiled run. scale_mask_softmax_in_place is the op that
    # dispatches to the *sharded* program factory (which honors cfg's 1x1 grid);
    # with no scale/mask and numeric_stable=True it is a plain stable softmax.
    for _ in range(3):
        out = ttnn.scale_mask_softmax_in_place(
            t, scale=1.0, mask=mask, program_config=cfg, numeric_stable=True
        )
    ttnn.synchronize_device(dev)

    # correctness
    res = ttnn.to_torch(out).float()
    ref = torch.softmax(a, dim=-1)
    pcc = torch.corrcoef(torch.stack([ref.flatten().double(), res.flatten().double()]))[
        0, 1
    ].item()
    print(f"RESULT ttnn-1core {K}x{K} PCC={pcc:.6f}")
    if os.environ.get("DIAG"):
        rr = res.reshape(S, S)
        print(
            "res row0 sum =",
            rr[0].sum().item(),
            " row-sum range:",
            rr.sum(-1).min().item(),
            rr.sum(-1).max().item(),
        )
        print("ref[0,:6]=", ref.reshape(S, S)[0, :6].tolist())
        print("res[0,:6]=", rr[0, :6].tolist())
        # is it softmax over the OTHER axis?
        alt = torch.softmax(a.reshape(S, S), dim=0)
        print(
            "PCC vs softmax(dim=0)=",
            torch.corrcoef(
                torch.stack([alt.flatten().double(), rr.flatten().double()])
            )[0, 1].item(),
        )

    try:
        ttnn.ReadDeviceProfiler(dev)
    except Exception as e:
        print("ReadDeviceProfiler:", e)
finally:
    ttnn.close_device(dev)
