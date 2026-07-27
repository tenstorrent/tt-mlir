# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# MINIMAL REPRO: the fused distributed_rms_norm stats buffer is never freed from L1.
#
# WHAT THIS SHOWS
#   The fused path (ttnn.distributed_rms_norm) needs a persistent all-gather
#   "stats" scratch. tt-mlir materializes it as a ttnn.empty hoisted into the
#   function prelude and CSE-shared across every norm, and NO ttnn.deallocate is
#   emitted for it -- so it stays resident in L1 after the program finishes.
#   In a real model that resident buffer (grown to num_devices tiles) eventually
#   collides in L1 with an unrelated op's circular buffers (the "Statically
#   allocated circular buffers clash with L1 buffers" OOM).
#
#   This repro isolates that: two fused norms chained together share ONE hoisted
#   stats ttnn.empty.  The test asserts L1 usage returns to baseline after the
#   program completes.  In the buggy state (no ttnn.deallocate for the stats
#   buffer) the assertion FAILS because the buffer remains allocated in L1.
#
#   Runs on a WH Galaxy (4x8) mesh.
#
# RUN
#   Requires SYSTEM_DESC_PATH set (see repo CLAUDE.md) and the golden harness
#   deps (this file lives next to test_normalization.py so it inherits its
#   conftest).  Use -s to see the per-op L1 printout:
#     pytest -svv \
#       test/python/golden/ttir_ops/normalization/test_stats_l1_persistence_repro.py
#
# WHAT TO EXPECT
#   FAIL  (before the compiler fix): L1 leaked > 0 bytes after program exit.
#   PASS  (after the fix): L1 returns to baseline; delta == 0.
#
#   Optional per-op memory log (requires -DTT_RUNTIME_DEBUG=1 build AND
#   TT_METAL_LOGGER_LEVEL=Info set before pytest):
#     * An EmptyOp line where totalBytesAllocatedPerBank jumps up.
#     * That elevated figure persists through both DistributedRMSNormOps.
#     * No DeallocateOp for the stats tensor -- unlike every other tensor.
# =============================================================================

from collections import OrderedDict
from typing import List, Optional

import pytest

import _ttmlir_runtime as _rt_mod

from conftest import get_request_kwargs
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from builder.base.builder_enums import MeshShardDirection, MeshShardType
from test_utils import shape_str, make_shard_shape

pytestmark = pytest.mark.frontend("ttir")

_rt = _rt_mod.runtime


def _l1_allocated_bytes(device) -> int:
    """Return total L1 bytes currently allocated across all banks on the device."""
    views = device.get_memory_view()
    v = views[_rt.MemoryBufferType.L1]
    return v.total_bytes_allocated_per_bank * v.num_banks


def _enable_operation_memory_logging():
    """Best-effort: turn on per-op spdlog memory logging if the runtime supports it.

    Requires a build with -DTT_RUNTIME_DEBUG=1 (DebugHooks.get() returns None
    otherwise) AND TT_METAL_LOGGER_LEVEL=Info set before libraries load AND
    pytest -s.  Returns None silently when unavailable -- the test still runs.

    Uses _ttmlir_runtime directly (already loaded by conftest) rather than
    importing ttrt.runtime, which loads a second _ttmlir_runtime.so and causes a
    nanobind duplicate-enum-key abort.
    """
    if _rt.DebugHooks.get() is None:
        print(
            "\n[memory-log] TT_RUNTIME_DEBUG=1 not compiled in; "
            "per-op spdlog logging unavailable (L1 assertion still runs)."
        )
        return None
    _rt.set_memory_log_level(_rt.MemoryLogLevel.OPERATION)
    return _rt


# (1,1,32,896) is the exact shape from the failing model. With cluster_axis=1
# and 8 galaxy columns each shard is (1,1,32,112).
@pytest.mark.parametrize("shape", [(1, 1, 32, 896)], ids=shape_str)
@pytest.mark.parametrize("mesh_shape", [(4, 8)], ids=shape_str)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("target", ["ttnn"])
def test_stats_buffer_not_cleared_from_l1(
    shape,
    mesh_shape,
    cluster_axis,
    target,
    request,
    device,
):
    _enable_operation_memory_logging()

    input_shape = shape
    weight_shape = (shape[-1],)
    shapes = [input_shape, weight_shape]

    shard_dims = [-1, len(input_shape) - 1]
    weight_shard_dims = [-1, 0]

    def module(builder: TTIRBuilder):
        import torch

        @builder.func(shapes, [torch.bfloat16] * len(shapes))
        def two_fused_norms(*inputs, unit_attrs: Optional[List[str]] = None):
            builder = inputs[-1]
            in0 = inputs[0]
            weight = inputs[1]

            sharded_input = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=make_shard_shape(len(input_shape), shard_dims, mesh_shape),
                shard_dims=shard_dims,
            )
            sharded_weight = builder.mesh_shard(
                weight,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=make_shard_shape(
                    len(weight_shape), weight_shard_dims, mesh_shape
                ),
                shard_dims=weight_shard_dims,
            )

            # Two fused norms share ONE prelude-hoisted stats ttnn.empty (CSE).
            # That buffer is never deallocated -- the L1 assertion below catches this.
            r1 = builder.distributed_rms_norm(
                sharded_input,
                cluster_axis=cluster_axis,
                weight=sharded_weight,
                residual=None,
                epsilon=1e-5,
            )
            r2 = builder.distributed_rms_norm(
                r1,
                cluster_axis=cluster_axis,
                weight=sharded_weight,
                residual=None,
                epsilon=1e-5,
            )

            gathered = builder.mesh_shard(
                r2,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=make_shard_shape(len(input_shape), shard_dims, mesh_shape),
                shard_dims=shard_dims,
            )
            return gathered

    l1_before = _l1_allocated_bytes(device)

    compile_and_execute_ttir(
        module,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        **get_request_kwargs(request),
        device=device,
        target=target,
    )

    l1_after = _l1_allocated_bytes(device)
    leaked = l1_after - l1_before

    print(
        f"\nL1 before: {l1_before:,} bytes  |  "
        f"after: {l1_after:,} bytes  |  "
        f"leaked: {leaked:+,} bytes"
    )

    assert leaked == 0, (
        f"Stats buffer leaked {leaked:,} bytes in L1 after program exit. "
        "A ttnn.deallocate for the hoisted stats ttnn.empty is missing."
    )
