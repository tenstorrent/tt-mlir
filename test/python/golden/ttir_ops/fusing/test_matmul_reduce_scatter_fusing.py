# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
from collections import OrderedDict
from typing import Callable, List, Tuple

import _ttmlir_runtime as tt_runtime
from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape, get_artifact_dir
from builder.base.builder_enums import MeshShardDirection, MeshShardType, ReduceType
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import (
    compile_ttir_to_flatbuffer,
    compile_and_execute_ttir,
)
from test_utils import shape_str

pytestmark = pytest.mark.frontend("ttir")

# The TTIR MatmulReduceScatter fusing patterns rewrite
#
#   proj = reduce_scatter(matmul/linear(input, weight)[+ bias])
#   [out  = residual + gate * proj]                     (gated-residual epilogue)
#
# into a `ttcore.composite "minimal_matmul_strided_reduce_scatter_async"`, which
# TTNNResolveComposites then promotes to the typed
# `ttnn.minimal_matmul_strided_reduce_scatter_async` op.
#
# This is the row-parallel FFN down-projection: the activation `[M, K]` and
# weight `[K, N]` are both K-sharded across the cluster axis, each device
# computes a partial `input @ weight`, and the reduce_scatter sums the partials
# across ranks while scattering the result along N.
#
# This module has two kinds of tests:
#   * test_matmul_reduce_scatter_fusing         -- compile-only IR check that the
#     full TTIR -> TTNN pipeline emits the fused op (runs on any target).
#   * test_matmul_reduce_scatter_fusing_execute -- runs the fused op on a real
#     multi-chip mesh and PCC-checks it against the golden, i.e. verifies the
#     fused matmul+collective is numerically correct on device.
#
# The individual compiler stages are also covered by lit tests
# (test/ttmlir/Dialect/TTIR/fusing/matmul_reduce_scatter_fusing.mlir and
# test/ttmlir/Dialect/TTNN/minimal_matmul_strided_reduce_scatter_async/*).

FUSED_OP = "ttnn.minimal_matmul_strided_reduce_scatter_async"


def _promote_options() -> List[str]:
    """Pipeline options that force-promote the composite the fusing pattern
    emits into the typed ttnn op. Without the optimizer/OpModel the default
    (Auto) resolution inlines the composite back into matmul + reduce_scatter,
    which would bypass the fused op.

    Returns a fresh list on each call: the compile helper appends
    system-desc-path/mesh-shape to this list in place, so a shared list would
    accumulate duplicate options across compiles.
    """
    return ["composite-resolution=force-promote"]


def _full_to_shard_device(builder: TTIRBuilder, input: Operand, dim: int) -> Operand:
    """Shard `input` along `dim` across the mesh's second (y) axis."""
    rank = len(builder._get_golden_tensor(input).shape)
    num_devices = builder.mesh_shape[1]
    shard_shape = [1] * rank
    shard_shape[dim] = num_devices
    return builder.mesh_shard(
        input,
        shard_direction=MeshShardDirection.FullToShard.value,
        shard_type=MeshShardType.Devices.value,
        shard_shape=shard_shape,
        shard_dims=[-1, dim],
    )


def _shard_to_full_device(builder: TTIRBuilder, input: Operand, dim: int) -> Operand:
    """Concatenate `input`'s shards along `dim` across the mesh's second (y) axis."""
    rank = len(builder._get_golden_tensor(input).shape)
    num_devices = builder.mesh_shape[1]
    shard_shape = [1] * rank
    shard_shape[dim] = num_devices
    return builder.mesh_shard(
        input,
        shard_direction=MeshShardDirection.ShardToFull.value,
        shard_type=MeshShardType.Devices.value,
        shard_shape=shard_shape,
        shard_dims=[-1, dim],
    )


def _full_to_shard_replicate(builder: TTIRBuilder, input: Operand) -> Operand:
    return builder.mesh_shard(
        input,
        shard_direction=MeshShardDirection.FullToShard.value,
        shard_type=MeshShardType.Replicate.value,
        shard_shape=[1],
        shard_dims=[-1],
    )


def _make_module(variant: str, m: int, k: int, n: int, cluster_axis: int) -> Callable:
    """Return a TTIRBuilder module that builds the pre-fusion pattern.

    tt-metal's minimal_matmul_strided_reduce_scatter kernel operates on 4D
    tensors (it asserts a batch of 1 and scatters the innermost dim, dim 3), so
    all operands are 4D `[1, 1, M/K/N ...]`. Activation `[1, 1, M, K]` and weight
    `[1, 1, K, N]` are both K-sharded across the mesh; each device computes a
    partial `input @ weight` and the reduce_scatter sums the partials across
    ranks while scattering the result along N (dim 3).
    `variant` selects the epilogue folded into the fused op:
      - "matmul":            proj = reduce_scatter(matmul(x, W))
      - "linear":            proj = reduce_scatter(linear(x, W, bias))  (row-broadcast bias)
      - "addcmul":           out  = residual + gate * reduce_scatter(matmul(x, W))
      - "addcmul_broadcast": as "addcmul" but gate is a row-broadcast `[1, 1, 1, N]`
        (the per-channel gate DiT actually uses; the full-gate case is "addcmul").
    """
    dtype = torch.bfloat16
    shapes: List[Shape] = [(1, 1, m, k), (1, 1, k, n)]
    if variant == "linear":
        shapes.append((1, 1, 1, n))  # bias
    elif variant in ("addcmul", "addcmul_broadcast"):
        gate_shape = (1, 1, 1, n) if variant == "addcmul_broadcast" else (1, 1, m, n)
        shapes.extend([gate_shape, (1, 1, m, n)])  # gate, residual

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def matmul_reduce_scatter(*args):
            operands = list(args[:-1])
            b: TTIRBuilder = args[-1]
            act, weight = operands[0], operands[1]

            # Row-parallel: shard the contraction dim K of both operands
            # (K is dim 3 of the activation, dim 2 of the weight).
            act_sharded = _full_to_shard_device(b, act, dim=3)
            weight_sharded = _full_to_shard_device(b, weight, dim=2)

            if variant == "linear":
                bias_replicated = _full_to_shard_replicate(b, operands[2])
                proj = b.linear(act_sharded, weight_sharded, bias_replicated)
            else:
                proj = b.matmul(act_sharded, weight_sharded)

            scattered = b.reduce_scatter(
                proj,
                reduce_type=ReduceType.Sum.value,
                scatter_dim=3,
                cluster_axis=cluster_axis,
            )

            if variant in ("addcmul", "addcmul_broadcast"):
                # The gate/residual apply to the scattered output, so they are
                # scattered along N (dim 3) the same way reduce_scatter is.
                gate_sharded = _full_to_shard_device(b, operands[2], dim=3)
                res_sharded = _full_to_shard_device(b, operands[3], dim=3)
                gated = b.multiply(scattered, gate_sharded)
                scattered = b.add(res_sharded, gated)

            return _shard_to_full_device(b, scattered, dim=3)

    return module


def _assert_fused(ir: str, variant: str) -> None:
    # The pattern must fuse into the typed op...
    assert FUSED_OP in ir, f"expected {FUSED_OP} in compiled IR for variant={variant}"
    # ...and the standalone matmul/collective it subsumes must be gone.
    assert (
        '"ttnn.reduce_scatter"' not in ir
    ), "standalone ttnn.reduce_scatter should be fused away"
    assert '"ttnn.matmul"' not in ir, "standalone ttnn.matmul should be fused away"


def _compile_to_ttnn_ir(module, mesh_shape: Tuple[int, int], request) -> str:
    """Run the TTIR -> TTNN pipeline and return the compiled TTNN IR text."""
    kwargs = get_request_kwargs(request)
    artifact_dir = get_artifact_dir(
        kwargs["output_root"], "TTIRBuilder", kwargs["test_base"], make_dir=True
    )
    compile_ttir_to_flatbuffer(
        module,
        system_desc_path=kwargs["system_desc_path"],
        artifact_dir=artifact_dir,
        target="ttnn",
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        pipeline_options=_promote_options(),
        save_artifacts=True,
    )
    with open(os.path.join(artifact_dir, "ttnn_compiled.mlir"), "r") as f:
        return f.read()


# Only the row-broadcast gate ("addcmul_broadcast", `[1, N]`) fuses; the full
# `[M, N]` gate ("addcmul") is intentionally left unfused (see
# test_matmul_reduce_scatter_full_gate_not_fused and the isRowBroadcast guard in
# MatmulReduceScatterFusingPattern.cpp).
@pytest.mark.parametrize(
    "variant",
    ["matmul", "linear", "addcmul_broadcast"],
    ids=["matmul", "linear", "addcmul_broadcast"],
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("mesh_shape", [(1, 2)], ids=shape_str)
def test_matmul_reduce_scatter_fusing(
    variant: str,
    cluster_axis: int,
    mesh_shape: Tuple[int, int],
    request,
):
    """Compile-only: the full TTIR -> TTNN pipeline emits the fused op.

    A pure IR check (no execution), so it runs on any target regardless of how
    many chips are physically present.
    """
    module = _make_module(variant, m=64, k=512, n=64, cluster_axis=cluster_axis)
    ir = _compile_to_ttnn_ir(module, mesh_shape, request)
    _assert_fused(ir, variant)


# The fused addcmul epilogue applies the gate per-channel (broadcast across the
# M/row dim), so a full `[M, N]` gate would be silently collapsed to its first
# row. The isRowBroadcast guard in the fusing pattern must keep that case
# unfused, leaving the primitive matmul + reduce_scatter + multiply + add in place.
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("mesh_shape", [(1, 2)], ids=shape_str)
def test_matmul_reduce_scatter_full_gate_not_fused(
    cluster_axis: int,
    mesh_shape: Tuple[int, int],
    request,
):
    """Compile-only: a full `[M, N]` gate must NOT fuse into the composite."""
    module = _make_module("addcmul", m=64, k=512, n=64, cluster_axis=cluster_axis)
    ir = _compile_to_ttnn_ir(module, mesh_shape, request)
    assert FUSED_OP not in ir, f"full-gate addcmul must not fuse into {FUSED_OP}"
    # It must fall back to the standalone matmul the guard declined to fuse.
    assert '"ttnn.matmul"' in ir, "full-gate addcmul should leave a standalone matmul"


# Only the variants that actually fuse are exercised on device; the full-gate
# addcmul case is covered (compile-only) by test_matmul_reduce_scatter_full_gate_not_fused.
@pytest.mark.parametrize(
    "variant",
    ["matmul", "linear", "addcmul_broadcast"],
    ids=["matmul", "linear", "addcmul_broadcast"],
)
@pytest.mark.parametrize("cluster_axis", [1])
# tt-metal's minimal_matmul_strided_reduce_scatter kernel is only exercised on a
# full 8-device ring (its nightly tests use mesh (1, 8) exclusively); a
# degenerate 2-device ring deadlocks. Use (1, 8) so the ring matches what the
# kernel supports.
@pytest.mark.parametrize("mesh_shape", [(1, 8)], ids=shape_str)
@pytest.mark.parametrize("target", ["ttnn"])
# The fused op pins topology=Ring, so the mesh must be opened with a ring fabric
# (the conftest default FABRIC_1D is a line and the async reduce-scatter would
# deadlock on the unrouted ring wrap-around edge).
@pytest.mark.parametrize(
    "fabric_config",
    [tt_runtime.runtime.FabricConfig.FABRIC_1D_RING],
    ids=["fabric_1d_ring"],
)
def test_matmul_reduce_scatter_fusing_execute(
    variant: str,
    cluster_axis: int,
    mesh_shape: Tuple[int, int],
    target: str,
    fabric_config,
    request,
    device,
):
    """Run the fused matmul+reduce_scatter on a real multi-chip mesh and PCC-check it.

    Verifies that fusing the collective into the matmul is numerically correct
    end-to-end on device (compares against the unfused golden the builder
    computes from the primitive matmul/reduce_scatter ops). Auto-deselected by
    the conftest mesh filter when fewer than mesh_shape chips are present.

    Shape: M=128, K=2048, N=512. K is chosen so the per-device contraction
    after K-sharding across the 8-device ring is a non-degenerate 2048/8 = 256
    (8 tiles) -- the reference kernel is only ever exercised with a full,
    multi-tile K, and a 1-tile per-device K is degenerate. N_tiles (16) is
    divisible by the ring size.
    """
    module = _make_module(variant, m=128, k=2048, n=512, cluster_axis=cluster_axis)

    # check_pcc defaults to True; leave it implicit so it can't collide with the
    # value get_request_kwargs may inject via --disable-pcc. Match tt-metal's own
    # bar for this op (pcc > 0.9995); the builder harness has no relative-RMSE
    # check, so the RMSE half of metal's criterion can't be mirrored here.
    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        pipeline_options=_promote_options(),
        pcc=0.9995,
    )
