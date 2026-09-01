# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import torch
from collections import OrderedDict
from typing import Callable, List, Tuple

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape, get_artifact_dir
from builder.base.builder_enums import MeshShardDirection, MeshShardType
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import (
    compile_ttir_to_flatbuffer,
    compile_and_execute_ttir,
)
from test_utils import shape_str

pytestmark = pytest.mark.frontend("ttir")

# The TTIR AllGatherMatmul fusing patterns rewrite
#
#   proj = matmul/linear(all_gather(input), weight)[+ bias]
#   [out  = residual + gate * proj]                     (gated-residual epilogue)
#
# into a `ttcore.composite "all_gather_minimal_matmul_async"`, which
# TTNNResolveComposites then promotes to the typed
# `ttnn.all_gather_minimal_matmul_async` op.
#
# This module has two tests:
#   * test_all_gather_matmul_fusing         -- compile-only IR check that the
#     full TTIR -> TTNN pipeline emits the fused op (runs on any target).
#   * test_all_gather_matmul_fusing_execute -- runs the fused op on a real
#     multi-chip mesh and PCC-checks it against the golden, i.e. verifies the
#     fused collective+matmul is numerically correct on device.
#
# The individual compiler stages are also covered by lit tests
# (test/ttmlir/Dialect/TTIR/fusing/all_gather_matmul_fusing.mlir and
# test/ttmlir/Dialect/TTNN/all_gather_minimal_matmul_async/*).

FUSED_OP = "ttnn.all_gather_minimal_matmul_async"


def _promote_options() -> List[str]:
    """Pipeline options that force-promote the composite the fusing pattern
    emits into the typed ttnn op. Without the optimizer/OpModel the default
    (Auto) resolution inlines the composite back into all_gather + matmul,
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


def _full_to_shard_replicate(builder: TTIRBuilder, input: Operand) -> Operand:
    return builder.mesh_shard(
        input,
        shard_direction=MeshShardDirection.FullToShard.value,
        shard_type=MeshShardType.Replicate.value,
        shard_shape=[1],
        shard_dims=[-1],
    )


def _shard_to_full_replicate(builder: TTIRBuilder, input: Operand) -> Operand:
    return builder.mesh_shard(
        input,
        shard_direction=MeshShardDirection.ShardToFull.value,
        shard_type=MeshShardType.Replicate.value,
        shard_shape=[1],
        shard_dims=[-1],
    )


def _make_module(variant: str, m: int, k: int, n: int, cluster_axis: int) -> Callable:
    """Return a TTIRBuilder module that builds the pre-fusion pattern.

    Activation `[M, K]` is K-sharded across the mesh and all-gathered back to the
    full `[M, K]` before the matmul/linear with a replicated weight `[K, N]`.
    `variant` selects the epilogue folded into the fused op:
      - "matmul":            proj = matmul(all_gather(x), W)
      - "linear":            proj = linear(all_gather(x), W, bias)  (row-broadcast bias)
      - "addcmul":           out  = residual + gate * matmul(all_gather(x), W)
      - "addcmul_broadcast": as "addcmul" but gate is a row-broadcast `[1, N]`
        (the per-channel gate DiT actually uses; the full-gate case is "addcmul").
    """
    dtype = torch.bfloat16
    shapes: List[Shape] = [(m, k), (k, n)]
    if variant == "linear":
        shapes.append((1, n))  # bias
    elif variant in ("addcmul", "addcmul_broadcast"):
        gate_shape = (1, n) if variant == "addcmul_broadcast" else (m, n)
        shapes.extend([gate_shape, (m, n)])  # gate, residual

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def all_gather_matmul(*args):
            operands = list(args[:-1])
            b: TTIRBuilder = args[-1]
            act, weight = operands[0], operands[1]

            act_sharded = _full_to_shard_device(b, act, dim=1)
            weight_replicated = _full_to_shard_replicate(b, weight)

            gathered = b.all_gather(
                act_sharded,
                all_gather_dim=1,
                cluster_axis=cluster_axis,
            )

            if variant == "linear":
                bias_replicated = _full_to_shard_replicate(b, operands[2])
                proj = b.linear(gathered, weight_replicated, bias_replicated)
            else:
                proj = b.matmul(gathered, weight_replicated)

            if variant in ("addcmul", "addcmul_broadcast"):
                gate_replicated = _full_to_shard_replicate(b, operands[2])
                res_replicated = _full_to_shard_replicate(b, operands[3])
                gated = b.multiply(proj, gate_replicated)
                proj = b.add(res_replicated, gated)

            return _shard_to_full_replicate(b, proj)

    return module


def _assert_fused(ir: str, variant: str) -> None:
    # The pattern must fuse into the typed op...
    assert FUSED_OP in ir, f"expected {FUSED_OP} in compiled IR for variant={variant}"
    # ...and the standalone collective/matmul it subsumes must be gone.
    assert (
        '"ttnn.all_gather"' not in ir
    ), "standalone ttnn.all_gather should be fused away"
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
# test_all_gather_matmul_full_gate_not_fused and the isRowBroadcast guard in
# AllGatherMatmulFusingPattern.cpp).
@pytest.mark.parametrize(
    "variant",
    ["matmul", "linear", "addcmul_broadcast"],
    ids=["matmul", "linear", "addcmul_broadcast"],
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("mesh_shape", [(1, 2)], ids=shape_str)
def test_all_gather_matmul_fusing(
    variant: str,
    cluster_axis: int,
    mesh_shape: Tuple[int, int],
    request,
):
    """Compile-only: the full TTIR -> TTNN pipeline emits the fused op.

    A pure IR check (no execution), so it runs on any target regardless of how
    many chips are physically present.
    """
    module = _make_module(variant, m=32, k=512, n=64, cluster_axis=cluster_axis)
    ir = _compile_to_ttnn_ir(module, mesh_shape, request)
    _assert_fused(ir, variant)


# The fused addcmul epilogue applies the gate per-channel (broadcast across the
# M/row dim), so a full `[M, N]` gate would be silently collapsed to its first
# row. The isRowBroadcast guard in the fusing pattern must keep that case
# unfused, leaving the primitive matmul + multiply + add in place.
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("mesh_shape", [(1, 2)], ids=shape_str)
def test_all_gather_matmul_full_gate_not_fused(
    cluster_axis: int,
    mesh_shape: Tuple[int, int],
    request,
):
    """Compile-only: a full `[M, N]` gate must NOT fuse into the composite."""
    module = _make_module("addcmul", m=32, k=512, n=64, cluster_axis=cluster_axis)
    ir = _compile_to_ttnn_ir(module, mesh_shape, request)
    assert FUSED_OP not in ir, f"full-gate addcmul must not fuse into {FUSED_OP}"
    # It must fall back to the standalone matmul the guard declined to fuse.
    assert '"ttnn.matmul"' in ir, "full-gate addcmul should leave a standalone matmul"


# Only the variants that actually fuse are exercised on device; the full-gate
# addcmul case is covered (compile-only) by test_all_gather_matmul_full_gate_not_fused.
@pytest.mark.parametrize(
    "variant",
    ["matmul", "linear", "addcmul_broadcast"],
    ids=["matmul", "linear", "addcmul_broadcast"],
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("mesh_shape", [(1, 2)], ids=shape_str)
@pytest.mark.parametrize("target", ["ttnn"])
def test_all_gather_matmul_fusing_execute(
    variant: str,
    cluster_axis: int,
    mesh_shape: Tuple[int, int],
    target: str,
    request,
    device,
):
    """Run the fused all_gather+matmul on a real multi-chip mesh and PCC-check it.

    Verifies that fusing the collective into the matmul is numerically correct
    end-to-end on device (compares against the unfused golden the builder
    computes from the primitive all_gather/matmul ops). Auto-deselected by the
    conftest mesh filter when fewer than mesh_shape chips are present.
    """
    module = _make_module(variant, m=32, k=512, n=64, cluster_axis=cluster_axis)

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


# Larger, perf-relevant shape to exercise the fused op beyond the tiny default
# case. Correctness (PCC) only -- this is not a timing benchmark. K (4096) is
# divisible by the 2-device cluster axis (2048 per device, tile-aligned), and
# M/N are tile-aligned multiples of 32. Bump mesh_shape to (1, 4)/(1, 8) if the
# cluster can form a larger line to also exercise multi-device gather.
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("mesh_shape", [(1, 2)], ids=shape_str)
@pytest.mark.parametrize("target", ["ttnn"])
def test_all_gather_matmul_fusing_execute_large(
    cluster_axis: int,
    mesh_shape: Tuple[int, int],
    target: str,
    request,
    device,
):
    """Run the fused all_gather+matmul at a larger shape and PCC-check it.

    Same end-to-end correctness check as test_all_gather_matmul_fusing_execute,
    but with M=1024, K=4096, N=4096 to stress the fused op at a size where the
    matmul and gather actually dominate. Auto-deselected by the conftest mesh
    filter when fewer than mesh_shape chips are present.
    """
    module = _make_module("matmul", m=1024, k=4096, n=4096, cluster_axis=cluster_axis)

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
