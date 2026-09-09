# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import pytest
import torch
from collections import OrderedDict
from typing import List, Optional, Tuple

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand, Shape, get_artifact_dir
from builder.base.builder_enums import MeshShardDirection, MeshShardType
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_ttir_to_flatbuffer
from test_utils import shape_str

pytestmark = pytest.mark.frontend("ttir")


def _has_bfp8_encoding(ir: str) -> bool:
    """True iff a tensor in `ir` actually carries a bfp_bf8 element type.

    Matches bfp_bf8 only inside a `!ttcore.tile<...>` element type (a real
    tensor encoding). This deliberately ignores the `supported_data_types`
    list embedded in the `#system_desc`, which always advertises `<bfp_bf8>`
    regardless of whether the pass ran.
    """
    return re.search(r"!ttcore\.tile<[^>]*bfp_bf8", ir) is not None


# Builds the projection + residual pattern the `ttnn-ccl-activation-dtype-lowering`
# pass targets, after the TTIR -> TTNN pipeline lowers it to:
#
#   ttnn.matmul -> ttnn.all_gather -> ttnn.add
#
# The pass should down-cast the matmul output (and the all_gather it propagates
# through) to bfp_bf8, while the residual add stays bf16. We assert the pass
# fires by compiling the *same* module twice -- once with the pass enabled and
# once without -- and checking that `bfp_bf8` only appears when it is enabled.
# A pure compile/IR check is used (rather than execution) because the pass is a
# precision-lowering transform: its observable effect is the rewritten tensor
# encoding, and this avoids relying on multi-device golden/PCC matching for a
# tensor-parallel matmul (cf. the skipped `test_llama_attention_1xn_tp`).


def _full_to_shard_replicate(builder: TTIRBuilder, input: Operand) -> Operand:
    return builder.mesh_shard(
        input,
        shard_direction=MeshShardDirection.FullToShard.value,
        shard_type=MeshShardType.Replicate.value,
        shard_shape=[1],
        shard_dims=[-1],
    )


def _full_to_shard_device(builder: TTIRBuilder, input: Operand, dim: int) -> Operand:
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


def _shard_to_full_replicate(builder: TTIRBuilder, input: Operand) -> Operand:
    return builder.mesh_shard(
        input,
        shard_direction=MeshShardDirection.ShardToFull.value,
        shard_type=MeshShardType.Replicate.value,
        shard_shape=[1],
        shard_dims=[-1],
    )


def _compile_proj_residual_ttnn(
    shapes: List[Shape],
    cluster_axis: int,
    mesh_shape: Tuple[int, int],
    request,
    enable_pass: bool,
) -> str:
    """Compile the projection+residual module to TTNN MLIR and return its text.

    A column-parallel projection (replicated activation x column-sharded weight)
    followed by an all_gather and a replicated residual add. With the pass
    enabled this lowers to a matmul -> all_gather -> add chain whose activation
    is cast to bfp_bf8 across the CCL.
    """
    act_shape, weight_shape, residual_shape = shapes

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.bfloat16] * len(shapes))
        def proj_residual(
            act: Operand,
            weight: Operand,
            residual: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            act_replicated = _full_to_shard_replicate(builder, act)
            weight_sharded = _full_to_shard_device(builder, weight, dim=1)

            # Projection matmul: [M, K] x [K, N/d] -> [M, N/d] per device.
            proj = builder.matmul(act_replicated, weight_sharded)

            # Gather the partial columns back to the full [M, N] activation.
            gathered = builder.all_gather(
                proj,
                all_gather_dim=1,
                cluster_axis=cluster_axis,
            )

            # Residual add restores bf16 (its result encoding is left bf16).
            residual_replicated = _full_to_shard_replicate(builder, residual)
            out = builder.add(gathered, residual_replicated)

            return _shard_to_full_replicate(builder, out)

    kwargs = get_request_kwargs(request)
    test_base = kwargs["test_base"] + ("__pass_on" if enable_pass else "__pass_off")
    artifact_dir = get_artifact_dir(
        kwargs["output_root"], "TTIRBuilder", test_base, make_dir=True
    )

    pipeline_options = [f"enable-activation-dtype-lowering={str(enable_pass).lower()}"]

    compile_ttir_to_flatbuffer(
        module,
        system_desc_path=kwargs["system_desc_path"],
        artifact_dir=artifact_dir,
        target="ttnn",
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        pipeline_options=pipeline_options,
        save_artifacts=True,
    )

    with open(os.path.join(artifact_dir, "ttnn_compiled.mlir"), "r") as f:
        return f.read()


@pytest.mark.parametrize(
    "shapes",
    [[(32, 128), (128, 256), (32, 256)]],
    ids=["m32_k128_n256"],
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("mesh_shape", [(1, 2)], ids=shape_str)
def test_ccl_activation_dtype_lowering(
    shapes: List[Shape],
    cluster_axis: int,
    mesh_shape: Tuple[int, int],
    request,
):
    ir_pass_on = _compile_proj_residual_ttnn(
        shapes, cluster_axis, mesh_shape, request, enable_pass=True
    )
    ir_pass_off = _compile_proj_residual_ttnn(
        shapes, cluster_axis, mesh_shape, request, enable_pass=False
    )

    # The pass down-casts the projection activation to bfp_bf8 across the CCL.
    assert _has_bfp8_encoding(
        ir_pass_on
    ), "ttnn-ccl-activation-dtype-lowering should emit a bfp_bf8 activation tensor"

    # Without the pass the same module stays bf16 -- proves the bfp_bf8 above is
    # the pass's doing, not something else in the pipeline.
    assert not _has_bfp8_encoding(
        ir_pass_off
    ), "no bfp_bf8 tensor should appear when the pass is disabled"

    # The cast is done by rewriting tensor encodings, never an explicit typecast.
    assert (
        "ttnn.typecast" not in ir_pass_on
    ), "the pass must not insert any ttnn.typecast op"
