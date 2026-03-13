# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.base.builder_enums import *
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import (
    compile_and_execute_ttir,
)
from test_utils import (
    Marks,
    shape_str,
    shapes_list_str,
    make_shard_shape,
)
from ttmlir.dialects import ttir

pytestmark = pytest.mark.frontend("ttir")

def check_op(mlir_file: str, op_name: str) -> bool:
    op_name = "ttnn." + op_name
    with open(mlir_file, "r") as f:
        for line in f:
            if op_name in line:
                return True
    return False

@pytest.mark.parametrize(
    "shapes", [((3, 128, 128), (3, 128, 128), (128,))], ids=shapes_list_str
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="Batched input not supported when bias exists (linear operation). https://github.com/tenstorrent/tt-metal/issues/31634"
)
def test_linear_without_workaround(
    shapes: List[Shape],
    dtype: torch.dtype,
    transpose_a: bool,
    transpose_b: bool,
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype, dtype, dtype])
        def linear_wrapper(
            in0: Operand,
            weight: Operand,
            bias: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.linear(
                in0, weight, bias, transpose_a, transpose_b, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "shape",
    [
        # 3D input triggers the workaround (rank < 4)
        (32, 128, 128),
        # 2D input also triggers the workaround
        (128, 128),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="ttnn.argmax requires 4D input tensors. Without the workaround, "
    "input tensors with rank < 4 are not unsqueezed to 4D before the op. "
    "Metal issue: https://github.com/tenstorrent/tt-metal/issues/18241"
)
def test_argmax_without_workaround(
    shape: Shape,
    dtype: torch.dtype,
    dim: int,
    keep_dim: bool,
    target: str,
    request,
    device,
):
    """
    Test argmax with workarounds disabled.
    Workaround: ArgMaxOpRewritePattern - unsqueezes input to 4D and reshapes
    output back to original rank.
    Trigger condition: input tensor rank < 4.
    """

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def argmax_no_workaround_wrapper(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.argmax(
                in0, dim_arg=[dim], keep_dim=keep_dim, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "shape,dim",
    [
        # CumSumOpRankRewritePattern: rank < 4 triggers unsqueeze to 4D
        pytest.param(
            (128, 128),
            0,
            marks=pytest.mark.xfail(
                reason="tt-metal cumsum requires 4D input tensors. "
                "Rank 2 input is not unsqueezed without the workaround."
            ),
            id="rank2_dim0",
        ),
        pytest.param(
            (32, 64, 128),
            1,
            marks=pytest.mark.xfail(
                reason="tt-metal cumsum requires 4D input tensors. "
                "Rank 3 input is not unsqueezed without the workaround."
            ),
            id="rank3_dim1",
        ),
        # CumSumOpDimRewritePattern: dim > 1 triggers permutation workaround
        pytest.param(
            (4, 4, 128, 128),
            2,
            marks=pytest.mark.xfail(
                reason="tt-metal cumsum only supports dim 0 or 1. "
                "dim=2 requires permutation workaround."
            ),
            id="rank4_dim2",
        ),
        pytest.param(
            (4, 4, 128, 128),
            3,
            marks=pytest.mark.xfail(
                reason="tt-metal cumsum only supports dim 0 or 1. "
                "dim=3 requires permutation workaround."
            ),
            id="rank4_dim3",
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_cumsum_without_workaround(
    shape: Shape,
    dim: int,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """
    Test cumsum with workarounds disabled.
    Workaround: CumSumOpRankRewritePattern - unsqueezes input to 4D when rank < 4.
    Workaround: CumSumOpDimRewritePattern - permutes axes when dim > 1 so cumsum
    runs on dim 0.
    """

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def cumsum_no_workaround_wrapper(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.cumsum(in0, dim=dim, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "shapes",
    [
        # 5D weight tensor triggers the workaround (rank > 4)
        ((1, 32), (1, 1, 1, 512, 128)),
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="TTNN EmbeddingOp supports at most 4D weight tensor. Without the "
    "workaround, weight tensors with rank > 4 are not squeezed to 4D."
)
def test_embedding_without_workaround(
    shapes: List[Shape],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """
    Test embedding with workarounds disabled.
    Workaround: EmbeddingOpSqueezeWeightRewritePattern - squeezes weight tensor
    to 4D when rank > 4 (all dims except last 2 must be 1).
    Trigger condition: weight tensor rank > 4.
    """

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype, dtype])
        def embedding_no_workaround_wrapper(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.embedding(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "shape,padding",
    [
        # 5D tensor with padding on last 2 dims (2 padded dims <= 3 max)
        pytest.param(
            (1, 4, 8, 8, 32),
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            marks=pytest.mark.xfail(
                reason="tt-metal pad corrupts shapes for rank > 4 tensors. "
                "Metal issue: https://github.com/tenstorrent/tt-metal/issues/38144"
            ),
            id="5d_pad_dim2_dim3",
        ),
        # 5D tensor with padding on 3 dims
        pytest.param(
            (2, 6, 4, 4, 16),
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            marks=pytest.mark.xfail(
                reason="tt-metal pad corrupts shapes for rank > 4 tensors. "
                "Metal issue: https://github.com/tenstorrent/tt-metal/issues/38144"
            ),
            id="5d_pad_3dims",
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_pad_high_dim_without_workaround(
    shape: Shape,
    padding: List[int],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """
    Test pad on rank > 4 tensors with workarounds disabled.
    Workaround: PadHighDimRewritePattern - squeezes to <= 4D by merging
    non-padded dims, pads, then unsqueezes back.
    Trigger condition: input rank > 4 and at most 3 padded dimensions.
    """

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def pad_high_dim_no_workaround_wrapper(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.pad(in0, padding=padding, value=0)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "shapes",
    [
        # Multi-row bias: bias shape [17, 32] has 17 rows.
        # The fused bias kernel (add_tiles_bcast_rows) only broadcasts row 0,
        # silently ignoring rows 1-16 and producing incorrect results.
        # Small K=32 so bias error is not masked by matmul output variance.
        pytest.param(
            ((17, 32), (32, 32), (17, 32)),
            marks=pytest.mark.xfail(
                reason="Fused bias kernel broadcasts only row 0 of bias tile, "
                "silently producing incorrect results for multi-row bias. "
                "https://github.com/tenstorrent/tt-metal/issues/39390"
            ),
        ),
        # Batched bias [2, 1, 1] with non-batched weight.
        # Bias batch dim > 1 triggers TT_FATAL(bias_batch_size == 1).
        pytest.param(
            ((2, 33, 1024), (1024, 1024), (2, 1, 1)),
            marks=pytest.mark.xfail(
                reason="Bias with non-unit batch dimensions not supported. "
                "https://github.com/tenstorrent/tt-metal/issues/31634"
            ),
        ),
        # Output shape mismatch on fused kernel path.
        # When bias is effectively 1D (padded height == TILE_HEIGHT) and the
        # broadcasted output shape differs from the matmul shape, the fused
        # kernel produces output with matmul shape [256, 512] instead of the
        # expected broadcasted shape [1, 256, 512].
        pytest.param(
            ((256, 1024), (1024, 512), (1, 1, 512)),
            marks=pytest.mark.xfail(
                reason="Fused linear kernel produces matmul-shaped output "
                "instead of broadcasted shape when bias triggers fused path. "
                "https://github.com/tenstorrent/tt-metal/issues/39392"
            ),
        ),
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_linear_bias_decomposition_without_workaround(
    shapes: List[Shape],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """
    Test linear op bias decomposition scenarios with workarounds disabled.
    Each param tests a different reason the bias cannot stay fused:
    - Multi-row bias (silent correctness bug in fused kernel)
    - Batched bias with non-batched weight
    """

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype, dtype, dtype])
        def linear_bias_decomp_wrapper(
            in0: Operand,
            weight: Operand,
            bias: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.linear(
                in0, weight, bias, False, False, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "shape",
    [
        # head_size=30 (not divisible by 32) triggers the workaround
        (2, 8, 128, 30),
    ],
    ids=shape_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="ConcatenateHeadsOp requires head_size divisible by tile size (32). "
    "Without the workaround, non-aligned head sizes are not decomposed into "
    "permute + reshape."
)
def test_concatenate_heads_without_workaround(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """
    Test concatenate_heads with workarounds disabled.
    Workaround: ConcatenateHeadsOpRewritePattern - decomposes into permute + reshape
    when head_size is not divisible by tile size (32).
    Trigger condition: head_size (input_shape[3]) % 32 != 0.
    """

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def concatenate_heads_no_workaround_wrapper(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.concatenate_heads(in0, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "input_shape,weight_shape,stride,padding",
    [
        # Conv3dDepthPaddingRewritePattern: non-zero padding triggers the workaround
        pytest.param(
            (1, 8, 28, 28, 16),
            (32, 16, 3, 3, 3),
            [1, 1, 1],
            [1, 1, 1],
            marks=pytest.mark.xfail(
                reason="tt-metal conv3d does not handle padding correctly. "
                "Metal issue: https://github.com/tenstorrent/tt-metal/issues/38143"
            ),
            id="depth_padding",
        ),
        # Conv3dPadOutputChannelsRewritePattern: output channels not multiple of 32
        pytest.param(
            (1, 8, 28, 28, 4),
            (17, 4, 3, 3, 3),
            [1, 1, 1],
            [0, 0, 0],
            marks=pytest.mark.xfail(
                reason="tt-metal conv3d requires output channels to be a multiple "
                "of tile width (32). "
                "Metal issue: https://github.com/tenstorrent/tt-metal/issues/38126"
            ),
            id="pad_output_channels",
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_conv3d_without_workaround(
    input_shape: Shape,
    weight_shape: Shape,
    stride: List[int],
    padding: List[int],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """
    Test conv3d with workarounds disabled.
    Workaround: Conv3dDepthPaddingRewritePattern - extracts padding into explicit PadOp.
    Trigger: non-zero padding with padding_mode="zeros".
    Workaround: Conv3dPadOutputChannelsRewritePattern - pads output channels to
    multiple of 32.
    Trigger: output channels not multiple of 32.
    """

    def module(builder: TTIRBuilder):
        @builder.func([input_shape, weight_shape], [dtype, dtype])
        def conv3d_no_workaround_wrapper(
            in0: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.conv3d(
                in0,
                weight,
                None,
                stride=stride,
                padding=padding,
                groups=1,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "shape,normalized_shape",
    [
        ((32, 128), [128]),
        ((2, 4, 64), [64]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="RMSNorm without high-precision compute config may produce inaccurate "
    "results. The workaround sets HiFi4 math fidelity and FP32 accumulation "
    "to match LayerNorm precision."
)
def test_rms_norm_without_workaround(
    shape: Shape,
    normalized_shape: List[int],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """
    Test rms_norm with workarounds disabled.
    Workaround: RMSNormConfigRewritePattern - applies high-precision
    DeviceComputeKernelConfig (HiFi4, FP32 accumulation) when no compute
    config is set.
    Trigger condition: RMSNormOp has no existing compute config.
    """

    def module(builder: TTIRBuilder):
        @builder.func([shape, tuple(normalized_shape)], [dtype, dtype])
        def rms_norm_no_workaround_wrapper(
            in0: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.rms_norm(
                in0,
                normalized_shape=normalized_shape,
                weight=weight,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "input_shape,num_heads,transpose_key",
    [
        # MHA case: head_size=30 (not divisible by 32) triggers the workaround
        # batch=2, seq=128, hidden=240, num_heads=8, head_size=30
        # input: [batch, seq, 3 * num_heads * head_size] = [2, 128, 720]
        ((2, 128, 720), 8, False),
    ],
    ids=["mha_head30_no_transpose"],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="SplitQueryKeyValueAndSplitHeadsOp requires head_size divisible by "
    "tile size (32). Without the workaround, head_size=30 is not decomposed "
    "into individual slice + reshape + permute operations."
)
def test_split_qkv_without_workaround(
    input_shape: Shape,
    num_heads: int,
    transpose_key: bool,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """
    Test split_query_key_value_and_split_heads with workarounds disabled.
    Workaround: SplitQueryKeyValueAndSplitHeadsOpRewritePattern - decomposes
    fused op into individual slice + reshape + permute operations when head_size
    is not divisible by tile size (32).
    Trigger condition: head_size % 32 != 0.
    """

    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [dtype])
        def split_qkv_no_workaround_wrapper(
            input_tensor: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            query, key, value = builder.split_query_key_value_and_split_heads(
                input_tensor,
                num_heads=num_heads,
                transpose_key=transpose_key,
            )
            return query, key, value

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "shapes",
    [
        # lhs and rhs shapes differ, triggering implicit broadcast workaround
        ((64, 128), (1, 128)),
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="ttnn.subtract with implicit broadcasting requires commutative operation. "
    "Without the workaround, subtract is not decomposed to add(lhs, neg(rhs)). "
    "Metal issue: https://github.com/tenstorrent/tt-metal/issues/24635"
)
def test_subtract_without_workaround(
    shapes: List[Shape],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """
    Test subtract with workarounds disabled.
    Workaround: SubtractOpImplicitBroadcastRewritePattern - decomposes
    subtract(lhs, rhs) into add(lhs, neg(rhs)) when shapes differ.
    Trigger condition: lhs and rhs shapes are different (implicit broadcast needed).
    """

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype, dtype])
        def subtract_no_workaround_wrapper(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.subtract(in0, in1, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "shapes,scale_factor",
    [
        # Bilinear mode with channel=30 (not multiple of 32) triggers the workaround
        # Input: (1, 8, 8, 30) in NHWC, Output: (1, 16, 16, 30) after 2x upscale
        ([(1, 8, 8, 30), (1, 16, 16, 30)], [2, 2]),
    ],
    ids=["bilinear_ch30"],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="ttnn.upsample bilinear mode requires channel dimension to be a "
    "multiple of tile width (32). Without the workaround, non-aligned "
    "channels are not padded before the operation."
)
def test_upsample_without_workaround(
    shapes: List[Shape],
    scale_factor: List[int],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """
    Test upsample2d bilinear with workarounds disabled.
    Workaround: UpsampleOpRewritePattern - pads channel dimension to multiple
    of 32, applies bilinear upsample, then slices to remove padding.
    Trigger condition: bilinear mode AND channel dimension not multiple of 32.
    """

    def module(builder: TTIRBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def upsample_no_workaround_wrapper(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.upsample2d(
                in0,
                in1,
                scale_factor=scale_factor,
                mode="bilinear",
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )


@pytest.mark.parametrize(
    "shape,all_gather_dim",
    [
        # 3D input triggers the workaround (rank < 4)
        ((32, 128, 128), 2),
        # 2D input also triggers the workaround
        ((128, 128), 1),
    ],
    ids=["rank3_dim2", "rank2_dim1"],
)
@pytest.mark.parametrize("mesh_shape", [(1, 2)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="ttnn.all_gather requires 4D input tensors. Without the workaround, "
    "input tensors with rank < 4 are not unsqueezed to 4D before the op."
)
def test_all_gather_without_workaround(
    shape: Shape,
    all_gather_dim: int,
    mesh_shape: Tuple[int, int],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """
    Test all_gather with workarounds disabled.
    Workaround: AllGatherOpRewritePattern - unsqueezes input to 4D, adjusts
    gather dim to account for padding, then reshapes output back to original rank.
    Trigger condition: input tensor rank < 4.
    """
    cluster_axis = 1  # gather along y-axis (2 devices in mesh_shape)
    rank_in = len(shape)
    rank_mesh = len(mesh_shape)

    # Take the last rank_mesh dims as sharded dims
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    # Full (un-sharded) input shape, scaled up by the mesh along shard dims
    full_input_shape = list(shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    def module(builder: TTIRBuilder):
        @builder.func([full_input_shape], [dtype])
        def all_gather_no_workaround_wrapper(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            in_shard = builder.mesh_shard(
                in0,
                shard_direction=MeshShardDirection.FullToShard.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )
            all_gather0 = builder.all_gather(
                in_shard,
                all_gather_dim=all_gather_dim,
                cluster_axis=cluster_axis,
                unit_attrs=unit_attrs,
            )
            return builder.mesh_shard(
                all_gather0,
                shard_direction=MeshShardDirection.ShardToFull.value,
                shard_type=MeshShardType.Devices.value,
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        mesh_name="mesh",
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
        pipeline_options=["disable-workarounds=true"],
    )
