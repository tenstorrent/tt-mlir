# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import sys
import torch
from typing import List, Optional, Tuple, Union
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


@pytest.fixture(autouse=True)
def clear_program_cache_after_test(device):
    """Clear program cache after each conv3d test to free L1 memory.

    Conv operations allocate tensors in L1 small that are only deallocated
    when the program cache is cleared. This fixture ensures the program cache
    is cleared after each test to prevent OOM errors from accumulated allocations.
    """
    yield
    conftest = sys.modules.get("conftest")
    if conftest and conftest._current_device is not None:
        conftest._current_device.clear_program_cache()


@pytest.mark.parametrize(
    "input_shape, weight_shape, bias_shape, stride, padding, groups",
    [
        # Basic 3x3x3 kernel, no padding, no bias
        ((1, 8, 28, 28, 4), (16, 4, 3, 3, 3), None, [1, 1, 1], [0, 0, 0], 1),
        # 3x3x3 kernel with bias
        (
            (1, 8, 28, 28, 4),
            (16, 4, 3, 3, 3),
            (1, 1, 1, 1, 16),
            [1, 1, 1],
            [0, 0, 0],
            1,
        ),
        # Stride=2 with padding
        (
            (1, 8, 28, 28, 16),
            (32, 16, 3, 3, 3),
            (1, 1, 1, 1, 32),
            [2, 2, 2],
            [1, 1, 1],
            1,
        ),
        # 3x3x3 with same-padding (padding=1)
        (
            (1, 8, 28, 28, 32),
            (32, 32, 3, 3, 3),
            (1, 1, 1, 1, 32),
            [1, 1, 1],
            [1, 1, 1],
            1,
        ),
        # Larger 5x5x5 kernel
        ((1, 16, 32, 32, 8), (32, 8, 5, 5, 5), None, [1, 1, 1], [0, 0, 0], 1),
        # Stride=2, no padding, no bias (downsampling)
        ((1, 8, 28, 28, 32), (64, 32, 3, 3, 3), None, [2, 2, 2], [0, 0, 0], 1),
        # 1x1x1 kernel (pointwise 3D convolution)
        (
            (1, 8, 16, 16, 64),
            (128, 64, 1, 1, 1),
            (1, 1, 1, 1, 128),
            [1, 1, 1],
            [0, 0, 0],
            1,
        ),
        # 3x1x1 kernel, stride=[2,1,1] (temporal downsampling)
        ((1, 5, 64, 64, 192), (192, 192, 3, 1, 1), None, [2, 1, 1], [0, 0, 0], 1),
    ],
    ids=[
        "basic_3x3x3_no_bias",
        "basic_3x3x3_with_bias",
        "stride2_with_padding",
        "same_padding_3x3x3",
        "large_kernel_5x5x5",
        "stride2_downsample_no_bias",
        "pointwise_1x1x1",
        "temporal_downsampling_192ch_s211",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_conv3d(
    input_shape: Shape,
    weight_shape: Shape,
    bias_shape: Optional[Shape],
    stride: List[int],
    padding: List[int],
    groups: int,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    if bias_shape:
        input_shapes = [input_shape, weight_shape, bias_shape]
        input_types = [dtype, dtype, dtype]

        def module(builder: TTIRBuilder):
            @builder.func(input_shapes, input_types)
            def conv3d_wrapper(
                in0: Operand,
                weight: Operand,
                bias: Operand,
                builder: TTIRBuilder,
                unit_attrs: Optional[List[str]] = None,
            ):
                return builder.conv3d(
                    in0,
                    weight,
                    bias,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    unit_attrs=unit_attrs,
                )

    else:
        input_shapes = [input_shape, weight_shape]
        input_types = [dtype, dtype]

        def module(builder: TTIRBuilder):
            @builder.func(input_shapes, input_types)
            def conv3d_wrapper(
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
                    groups=groups,
                    unit_attrs=unit_attrs,
                )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@pytest.mark.parametrize("enable_greedy", [False, True], ids=["chain", "greedy"])
def test_conv3d_optimizer(enable_greedy: bool, request, device):
    """Golden execution of conv3d through the optimizer-enabled pipeline.

    Covers the path introduced alongside post-optimizer conv3d weight
    preparation: Conv3dOp is handled by LegalOpConfigAnalysis, a Conv3dConfigAttr
    (here pinned via --override-conv3d-config) is applied, and the
    TTNNPrepareConv3dWeights pass materializes the prepare op using the
    optimizer-chosen c_in_block. The default (no-optimizer) pipeline exercised by
    test_conv3d above never runs the optimizer, so this is the only golden
    coverage of that path.

    Note: c_in_block only changes how the input-channel reduction is tiled, not
    the conv result, so this golden test cannot distinguish whether the override
    was applied — it would pass even if c_in_block were ignored. It validates
    that the optimizer / override pipeline compiles and executes with correct
    numerics; that c_in_block=64 was actually selected is verified at the IR
    level.
    """
    # in_channels=128 => c_in_aligned=128, so c_in_block=64 is a legal block.
    input_shape = (1, 8, 28, 28, 128)
    weight_shape = (32, 128, 3, 3, 3)
    input_shapes = [input_shape, weight_shape]
    input_types = [torch.bfloat16, torch.bfloat16]

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, input_types)
        def conv3d_optimizer_wrapper(
            in0: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.conv3d(
                in0,
                weight,
                None,
                stride=[1, 1, 1],
                padding=[0, 0, 0],
                groups=1,
                loc="conv3d_opt",
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target="ttnn",
        pipeline_options=[
            "optimization-level=1",
            f"enable-greedy-optimizer={'true' if enable_greedy else 'false'}",
            "override-conv3d-config=conv3d_opt=c_in_block#64",
        ],
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_conv3d_c_in_block_divergent(dtype, target, request, device):
    """Conv3d whose kernel volume is even, so tt-metal's auto-derived c_in_block
    differs from the TILE_WIDTH=32 the compiler pins.

    tt-metal derives its default c_in_block as
        lcm(l1_alignment, TILE_WIDTH / gcd(kernel_vol, TILE_WIDTH)).
    For an even kernel volume that gcd is > 1, so the default is 16, not 32.
    Here kernel = (2, 2, 2) => kernel_vol = 8, gcd(8, 32) = 8, giving
        lcm(16, 32 / 8) = lcm(16, 4) = 16  !=  32.

    This guards the "config is the single source of truth" invariant: the weight
    is prepared with the same c_in_block the runtime kernel consumes (the pinned
    32), so the result stays numerically correct even when that diverges from
    the value tt-metal would have chosen on its own (16). It also pins the rest
    of the config (c_out_block, spatial out-blocks, compute grid); before that
    fix this path handed tt-metal a partial config and tripped its struct
    defaults (C_out_block=0 / 1x1 grid).
    """
    # in_channels=32 => C_in % c_in_block == 0 for the pinned c_in_block=32.
    input_shape = (1, 4, 16, 16, 32)
    weight_shape = (16, 32, 2, 2, 2)
    input_shapes = [input_shape, weight_shape]
    input_types = [dtype, dtype]

    def module(builder: TTIRBuilder):
        @builder.func(input_shapes, input_types)
        def conv3d_wrapper(
            in0: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.conv3d(
                in0,
                weight,
                None,
                stride=[1, 1, 1],
                padding=[0, 0, 0],
                groups=1,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )
