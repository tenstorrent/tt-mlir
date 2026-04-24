# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# ttmetal-only mirrors of multi-backend tests in `test_stablehlo_ops.py`.

import pytest
import torch
from conftest import get_request_kwargs
from typing import Callable, List, Optional, Tuple
from collections import OrderedDict

from builder.base.builder_utils import Operand, Shape
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.base.builder_apis import compile_and_execute_shlo
from test_utils import shape_str, Marks

pytestmark = pytest.mark.frontend("shlo")


# --- Helpers (extracted from original) ---
def module_remainder(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def remainder(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.remainder(in0, in1, unit_attrs=unit_attrs)


def module_atan2(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def atan2(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.atan2(in0, in1, unit_attrs=unit_attrs)


def module_add(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def add(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.add(in0, in1)


def module_max(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def max(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.max(in0, in1, unit_attrs=unit_attrs)


def module_min(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def minimum(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.min(in0, in1, unit_attrs=unit_attrs)


def module_mul(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def multiply(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.mul(in0, in1, unit_attrs=unit_attrs)


def module_pow(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def pow(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        randn_base_tensor = builder._get_golden_tensor(in0)
        randn_exponent_tensor = builder._get_golden_tensor(in1)
        randn_base_tensor = randn_base_tensor.apply_shardwise(
            lambda shard: (
                shard.abs() if torch.is_floating_point(randn_exponent_tensor) else shard
            )
        )
        if torch.is_floating_point(randn_exponent_tensor):
            randn_base_tensor = torch.abs(randn_base_tensor)
        builder.set_goldens_from_builder_tensor(
            {in0: randn_base_tensor, in1: randn_exponent_tensor}
        )
        builder.set_graph_level_check(True)
        return builder.pow(in0, in1, unit_attrs=unit_attrs)


def module_subtract(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def subtract(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.subtract(in0, in1, unit_attrs=unit_attrs)


def module_compare_eq(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def compare_eq(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.compare(in0, in1, "EQ", unit_attrs=unit_attrs)


def module_compare_ne(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def compare_ne(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.compare(in0, in1, "NE", unit_attrs=unit_attrs)


def module_compare_ge(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def compare_ge(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.compare(in0, in1, "GE", unit_attrs=unit_attrs)


def module_compare_gt(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def compare_gt(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.compare(in0, in1, "GT", unit_attrs=unit_attrs)


def module_compare_le(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def compare_le(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.compare(in0, in1, "LE", unit_attrs=unit_attrs)


def module_compare_lt(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def compare_lt(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.compare(in0, in1, "LT", unit_attrs=unit_attrs)


# --- Tests (target rewritten to ["ttmetal"]) ---
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "test_fn",
    [
        module_add,
        module_max,
        module_min,
        module_mul,
        module_pow,
        module_subtract,
        module_remainder | Marks(pytest.mark.skip_config(["ttmetal"])),
        module_atan2 | Marks(pytest.mark.skip_config(["ttmetal"])),
    ],
)
def test_binary_ops(test_fn: Callable, target: str, request, device):
    compile_and_execute_shlo(
        test_fn,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "test_fn",
    [
        module_compare_eq | Marks(pytest.mark.skip_config(["ttmetal"])),
        module_compare_ne | Marks(pytest.mark.skip_config(["ttmetal"])),
        module_compare_ge | Marks(pytest.mark.skip_config(["ttmetal"])),
        module_compare_gt | Marks(pytest.mark.skip_config(["ttmetal"])),
        module_compare_le | Marks(pytest.mark.skip_config(["ttmetal"])),
        module_compare_lt | Marks(pytest.mark.skip_config(["ttmetal"])),
    ],
)
def test_compare_ops(test_fn: Callable, target: str, request, device):
    compile_and_execute_shlo(
        test_fn,
        **get_request_kwargs(request),
        target=target,
        device=device,
        check_pcc=False,
    )


@pytest.mark.parametrize("target", ["ttmetal"])
def test_reshape_mismatch_raises(target, request, device):
    """
    Element-count mismatch must raise. It may surface as a ValueError from the
    builder or as a TTBuilderCompileException during compile/exec; accept any Exception.
    """
    input_shape = (2, 3)  # 6
    output_shape = (4, 2)  # 8 -> not match

    def module(builder: StableHLOBuilder):
        @builder.func([input_shape], [torch.float32])
        def reshape_wrapper(in0: Operand, builder: StableHLOBuilder):
            if hasattr(builder, "set_graph_level_check"):
                builder.set_graph_level_check(True)
            return builder.reshape(in0, output_shape)

    with pytest.raises(Exception):
        compile_and_execute_shlo(
            module,
            **get_request_kwargs(request),
            target=target,
            device=device,
        )


@pytest.mark.parametrize("shape", [(2, 3, 4), (128, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "permutation",
    [
        [1, 0],
        [2, 1, 0],
        [0, 2, 1],
    ],
)
def test_transpose(
    shape: Shape,
    dtype: torch.dtype,
    permutation: List[int],
    target: str,
    request,
    device,
):
    if len(shape) != len(permutation):
        pytest.skip(f"Permutation {permutation} doesn't match shape rank {len(shape)}")

    # Skip ttmetal for dimensions > 2
    if target == "ttmetal" and len(shape) > 2:
        pytest.skip(
            f"ttmetal does not support transpose for dimensions > 2, got shape with {len(shape)} dimensions"
        )

    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def transpose(
            in0: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.transpose(in0, permutation, unit_attrs=unit_attrs)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(2, 3)], ids=shape_str)
@pytest.mark.parametrize("padding", [[1, 1, 1, 1], [1, 0, 0, 1]])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize(
    "target",
    [
        pytest.param(
            "ttmetal",
            marks=pytest.mark.skip(
                reason="ttir.pad lowering not supported on ttmetal, failed to legalize"
            ),
        )
    ],
)
def test_pad(
    shape: Shape,
    padding: List[int],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def pad(
            in0: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # 0-rank tensor constant
            padding_value = builder.constant(torch.tensor(0.0, dtype=dtype))
            return builder.pad(in0, padding_value, padding, unit_attrs=unit_attrs)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "shape,start_indices,limit_indices,strides",
    [
        ((128, 128), [0, 0], [64, 64], [1, 1]),
        ((128, 128), [32, 32], [96, 96], [1, 1]),
        ((128, 128), [0, 0], [128, 64], [2, 1]),
        ((256, 256), [64, 64], [192, 192], [1, 1]),
    ],
    ids=["128x128_basic", "128x128_offset", "128x128_stride", "256x256_large"],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_slice(
    shape: Shape,
    start_indices: List[int],
    limit_indices: List[int],
    strides: List[int],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def slice(in0: Operand, builder: StableHLOBuilder):
            builder.set_graph_level_check(True)
            return builder.slice(in0, start_indices, limit_indices, strides)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# ---------------------------------------------------------------------------
# test_reshape ttmetal mirror — original used a dynamically-built
# `_RESHAPE_PARAMS` list where every ttmetal entry was marked
# `pytest.mark.skip(reason="reshape lowering not yet supported in TTMetal
# backend")`. Preserve that here.
# ---------------------------------------------------------------------------
_RESHAPE_CASES_TTMETAL = [
    ([(2, 3), (3, 2)], "swap"),
    ([(2, 3), (6,)], "flatten"),
    ([(1, 784), (1, 28, 28)], "unflatten"),
    ([(4, 8, 16), (4, 128)], "3d_to_2d"),
    ([(64, 512), (64, 1, 512)], "expand_dims"),
    ([(128, 128), (64, 256)], "rearrange_2d"),
    ([(10,), (10,)], "identity"),
    ([(0, 6), (0, 2, 3)], "zero_dim"),
]

_RESHAPE_PARAMS_TTMETAL = [
    pytest.param(
        shapes,
        "ttmetal",
        id=f"{case_id}-ttmetal",
        marks=pytest.mark.skip(
            reason="reshape lowering not yet supported in TTMetal backend"
        ),
    )
    for shapes, case_id in _RESHAPE_CASES_TTMETAL
]


@pytest.mark.parametrize("shapes, target", _RESHAPE_PARAMS_TTMETAL)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_reshape(shapes: tuple, target: str, dtype: torch.dtype, request, device):
    input_shape, output_shape = shapes

    def module(builder: StableHLOBuilder):
        @builder.func([input_shape], [dtype])
        def reshape_wrapper(in0: Operand, builder: StableHLOBuilder):
            if hasattr(builder, "set_graph_level_check"):
                builder.set_graph_level_check(True)
            return builder.reshape(in0, output_shape)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )
