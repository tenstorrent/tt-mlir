# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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


def module_abs(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def abs(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.abs(in0, unit_attrs=unit_attrs)


def module_ceil(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def ceil(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.ceil(in0, unit_attrs=unit_attrs)


def module_cosine(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def cosine(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.cosine(in0, unit_attrs=unit_attrs)


def module_exp(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def exp(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.exp(in0, unit_attrs=unit_attrs)


def module_floor(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def floor(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.floor(in0, unit_attrs=unit_attrs)


def module_neg(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def neg(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.neg(in0, unit_attrs=unit_attrs)


def module_rsqrt(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def rsqrt(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.rsqrt(in0, unit_attrs=unit_attrs)


def module_sine(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def sine(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.sine(in0, unit_attrs=unit_attrs)


def module_sqrt(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def sqrt(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.sqrt(in0, unit_attrs=unit_attrs)


def module_tan(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def tan(
        in0: Operand, builder: StableHLOBuilder, unit_attrs: Optional[List[str]] = None
    ):
        import math

        if str(in0.type.element_type) not in ["bf16", "f32"]:
            raise ValueError("tan op only supports bf16 and f32 data types")
        dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
        randn_tensor = torch.randn(in0.type.shape, dtype=dtype)
        input_golden = randn_tensor.uniform_(
            (-math.pi / 2 + 0.05), (math.pi / 2 - 0.05)
        )
        builder.set_goldens({in0: input_golden})
        builder.set_graph_level_check(True)
        return builder.tan(in0, unit_attrs=unit_attrs)


def module_tanh(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def tanh(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.tanh(in0, unit_attrs=unit_attrs)


def module_log1p(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def log1p(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        if str(in0.type.element_type) not in ["bf16", "f32"]:
            raise ValueError("log1p op only supports bf16 and f32 data types")
        dtype = torch.bfloat16 if in0.type.element_type == "bf16" else torch.float32
        randn_tensor = torch.randn(in0.type.shape, dtype=dtype)
        abs_tensor = torch.abs(randn_tensor)
        error_margin = torch.full(randn_tensor.shape, -0.99)
        input_golden = torch.add(abs_tensor, error_margin)
        builder.set_goldens({in0: input_golden})
        builder.set_graph_level_check(True)
        return builder.log1p(in0, unit_attrs=unit_attrs)


def module_logistic(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def logistic(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.logistic(in0, unit_attrs=unit_attrs)


def module_shift_right_logical(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.int32, torch.int32])
    def shift_right_logical(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.shift_right_logical(in0, in1, unit_attrs=unit_attrs)


def module_clamp(builder: StableHLOBuilder):
    @builder.func(
        [(128, 128), (128, 128), (128, 128)],
        [torch.float32, torch.float32, torch.float32],
    )
    def clamp(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.clamp(in0, in1, in2, unit_attrs=unit_attrs)


def module_select(builder: StableHLOBuilder):
    @builder.func(
        [(32, 32), (32, 32), (32, 32)], [torch.bool, torch.float32, torch.float32]
    )
    def select(
        pred: Operand,
        on_true: Operand,
        on_false: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.select(pred, on_true, on_false, unit_attrs=unit_attrs)


def module_log(builder: StableHLOBuilder):
    @builder.func([(128, 128)], [torch.float32])
    def log(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.log(in0, unit_attrs=unit_attrs)


def module_and_int(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.int32, torch.int32])
    def logical_and(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.logical_and(in0, in1, unit_attrs=unit_attrs)


def module_or_int(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.int32, torch.int32])
    def or_(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.or_(in0, in1, unit_attrs=unit_attrs)


def module_xor_int(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.int32, torch.int32])
    def xor(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.xor(in0, in1, unit_attrs=unit_attrs)


def module_and_bool(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.bool, torch.bool])
    def logical_and(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.logical_and(in0, in1, unit_attrs=unit_attrs)


def module_or_bool(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.bool, torch.bool])
    def or_(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.or_(in0, in1, unit_attrs=unit_attrs)


def module_xor_bool(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.bool, torch.bool])
    def xor(
        in0: Operand,
        in1: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.xor(in0, in1, unit_attrs=unit_attrs)


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


def module_broadcast_in_dim(builder: StableHLOBuilder):
    @builder.func([(128, 128), (128, 128)], [torch.float32, torch.float32])
    def broadcast_in_dim(
        in0: Operand,
        builder: StableHLOBuilder,
        broadcast_dimensions: List[int],
        output_shape: List[int],
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.broadcast_in_dim(
            in0,
            broadcast_dimensions=broadcast_dimensions,
            output_shape=output_shape,
            unit_attrs=unit_attrs,
        )


@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.parametrize(
    "test_fn",
    [
        module_add,
        module_max
        | Marks(
            pytest.mark.skip_config(
                ["ttmetal"], reason="https://github.com/tenstorrent/tt-mlir/issues/5016"
            )
        ),
        module_min | Marks(pytest.mark.skip_config(["ttmetal"])),
        module_mul,
        module_pow
        | Marks(
            pytest.mark.skip_config(
                ["ttnn"], reason="https://github.com/tenstorrent/tt-metal/pull/33904"
            )
        ),
        module_subtract,
    ],
)
def test_binary_ops(test_fn: Callable, target: str, request, device):
    compile_and_execute_shlo(
        test_fn,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [
        module_abs,
        module_ceil,
        module_cosine,
        module_exp,
        module_floor,
        module_log,
        module_log1p,
        module_logistic,
        module_neg,
        module_rsqrt,
        module_sine,
        module_sqrt,
        module_tan,
        module_tanh,
    ],
)
def test_unary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    # FP32 sqrt/rsqrt/log fail due to tt-metal untilize NaN handling.
    # See: https://github.com/tenstorrent/tt-metal/pull/33904
    if (
        test_fn.__name__ in ["module_sqrt", "module_rsqrt", "module_log"]
        and dtype == torch.float32
        and target == "ttnn"
    ):
        pytest.xfail(
            f"FP32 {test_fn.__name__} fails due to tt-metal untilize NaN handling. "
            "See: https://github.com/tenstorrent/tt-metal/pull/33904"
        )

    compile_and_execute_shlo(
        test_fn,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(64, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("dimension", [0])
def test_get_dimension_size(
    shape: Shape,
    dtype: torch.dtype,
    dimension: int,
    target: str,
    request,
    device,
):
    def module_get_dimension_size(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def get_dimension_size(
            in0: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_graph_level_check(True)
            return builder.get_dimension_size(in0, dimension, unit_attrs=unit_attrs)

    compile_and_execute_shlo(
        module_get_dimension_size,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


_RESHAPE_CASES = [
    # shapes, semantic id, skip_ttmetal?
    ([(2, 3), (3, 2)], "swap", True),
    ([(2, 3), (6,)], "flatten", True),
    ([(1, 784), (1, 28, 28)], "unflatten", True),
    ([(4, 8, 16), (4, 128)], "3d_to_2d", True),
    ([(64, 512), (64, 1, 512)], "expand_dims", True),
    ([(128, 128), (64, 256)], "rearrange_2d", True),
    ([(10,), (10,)], "identity", True),
    ([(0, 6), (0, 2, 3)], "zero_dim", True),
]

_RESHAPE_PARAMS = []
for shapes, case_id, skip_ttmetal in _RESHAPE_CASES:
    # ttnn: expected to pass
    _RESHAPE_PARAMS.append(pytest.param(shapes, "ttnn", id=f"{case_id}-ttnn"))
    # ttmetal: skip cases known to be unsupported
    marks = []
    if skip_ttmetal:
        marks.append(
            pytest.mark.skip(
                reason="reshape lowering not yet supported in TTMetal backend"
            )
        )
    _RESHAPE_PARAMS.append(
        pytest.param(shapes, "ttmetal", id=f"{case_id}-ttmetal", marks=marks)
    )


@pytest.mark.parametrize("shapes, target", _RESHAPE_PARAMS)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
def test_reshape(
    shapes: tuple[Shape, Shape], target: str, dtype: torch.dtype, request, device
):
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


@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [
        module_select,
        module_clamp,
    ],
)
def test_ternary_ops(test_fn: Callable, target: str, request, device):
    compile_and_execute_shlo(
        test_fn,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
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


@pytest.mark.parametrize(
    "shapes,batch_dims_lhs,contract_dims_lhs,batch_dims_rhs,contract_dims_rhs",
    [
        (
            [(4, 10, 3, 5, 7), (4, 10, 5, 7, 3)],
            [0],
            [3],
            [0],
            [2],
        )
    ],
)
def test_dot_general(
    shapes: List[Shape],
    batch_dims_lhs: List[int],
    contract_dims_lhs: List[int],
    batch_dims_rhs: List[int],
    contract_dims_rhs: List[int],
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func(shapes, [torch.float32] * len(shapes))
        def dot_general(
            in0: Operand,
            in1: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.dot_general(
                in0,
                in1,
                batch_dims_lhs,
                contract_dims_lhs,
                batch_dims_rhs,
                contract_dims_rhs,
            )

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target="ttnn",
        device=device,
    )


@pytest.mark.parametrize("shape", [(2, 3, 4), (128, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
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
        "ttnn",
        pytest.param(
            "ttmetal",
            marks=pytest.mark.skip(
                reason="ttir.pad lowering not supported on ttmetal, failed to legalize"
            ),
        ),
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
    "shapes,dim",
    [
        ([(64, 128), (64, 128)], 0),  # 2 tensors, dim 0
        ([(128, 64), (128, 64)], 1),  # 2 tensors, dim 1
        ([(64, 128), (32, 128), (16, 128)], 0),  # 3 tensors, dim 0
        ([(32, 64), (32, 128)], 1),  # Different sizes in dim
        ([(64, 64), (64, 64), (64, 64)], 0),  # 3 identical tensors
        ([(128, 64), (128, 64), (128, 64), (128, 64)], 1),  # 4 tensors
    ],
    ids=["2t_dim0", "2t_dim1", "3t_dim0_ttir", "diff_size", "3t_same", "4t_dim1"],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_concatenate(
    shapes: List[Shape],
    dim: int,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    # Create a wrapper function
    def module(builder: StableHLOBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def concatenate_wrapper(*inputs_and_builder):
            *inputs, builder = inputs_and_builder
            builder.set_graph_level_check(True)
            return builder.concatenate(list(inputs), dim=dim)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shapes", [[(64, 64), (64, 64), (64, 64)]], ids=["64x64"])
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_stablehlo_multi_return_support(
    shapes: List[Shape], dtypes: List[torch.dtype], target: str, request, device
):
    def module(builder: StableHLOBuilder):
        @builder.func(shapes, dtypes)
        def multi_return_model(
            in0: Operand, in1: Operand, in2: Operand, builder: StableHLOBuilder
        ):
            builder.set_graph_level_check(True)

            add_result = builder.add(in0, in1)
            exp_result = builder.exp(in2)
            sqrt_result = builder.sqrt(exp_result)

            return exp_result, sqrt_result

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# Logical operations tests (boolean tensors)
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.bool], ids=["bool"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [
        module_and_bool,
        module_or_bool,
        module_xor_bool,
        module_shift_right_logical,
    ],
)
def test_logical_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_shlo(
        test_fn,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pcc=-1.0,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.bool], ids=["bool"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_logical_unary_ops(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module_not_(builder: StableHLOBuilder):
        @builder.func([(128, 128)], [torch.bool])
        def not_(
            in0: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_graph_level_check(True)
            return builder.not_(in0, unit_attrs=unit_attrs)

    compile_and_execute_shlo(
        module_not_,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pcc=-1.0,
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
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
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


@pytest.mark.parametrize("shape", [(1, 1, 64, 32), (1, 3, 256, 256)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
@pytest.mark.parametrize("is_splat", [True, False], ids=["splat", "non-splat"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_constant(
    shape: Shape, dtype: torch.dtype, is_splat: bool, target: str, request, device
):
    def constant_fn(builder: StableHLOBuilder):
        builder.set_graph_level_check(True)
        if is_splat:
            if dtype.is_floating_point:
                splat_value = torch.randn([])
            else:
                splat_value = torch.randint(-100, 100, [])
            tensor = torch.full(shape, splat_value.item(), dtype=dtype)
        else:
            if dtype.is_floating_point:
                tensor = torch.randn(shape, dtype=dtype)
            else:
                tensor = torch.randint(-100, 100, shape, dtype=dtype)

        result = builder.constant(tensor)
        return result

    compile_and_execute_shlo(
        constant_fn,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "shape,start_indices_val,slice_sizes",
    [
        ((128, 128), [0, 0], [64, 64]),
        ((128, 128), [32, 32], [64, 64]),
        ((128, 128), [0, 0], [128, 64]),
        ((256, 256), [64, 64], [128, 128]),
    ],
    ids=[
        "dyn_128x128_basic",
        "dyn_128x128_offset",
        "dyn_128x128_fullrow",
        "dyn_256x256_large",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_dynamic_slice(
    shape: Shape,
    start_indices_val: List[int],
    slice_sizes: List[int],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def dynamic_slice(in0: Operand, builder: StableHLOBuilder):
            builder.set_graph_level_check(True)
            return builder.dynamic_slice(
                in0, start_indices_val, slice_sizes=slice_sizes
            )

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# Bitwise operations tests (integer tensors)
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [
        module_and_int,
        module_or_int,
        module_xor_int,
    ],
)
def test_bitwise_binary_ops(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    compile_and_execute_shlo(
        test_fn,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.int32], ids=["i32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_bitwise_unary_ops(
    shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module_not_(builder: StableHLOBuilder):
        @builder.func([(128, 128)], [torch.int32])
        def not_(
            in0: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_graph_level_check(True)
            return builder.not_(in0, unit_attrs=unit_attrs)

    compile_and_execute_shlo(
        module_not_,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# ----- Reduce Operations -----


def reduce_sum(
    in0: Operand,
    builder: StableHLOBuilder,
    dimensions: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    return builder.reduce_sum(in0, dimensions, unit_attrs=unit_attrs)


def reduce_max(
    in0: Operand,
    builder: StableHLOBuilder,
    dimensions: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    return builder.reduce_max(in0, dimensions, unit_attrs=unit_attrs)


def reduce_min(
    in0: Operand,
    builder: StableHLOBuilder,
    dimensions: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    return builder.reduce_min(in0, dimensions, unit_attrs=unit_attrs)


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("dimensions", [[0], [1]])
def test_reduce_sum(
    shape: Shape,
    dtype: torch.dtype,
    dimensions: List[int],
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def reduce_sum_wrapper(in0: Operand, builder: StableHLOBuilder):
            return reduce_sum(in0, builder, dimensions)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("dimensions", [[0], [1]])
def test_reduce_max(
    shape: Shape,
    dtype: torch.dtype,
    dimensions: List[int],
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def reduce_max_wrapper(in0: Operand, builder: StableHLOBuilder):
            return reduce_max(in0, builder, dimensions)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("dimensions", [[0], [1]])
def test_reduce_min(
    shape: Shape,
    dtype: torch.dtype,
    dimensions: List[int],
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def reduce_min_wrapper(in0: Operand, builder: StableHLOBuilder):
            return reduce_min(in0, builder, dimensions)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


# ----- Pooling Operations -----


def max_pool_2d(
    in0: Operand,
    builder: StableHLOBuilder,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.pool_2d(
        in0,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        pool_type="max",
        unit_attrs=unit_attrs,
    )


def avg_pool_2d(
    in0: Operand,
    builder: StableHLOBuilder,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    builder.set_graph_level_check(True)
    return builder.pool_2d(
        in0,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        pool_type="avg",
        unit_attrs=unit_attrs,
    )


@pytest.mark.parametrize(
    "shape,kernel_size,stride,padding",
    [
        ((32, 32), [3, 3], [2, 2], [1, 1, 1, 1]),
        ((64, 64), [2, 2], [2, 2], [0, 0, 0, 0]),
        ((128, 128), [3, 3], [1, 1], [1, 1, 1, 1]),
        ((1, 32, 64, 64), [1, 1, 3, 3], [1, 1, 2, 2], [0, 0, 0, 0, 1, 1, 1, 1]),
        ((1, 64, 128, 128), [1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0]),
        ((1, 32, 64, 64), [1, 1, 3, 3], [1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]),
    ],
    ids=[
        "rank2_k3s2p1",
        "rank2_k2s2p0",
        "rank2_k3s1p1",
        "rank4_k3s2p1",
        "rank4_k2s2p0",
        "rank4_k3s1p1",
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_max_pool_2d(
    shape: Shape,
    dtype: torch.dtype,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def max_pool_2d_wrapper(in0: Operand, builder: StableHLOBuilder):
            return max_pool_2d(in0, builder, kernel_size, stride, padding)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "shape,kernel_size,stride,padding",
    [
        ((32, 32), [3, 3], [2, 2], [1, 1, 1, 1]),
        ((64, 64), [2, 2], [2, 2], [0, 0, 0, 0]),
        ((128, 128), [3, 3], [1, 1], [1, 1, 1, 1]),
        ((1, 32, 64, 64), [1, 1, 3, 3], [1, 1, 2, 2], [0, 0, 0, 0, 1, 1, 1, 1]),
        ((1, 64, 128, 128), [1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 0, 0, 0, 0, 0, 0]),
        ((1, 32, 64, 64), [1, 1, 3, 3], [1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1]),
    ],
    ids=[
        "rank2_k3s2p1",
        "rank2_k2s2p0",
        "rank2_k3s1p1",
        "rank4_k3s2p1",
        "rank4_k2s2p0",
        "rank4_k3s1p1",
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_avg_pool_2d(
    shape: Shape,
    dtype: torch.dtype,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def avg_pool_2d_wrapper(in0: Operand, builder: StableHLOBuilder):
            return avg_pool_2d(in0, builder, kernel_size, stride, padding)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "shapes, dtype, dimensions",
    [
        ([(2, 3, 4)], torch.float32, [1]),
        ([(2, 3, 4)], torch.float32, [0, 1]),
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_reverse(shapes, dtype, dimensions, target: str, request, device):
    def module(builder: StableHLOBuilder):
        @builder.func(shapes, [dtype])
        def reverse(in0: Operand, builder: StableHLOBuilder):
            builder.set_graph_level_check(True)
            return builder.reverse(in0, dimensions)

    compile_and_execute_shlo(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("target", ["ttnn"])
def test_select(target: str, request, device):
    compile_and_execute_shlo(
        module_select,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


def module_batch_norm_training(builder: StableHLOBuilder):
    @builder.func(
        [(1, 32, 64, 64), (64,), (64,)],
        [torch.float32, torch.float32, torch.float32],
    )
    def batch_norm_training(
        operand: Operand,
        scale: Operand,
        offset: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.batch_norm_training(
            operand, scale, offset, epsilon=1e-5, feature_index=3
        )


def module_batch_norm_inference(builder: StableHLOBuilder):
    @builder.func(
        [(1, 32, 64, 64), (64,), (64,), (64,), (64,)],
        [torch.float32] * 5,
    )
    def batch_norm_inference(
        operand: Operand,
        scale: Operand,
        offset: Operand,
        mean: Operand,
        variance: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.batch_norm_inference(
            operand,
            scale,
            offset,
            mean,
            variance,
            epsilon=1e-5,
            feature_index=3,
        )


def module_batch_norm_grad(builder: StableHLOBuilder):
    @builder.func(
        [(1, 32, 64, 64), (64,), (64,), (64,), (1, 32, 64, 64)],
        [torch.float32] * 5,
    )
    def batch_norm_grad(
        operand: Operand,
        scale: Operand,
        mean: Operand,
        variance: Operand,
        grad_output: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.batch_norm_grad(
            operand,
            scale,
            mean,
            variance,
            grad_output,
            epsilon=1e-5,
            feature_index=3,
        )


@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [
        module_batch_norm_training,
    ],
)
def test_batch_norm_training_op(test_fn: Callable, target: str, request, device):

    compile_and_execute_shlo(
        test_fn,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [
        module_batch_norm_inference,
    ],
)
def test_batch_norm_inference_op(test_fn: Callable, target: str, request, device):
    compile_and_execute_shlo(
        test_fn,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [
        module_batch_norm_grad,
    ],
)
def test_batch_norm_grad_op(test_fn: Callable, target: str, request, device):
    compile_and_execute_shlo(
        test_fn,
        **get_request_kwargs(request),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "output_shape, output_type, iota_dimension",
    [
        ([4, 5], torch.float32, 0),
        ([4, 5], torch.float32, 1),
        ([2, 3, 4], torch.float32, 1),
        ([2, 3, 4], torch.float32, 2),
        ([32, 64], torch.bfloat16, 0),
        ([32, 64], torch.bfloat16, 1),
    ],
    ids=[
        "shape_4x5_dim0",
        "shape_4x5_dim1",
        "shape_2x3x4_dim1",
        "shape_2x3x4_dim2",
        "shape_32x64_bf16_dim0",
        "shape_32x64_bf16_dim1",
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_iota(
    output_shape: List[int],
    output_type: torch.dtype,
    iota_dimension: int,
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([output_shape], [output_type])
        def iota(
            in0: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_graph_level_check(True)
            return builder.iota(
                output=in0.type,
                iota_dimension=iota_dimension,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_shlo(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "output_shape, output_type, iota_dimension",
    [
        ([4, 5], torch.float32, 0),
        ([4, 5], torch.float32, 1),
        ([2, 3, 4], torch.float32, 1),
        ([2, 3, 4], torch.float32, 2),
        ([32, 64], torch.bfloat16, 0),
        ([32, 64], torch.bfloat16, 1),
    ],
    ids=[
        "shape_4x5_dim0",
        "shape_4x5_dim1",
        "shape_2x3x4_dim1",
        "shape_2x3x4_dim2",
        "shape_32x64_bf16_dim0",
        "shape_32x64_bf16_dim1",
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
def test_dynamic_iota(
    output_shape: List[int],
    output_type: torch.dtype,
    iota_dimension: int,
    target: str,
    request,
    device,
):
    def module(builder: StableHLOBuilder):
        @builder.func([output_shape], [output_type])
        def dynamic_iota(
            in0: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_graph_level_check(True)
            shape_tensor = builder.constant(
                torch.tensor(output_shape, dtype=torch.int64)
            )
            return builder.dynamic_iota(
                output=in0.type,
                output_shape=shape_tensor,
                iota_dimension=iota_dimension,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_shlo(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


def module_reduce_window_sum(builder: StableHLOBuilder):
    @builder.func([(1, 1, 8, 8)], [torch.float32])
    def reduce_window_sum(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.reduce_window(
            in0,
            init_value=0.0,
            window_dimensions=[1, 1, 2, 2],
            window_strides=[1, 1, 2, 2],
            padding=[[0, 0], [0, 0], [0, 0], [0, 0]],
            body="add",
        )


def module_reduce_window_avg(builder: StableHLOBuilder):
    @builder.func([(1, 1, 8, 8)], [torch.float32])
    def reduce_window_avg(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        window_h, window_w = 2, 2
        window_area = float(window_h * window_w)
        output_shape = (1, 1, 4, 4)
        summed = builder.reduce_window(
            in0,
            init_value=0.0,
            window_dimensions=[1, 1, window_h, window_w],
            window_strides=[1, 1, 2, 2],
            padding=[[0, 0], [0, 0], [0, 0], [0, 0]],
            body="add",
        )
        divisor = builder.constant(
            torch.full(output_shape, window_area, dtype=torch.float32)
        )
        return builder.divide(summed, divisor)


def module_reduce_window_max(builder: StableHLOBuilder):
    @builder.func([(1, 1, 8, 8)], [torch.float32])
    def reduce_window_max(
        in0: Operand,
        builder: StableHLOBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        builder.set_graph_level_check(True)
        return builder.reduce_window(
            in0,
            init_value=float("-inf"),
            window_dimensions=[1, 1, 3, 3],
            window_strides=[1, 1, 1, 1],
            padding=[[0, 0], [0, 0], [1, 1], [1, 1]],
            body="max",
        )


@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize(
    "test_fn",
    [module_reduce_window_sum, module_reduce_window_avg, module_reduce_window_max],
)
def test_reduce_window_op(test_fn: Callable, target: str, request, device):
    compile_and_execute_shlo(
        test_fn,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize("target", ["ttnn"])
def test_all_gather(target: str, request, device):
    def module_all_gather(builder: StableHLOBuilder):
        @builder.func([(1, 1, 32, 32)], [torch.float32])
        def my_modela(in0: Operand, builder: StableHLOBuilder):
            def single_device_func(in0: Operand, builder: StableHLOBuilder):
                all_gather0 = builder.all_gather(in0, 3, [[0]])
                return all_gather0

            tensor_sharding_attr = builder.tensor_sharding_attr(
                mesh_name="mesh",
                dimension_shardings=[
                    builder.dimension_sharding_attr(
                        axes=[],
                        is_closed=True,
                    ),
                    builder.dimension_sharding_attr(
                        axes=[],
                        is_closed=True,
                    ),
                    builder.dimension_sharding_attr(
                        axes=[builder.axis_ref_attr(name="x")],
                        is_closed=True,
                    ),
                    builder.dimension_sharding_attr(
                        axes=[builder.axis_ref_attr(name="y")],
                        is_closed=True,
                    ),
                ],
            )

            manual_computation_op0 = builder.manual_computation(
                single_device_func,
                [in0],
                in_shardings=[tensor_sharding_attr],
                out_shardings=[tensor_sharding_attr],
                manual_axes=["x", "y"],
            )
            return manual_computation_op0

    compile_and_execute_shlo(
        module_all_gather,
        **get_request_kwargs(request),
        target=target,
        device=device,
        mesh_dict=OrderedDict([("x", 1), ("y", 1)]),
    )


@pytest.mark.parametrize(
    "shapes,stride,padding,dilation,groups",
    [
        # ResNet initial 7x7 conv: stride=2, padding=3
        (
            [(1, 3, 224, 224), (64, 3, 7, 7)],
            [2, 2],
            [3, 3, 3, 3],
            [1, 1],
            1,
        ),
        # ResNet 1x1 conv: stride=1, no padding
        (
            [(1, 64, 56, 56), (64, 64, 1, 1)],
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
            1,
        ),
        # ResNet 3x3 conv: stride=1, padding=1
        (
            [(1, 64, 56, 56), (64, 64, 3, 3)],
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            1,
        ),
        # ResNet bottleneck 1x1 expansion: stride=1, no padding
        (
            [(1, 64, 56, 56), (256, 64, 1, 1)],
            [1, 1],
            [0, 0, 0, 0],
            [1, 1],
            1,
        ),
        # ResNet stride 2 downsampling: 3x3 conv
        (
            [(1, 64, 56, 56), (128, 64, 3, 3)],
            [2, 2],
            [1, 1, 1, 1],
            [1, 1],
            1,
        ),
        # Small test case
        (
            [(1, 16, 32, 32), (32, 16, 3, 3)],
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            1,
        ),
    ],
    ids=[
        "resnet_initial_7x7",
        "resnet_1x1_conv",
        "resnet_3x3_conv",
        "resnet_bottleneck_expansion",
        "resnet_stride2_downsample",
        "small_3x3",
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_convolution(
    shapes: List[Shape],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Test the stablehlo.convolution op with various ResNet-style configurations"""

    def module(builder: StableHLOBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def convolution(
            in0: Operand,
            weight: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.convolution(
                in0,
                weight,
                window_strides=stride,
                padding=padding,
                lhs_dilation=dilation,
                rhs_dilation=dilation,
                input_batch_dimension=0,
                input_feature_dimension=1,
                input_spatial_dimensions=list(range(2, 2 + len(dilation))),
                kernel_output_feature_dimension=0,
                kernel_input_feature_dimension=1,
                kernel_spatial_dimensions=list(range(2, 2 + len(dilation))),
                output_batch_dimension=0,
                output_feature_dimension=1,
                output_spatial_dimensions=list(range(2, 2 + len(dilation))),
                feature_group_count=groups,
                batch_group_count=1,
            )

    compile_and_execute_shlo(
        module,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )


@pytest.mark.parametrize(
    "shapes,stride,padding,dilation,groups",
    [
        # Depthwise convolution (groups = input_channels)
        (
            [(1, 32, 28, 28), (32, 1, 3, 3)],
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            32,
        ),
        # Group convolution (4 groups)
        (
            [(1, 64, 32, 32), (64, 16, 3, 3)],
            [1, 1],
            [1, 1, 1, 1],
            [1, 1],
            4,
        ),
        # Dilated convolution
        (
            [(1, 32, 32, 32), (64, 32, 3, 3)],
            [1, 1],
            [2, 2, 2, 2],
            [2, 2],
            1,
        ),
    ],
    ids=["depthwise_conv", "group_conv_4groups", "dilated_conv"],
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_convolution_groups_dilation(
    shapes: List[Shape],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Test convolution with group and dilation patterns"""

    def module(builder: StableHLOBuilder):
        @builder.func(shapes, [dtype] * len(shapes))
        def convolution(
            in0: Operand,
            weight: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.convolution(
                in0,
                weight,
                window_strides=stride,
                padding=padding,
                lhs_dilation=[1, 1],
                rhs_dilation=dilation,
                input_batch_dimension=0,
                input_feature_dimension=1,
                input_spatial_dimensions=[2, 3],
                kernel_output_feature_dimension=0,
                kernel_input_feature_dimension=1,
                kernel_spatial_dimensions=[2, 3],
                output_batch_dimension=0,
                output_feature_dimension=1,
                output_spatial_dimensions=[2, 3],
                feature_group_count=groups,
                batch_group_count=1,
            )

    compile_and_execute_shlo(
        module,
        test_base=request.node.name,
        target=target,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


@pytest.mark.parametrize("shape", [(128,)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("broadcast_dimensions", [[1]])
@pytest.mark.parametrize("output_shape", [[32, 128]])
def test_broadcast_ops(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    broadcast_dimensions: List[int],
    output_shape: List[int],
    request,
    device,
):
    # Create a wrapper function that captures broadcast_dimensions and output_shape
    def broadcast_wrapper(builder: StableHLOBuilder):
        @builder.func([shape], [dtype])
        def broadcast(
            in0: Operand,
            builder: StableHLOBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_graph_level_check(True)
            return builder.broadcast_in_dim(
                in0,
                broadcast_dimensions=broadcast_dimensions,
                output_shape=output_shape,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_shlo(
        broadcast_wrapper,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
    )
