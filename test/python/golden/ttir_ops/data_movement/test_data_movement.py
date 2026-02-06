# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from typing import List, Optional
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import (
    shapes_list_str,
    shape_str,
)

pytestmark = pytest.mark.frontend("ttir")

# Concat tests
@pytest.mark.parametrize(
    "shapes",
    [
        [
            (64, 128),
            (32, 128),
            (16, 128),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_concat(shapes: List[Shape], dim: int, target: str, request, device):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32, torch.float32, torch.float32])
        def concat_wrapper(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.concat([in0, in1, in2], dim, unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@x86_only
@pytest.mark.parametrize(
    "shapes",
    [
        [
            (64, 128),
            (32, 128),
            (16, 128),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
@pytest.mark.xfail(reason="Compile failure on CPU target")
def test_cpu_hoistable_concat_op(
    shapes: List[Shape],
    dim: int,
    request,
    target: str,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32, torch.float32, torch.float32])
        def hoisted_concat_wrapper(
            in0: Operand,
            in1: Operand,
            in2: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.concat(
                [in0, in1, in2], dim, unit_attrs=["ttir.should_hoist"]
            )

    """Test unary ops that support CPU hoisting"""
    compile_and_execute_ttir(
        module,
        test_base=f"{request.node.name}",
        target=target,
        device=device,
    )


# Pad tests
@pytest.mark.parametrize("shape", [(1, 1, 5, 5)], ids=shape_str)
@pytest.mark.parametrize("padding", [[0, 1, 2, 3, 4, 5, 6, 7]])
@pytest.mark.parametrize("value", [0])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_pad(
    shape: Shape, padding: List[int], value: int, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def pad_wrapper(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.pad(in0, padding=padding, value=value)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# Permute tests
@pytest.mark.parametrize("shapes", [[(2, 3, 4)]], ids=shapes_list_str)
@pytest.mark.parametrize("permutation", [[1, 2, 0]])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_permute(
    shapes: List[Shape], permutation: List[int], target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32])
        def permute_wrapper(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.permute(
                in0,
                permutation=permutation,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# RepeatInterleave tests
@pytest.mark.parametrize(
    "shapes",
    [[(1, 8, 1, 12, 64)]],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("repeats", [1])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_repeat_interleave(
    shapes: List[Shape], repeats: int, dim: int, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func(shapes, [torch.float32])
        def repeat_interleave_wrapper(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.repeat_interleave(
                in0, repeats=repeats, dim=dim, unit_attrs=unit_attrs
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# Repeat tests
@pytest.mark.parametrize("shape", [(1, 32, 32), (2, 16, 16), (1, 1, 64)], ids=shape_str)
@pytest.mark.parametrize("dims", [[32, 1, 1], [1, 2, 2], [2, 3, 4], [1, 1, 1]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_repeat(shape: Shape, dims: List[int], dtype, target: str, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def repeat_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.repeat(in0, repeat_dimensions=dims, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# Reshape tests
@pytest.mark.parametrize(
    "shapes",
    [
        # [input_shape, output_shape]
        [(128, 128), (16384,)],  # Flatten 2D to 1D
        [(24,), (2, 3, 4)],  # Unflatten 1D to 3D
        [(2, 3, 4), (6, 4)],  # 3D to 2D reshape
        [(128, 128), (64, 256)],  # 2D to 2D different arrangement
        [(1, 1, 1), (1,)],  # Edge case: all dimensions are 1
        [(10,), (10,)],  # Identity reshape
        [(64, 512), (64, 1, 512)],  # Common ML pattern: expand dims
        [(256, 256), (512, 128)],  # Power of 2 reshape
        [(32, 3, 224, 224), (32, 150528)],  # Large ML pattern: batch flatten
        [(0, 32, 128), (0,)],  # Edge case: zero-sized dimension
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.int64, torch.int32, torch.uint8],
    ids=["f32", "i64", "i32", "ui8"],
)
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_reshape(shapes, dtype: torch.dtype, target: str, request, device):
    input_shape, output_shape = shapes

    # Large tensor reshape with int types fails due to tt-metal untilize issue
    # for tensors wider than MAX_PACK_UNTILIZE_WIDTH (8 tiles).
    # See: https://github.com/tenstorrent/tt-metal/issues/34072
    if dtype in [torch.int32, torch.int64]:
        pytest.xfail(
            "Large tensor reshape with int types fails due to tt-metal untilize issue. "
            "See: https://github.com/tenstorrent/tt-metal/issues/34072"
        )

    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [dtype])
        def reshape_wrapper(in0: Operand, builder: TTIRBuilder):
            return builder.reshape(in0, output_shape)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@pytest.mark.parametrize("shape", [(1, 128, 128, 1)], ids=shape_str)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_squeeze(shape: Shape, dim: int, target: str, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def squeeze_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.squeeze(in0, dim, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_unsqueeze(shape: Shape, dim: int, target: str, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def unsqueeze_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.unsqueeze(in0, dim, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@x86_only
@pytest.mark.parametrize("shapes", [[(128, 128), (16384,)]], ids=shapes_list_str)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_cpu_hoistable_reshape_op(
    shapes: List[Shape],
    request,
    target: str,
    device,
):
    input_shape, output_shape = shapes

    def module(builder: TTIRBuilder):
        @builder.func([input_shape], [torch.float32])
        def hoisted_reshape_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.reshape(
                in0, shape=output_shape, unit_attrs=["ttir.should_hoist"]
            )

    """Test unary ops that support CPU hoisting"""
    compile_and_execute_ttir(
        module,
        test_base=f"{request.node.name}",
        target=target,
        device=device,
    )


# Slice tests
@pytest.mark.parametrize(
    "shape,begins,ends,step",
    [
        # Simple 2D
        ((64, 64), [0, 0], [32, 32], None),
        # Crop 2D
        ((64, 64), [10, 20], [50, 60], [1, 1]),
        # Every three rows/cols
        ((192, 64), [2, 0], [192, 64], [3, 1]),
        ((64, 192), [0, 2], [64, 192], [1, 3]),
        # Sample large 2D tensors
        ((32, 131072), [0, 3], [32, 128 * 991], [2, 991]),
        ((131072, 32), [5, 1], [128 * 997, 32], [997, 2]),
        ((1024, 1024), [3, 2], [64 * 11, 64 * 13], [11, 13]),
        # Simple 3D
        ((2, 64, 32), [0, 0, 0], [1, 64, 32], None),
        # Interleaved 3D
        ((2, 64, 32), [0, 1, 0], [1, 64, 32], [1, 2, 1]),
        ((2, 64, 32), [0, 1, 0], [1, 64, 32], [1, 2, 2]),
        # Strided crop 3D
        ((64, 64, 64), [10, 20, 28], [50, 60, 64], [2, 2, 1]),
        ((64, 64, 64), [10, 20, 30], [50, 60, 64], [2, 2, 1]),
        ((64, 64, 64), [5, 30, 12], [11, 34, 36], [3, 1, 4]),
        # Minus 1
        ((3, 512, 256), [0, 1, 0], [3, 512, 255], None),
        ((5, 65, 1025), [0, 0, 1], [5, 64, 1025], None),
        # Simple 4D
        ((2, 24, 32, 128), [1, 8, 3, 64], [2, 16, 7, 128], None),
        # NCHW: 2nd half - green - down sample & make square
        ((4, 3, 64, 96), [2, 1, 1, 0], [4, 2, 64, 96], [1, 1, 2, 3]),
        # NHWC: odd - crop - blue & alpha
        ((6, 64, 64, 4), [1, 15, 16, 2], [6, 47, 48, 4], [2, 1, 1, 1]),
        # Simple 5D
        ((2, 4, 6, 64, 64), [0, 1, 0, 0, 0], [1, 2, 1, 32, 32], None),
        # Mixed 5D
        ((3, 4, 5, 128, 128), [1, 0, 3, 32, 64], [3, 4, 4, 96, 128], [1, 2, 1, 1, 1]),
        # Pick a single element
        ((3, 20, 14, 64, 64), [1, 5, 6, 31, 32], [2, 6, 7, 32, 33], None),
    ],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
def test_slice(
    shape: Shape,
    begins: List[int],
    ends: List[int],
    step: List[int],
    target: str,
    request,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def slice_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.slice(in0, begins, ends, step, unit_attrs=unit_attrs)

    # NoC alignment is at least 16B => must align to 4 floats.
    special_dma = (begins[-1] % 4 != 0) or (step is not None and step[-1] != 1)
    if target == "ttmetal" and special_dma:
        request.node.add_marker(
            pytest.mark.xfail(
                reason="Unaligned and/or strided DMA in the last dim #6475", run=True
            )
        )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# Sort tests
@pytest.mark.parametrize("shape", [(1, 64, 64)], ids=shape_str)
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float32, torch.int32, torch.int16],
    ids=["bf16", "f32", "int32", "int16"],
)
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("stable", [True, False])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_sort(
    shape: Shape,
    dtype: torch.dtype,
    dim: int,
    descending: bool,
    stable: bool,
    target: str,
    request,
    device,
):
    if dim == 0:
        pytest.skip(
            "Sorting along batch dimension is not supported: https://github.com/tenstorrent/tt-metal/issues/31187"
        )

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def sort_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            sort_0_values, sort_0_indices = builder.sort(
                in0,
                dim=dim,
                descending=descending,
                stable=stable,
                unit_attrs=unit_attrs,
            )

            return sort_0_values  # Return only sorted values for testing

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


# Transpose tests
@pytest.mark.parametrize("shape", [(64, 32)], ids=shape_str)
@pytest.mark.parametrize("transpose_dims", [(0, 1)])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
def test_transpose(
    shape: Shape, transpose_dims: List[int], target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def transpose_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.transpose(
                in0,
                dim0=transpose_dims[0],
                dim1=transpose_dims[1],
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
    )


@x86_only
@pytest.mark.parametrize("shape", [(32, 64, 128)], ids=shape_str)
@pytest.mark.parametrize("transpose_dims", [(0, 1)])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_cpu_hoistable_transpose_op(
    shape: Shape,
    transpose_dims: List[int],
    request,
    target: str,
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def hoisted_transpose_wrapper(
            in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
        ):
            return builder.transpose(
                in0,
                dim0=transpose_dims[0],
                dim1=transpose_dims[1],
                unit_attrs=["ttir.should_hoist"],
            )

    """Test unary ops that support CPU hoisting"""
    compile_and_execute_ttir(
        module,
        test_base=f"{request.node.name}",
        target=target,
        device=device,
    )


# Typecast tests
@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize(
    "from_type,to_type",
    [
        (torch.int32, torch.float32),
        (torch.float32, torch.int32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
    ],
    ids=["i32-f32", "f32-i32", "bf16-f32", "f32-bf16"],
)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
def test_typecast(
    shape: Shape,
    from_type: torch.dtype,
    to_type: torch.dtype,
    target: str,
    request,
    device,
):
    if from_type == torch.float32 and to_type == torch.int32 and target == "ttmetal":
        pytest.xfail("ttmetal does not support float32 to int32 typecast")

    if from_type == torch.float32 and to_type == torch.bfloat16 and target == "ttmetal":
        pytest.xfail(
            "f32->bf16 typecast fails due to LLK tiling issue. "
            "See comment at: https://github.com/tenstorrent/tt-metal/issues/35302"
        )

    def module(builder: TTIRBuilder):
        @builder.func([shape], [from_type])
        def typecast(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.typecast(in0, output_type=to_type, unit_attrs=unit_attrs)

    pipeline_options = []
    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        target=target,
        pipeline_options=pipeline_options,
    )
