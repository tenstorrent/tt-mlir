# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from typing import List, Optional
from conftest import x86_only
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir
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
    def concat_wrapper(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.concat([in0, in1, in2], dim, unit_attrs)

    # Set the name for better test identification.
    concat_wrapper.__name__ = "concat"

    compile_and_execute_ttir(
        concat_wrapper,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
    def hoisted_concat_wrapper(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.concat([in0, in1, in2], dim, unit_attrs=["ttir.should_hoist"])

    hoisted_concat_wrapper.__name__ = "hoisted_concat"

    """Test unary ops that support CPU hoisting"""
    compile_and_execute_ttir(
        hoisted_concat_wrapper,
        inputs_shapes=shapes,
        test_base=f"{request.node.name}",
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Pad tests
@pytest.mark.parametrize("shape", [(1, 1, 5, 5)], ids=shape_str)
@pytest.mark.parametrize("padding", [[0, 1, 2, 3, 4, 5, 6, 7]])
@pytest.mark.parametrize("value", [0])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_pad(
    shape: Shape, padding: List[int], value: int, target: str, request, device
):
    def pad_wrapper(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.pad(in0, padding=padding, value=value, unit_attrs=unit_attrs)

    pad_wrapper.__name__ = "pad"

    compile_and_execute_ttir(
        pad_wrapper,
        inputs_shapes=[shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


# Permute tests
@pytest.mark.parametrize("shapes", [[(2, 3, 4)]], ids=shapes_list_str)
@pytest.mark.parametrize("permutation", [[1, 2, 0]])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_permute(
    shapes: List[Shape], permutation: List[int], target: str, request, device
):
    def permute_wrapper(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.permute(
            in0,
            permutation=permutation,
            unit_attrs=unit_attrs,
        )

    permute_wrapper.__name__ = "permute"

    compile_and_execute_ttir(
        permute_wrapper,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


# RepeatInterleave tests
@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 8, 1, 12, 64),
            (1, 8, 1, 12, 64),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("repeats", [1])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_repeat_interleave(
    shapes: List[Shape], repeats: int, dim: int, target: str, request, device
):
    def repeat_interleave_wrapper(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.repeat_interleave(
            in0, in1, repeats=repeats, dim=dim, unit_attrs=unit_attrs
        )

    repeat_interleave_wrapper.__name__ = "repeat_interleave"

    compile_and_execute_ttir(
        repeat_interleave_wrapper,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


# Repeat tests
@pytest.mark.parametrize("shape", [(1, 32, 32), (2, 16, 16), (1, 1, 64)], ids=shape_str)
@pytest.mark.parametrize("dims", [[32, 1, 1], [1, 2, 2], [2, 3, 4], [1, 1, 1]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=["f32", "i32"])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_repeat(shape: Shape, dims: List[int], dtype, target: str, request, device):
    def repeat_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.repeat(in0, dims=dims, unit_attrs=unit_attrs)

    repeat_wrapper.__name__ = "repeat"

    compile_and_execute_ttir(
        repeat_wrapper,
        [shape],
        [dtype],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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

    def reshape_wrapper(in0: Operand, builder: TTIRBuilder):
        return builder.reshape(in0, output_shape)

    reshape_wrapper.__name__ = "reshape"

    compile_and_execute_ttir(
        reshape_wrapper,
        [input_shape],
        [dtype],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


@pytest.mark.parametrize("shape", [(1, 128, 128, 1)], ids=shape_str)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_squeeze(shape: Shape, dim: int, target: str, request, device):
    def squeeze_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.squeeze(in0, dim, unit_attrs=unit_attrs)

    squeeze_wrapper.__name__ = "squeeze"

    compile_and_execute_ttir(
        squeeze_wrapper,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


@pytest.mark.parametrize("shape", [(128, 128)], ids=shape_str)
@pytest.mark.parametrize("dim", [0])
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_unsqueeze(shape: Shape, dim: int, target: str, request, device):
    def unsqueeze_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.unsqueeze(in0, dim, unit_attrs=unit_attrs)

    unsqueeze_wrapper.__name__ = "unsqueeze"

    compile_and_execute_ttir(
        unsqueeze_wrapper,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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

    def hoisted_reshape_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.reshape(
            in0, shape=output_shape, unit_attrs=["ttir.should_hoist"]
        )

    hoisted_reshape_wrapper.__name__ = "hoisted_reshape"

    """Test unary ops that support CPU hoisting"""
    compile_and_execute_ttir(
        hoisted_reshape_wrapper,
        inputs_shapes=[input_shape],
        test_base=f"{request.node.name}",
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Slice tests
@pytest.mark.parametrize(
    "shape,begins,ends,step",
    [
        ((64, 64), [0, 0], [32, 32], None),
        ((64, 64), [10, 20], [50, 60], [1, 1]),
        ((64, 64, 64), [10, 20, 30], [50, 60, 64], [2, 2, 1]),
    ],
    ids=["basic_slice", "explicit_step", "3d_slice"],
)
@pytest.mark.parametrize("target", ["ttnn", "emitpy"])
def test_slice(
    shape: Shape,
    begins: List[int],
    ends: List[int],
    step: List[int],
    target: str,
    request,
    device,
):
    def slice_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.slice(in0, begins, ends, step, unit_attrs=unit_attrs)

    slice_wrapper.__name__ = "slice"

    compile_and_execute_ttir(
        slice_wrapper,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


# Sort tests
@pytest.mark.parametrize("shape", [(1, 64, 64)], ids=shape_str)
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
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

    def sort_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        sort_0 = builder.sort(
            in0, dim=dim, descending=descending, stable=stable, unit_attrs=unit_attrs
        )
        # Calculate golden for values and indices
        in0_golden = builder._get_golden_tensor(in0)

        values, indicies = torch.sort(
            in0_golden, dim=dim, descending=descending, stable=stable
        )
        builder.set_goldens_from_builder_tensor(
            {in0: in0_golden}, {sort_0.values: values}
        )

        return sort_0.values  # Return only sorted values for testing

    sort_wrapper.__name__ = "sort"

    compile_and_execute_ttir(
        sort_wrapper,
        [shape],
        [dtype],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )


# Transpose tests
@pytest.mark.parametrize("shape", [(64, 32)], ids=shape_str)
@pytest.mark.parametrize("transpose_dims", [(0, 1)])
@pytest.mark.parametrize("target", ["ttnn", "ttmetal", "emitpy"])
def test_transpose(
    shape: Shape, transpose_dims: List[int], target: str, request, device
):
    def transpose_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.transpose(
            in0, dim0=transpose_dims[0], dim1=transpose_dims[1], unit_attrs=unit_attrs
        )

    transpose_wrapper.__name__ = "transpose"

    compile_and_execute_ttir(
        transpose_wrapper,
        [shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
    def hoisted_transpose_wrapper(
        in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
    ):
        return builder.transpose(
            in0,
            dim0=transpose_dims[0],
            dim1=transpose_dims[1],
            unit_attrs=["ttir.should_hoist"],
        )

    hoisted_transpose_wrapper.__name__ = "hoisted_transpose"

    """Test unary ops that support CPU hoisting"""
    compile_and_execute_ttir(
        hoisted_transpose_wrapper,
        inputs_shapes=[shape],
        test_base=f"{request.node.name}",
        target=target,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
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
        pytest.xfail("ttmetal does not support int32 to float32 typecast")

    def typecast(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.typecast(in0, output_type=to_type, unit_attrs=unit_attrs)

    pipeline_options = []
    compile_and_execute_ttir(
        typecast,
        [shape],
        [from_type],
        test_base=request.node.name,
        device=device,
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        pipeline_options=pipeline_options,
    )
