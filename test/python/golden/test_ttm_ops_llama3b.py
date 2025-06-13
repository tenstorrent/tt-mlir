# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import pytest
import torch
from typing import Callable, List

from ttir_builder import Operand, TTIRBuilder, Shape, TypeInfo
from ttir_builder.utils import compile_to_flatbuffer, Marks, shape_str
from ttmlir.ir import (
    DenseI64ArrayAttr,
    DenseI32ArrayAttr,
)


TORCH_DTYPE_IDS = {
    torch.float32: "fp32",
    torch.bfloat16: "bf16",
    torch.int64: "i64",
    torch.float64: "fp64",
}


@dataclass
class OpAndTensors:
    test_op: callable
    shapes: list[Shape]
    dtypes: list[torch.dtype]
    base_name: str

    def __str__(self):
        return (
            self.base_name
            + "_"
            + "_".join(
                map(
                    lambda x: f"{shape_str(x[0])}-{TORCH_DTYPE_IDS[x[1]]}",
                    zip(self.shapes, self.dtypes),
                )
            )
        )


def add(in0: Operand, in1: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None):
    return builder.add(in0, in1, unit_attrs=unit_attrs)


def div(in0: Operand, in1: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None):
    return builder.div(in0, in1, unit_attrs=unit_attrs)


def multiply(
    in0: Operand, in1: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None
):
    return builder.multiply(in0, in1, unit_attrs=unit_attrs)


def pow(in0: Operand, in1: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None):
    return builder.pow(in0, in1, unit_attrs=unit_attrs)


def subtract(
    in0: Operand, in1: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None
):
    return builder.subtract(in0, in1, unit_attrs=unit_attrs)


binary_eltwise_cases = [
    OpAndTensors(add, [(1, 32, 1)], [torch.float32], "add"),
    OpAndTensors(add, [(1, 8, 32, 128)], [torch.bfloat16], "add"),
    OpAndTensors(add, [(1, 24, 24, 32)], [torch.bfloat16], "add"),
    OpAndTensors(add, [(1, 32, 3072)], [torch.bfloat16], "add"),
    OpAndTensors(add, [(1, 24, 32, 128)], [torch.bfloat16], "add"),
    OpAndTensors(add, [(1, 1, 32, 32)], [torch.bfloat16], "add"),
    OpAndTensors(add, [(32,)], [torch.int64], "add"),
    OpAndTensors(div, [(1, 24, 32, 32)], [torch.float32], "div"),
    OpAndTensors(div, [(1, 32, 1)], [torch.float32], "div"),
    OpAndTensors(div, [(1,)], [torch.float64], "div"),
    OpAndTensors(multiply, [(1, 8, 32, 128)], [torch.bfloat16], "multiply"),
    OpAndTensors(multiply, [(1, 24, 128, 32)], [torch.bfloat16], "multiply"),
    OpAndTensors(multiply, [(1, 32, 3072)], [torch.bfloat16], "multiply"),
    OpAndTensors(multiply, [(1, 24, 32, 128)], [torch.bfloat16], "multiply"),
    OpAndTensors(multiply, [(32,)], [torch.int64], "multiply"),
    OpAndTensors(multiply, [(1, 32, 8192)], [torch.bfloat16], "multiply"),
    OpAndTensors(multiply, [(32, 32)], [torch.bfloat16], "multiply"),
    OpAndTensors(multiply, [(1, 32, 128)], [torch.float32], "multiply"),
    OpAndTensors(multiply, [(1, 32, 3072)], [torch.float32], "multiply"),
    OpAndTensors(pow, [(1, 32, 3072)], [torch.float32], "pow"),
    OpAndTensors(subtract, [(32, 32)], [torch.int64], "subtract"),
    OpAndTensors(subtract, [(1, 24, 32, 32)], [torch.float32], "subtract"),
]


@pytest.mark.parametrize("case", binary_eltwise_cases, ids=str)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_binary(
    case: OpAndTensors,
    target: str,
    request,
):

    assert len(case.shapes) == len(case.dtypes) == 1
    pipeline_options = []
    compile_to_flatbuffer(
        case.test_op,
        case.shapes * 2,
        case.dtypes * 2,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        pipeline_options=pipeline_options,
    )


def ceil(in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None):
    return builder.ceil(in0, unit_attrs=unit_attrs)


def cos(in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None):
    return builder.cos(in0, unit_attrs=unit_attrs)


def exp(in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None):
    return builder.exp(in0, unit_attrs=unit_attrs)


def neg(in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None):
    return builder.neg(in0, unit_attrs=unit_attrs)


def rsqrt(in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None):
    return builder.rsqrt(in0, unit_attrs=unit_attrs)


def sigmoid(in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None):
    return builder.sigmoid(in0, unit_attrs=unit_attrs)


def sin(in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None):
    return builder.sin(in0, unit_attrs=unit_attrs)


unary_eltwise_cases = [
    OpAndTensors(ceil, [(1,)], [torch.float64], "ceil"),
    OpAndTensors(cos, [(1, 32, 128)], [torch.float32], "cos"),
    OpAndTensors(exp, [(1, 24, 32, 32)], [torch.float32], "exp"),
    OpAndTensors(neg, [(1, 8, 32, 64)], [torch.bfloat16], "neg"),
    OpAndTensors(neg, [(1, 24, 32, 64)], [torch.bfloat16], "neg"),
    OpAndTensors(rsqrt, [(1, 32, 1)], [torch.float32], "rsqrt"),
    OpAndTensors(sigmoid, [(1, 32, 8192)], [torch.bfloat16], "sigmoid"),
    OpAndTensors(sin, [(1, 32, 128)], [torch.float32], "sin"),
]


@pytest.mark.parametrize("case", unary_eltwise_cases, ids=str)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_unary(
    case: OpAndTensors,
    target: str,
    request,
):

    assert len(case.shapes) == len(case.dtypes) == 1
    pipeline_options = []
    compile_to_flatbuffer(
        case.test_op,
        case.shapes,
        case.dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        pipeline_options=pipeline_options,
    )


def matmul(
    in0: Operand, in1: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None
):
    return builder.matmul(in0, in1, unit_attrs=unit_attrs)


binary_cases = [
    OpAndTensors(
        matmul, [(32, 8192), (8192, 3072)], [torch.bfloat16, torch.bfloat16], "matmul"
    ),
    OpAndTensors(
        matmul, [(32, 3072), (3072, 3072)], [torch.bfloat16, torch.bfloat16], "matmul"
    ),
    OpAndTensors(
        matmul, [(1, 64, 1), (1, 1, 32)], [torch.float32, torch.float32], "matmul"
    ),
    OpAndTensors(
        matmul, [(32, 3072), (3072, 1024)], [torch.bfloat16, torch.bfloat16], "matmul"
    ),
    OpAndTensors(
        matmul, [(32, 3072), (3072, 8192)], [torch.bfloat16, torch.bfloat16], "matmul"
    ),
    OpAndTensors(
        matmul,
        [(24, 32, 128), (24, 128, 32)],
        [torch.bfloat16, torch.bfloat16],
        "matmul",
    ),
    OpAndTensors(
        matmul, [(32, 3072), (3072, 128256)], [torch.bfloat16, torch.bfloat16], "matmul"
    ),
    OpAndTensors(
        matmul,
        [(24, 32, 32), (24, 32, 128)],
        [torch.bfloat16, torch.bfloat16],
        "matmul",
    ),
]


@pytest.mark.parametrize("case", binary_cases, ids=str)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_binary(
    case: OpAndTensors,
    target: str,
    request,
):

    assert len(case.shapes) == len(case.dtypes) == 2
    pipeline_options = []
    compile_to_flatbuffer(
        case.test_op,
        case.shapes,
        case.dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        pipeline_options=pipeline_options,
    )


def concat(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    dim: int,
    builder: TTIRBuilder,
    unit_attrs: List[str] = None,
):
    return builder.concat([in0, in1, in2], dim=dim, unit_attrs=unit_attrs)


@pytest.mark.parametrize(
    "shapes_and_dim",
    [
        [(1, 8, 32, 64), (1, 8, 32, 64), (1, 8, 32, 128), 3],
        [(1, 32, 64), (1, 32, 64), (1, 32, 128), 2],
        [(1, 24, 32, 64), (1, 24, 32, 64), (1, 24, 32, 128), 3],
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_concat(shapes_and_dim, target: str, request):
    # Create a wrapper function that captures dim
    shapes, dim = shapes_and_dim[:3], shapes_and_dim[3]

    def concat_wrapper(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        return concat(in0, in1, in2, dim, builder, unit_attrs)

    # Set the name for better test identification
    concat_wrapper.__name__ = "concat"
    pipeline_options = []
    compile_to_flatbuffer(
        concat_wrapper,
        shapes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        pipeline_options=pipeline_options,
    )


def broadcast(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
    broadcast_dimensions: List[int] = None,
    unit_attrs: List[str] = None,
):
    return builder.broadcast(
        in0, in1, broadcast_dimensions=broadcast_dimensions, unit_attrs=unit_attrs
    )


@dataclass
class BroadcastCase:
    shapes: list[Shape]
    dims: list[int]
    dtypes: list[torch.dtype]

    def __str__(self):
        return "broadcast_" + "_".join(
            map(
                lambda x: f"{shape_str(x[0])}-{TORCH_DTYPE_IDS[x[1]]}",
                zip(self.shapes, self.dtypes),
            )
        )


broadcast_cases = [
    BroadcastCase([(32,), (32,)], [1], [torch.int64]),
    BroadcastCase([(1, 1, 3072), (1, 32, 3072)], [1, 32, 1], [torch.bfloat16]),
    BroadcastCase([(1, 24, 32, 32), (1, 24, 32, 32)], [1, 1, 1, 1], [torch.bfloat16]),
    BroadcastCase([(1, 32, 3072), (1, 32, 3072)], [1, 1, 1], [torch.float32]),
    BroadcastCase([(1, 1), (32, 32)], [32, 32], [torch.bfloat16]),
    BroadcastCase([(32, 32), (32, 32)], [1, 1], [torch.bfloat16]),
    BroadcastCase([(24, 32, 128), (24, 32, 128)], [1, 1, 1], [torch.bfloat16]),
    BroadcastCase([(1, 1, 1, 1), (1, 24, 128, 32)], [1, 24, 128, 32], [torch.bfloat16]),
    BroadcastCase(
        [(1, 8, 1, 32, 128), (1, 8, 3, 32, 128)], [1, 1, 3, 1, 1], [torch.bfloat16]
    ),
    BroadcastCase([(1, 1, 32, 32), (1, 24, 32, 32)], [1, 24, 1, 1], [torch.bfloat16]),
    BroadcastCase([(1, 1, 1, 32), (1, 1, 32, 32)], [1, 1, 32, 1], [torch.bfloat16]),
    BroadcastCase([(1, 8, 32, 128), (1, 8, 32, 128)], [1, 1, 1, 1], [torch.bfloat16]),
    BroadcastCase([(1, 24, 32, 128), (1, 24, 32, 128)], [1, 1, 1, 1], [torch.bfloat16]),
    BroadcastCase([(1, 32, 128), (1, 32, 128)], [1, 1, 1], [torch.float32]),
    BroadcastCase([(1, 32, 1), (1, 32, 1)], [1, 1, 1], [torch.float32]),
    BroadcastCase([(1, 1, 1, 1), (1, 24, 32, 32)], [1, 24, 32, 32], [torch.bfloat16]),
    BroadcastCase([(1, 24, 32, 32), (1, 24, 32, 32)], [1, 1, 1, 1], [torch.float32]),
    BroadcastCase([(1, 1, 32, 128), (1, 8, 32, 128)], [1, 8, 1, 1], [torch.bfloat16]),
    BroadcastCase([(1, 1, 32), (1, 1, 32)], [1, 1, 1], [torch.float32]),
    BroadcastCase([(32, 1), (32, 32)], [1, 32], [torch.int64]),
    BroadcastCase([(1, 32), (32, 32)], [32, 1], [torch.int64]),
    BroadcastCase([(32, 32), (32, 32)], [1, 1], [torch.int64]),
    BroadcastCase([(1, 1, 1, 1), (1, 24, 32, 128)], [1, 24, 32, 128], [torch.bfloat16]),
    BroadcastCase([(1, 1, 1), (1, 32, 1)], [1, 32, 1], [torch.float32]),
    BroadcastCase([(24, 128, 32), (24, 128, 32)], [1, 1, 1], [torch.bfloat16]),
    BroadcastCase([(1, 32, 3072), (1, 32, 3072)], [1, 1, 1], [torch.bfloat16]),
    BroadcastCase([(1, 1, 32, 32), (1, 1, 32, 32)], [1, 1, 1, 1], [torch.bfloat16]),
    BroadcastCase([(1, 1, 1), (1, 32, 128)], [1, 32, 128], [torch.float32]),
    BroadcastCase([(1, 32, 1), (1, 32, 3072)], [1, 1, 3072], [torch.float32]),
    BroadcastCase([(1, 1, 32, 128), (1, 24, 32, 128)], [1, 24, 1, 1], [torch.bfloat16]),
    BroadcastCase([(1, 1, 1), (1, 32, 3072)], [1, 32, 3072], [torch.float32]),
    BroadcastCase([(1, 24, 128, 32), (1, 24, 128, 32)], [1, 1, 1, 1], [torch.bfloat16]),
]


@pytest.mark.parametrize("case", broadcast_cases, ids=str)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_broadcast(case, target: str, request):
    # Create a wrapper function that captures broadcast_dimensions
    def broadcast_wrapper(
        in0: Operand, in1: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None
    ):
        return broadcast(in0, in1, builder, case.dims, unit_attrs)

    # Set the name for better test identification
    broadcast_wrapper.__name__ = "broadcast"
    pipeline_options = []
    compile_to_flatbuffer(
        broadcast_wrapper,
        case.shapes,
        case.dtypes * 2,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        pipeline_options=pipeline_options,
    )
