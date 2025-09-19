# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Callable, Sequence, Optional

from ttmlir.ir import *
from ttmlir.passes import ttir_to_ttmetal_backend_pipeline

from builder.base.builder import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import build_ttir_module, compile_ttir_to_flatbuffer
from test_utils import Marks, shape_str

# Generic utility to build a repeated op chain for unary or binary ops
def repeat_op_chain(
    op: Callable[..., Operand],
    inputs: Sequence[Operand],
    arity: int,
    repeat_count: Optional[int] = None,
    num_inputs: Optional[int] = None,
) -> Operand:
    """
    Apply a builder op repeatedly to form a chain.

    - arity == 1: op is unary. Apply it to the single input `repeat_count` times.
    - arity == 2: op is binary. Left-reduce it across `inputs` in order.
      If `repeat_count` is provided, it must equal len(inputs) - 1.
    """
    if arity not in (1, 2):
        raise ValueError("arity must be 1 or 2")

    if num_inputs is not None and num_inputs != len(inputs):
        raise ValueError("num_inputs must equal len(inputs)")

    if arity == 1:
        if len(inputs) != 1:
            raise ValueError("unary op requires exactly one input")
        if repeat_count is None:
            raise ValueError("repeat_count is required for unary op")
        result = inputs[0]
        for _ in range(repeat_count):
            result = op(result)
        return result

    # arity == 2
    if len(inputs) < 2:
        raise ValueError("binary op requires at least two inputs")
    if repeat_count is not None and repeat_count != len(inputs) - 1:
        raise ValueError("repeat_count must equal len(inputs) - 1 for binary op")
    result = op(inputs[0], inputs[1])
    for operand in inputs[2:]:
        result = op(result, operand)
    return result

def unary_chain(in0: Operand, builder: TTIRBuilder):
    neg_0 = builder.neg(in0)
    recip_0 = builder.reciprocal(neg_0)
    return builder.abs(recip_0)

# def adder_sawtooth(
#     in0: Operand,
#     in1: Operand,
#     in2: Operand,
#     in3: Operand,
#     builder: TTIRBuilder,
# ):
#     return repeat_op_chain(
#         op=builder.add,
#         inputs=[in0, in1, in2, in3],
#         arity=2,
#     )

# @pytest.mark.parametrize("shape", [(128, 128)])
# @pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
# @pytest.mark.parametrize("target", ["ttmetal"])
# def test_eltwise_adder_branch_33(shape: Shape, dtype: torch.dtype, target: str, request):
#     options = []
#     compile_ttir_to_flatbuffer(
#         adder_sawtooth,
#         [shape]*4,
#         [dtype]*4,
#         target=target,
#         custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
#         test_base=request.node.name,
#         module_dump=True,
#         print_ir=True,
#         output_root=request.config.getoption("--path"),
#         system_desc_path=request.config.getoption("--sys-desc"),
#     )

unary_ops = [
    # exp | Marks(pytest.mark.skip_config(["ttmetal", "p150"], reason="Issue #4078")),
    # expm1 | Marks(pytest.mark.skip_config(["ttmetal"])),
    # floor | Marks(pytest.mark.fails_golden),
    # abs,
    # neg,
#     sign | Marks(pytest.mark.skip_config(["ttmetal"])),
#     cos | Marks(pytest.mark.skip_config(["ttmetal", "p150"], reason="Issue #4083")),
#     sin | Marks(pytest.mark.skip_config(["ttmetal", "p150"], reason="Issue #4083")),
#     atan | Marks(pytest.mark.skip_config(["ttmetal"])),
#     tanh | Marks(pytest.mark.skip_config(["ttmetal"])),
#     relu | Marks(pytest.mark.skip_config(["ttmetal"])),
#     gelu | Marks(pytest.mark.skip_config(["ttmetal"])),
#     leaky_relu | Marks(pytest.mark.skip_config(["ttmetal"])),
#     cbrt | Marks(pytest.mark.skip_config(["ttmetal"])),
#     sigmoid | Marks(pytest.mark.fails_golden),
#     reciprocal,
#     is_finite | Marks(pytest.mark.skip_config(["ttmetal"])),
#     ceil | Marks(pytest.mark.skip_config(["ttmetal"])),
#     sum | Marks(pytest.mark.skip_config(["ttmetal"])),
#     mean | Marks(pytest.mark.skip_config(["ttmetal"])),
#     max | Marks(pytest.mark.fails_golden, pytest.mark.skip_config(["ttmetal"])),
#     min | Marks(pytest.mark.fails_golden, pytest.mark.skip_config(["ttmetal"])),
#     get_dimension_size
#     | Marks(
#         pytest.mark.skip_config(["ttmetal"]),
#         pytest.mark.skip_config(["ttnn-standalone"]),
#     ),
]

@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_unary_chain(
    shape: Shape, dtype: torch.dtype, target: str, request
):
    options = []
    compile_ttir_to_flatbuffer(
        unary_chain,
        [shape],
        [dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )

# def add_multiply(
#     in0: Operand,
#     in1: Operand,
#     in2: Operand,
#     in3: Operand,
#     builder: TTIRBuilder
# ):

#     add_0 = builder.add(in0, in1)
#     multiply_1 = builder.multiply(in2, add_0)
#     return builder.multiply(multiply_1, in3)

# @pytest.mark.parametrize("shape", [(128, 128)])
# @pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
# @pytest.mark.parametrize("target", ["ttmetal"])
# def test_eltwise_fusion_add_multiply(shape: Shape, dtype: torch.dtype, target: str, request):
#     options = []
#     compile_ttir_to_flatbuffer(
#         add_multiply,
#         [shape]*4,
#         [dtype]*4,
#         target=target,
#         custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
#         test_base=request.node.name,
#         module_dump=True,
#         print_ir=True,
#         output_root=request.config.getoption("--path"),
#         system_desc_path=request.config.getoption("--sys-desc"),
#     )

# ##--##-------------------------------------------------------------------##--##

def multiuse_diamond_broadcast_reduce_no_fuse(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    in3: Operand,
    builder: TTIRBuilder
):

    shared_intermediate = builder.add(in0, in1)
    user_0 = builder.multiply(shared_intermediate, in2)
    user_1 = builder.multiply(shared_intermediate, in3)
    return builder.multiply(user_0, user_1)


@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_fusion_multiuse_operand_no_fuse(
    shape: Shape, 
    dtype: torch.dtype, 
    target: str, 
    request
):
    options = []
    compile_ttir_to_flatbuffer(
        multiuse_diamond_broadcast_reduce_no_fuse,
        [shape]*4,
        [dtype]*4,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )

# ##--##-------------------------------------------------------------------##--##

# def multiuse_diamond_with_tail(
#     in0: Operand,
#     in1: Operand,
#     in2: Operand,
#     in3: Operand,
#     builder: TTIRBuilder
# ):

#     shared_intermediate = builder.add(in0, in1)
#     user_0 = builder.multiply(shared_intermediate, in2)
#     user_1 = builder.multiply(shared_intermediate, in3)
#     merged = builder.multiply(user_0, user_1)
#     temp_0 = builder.abs(merged)
#     temp_1 = builder.ceil(temp_0)
#     return builder.neg(temp_1)

# @pytest.mark.parametrize("shape", [(128, 128)])
# @pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
# @pytest.mark.parametrize("target", ["ttmetal"])
# def test_eltwise_fusion_multiuse_diamond_with_tail(
#     shape: Shape, 
#     dtype: torch.dtype, 
#     target: str, 
#     request
# ):
#     options = []
#     compile_ttir_to_flatbuffer(
#         multiuse_diamond_with_tail,
#         [shape]*4,
#         [dtype]*4,
#         target=target,
#         custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
#         test_base=request.node.name,
#         module_dump=True,
#         print_ir=True,
#         output_root=request.config.getoption("--path"),
#         system_desc_path=request.config.getoption("--sys-desc"),
#     )

# ##--##-------------------------------------------------------------------##--##

# # def multiuse_diamond_broadcast_with_tail(
# #     in0: Operand,
# #     in1: Operand,
# #     builder: TTIRBuilder
# # ):

# #     chain_0_0 = builder.exp(in0)
# #     chain_0_1 = builder.log(chain_0_0)

# #     chain_1_0 = builder.exp(in1)

# #     shared_intermediate = builder.add(in0, in1)
# #     user_0 = builder.multiply(shared_intermediate, in2)
# #     user_1 = builder.multiply(shared_intermediate, in3)
# #     merged = builder.multiply(user_0, user_1)
# #     temp_0 = builder.abs(merged)
# #     temp_1 = builder.ceil(temp_0)
# #     return builder.neg(temp_1)



# rename to giraffe_adder()
def add(
    in0: Operand,
    in1: Operand,
    in2: Operand,
    in3: Operand,
    builder: TTIRBuilder,
):
    add_0 = builder.add(in0, in1)
    add_1 = builder.add(add_0, in2)
    add_2 = builder.add(add_1, in3)
    return add_2

@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_add(shape: Shape, dtype: torch.dtype, target: str, request):
    options = []
    compile_ttir_to_flatbuffer(
        add,
        [shape]*4,
        [dtype]*4,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        # print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )