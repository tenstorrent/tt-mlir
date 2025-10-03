# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Callable, Sequence, Optional

from ttmlir.ir import *
from ttmlir.passes import ttir_to_ttmetal_backend_pipeline
from ttmlir.dialects import ttir

from builder.base.builder import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import build_ttir_module, compile_ttir_to_flatbuffer
from test_utils import (
    Marks,
    shape_str,
    make_shard_shape,
    shard_wrap_factory,
)


pytestmark = pytest.mark.frontend("ttir")

### ----------------------------------------------------------------------- ###
# Test Manifest
### ----------------------------------------------------------------------- ###


### ----------------------------------------------------------------------- ###
# Main Parameter Sets to Reduce Verbiage
### ----------------------------------------------------------------------- ###

enablePrintIR = True

gridParams = [
    "override-device-shape=1,1",
    "override-device-shape=2,2",
    "override-device-shape=4,4",
    "override-device-shape=8,8",
]

### ----------------------------------------------------------------------- ###
# Utilities
### ----------------------------------------------------------------------- ###

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


##--##-------------------------------------------------------------------##--##

# Generic utility to build a binary reduction tree for binary ops
def binary_reduction_tree(
    op: Callable[[Operand, Operand], Operand],
    inputs: Sequence[Operand],
    num_inputs: Optional[int] = None,
) -> Operand:
    """
    Reduce `inputs` using a binary tree pattern.

    Pairs adjacent operands and applies `op` to each pair, carrying forward the
    last unpaired operand when the count is odd, and repeats until a single
    result remains.
    """
    if len(inputs) < 2:
        raise ValueError("binary op requires at least two inputs")
    if num_inputs is not None and num_inputs != len(inputs):
        raise ValueError("num_inputs must equal len(inputs)")

    level: List[Operand] = list(inputs)
    while len(level) > 1:
        next_level: List[Operand] = []
        i = 0
        while i + 1 < len(level):
            next_level.append(op(level[i], level[i + 1]))
            i += 2
        if i < len(level):
            next_level.append(level[i])
        level = next_level

    return level[0]


##--##-------------------------------------------------------------------##--##


def unary_op_builder(op_name: str, builder: TTIRBuilder):
    if op_name == "abs":
        return builder.abs
    if op_name == "ceil":
        return builder.ceil
    if op_name == "cos":
        return builder.cos
    if op_name == "exp":
        return builder.exp
    if op_name == "floor":
        return builder.floor
    if op_name == "gelu":
        return builder.gelu
    if op_name == "log":
        return builder.log
    if op_name == "logical_not":
        return builder.logical_not
    if op_name == "negative":
        return builder.neg
    if op_name == "recip":
        return builder.recip
    if op_name == "rsqrt":
        return builder.rsqrt
    if op_name == "sqrt":
        return builder.sqrt
    if op_name == "sigmoid":
        return builder.sigmoid
    if op_name == "sin":
        return builder.sin
    if op_name == "tan":
        return builder.tan
    if op_name == "eqz":
        return builder.eqz
    if op_name == "nez":
        return builder.nez
    if op_name == "gtz":
        return builder.gtz
    if op_name == "gez":
        return builder.gez
    if op_name == "ltz":
        return builder.ltz
    if op_name == "lez":
        return builder.lez


##--##-------------------------------------------------------------------##--##


def binary_op_builder(op_name: str, builder: TTIRBuilder):
    if op_name == "add":
        return builder.add
    # if op_name == "div":
    #     return builder.div
    # if op_name == "maximum":
    #     return builder.max
    # if op_name == "pow":
    #     return builder.pow
    # if op_name == "sub":
    #     return builder.sub


### ----------------------------------------------------------------------- ###
# # Test: Key Composite Ops
### ----------------------------------------------------------------------- ###


def cosh(in0: Operand, in1: Operand, builder: TTIRBuilder):
    neg_x = builder.neg(in0)

    e_neg_x = builder.exp(neg_x)
    e_pos_x = builder.exp(in0)

    nr_term = builder.add(e_pos_x, e_neg_x)
    ret_val = builder.multiply(nr_term, in1)

    return ret_val


# Everything should pass
@pytest.mark.parametrize("grid", gridParams)
@pytest.mark.parametrize(
    "shape", [(128, 128)]
)  # , (32, 64), (64, 64), (64, 128), (128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_fuse_cosh(
    grid: str, shape: Shape, dtype: torch.dtype, target: str, request
):
    options = [grid]

    compile_ttir_to_flatbuffer(
        cosh,
        [shape] * 2,
        [dtype] * 2,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        print_ir=enablePrintIR,
    )


### ----------------------------------------------------------------------- ###
# # Test: Unary Op Sanity Checks
### ----------------------------------------------------------------------- ###


@pytest.mark.parametrize(
    "op_name",
    [
        "abs",  # passes 128x128
        "ceil",  # passes 128x128
        "cos",  # passes 128x128
        "exp",  # passes 128x128
        "floor",  # passes 128x128
        # "gelu", # fails 128x128
        # "log", # fails 128x128
        "logical_not",  # passes 128x128
        "negative",  # passes 128x128
        # "recip",
        # "rsqrt",  # fails 128x128
        # "sqrt",  # fails 128x128
        # "sigmoid", # fails 128x128
        "sin",  # passes 128x128
        # "tan", # fails 128x128
        # "eqz", # fails
        # "nez", # fails
        # "gtz", # fails
        # "gez", # fails
        # "ltz", # fails
        # "lez", # fails
    ],
)
@pytest.mark.parametrize("grid", gridParams)
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_sanity_check_unary_op(
    op_name: str,
    grid: str,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def unary_op_wrapper(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return repeat_op_chain(
            op=unary_op_builder(op_name, builder),
            inputs=[in0],
            arity=1,
            repeat_count=1,
            num_inputs=1,
        )

    options = [grid]

    compile_ttir_to_flatbuffer(
        unary_op_wrapper,
        [shape],
        [dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        print_ir=enablePrintIR,
    )


### ----------------------------------------------------------------------- ###
# Test: Long Unary Chain
### ----------------------------------------------------------------------- ###

# unary ops are done in place, should be able to fuse
# indefinitely
def unary_chain(in0: Operand, builder: TTIRBuilder):
    res_0 = builder.abs(in0)
    res_1 = builder.sin(res_0)
    res_2 = builder.neg(res_1)
    res_3 = builder.exp(res_2)

    res_4 = builder.abs(res_3)
    res_5 = builder.cos(res_4)
    res_6 = builder.neg(res_5)
    res_7 = builder.exp(res_6)

    res_8 = builder.neg(res_7)
    res_9 = builder.sin(res_8)
    res_10 = builder.neg(res_9)
    res_11 = builder.exp(res_10)

    res_12 = builder.abs(res_11)
    res_13 = builder.cos(res_12)
    res_14 = builder.neg(res_13)
    res_15 = builder.exp(res_14)

    res_16 = builder.neg(res_15)
    res_17 = builder.sin(res_16)
    res_18 = builder.neg(res_17)
    res_19 = builder.exp(res_18)

    return res_19


@pytest.mark.parametrize("grid", gridParams)
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_fuse_unary_chain(
    grid: str, shape: Shape, dtype: torch.dtype, target: str, request
):

    options = [grid]

    compile_ttir_to_flatbuffer(
        unary_chain,
        [shape],
        [dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        print_ir=enablePrintIR,
    )


### ----------------------------------------------------------------------- ###
# Test: Joins and Binary Reduction Trees
### ----------------------------------------------------------------------- ###


def converging_unary_branches(
    in0: Operand,
    in1: Operand,
    builder: TTIRBuilder,
):
    branch_0_0 = builder.abs(in0)
    branch_0_1 = builder.exp(branch_0_0)
    branch_0_2 = builder.neg(branch_0_1)

    branch_1_0 = builder.neg(in1)
    branch_1_1 = builder.exp(branch_1_0)
    branch_1_2 = builder.abs(branch_1_1)

    return builder.div(branch_0_2, branch_1_2)


@pytest.mark.parametrize("grid", gridParams)
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_fuse_converging_unary_branches(
    grid: str, shape: Shape, dtype: torch.dtype, target: str, request
):

    options = [grid]

    compile_ttir_to_flatbuffer(
        converging_unary_branches,
        [shape] * 2,
        [dtype] * 2,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        print_ir=enablePrintIR,
    )


##--##-------------------------------------------------------------------##--##

# TODO(mbagherbeikTT): figure out why a 4 input add fails without setting goldens
# so we can use the helper functions and not have to manually copy paste
# the same code for different tests. Or any way of setting goldens easily in
# the tree and ladder buidlers
@pytest.mark.parametrize("grid", gridParams)
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_fuse_binary_reduction_tree(
    grid: str, shape: Shape, dtype: torch.dtype, target: str, request
):
    options = [grid]

    def add_tree_8_to_1(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        in3: Operand,
        in4: Operand,
        in5: Operand,
        in6: Operand,
        in7: Operand,
        builder: TTIRBuilder,
    ):
        input_0 = torch.full(shape, 1).to(dtype)
        input_1 = torch.full(shape, 2).to(dtype)
        input_2 = torch.full(shape, 3).to(dtype)
        input_3 = torch.full(shape, 4).to(dtype)
        input_4 = torch.full(shape, 5).to(dtype)
        input_5 = torch.full(shape, 6).to(dtype)
        input_6 = torch.full(shape, 7).to(dtype)
        input_7 = torch.full(shape, 8).to(dtype)

        add_0_0 = builder.add(in0, in1)
        add_0_1 = builder.add(in2, in3)
        add_1_0 = builder.add(add_0_0, add_0_1)

        add_0_2 = builder.add(in4, in5)
        add_0_3 = builder.add(in6, in7)
        add_1_1 = builder.add(add_0_2, add_0_3)

        add_2_0 = builder.add(add_1_0, add_1_1)

        output_0 = torch.full(shape, 36).to(dtype)

        builder.set_goldens(
            {
                in0: input_0,
                in1: input_1,
                in2: input_2,
                in3: input_3,
                in4: input_4,
                in5: input_5,
                in6: input_6,
                in7: input_7,
            },
            {add_2_0: output_0},
        )

        return add_2_0

    compile_ttir_to_flatbuffer(
        add_tree_8_to_1,
        [shape] * 8,
        [dtype] * 8,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        print_ir=enablePrintIR,
    )


### ----------------------------------------------------------------------- ###
# # Diamond Fork and Join Patterns
### ----------------------------------------------------------------------- ###


def diamond_unary_op_fanout(
    in0: Operand,
    builder: TTIRBuilder,
):
    abs_0 = builder.abs(in0)

    ceil_0 = builder.ceil(abs_0)
    neg_0 = builder.neg(ceil_0)

    floor_0 = builder.floor(abs_0)
    neg_1 = builder.neg(floor_0)

    return builder.add(neg_0, neg_1)


@pytest.mark.parametrize("grid", gridParams)
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_diamond_unary_op_fanout(
    grid: str, shape: Shape, dtype: torch.dtype, target: str, request
):

    options = [grid]

    compile_ttir_to_flatbuffer(
        diamond_unary_op_fanout,
        [shape],
        [dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        print_ir=enablePrintIR,
    )


### ----------------------------------------------------------------------- ###
# # Elementwise SFPU Binary - Ladder Fusion Methods : max inputs
### ----------------------------------------------------------------------- ###

# def eltwise_fuse_div_ladder_max_inputs(
#     in0: Operand, in1: Operand, in2: Operand, in3: Operand,
#     in4: Operand,
#     builder: TTIRBuilder,
# ):
#     return repeat_op_chain(
#         op=builder.div,
#         inputs=[
#             in0, in1, in2, in3,
#             in4,
#         ],
#         arity=2,
#     )

# @pytest.mark.parametrize("shape", [(128, 128)])
# @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
# @pytest.mark.parametrize("target", ["ttmetal"])
# @pytest.mark.parametrize(
#     "test_fn",
#     [
#         # eltwise_fuse_sub_binary_ladder_max_inputs,
#         eltwise_fuse_div_ladder_max_inputs,
#         # eltwise_fuse_maximum_ladder_max_inputs,
#         # eltwise_fuse_pow_ladder_max_inputs,
#     ]
# )
# def test_eltwise_binary_op_ladder_max_inputs(
#     test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request
# ):
#     options = []
#     compile_ttir_to_flatbuffer(
#         test_fn,
#         [shape]*5,
#         [dtype]*5,
#         target=target,
#         custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
#         test_base=request.node.name,
#         module_dump=True,
#         output_root=request.config.getoption("--path"),
#         system_desc_path=request.config.getoption("--sys-desc"),
#         print_ir=True,
#     )

### ----------------------------------------------------------------------- ###
# # Elementwise SFPU Binary - Ladder Fusion Methods : max inputs PLUS ONE MORE
### ----------------------------------------------------------------------- ###

# def eltwise_fuse_div_ladder_max_inputs_plus_1(
#     in0: Operand, in1: Operand, in2: Operand, in3: Operand,
#     in4: Operand, in5: Operand,
#     builder: TTIRBuilder,
# ):
#     return repeat_op_chain(
#         op=builder.div,
#         inputs=[
#             in0, in1, in2, in3,
#             in4, in5,
#         ],
#         arity=2,
#     )

# @pytest.mark.parametrize("shape", [(128, 128)])
# @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
# @pytest.mark.parametrize("target", ["ttmetal"])
# @pytest.mark.parametrize(
#     "test_fn",
#     [
#         # eltwise_fuse_sub_binary_ladder_max_inputs_plus_1,
#         eltwise_fuse_div_ladder_max_inputs_plus_1,
#         # eltwise_fuse_maximum_ladder_max_inputs_plus_1,
#         # eltwise_fuse_pow_ladder_max_inputs_plus_1,
#     ]
# )
# def test_eltwise_binary_op_ladder_max_inputs_plus_1(
#     test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request
# ):
#     options = []
#     compile_ttir_to_flatbuffer(
#         test_fn,
#         [shape]*6,
#         [dtype]*6,
#         target=target,
#         custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
#         test_base=request.node.name,
#         module_dump=True,
#         output_root=request.config.getoption("--path"),
#         system_desc_path=request.config.getoption("--sys-desc"),
#         print_ir=True,
#     )

### ----------------------------------------------------------------------- ###
# # Elementwise SFPU Binary - Tree Fusion Methods - max inputs
### ----------------------------------------------------------------------- ###

# def eltwise_fuse_div_tree_max_inputs(
#     in0: Operand, in1: Operand, in2: Operand, in3: Operand,
#     in4: Operand, in5: Operand,
#     builder: TTIRBuilder,
# ):
#     return reduction_op_tree(
#         op=builder.div,
#         inputs=[
#             in0, in1, in2, in3,
#             in4, in5,
#         ]
#     )

# @pytest.mark.parametrize("shape", [(128, 128)])
# @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
# @pytest.mark.parametrize("target", ["ttmetal"])
# @pytest.mark.parametrize(
#     "test_fn",
#     [
#         # eltwise_fuse_sub_binary_tree_max_inputs,
#         eltwise_fuse_div_tree_max_inputs,
#         # eltwise_fuse_maximum_tree_max_inputs,
#         # eltwise_fuse_pow_tree_max_inputs,
#     ]
# )
# def test_eltwise_binary_op_tree_max_inputs(
#     test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request
# ):
#     options = []
#     compile_ttir_to_flatbuffer(
#         test_fn,
#         [shape]*6,
#         [dtype]*6,
#         target=target,
#         custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
#         test_base=request.node.name,
#         module_dump=True,
#         output_root=request.config.getoption("--path"),
#         system_desc_path=request.config.getoption("--sys-desc"),
#         print_ir=True,
#     )

### ----------------------------------------------------------------------- ###
# # Elementwise SFPU Binary - Tree Fusion Methods - max inputs
### ----------------------------------------------------------------------- ###

# def eltwise_fuse_div_tree_max_inputs_plus_1(
#     in0: Operand, in1: Operand, in2: Operand, in3: Operand,
#     in4: Operand, in5: Operand, in6: Operand, in7: Operand,
#     # in8: Operand, in9: Operand, in10: Operand, in11: Operand,
#     # in12: Operand, in13: Operand, in14: Operand, in15: Operand,
#     builder: TTIRBuilder,
#     shape: Shape,
#     dtype: torch.dtype,
#     unit_attrs: Optional[List[str]] = None,
# ):
#     input_0 = torch.full(shape, 16).to(dtype)
#     input_1 = torch.full(shape, 2).to(dtype)

#     input_2 = torch.full(shape, 4).to(dtype)
#     input_3 = torch.full(shape, 2).to(dtype)

#     input_4 = torch.full(shape, 2).to(dtype)
#     input_5 = torch.full(shape, 1).to(dtype)

#     input_6 = torch.full(shape, 16).to(dtype)
#     input_7 = torch.full(shape, 16).to(dtype)

#     # input_8 = torch.full(shape, 16).to(dtype)
#     # input_9 = torch.full(shape, 2).to(dtype)

#     # input_10 = torch.full(shape, 1).to(dtype)
#     # input_11 = torch.full(shape, 1).to(dtype)

#     # input_12 = torch.full(shape, 1).to(dtype)
#     # input_13 = torch.full(shape, 1).to(dtype)

#     # input_14 = torch.full(shape, 1).to(dtype)
#     # input_15 = torch.full(shape, 1).to(dtype)


#     output_0 = torch.full(shape, 2).to(dtype)

#     # l_0_0 = builder.div(in0, in1)
#     # l_0_1 = builder.div(in2, in3)
#     # l_0_2 = builder.div(in4, in5)
#     # l_0_3 = builder.div(in6, in7)
#     # # l_0_4 = builder.div(in8, in9)
#     # # l_0_5 = builder.div(in10, in11)
#     # # l_0_6 = builder.div(in12, in13)
#     # # l_0_7 = builder.div(in14, in15)

#     # l_1_0 = builder.div(l_0_0, l_0_1)
#     # l_1_1 = builder.div(l_0_2, l_0_3)
#     # # l_1_2 = builder.div(l_0_4, l_0_5)
#     # # l_1_3 = builder.div(l_0_6, l_0_7)

#     # l_2_0 = builder.div(l_1_0, l_1_1)
#     # l_2_1 = builder.div(l_1_2, l_1_3)

#     # tree_out = builder.div(l_2_0, l_2_1)

#     tree_out = reduction_op_tree(
#         op=builder.div,
#         inputs=[
#             in0, in1, in2, in3,
#             in4, in5, in6, in7,
#             # in8, in9, in10, in11,
#             # in12, in13, in14, in15
#         ]
#     )

#     builder.set_goldens(
#         {in0: input_0, in1: input_1, in2: input_2, in3: input_3,
#         in4: input_4, in5: input_5, in6: input_6, in7: input_7,
#         # in8: input_8, in9: input_9, in10: input_10, in11: input_11,
#         # in12: input_12, in13: input_13, in14: input_14, in15: input_15
#         }, {tree_out: output_0})

#     return tree_out


# @pytest.mark.parametrize("shape", [(128, 128)])
# @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
# @pytest.mark.parametrize("target", ["ttmetal"])
# @pytest.mark.parametrize(
#     "test_fn",
#     [
#         # eltwise_fuse_sub_binary_tree_max_inputs_plus_1,
#         eltwise_fuse_div_tree_max_inputs_plus_1,
#         # eltwise_fuse_maximum_tree_max_inputs_plus_1,
#         # eltwise_fuse_pow_tree_max_inputs_plus_1,
#     ]
# )
# def test_eltwise_binary_op_tree_max_inputs_plus_1(
#     test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request
# ):
#     def eltwise_fuse_binary_tree_op_tree_max_inputs_plus_1_wrapper(
#         in0: Operand, in1: Operand, in2: Operand, in3: Operand,
#         in4: Operand, in5: Operand, in6: Operand, in7: Operand,
#         # in8: Operand, in9: Operand, in10: Operand, in11: Operand,
#         # in12: Operand, in13: Operand, in14: Operand, in15: Operand,
#         builder: TTIRBuilder,
#         unit_attrs: Optional[List[str]] = None,
#     ):
#         return test_fn(
#             in0, in1, in2, in3,
#             in4, in5, in6, in7,
#             # in8, in9, in10, in11,
#             # in12, in13, in14, in15,
#             builder, shape, dtype, unit_attrs)

#     options = []
#     compile_ttir_to_flatbuffer(
#         eltwise_fuse_binary_tree_op_tree_max_inputs_plus_1_wrapper,
#         [shape]*8,
#         [dtype]*8,
#         target=target,
#         custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
#         test_base=request.node.name,
#         module_dump=True,
#         output_root=request.config.getoption("--path"),
#         system_desc_path=request.config.getoption("--sys-desc"),
#         print_ir=True,
#     )


### ----------------------------------------------------------------------- ###
# #
### ----------------------------------------------------------------------- ###

# def diamond_binary_op_fanout(
#     in0: Operand,
#     in1: Operand,
#     in2: Operand,
#     in3: Operand,
#     builder: TTIRBuilder,
# ):
#     div_fanout = builder.div(in0, in1)

#     div_left = builder.div(div_0, in2)
#     div_right = builder.div(div_0, in3)

#     abs_left = builder.abs(div_left)
#     abs_right = builder.abs(div_right)

#     exp_left = builder.exp(abs_left)
#     exp_right = builder.exp(abs.right)

#     div_fanin = builder.div(exp_left, exp_right)

#     return div_fanin

# @pytest.mark.parametrize("shape", [(128, 128)])
# @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
# @pytest.mark.parametrize("target", ["ttmetal"])
# def test_diamond_unary_fanout(shape: Shape, dtype: torch.dtype, target: str, request):
#     options = []
#     compile_ttir_to_flatbuffer(
#         diamond_binary_op_fanout,
#         [shape]*4,
#         [dtype]*4,
#         target=target,
#         custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
#         test_base=request.node.name,
#         module_dump=True,
#         output_root=request.config.getoption("--path"),
#         system_desc_path=request.config.getoption("--sys-desc"),
#         print_ir=True,
#     )

### ----------------------------------------------------------------------- ###
# #
### ----------------------------------------------------------------------- ###


### ----------------------------------------------------------------------- ###
# #
### ----------------------------------------------------------------------- ###

# def big_one(
#     in0: Operand, in1: Operand, in2: Operand, in3: Operand,
#     in4: Operand, in5: Operand, in6: Operand,
#     builder: TTIRBuilder,
# ):
#     branch_0_0 = builder.abs(in0)
#     branch_0_1 = builder.exp(branch_0_0)
#     branch_0_2 = builder.neg(branch_0_1)

#     branch_1_0 = builder.neg(in1)
#     branch_1_1 = builder.exp(branch_1_0)
#     branch_1_2 = builder.abs(branch_1_1)

#     div_0f1 = builder.div(branch_0_2, branch_1_2)

#     branch_2_0 = builder.abs(in2)
#     branch_2_1 = builder.exp(branch_2_0)
#     branch_2_2 = builder.neg(branch_2_1)

#     branch_3_0 = builder.neg(in3)
#     branch_3_1 = builder.exp(branch_3_0)
#     branch_3_2 = builder.abs(branch_3_1)

#     div_2f3 = builder.div(branch_2_2, branch_3_2)

#     div_fuse_all = builder.div(div_0f1, div_2f3)

#     abs_0 = builder.abs(div_fuse_all)

#     div_left = builder.div(abs_0, in4)
#     div_right = builder.div(abs_0, in5)

#     div_final = builder.div(div_left, div_right)

#     return builder.div(div_final, in6)


# @pytest.mark.parametrize("shape", [(128, 128)])
# @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
# @pytest.mark.parametrize("target", ["ttmetal"])
# def test_big_one(shape: Shape, dtype: torch.dtype, target: str, request):
#     options = []
#     compile_ttir_to_flatbuffer(
#         big_one,
#         [shape]*7,
#         [dtype]*7,
#         target=target,
#         custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
#         test_base=request.node.name,
#         module_dump=True,
#         output_root=request.config.getoption("--path"),
#         system_desc_path=request.config.getoption("--sys-desc"),
#         print_ir=True,
#     )

# # ttkernel compute config attribute theres a flag for fp32 dest --> globally off?


# def eltwise_unary_chain_multi_tile(
#     in0: Operand,
#     builder: TTIRBuilder
# ):
#     res_0 = builder.abs(in0)
#     res_1 = builder.sin(res_0)
#     res_2 = builder.neg(res_1)
#     res_3 = builder.exp(res_2)

#     return res_3

# @pytest.mark.parametrize("grid",
#     [
#         "override-device-shape=1,1",
#         # "override-device-shape=2,2",
#         # "override-device-shape=4,4",
#     ]
# )
# @pytest.mark.parametrize("shape", [(128, 128)])
# @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
# @pytest.mark.parametrize("target", ["ttmetal"])
# def test_eltwise_unary_chain_multi_tile(grid: str, shape: Shape, dtype: torch.dtype, target: str, request):
#     options = [grid]

#     compile_ttir_to_flatbuffer(
#         eltwise_unary_chain_multi_tile,
#         [shape],
#         [dtype],
#         target=target,
#         custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
#         test_base=request.node.name,
#         print_ir=True,
#         module_dump=True,
#         output_root=request.config.getoption("--path"),
#         system_desc_path=request.config.getoption("--sys-desc"),
#     )
