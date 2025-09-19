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
from test_utils import (
    Marks,
    shape_str,
    make_shard_shape,
    shard_wrap_factory,
)

MAX_INPUT_OPERANDS = 15

pytestmark = pytest.mark.frontend("ttir")

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

# Generic utility to build a binary reduction tree for binary ops
def op_tree_reduce(
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

# ##--##-------------------------------------------------------------------##--##


### ------------------------------------------------------------------------ ###
# Elementwise SFPU Binary - Ladder Fusion Methods : max inputs
### ------------------------------------------------------------------------ ###

# def eltwise_fuse_sub_binary_ladder_max_inputs(
#     in0: Operand, in1: Operand, in2: Operand, in3: Operand,
#     in4: Operand, in5: Operand, in6: Operand, in7: Operand,
#     in8: Operand, in9: Operand, in10: Operand, in11: Operand, 
#     in12: Operand, in13: Operand, in14: Operand,
#     builder: TTIRBuilder,
# ):
#     return repeat_op_chain(
#         op=builder.sub_binary,
#         inputs=[
#             in0, in1, in2, in3, 
#             in4, in5, in6, in7, 
#             in8, in9, in10, in11, 
#             in12, in13, in14,
#         ],
#         arity=2,
#     )

def eltwise_fuse_div_ladder_max_inputs(
    in0: Operand, in1: Operand, in2: Operand, in3: Operand,
    in4: Operand, in5: Operand, in6: Operand, in7: Operand,
    in8: Operand, in9: Operand, in10: Operand, in11: Operand, 
    in12: Operand, in13: Operand, in14: Operand,
    builder: TTIRBuilder,
):
    return repeat_op_chain(
        op=builder.div,
        inputs=[
            in0, in1, in2, in3, 
            in4, in5, in6, in7, 
            in8, in9, in10, in11, 
            in12, in13, in14
        ],
        arity=2,
    )

def eltwise_fuse_maximum_ladder_max_inputs(
    in0: Operand, in1: Operand, in2: Operand, in3: Operand,
    in4: Operand, in5: Operand, in6: Operand, in7: Operand,
    in8: Operand, in9: Operand, in10: Operand, in11: Operand, 
    in12: Operand, in13: Operand, in14: Operand,
    builder: TTIRBuilder,
):
    return repeat_op_chain(
        op=builder.maximum,
        inputs=[
            in0, in1, in2, in3, 
            in4, in5, in6, in7, 
            in8, in9, in10, in11, 
            in12, in13, in14
        ],
        arity=2,
    )

def eltwise_fuse_pow_ladder_max_inputs(
    in0: Operand, in1: Operand, in2: Operand, in3: Operand,
    in4: Operand, in5: Operand, in6: Operand, in7: Operand,
    in8: Operand, in9: Operand, in10: Operand, in11: Operand, 
    in12: Operand, in13: Operand, in14: Operand,
    builder: TTIRBuilder,
):
    return repeat_op_chain(
        op=builder.pow,
        inputs=[
            in0, in1, in2, in3, 
            in4, in5, in6, in7, 
            in8, in9, in10, in11, 
            in12, in13, in14
        ],
        arity=2,
    )

@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "test_fn", 
    [
        # eltwise_fuse_sub_binary_ladder_max_inputs,
        eltwise_fuse_div_ladder_max_inputs,
        eltwise_fuse_maximum_ladder_max_inputs,
        eltwise_fuse_pow_ladder_max_inputs,
    ]
)
def test_eltwise_binary_op_ladder_max_inputs(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request
):
    options = []
    compile_ttir_to_flatbuffer(
        test_fn,
        [shape]*MAX_INPUT_OPERANDS,
        [dtype]*MAX_INPUT_OPERANDS,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        # print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )

### ------------------------------------------------------------------------ ###
# Elementwise SFPU Binary - Ladder Fusion Methods : max inputs PLUS ONE MORE
### ------------------------------------------------------------------------ ###

# def eltwise_fuse_sub_binary_ladder_max_inputs_plus_1(
#     in0: Operand, in1: Operand, in2: Operand, in3: Operand,
#     in4: Operand, in5: Operand, in6: Operand, in7: Operand,
#     in8: Operand, in9: Operand, in10: Operand, in11: Operand, 
#     in12: Operand, in13: Operand, in14: Operand, in15: Operand,
#     builder: TTIRBuilder,
# ):
#     return repeat_op_chain(
#         op=builder.sub_binary,
#         inputs=[
#             in0, in1, in2, in3, 
#             in4, in5, in6, in7, 
#             in8, in9, in10, in11, 
#             in12, in13, in14, in15
#         ],
#         arity=2,
#     )

def eltwise_fuse_div_ladder_max_inputs_plus_1(
    in0: Operand, in1: Operand, in2: Operand, in3: Operand,
    in4: Operand, in5: Operand, in6: Operand, in7: Operand,
    in8: Operand, in9: Operand, in10: Operand, in11: Operand, 
    in12: Operand, in13: Operand, in14: Operand, in15: Operand,
    builder: TTIRBuilder,
):
    return repeat_op_chain(
        op=builder.div,
        inputs=[
            in0, in1, in2, in3, 
            in4, in5, in6, in7, 
            in8, in9, in10, in11, 
            in12, in13, in14, in15
        ],
        arity=2,
    )

def eltwise_fuse_maximum_ladder_max_inputs_plus_1(
    in0: Operand, in1: Operand, in2: Operand, in3: Operand,
    in4: Operand, in5: Operand, in6: Operand, in7: Operand,
    in8: Operand, in9: Operand, in10: Operand, in11: Operand, 
    in12: Operand, in13: Operand, in14: Operand, in15: Operand,
    builder: TTIRBuilder,
):
    return repeat_op_chain(
        op=builder.maximum,
        inputs=[
            in0, in1, in2, in3, 
            in4, in5, in6, in7, 
            in8, in9, in10, in11, 
            in12, in13, in14, in15
        ],
        arity=2,
    )

def eltwise_fuse_pow_ladder_max_inputs_plus_1(
    in0: Operand, in1: Operand, in2: Operand, in3: Operand,
    in4: Operand, in5: Operand, in6: Operand, in7: Operand,
    in8: Operand, in9: Operand, in10: Operand, in11: Operand, 
    in12: Operand, in13: Operand, in14: Operand, in15: Operand,
    builder: TTIRBuilder,
):
    return repeat_op_chain(
        op=builder.pow,
        inputs=[
            in0, in1, in2, in3, 
            in4, in5, in6, in7, 
            in8, in9, in10, in11, 
            in12, in13, in14, in15
        ],
        arity=2,
    )

@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "test_fn", 
    [
        # eltwise_fuse_sub_binary_ladder_max_inputs_plus_1,
        eltwise_fuse_div_ladder_max_inputs_plus_1,
        eltwise_fuse_maximum_ladder_max_inputs_plus_1,
        eltwise_fuse_pow_ladder_max_inputs_plus_1,
    ]
)
def test_eltwise_binary_op_ladder_max_inputs_plus_1(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request
):
    options = []
    compile_ttir_to_flatbuffer(
        test_fn,
        [shape]*(MAX_INPUT_OPERANDS+1),
        [dtype]*(MAX_INPUT_OPERANDS+1),
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )

### ------------------------------------------------------------------------ ###
# Elementwise SFPU Binary - Tree Fusion Methods - max pow2 inputs fusable
### ------------------------------------------------------------------------ ###

# Generic utility to build a binary reduction tree for binary ops
def reduction_op_tree(
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

### ------------------------------------------------------------------------ ###
# Elementwise SFPU Binary - Tree Fusion Methods - max inputs
### ------------------------------------------------------------------------ ###

# def eltwise_fuse_sub_binary_tree_max_inputs(
#     in0: Operand, in1: Operand, in2: Operand, in3: Operand,
#     in4: Operand, in5: Operand, in6: Operand, in7: Operand,
#     in8: Operand, in9: Operand, in10: Operand, in11: Operand, 
#     in12: Operand, in13: Operand, in14: Operand,
#     builder: TTIRBuilder,
# ):
#     return reduction_op_tree(
#         op=builder.sub_binary,
#         inputs=[
#             in0, in1, in2, in3, 
#             in4, in5, in6, in7, 
#             in8, in9, in10, in11, 
#             in12, in13, in14,
#         ]
#     )

def eltwise_fuse_div_tree_max_inputs(
    in0: Operand, in1: Operand, in2: Operand, in3: Operand,
    in4: Operand, in5: Operand, in6: Operand, in7: Operand,
    in8: Operand, in9: Operand, in10: Operand, in11: Operand, 
    in12: Operand, in13: Operand, in14: Operand,
    builder: TTIRBuilder,
):
    return reduction_op_tree(
        op=builder.div,
        inputs=[
            in0, in1, in2, in3, 
            in4, in5, in6, in7, 
            in8, in9, in10, in11, 
            in12, in13, in14,
        ]
    )

def eltwise_fuse_maximum_tree_max_inputs(
    in0: Operand, in1: Operand, in2: Operand, in3: Operand,
    in4: Operand, in5: Operand, in6: Operand, in7: Operand,
    in8: Operand, in9: Operand, in10: Operand, in11: Operand, 
    in12: Operand, in13: Operand, in14: Operand,
    builder: TTIRBuilder,
):
    return reduction_op_tree(
        op=builder.maximum,
        inputs=[
            in0, in1, in2, in3, 
            in4, in5, in6, in7, 
            in8, in9, in10, in11, 
            in12, in13, in14,
        ]
    )

def eltwise_fuse_pow_tree_max_inputs(
    in0: Operand, in1: Operand, in2: Operand, in3: Operand,
    in4: Operand, in5: Operand, in6: Operand, in7: Operand,
    in8: Operand, in9: Operand, in10: Operand, in11: Operand, 
    in12: Operand, in13: Operand, in14: Operand,
    builder: TTIRBuilder,
):
    return reduction_op_tree(
        op=builder.pow,
        inputs=[
            in0, in1, in2, in3, 
            in4, in5, in6, in7, 
            in8, in9, in10, in11, 
            in12, in13, in14,
        ]
    )

@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "test_fn", 
    [
        # eltwise_fuse_sub_binary_tree_max_inputs,
        eltwise_fuse_div_tree_max_inputs,
        eltwise_fuse_maximum_tree_max_inputs,
        eltwise_fuse_pow_tree_max_inputs,
    ]
)
def test_eltwise_binary_op_tree_max_inputs(
    test_fn: Callable, shape: Shape, dtype: torch.dtype, target: str, request
):
    options = []
    compile_ttir_to_flatbuffer(
        test_fn,
        [shape]*MAX_INPUT_OPERANDS,
        [dtype]*MAX_INPUT_OPERANDS,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )

### ------------------------------------------------------------------------ ###
# 
### ------------------------------------------------------------------------ ###

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

@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_converging_unary_branches(shape: Shape, dtype: torch.dtype, target: str, request):
    options = []
    compile_ttir_to_flatbuffer(
        converging_unary_branches,
        [shape]*2,
        [dtype]*2,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )