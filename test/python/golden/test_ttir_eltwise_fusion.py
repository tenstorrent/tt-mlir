# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Callable, Sequence, Optional

from ttmlir.ir import *
from ttmlir.passes import ttir_to_ttmetal_backend_pipeline
from ttmlir.dialects import ttir
from conftest import get_request_kwargs

from builder.base.builder_utils import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import (
    Marks,
    shape_str,
    make_shard_shape,
    shard_wrap_factory,
)


pytestmark = pytest.mark.frontend("ttir")


### ----------------------------------------------------------------------- ###
# Main Parameter Sets to Reduce Verbiage
### ----------------------------------------------------------------------- ###

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


# Everything should pass
@pytest.mark.parametrize("grid", gridParams)
@pytest.mark.parametrize(
    "shape", [(128, 128)]
)  # , (32, 64), (64, 64), (64, 128), (128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_fuse_cosh(
    grid: str, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    options = [grid]

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def cosh(in0: Operand, in1: Operand, builder: TTIRBuilder):
            neg_x = builder.neg(in0)

            e_neg_x = builder.exp(neg_x)
            e_pos_x = builder.exp(in0)

            nr_term = builder.add(e_pos_x, e_neg_x)
            ret_val = builder.multiply(nr_term, in1)

            return ret_val

    compile_and_execute_ttir(
        module,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        device=device,
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
    device,
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
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

    compile_and_execute_ttir(
        module,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        device=device,
    )


### ----------------------------------------------------------------------- ###
# Test: Long Unary Chain
### ----------------------------------------------------------------------- ###


@pytest.mark.parametrize("grid", gridParams)
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_fuse_unary_chain(
    grid: str, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    # unary ops are done in place, should be able to fuse
    # indefinitely
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
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

    options = [grid]

    compile_and_execute_ttir(
        module,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        device=device,
    )


### ----------------------------------------------------------------------- ###
# Test: Joins and Binary Reduction Trees
### ----------------------------------------------------------------------- ###


@pytest.mark.parametrize("grid", gridParams)
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_fuse_converging_unary_branches(
    grid: str, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
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

    options = [grid]

    compile_and_execute_ttir(
        module,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        device=device,
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
@pytest.mark.skip(
    reason="TODO(ckaravasilisTT): reenable when binary FPU fusion is supported"
)
def test_eltwise_fuse_binary_reduction_tree(
    grid: str, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    options = [grid]

    def module(builder: TTIRBuilder):
        @builder.func([shape] * 8, [dtype] * 8)
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

    compile_and_execute_ttir(
        module,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        device=device,
    )


### ----------------------------------------------------------------------- ###
# # Test: Where (Ternary) Fusion
### ----------------------------------------------------------------------- ###


@pytest.mark.parametrize("grid", gridParams)
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_fuse_where_simple(
    grid: str, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    """Test simple where with unary ops on true/false branches."""

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape, shape], [dtype, dtype, dtype])
        def where_simple(
            cond: Operand,
            true_val: Operand,
            false_val: Operand,
            builder: TTIRBuilder,
        ):
            # Condition must be strictly 0s or 1s
            condition_tensor = torch.randint(0, 2, shape).to(dtype)
            builder.set_goldens(inputs={cond: condition_tensor})

            # Apply unary ops to true and false branches before where
            true_branch = builder.abs(true_val)
            false_branch = builder.neg(false_val)

            return builder.where(cond, true_branch, false_branch)

    options = [grid]

    compile_and_execute_ttir(
        module,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        device=device,
    )


##--##-------------------------------------------------------------------##--##


@pytest.mark.parametrize("grid", gridParams)
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_fuse_where_with_unary_chains(
    grid: str, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    """Test where with longer unary chains on inputs and output."""

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape, shape], [dtype, dtype, dtype])
        def where_with_unary_chains(
            cond: Operand,
            true_val: Operand,
            false_val: Operand,
            builder: TTIRBuilder,
        ):
            # Condition must be strictly 0s or 1s
            condition_tensor = torch.randint(0, 2, shape).to(dtype)
            builder.set_goldens(inputs={cond: condition_tensor})

            # Unary chain on true branch
            true_0 = builder.abs(true_val)
            true_1 = builder.exp(true_0)

            # Unary chain on false branch
            false_0 = builder.neg(false_val)
            false_1 = builder.sin(false_0)

            # Where operation
            result = builder.where(cond, true_1, false_1)

            # Unary chain on output
            out_0 = builder.abs(result)
            out_1 = builder.neg(out_0)

            return out_1

    options = [grid]

    compile_and_execute_ttir(
        module,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        device=device,
    )


##--##-------------------------------------------------------------------##--##


@pytest.mark.parametrize("grid", gridParams)
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_eltwise_fuse_where_with_binary_inputs(
    grid: str, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    """Test where with binary ops feeding into the true/false branches."""

    def module(builder: TTIRBuilder):
        @builder.func(
            [shape, shape, shape, shape, shape],
            [dtype, dtype, dtype, dtype, dtype],
        )
        def where_with_binary_inputs(
            cond: Operand,
            in0: Operand,
            in1: Operand,
            in2: Operand,
            in3: Operand,
            builder: TTIRBuilder,
        ):
            # Condition must be strictly 0s or 1s
            condition_tensor = torch.randint(0, 2, shape).to(dtype)
            builder.set_goldens(inputs={cond: condition_tensor})

            # Binary ops for true and false branches
            true_branch = builder.div(in0, in1)
            false_branch = builder.pow(in2, in3)

            return builder.where(cond, true_branch, false_branch)

    options = [grid]

    compile_and_execute_ttir(
        module,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        device=device,
    )


### ----------------------------------------------------------------------- ###
# # Diamond Fork and Join Patterns
### ----------------------------------------------------------------------- ###


@pytest.mark.parametrize("grid", gridParams)
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_diamond_unary_op_fanout(
    grid: str, shape: Shape, dtype: torch.dtype, target: str, request, device
):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def diamond_unary_op_fanout(
            in0: Operand,
            builder: TTIRBuilder,
        ):
            abs_0 = builder.abs(in0)

            ceil_0 = builder.ceil(abs_0)
            neg_0 = builder.neg(ceil_0)

            floor_0 = builder.floor(abs_0)
            neg_1 = builder.neg(floor_0)

            return builder.div(neg_0, neg_1)

    options = [grid]

    compile_and_execute_ttir(
        module,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
        device=device,
    )
