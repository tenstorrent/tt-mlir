# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %pytest -svv %s | FileCheck %s

from test_infra.ttir import Add, Multiply, Exp
from test_infra.ttir_builder import TTIRBuilder
from ttmlir.ir import *

import torch
import pytest

# ----- Fixtures -----


@pytest.fixture
def context() -> Context:
    return Context()


@pytest.fixture
def location(context: Context) -> Location:
    return Location.unknown(context)


@pytest.fixture
def builder(context: Context, location: Location) -> TTIRBuilder:
    return TTIRBuilder(context, location)


@pytest.fixture
def add(context: Context, location: Location) -> Add:
    return Add(context, location)


@pytest.fixture
def multiply(context: Context, location: Location) -> Multiply:
    return Multiply(context, location)


@pytest.fixture
def exp(context: Context, location: Location) -> Exp:
    return Exp(context, location)


# ----- Tests -----


def test_add(builder: TTIRBuilder, add: Add):
    in0_shape = (32, 32)
    in1_shape = (32, 32)
    out_shape = (32, 32)

    f = builder.emit_mlir_function(add, [in0_shape, in1_shape], [out_shape])
    print("\n", f)
    # CHECK: func.func @add(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32>
    # CHECK: %0 = tensor.empty() : tensor<32x32xf32>
    # CHECK: %1 = "ttir.add"(%arg0, %arg1, %0)
    # CHECK: operand_constraints = [#any, #any, #any]

    in0_seed = 0
    in1_seed = 1
    torch.manual_seed(in0_seed)
    t0 = torch.rand(in0_shape)
    torch.manual_seed(in1_seed)
    t1 = torch.rand(in1_shape)

    golden = add.golden(t0, t1)
    print("\n", golden)


def test_multiply(builder: TTIRBuilder, multiply: Multiply):
    in0_shape = (32, 32)
    in1_shape = (32, 32)
    out_shape = (32, 32)

    f = builder.emit_mlir_function(multiply, [in0_shape, in1_shape], [out_shape])
    print("\n", f)
    # CHECK: func.func @multiply(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32>
    # CHECK: %0 = tensor.empty() : tensor<32x32xf32>
    # CHECK: %1 = "ttir.multiply"(%arg0, %arg1, %0)
    # CHECK: operand_constraints = [#any, #any, #any]

    in0_seed = 0
    in1_seed = 1
    torch.manual_seed(in0_seed)
    t0 = torch.rand(in0_shape)
    torch.manual_seed(in1_seed)
    t1 = torch.rand(in1_shape)

    golden = multiply.golden(t0, t1)
    print("\n", golden)


def test_exp(builder: TTIRBuilder, exp: Exp):
    in0_shape = (32, 32)
    out_shape = (32, 32)

    f = builder.emit_mlir_function(exp, [in0_shape], [out_shape])
    print("\n", f)
    # CHECK: func.func @exp(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32>
    # CHECK: %0 = tensor.empty() : tensor<32x32xf32>
    # CHECK: %1 = "ttir.exp"(%arg0, %0)
    # CHECK: operand_constraints = [#any, #any]

    in0_seed = 0
    torch.manual_seed(in0_seed)
    t0 = torch.rand(in0_shape)

    golden = exp.golden(t0)
    print("\n", golden)
