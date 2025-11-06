# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttmlir.ir import *

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import (
    compile_ttir_to_flatbuffer,
    compile_and_execute_ttir,
)
from typing import Optional
import math
from random import randrange

pytestmark = pytest.mark.frontend("ttir")

####################### Gamma based functions #######################

# Support for the range (1, +inf)
def digamma_composite(
    x: Operand,
    shape: Shape,
    dtype: torch.dtype,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    constant_tensors = [
        torch.full(shape, 0.5).to(dtype),
        torch.full(shape, 0.083333333).to(dtype),
        torch.full(shape, 0.008333333333333333).to(dtype),
        torch.full(shape, 0.003968253968253968).to(dtype),
        torch.full(shape, 0.004166666666666667).to(dtype),
        torch.full(shape, 0.007575757575757576).to(dtype),
        torch.full(shape, 0.021092796092796094).to(dtype),
        torch.full(shape, 0.08333333333333333).to(dtype),
    ]
    constants = [builder.constant(i, unit_attrs=unit_attrs) for i in constant_tensors]

    # create builder output
    recip = builder.reciprocal(x, unit_attrs=unit_attrs)
    term1 = builder.multiply(recip, constants[0], unit_attrs=unit_attrs)

    recip_square = builder.multiply(recip, recip, unit_attrs=unit_attrs)
    term2 = builder.multiply(recip_square, constants[1], unit_attrs=unit_attrs)
    intermediate2 = builder.subtract(term1, term2)

    recip_pow_4 = builder.multiply(recip_square, recip_square, unit_attrs=unit_attrs)
    term3 = builder.multiply(recip_pow_4, constants[2], unit_attrs=unit_attrs)
    intermediate3 = builder.add(intermediate2, term3)

    recip_pow_6 = builder.multiply(recip_pow_4, recip_square, unit_attrs=unit_attrs)
    term4 = builder.multiply(recip_pow_6, constants[3], unit_attrs=unit_attrs)
    intermediate4 = builder.subtract(intermediate3, term4)

    recip_pow_8 = builder.multiply(recip_pow_6, recip_square, unit_attrs=unit_attrs)
    term5 = builder.multiply(recip_pow_8, constants[4], unit_attrs=unit_attrs)
    intermediate5 = builder.add(intermediate4, term5)

    recip_pow_10 = builder.multiply(recip_pow_8, recip_square, unit_attrs=unit_attrs)
    term6 = builder.multiply(recip_pow_10, constants[5], unit_attrs=unit_attrs)
    intermediate6 = builder.subtract(intermediate5, term6)

    recip_pow_12 = builder.multiply(recip_pow_10, recip_square, unit_attrs=unit_attrs)
    term7 = builder.multiply(recip_pow_12, constants[6], unit_attrs=unit_attrs)
    intermediate7 = builder.add(intermediate6, term7)

    recip_pow_14 = builder.multiply(recip_pow_12, recip_square, unit_attrs=unit_attrs)
    term8 = builder.multiply(recip_pow_14, constants[7], unit_attrs=unit_attrs)
    intermediate8 = builder.subtract(intermediate7, term8)

    log_x = builder.log(x)
    result = builder.subtract(log_x, intermediate8)

    return result


# Support for the range (1, +inf)
# TODO: Add root checks when ttir.where is supported
def lgamma_composite(
    x: Operand,
    shape: Shape,
    dtype: torch.dtype,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    constant_tensors = [
        torch.full(shape, 76.18009172947146).to(dtype),
        torch.full(shape, -86.50532032941677).to(dtype),
        torch.full(shape, 24.01409824083091).to(dtype),
        torch.full(shape, -1.231739572450155).to(dtype),
        torch.full(shape, 0.1208650973866179e-2).to(dtype),
        torch.full(shape, -0.5395239384953e-5).to(dtype),
        torch.full(shape, 1.0).to(dtype),
        torch.full(shape, 2.0).to(dtype),
        torch.full(shape, 3.0).to(dtype),
        torch.full(shape, 4.0).to(dtype),
        torch.full(shape, 5.0).to(dtype),
        torch.full(shape, 6.0).to(dtype),
        torch.full(shape, 5.5).to(dtype),
        torch.full(shape, 0.5).to(dtype),
        torch.full(shape, 0.918938531357171).to(dtype),
        # torch.full(shape, 0.0).to(dtype),
    ]
    constants = [builder.constant(i, unit_attrs=unit_attrs) for i in constant_tensors]

    # input = x - 1.0
    input_val = builder.subtract(x, constants[6], unit_attrs=unit_attrs)

    # Build temp accumulator
    # z1 = 1/(input + 1.0) * 76.18009172947146
    z1 = builder.multiply(
        builder.reciprocal(
            builder.add(input_val, constants[6], unit_attrs=unit_attrs),
            unit_attrs=unit_attrs,
        ),
        constants[0],
        unit_attrs=unit_attrs,
    )
    temp = builder.add(z1, constants[6], unit_attrs=unit_attrs)

    # z1 = 1/(input + 2.0) * -86.50532032941677
    z1 = builder.multiply(
        builder.reciprocal(
            builder.add(input_val, constants[7], unit_attrs=unit_attrs),
            unit_attrs=unit_attrs,
        ),
        constants[1],
        unit_attrs=unit_attrs,
    )
    temp = builder.add(temp, z1, unit_attrs=unit_attrs)

    # z1 = 1/(input + 3.0) * 24.01409824083091
    z1 = builder.multiply(
        builder.reciprocal(
            builder.add(input_val, constants[8], unit_attrs=unit_attrs),
            unit_attrs=unit_attrs,
        ),
        constants[2],
        unit_attrs=unit_attrs,
    )
    temp = builder.add(temp, z1, unit_attrs=unit_attrs)

    # z1 = 1/(input + 4.0) * -1.231739572450155
    z1 = builder.multiply(
        builder.reciprocal(
            builder.add(input_val, constants[9], unit_attrs=unit_attrs),
            unit_attrs=unit_attrs,
        ),
        constants[3],
        unit_attrs=unit_attrs,
    )
    temp = builder.add(temp, z1, unit_attrs=unit_attrs)

    # z1 = 1/(input + 5.0) * 0.1208650973866179e-2
    z1 = builder.multiply(
        builder.reciprocal(
            builder.add(input_val, constants[10], unit_attrs=unit_attrs),
            unit_attrs=unit_attrs,
        ),
        constants[4],
        unit_attrs=unit_attrs,
    )
    temp = builder.add(temp, z1, unit_attrs=unit_attrs)

    # z1 = 1/(input + 6.0) * -0.5395239384953e-5
    z1 = builder.multiply(
        builder.reciprocal(
            builder.add(input_val, constants[11], unit_attrs=unit_attrs),
            unit_attrs=unit_attrs,
        ),
        constants[5],
        unit_attrs=unit_attrs,
    )
    temp = builder.add(temp, z1, unit_attrs=unit_attrs)

    # t = input + 5.5
    t = builder.add(input_val, constants[12], unit_attrs=unit_attrs)
    t_log = builder.log(t, unit_attrs=unit_attrs)

    # temp_log = log(temp)
    temp_log = builder.log(temp, unit_attrs=unit_attrs)

    # result = (input + 0.5) * t_log + 0.918938531357171
    result = builder.add(
        builder.multiply(
            builder.add(input_val, constants[13], unit_attrs=unit_attrs),
            t_log,
            unit_attrs=unit_attrs,
        ),
        constants[14],
        unit_attrs=unit_attrs,
    )

    # result = result + temp_log
    result = builder.add(result, temp_log, unit_attrs=unit_attrs)

    # result = result - t
    result = builder.subtract(result, t, unit_attrs=unit_attrs)

    # ttir.where marked illegal
    # result = builder.where(builder.eq(x, constants[6]), constants[15], result)
    # result = builder.where(builder.eq(x, constants[7]), constants[15], result)

    return result


# Support for Range [1.5, +inf]
def multigammaln_composite(
    x: Operand,
    shape: Shape,
    dtype: torch.dtype,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    constant_tensors = [
        torch.full(shape, 0.5).to(dtype),
        torch.full(shape, 1.0).to(dtype),
        torch.full(shape, 1.5).to(dtype),
        torch.full(shape, 3.434189657547).to(dtype),
    ]
    constants = [builder.constant(i, unit_attrs=unit_attrs) for i in constant_tensors]

    # result = lgamma(x) + lgamma(x - 0.5) + lgamma(x - 1.0) + lgamma(x - 1.5) + 3.434189657547
    # Note: We use lgamma_composite for the intermediate lgamma calculations
    lgamma_x = lgamma_composite(x, shape, dtype, builder, unit_attrs=unit_attrs)

    x_minus_0_5 = builder.subtract(x, constants[0], unit_attrs=unit_attrs)
    lgamma_x_0_5 = lgamma_composite(
        x_minus_0_5, shape, dtype, builder, unit_attrs=unit_attrs
    )

    x_minus_1_0 = builder.subtract(x, constants[1], unit_attrs=unit_attrs)
    lgamma_x_1_0 = lgamma_composite(
        x_minus_1_0, shape, dtype, builder, unit_attrs=unit_attrs
    )

    x_minus_1_5 = builder.subtract(x, constants[2], unit_attrs=unit_attrs)
    lgamma_x_1_5 = lgamma_composite(
        x_minus_1_5, shape, dtype, builder, unit_attrs=unit_attrs
    )

    result = builder.add(lgamma_x, lgamma_x_0_5, unit_attrs=unit_attrs)
    result = builder.add(result, lgamma_x_1_0, unit_attrs=unit_attrs)
    result = builder.add(result, lgamma_x_1_5, unit_attrs=unit_attrs)
    result = builder.add(result, constants[3], unit_attrs=unit_attrs)

    return result


# Support for range of input(1, 10) and k(1, 10)
def polygamma_composite(
    x: Operand,
    k: int,
    shape: Shape,
    dtype: torch.dtype,
    builder: TTIRBuilder,
    unit_attrs: Optional[List[str]] = None,
):
    k_der = 1.0 + k
    fact_val = math.gamma(k_der)
    if k == 2 or k == 4 or k == 6 or k == 8 or k == 10:
        fact_val *= -1.0

    k_der_builder = builder.constant(
        torch.full(shape, k_der).to(dtype), unit_attrs=unit_attrs
    )
    temp = builder.reciprocal(builder.pow(x, k_der_builder), unit_attrs=unit_attrs)
    for i in range(1, 11):
        i_builder = builder.constant(
            torch.full(shape, i).to(dtype), unit_attrs=unit_attrs
        )
        z1 = builder.reciprocal(
            builder.pow(
                builder.add(x, i_builder, unit_attrs=unit_attrs),
                k_der_builder,
                unit_attrs=unit_attrs,
            ),
            unit_attrs=unit_attrs,
        )
        temp = builder.add(temp, z1, unit_attrs=unit_attrs)

    fact_val_builder = builder.constant(
        torch.full(shape, fact_val).to(dtype), unit_attrs=unit_attrs
    )
    result = builder.multiply(temp, fact_val_builder, unit_attrs=unit_attrs)
    return result


# Support for the range (1, +inf)
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_digamma(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def digamma(
        x: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # set input golden
        x_tensor = torch.rand(shape).to(dtype) * 1e5
        x_tensor = torch.clamp(x_tensor, min=1)

        # compute output golden
        output_golden = torch.digamma(x_tensor).to(dtype)

        # Create builder output following ttnn implementation
        result = digamma_composite(x, shape, dtype, builder, unit_attrs)

        # set goldens
        builder.set_goldens({x: x_tensor}, {result: output_golden})
        return result

    options = []
    compile_and_execute_ttir(
        digamma,
        [shape],
        [dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
        atol=3e-02,
        rtol=3e-02,
    )


# Support for the range (1, +inf)
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_lgamma(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def lgamma(
        x: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # set input golden
        x_tensor = torch.rand(shape).to(dtype) * 1e5
        x_tensor = torch.clamp(x_tensor, min=1)

        # compute output golden
        output_golden = torch.lgamma(x_tensor).to(dtype)

        # Create builder output following ttnn implementation
        result = lgamma_composite(x, shape, dtype, builder, unit_attrs)

        # Set goldens
        builder.set_goldens({x: x_tensor}, {result: output_golden})

        return result

    options = []
    compile_and_execute_ttir(
        lgamma,
        [shape],
        [dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


# Support for Range [1.5, +inf]
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_multigammaln(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def multigammaln(
        x: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # set input golden
        x_tensor = torch.rand(shape).to(dtype) * 1e5
        x_tensor = torch.clamp(x_tensor, min=1.5)

        # compute output golden (p=4 for multigammaln)
        output_golden = torch.special.multigammaln(x_tensor, 4).to(dtype)

        # Create builder output following ttnn implementation
        result = multigammaln_composite(x, shape, dtype, builder, unit_attrs=unit_attrs)

        # Set goldens
        builder.set_goldens({x: x_tensor}, {result: output_golden})

        return result

    options = []
    compile_and_execute_ttir(
        multigammaln,
        [shape],
        [dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


# Support for range of input(1, 10) and k(1, 10)
@pytest.mark.parametrize("k", list(range(1, 11)))
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_polygamma(
    k: int,
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    # choose
    def polygamma(
        x: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # set input golden
        x_tensor = (torch.rand(shape) * 9 + 1).to(dtype)

        # compute output golden
        output_golden = torch.special.polygamma(k, x_tensor).to(dtype)

        # Create builder output following ttnn implementation
        result = polygamma_composite(x, k, shape, dtype, builder, unit_attrs=unit_attrs)

        # Set goldens
        builder.set_goldens({x: x_tensor}, {result: output_golden})

        return result

    options = []
    compile_and_execute_ttir(
        polygamma,
        [shape],
        [dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


####################### GLU ops #######################
@pytest.mark.parametrize("shape", [(1, 1, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_glu_split(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def glu(
        x1: Operand,
        x2: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        result = builder.multiply(
            x1, builder.sigmoid(x2, unit_attrs=unit_attrs), unit_attrs=unit_attrs
        )

        return result

    options = []
    compile_and_execute_ttir(
        glu,
        [shape, shape],
        [dtype, dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


@pytest.mark.parametrize("shape", [(1, 1, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reglu_split(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def reglu(
        x1: Operand,
        x2: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        result = builder.multiply(
            x1, builder.relu(x2, unit_attrs=unit_attrs), unit_attrs=unit_attrs
        )

        return result

    options = []
    compile_and_execute_ttir(
        reglu,
        [shape, shape],
        [dtype, dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


@pytest.mark.parametrize("shape", [(1, 1, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_geglu_split(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def geglu(
        x1: Operand,
        x2: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        result = builder.multiply(
            x1, builder.gelu(x2, unit_attrs=unit_attrs), unit_attrs=unit_attrs
        )

        return result

    options = []
    compile_and_execute_ttir(
        geglu,
        [shape, shape],
        [dtype, dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )


@pytest.mark.parametrize("shape", [(1, 1, 128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_swiglu_split(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    def swiglu(
        x1: Operand,
        x2: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        result = builder.multiply(
            x1, builder.silu(x2, unit_attrs=unit_attrs), unit_attrs=unit_attrs
        )

        return result

    options = []
    compile_and_execute_ttir(
        swiglu,
        [shape, shape],
        [dtype, dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )
