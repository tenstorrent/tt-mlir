# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttmlir.ir import *

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer
from typing import Optional

pytestmark = pytest.mark.frontend("ttir")




@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_custom_add(
    shape: Shape, dtype: torch.dtype, target: str, request
):
    options = []

    def custom_add(
        in0: Operand,
        in1: Operand,
        builder: TTIRBuilder,
    ):

        input_0 = torch.randn(shape).to(dtype)
        input_1 = torch.full(shape, 1).to(dtype)
        output_0 = torch.add(input_0, input_1).to(dtype)

        sum = builder.add(in0, in1)

        builder.set_goldens(
            {
                in0: input_0,
                in1: input_1,
            },
            {sum: output_0},
        )

        return sum

    compile_ttir_to_flatbuffer(
        custom_add,
        [shape] * 2,
        [dtype] * 2,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc")
    )



@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_atanh(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def atanh(
        x: Operand,
        ones: Operand, 
        twos: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # compute torch inputs and outputs    
        x_tensor = torch.randn(shape).to(dtype)    
        x_tensor = x_tensor.uniform_(-1.0, 1.0)
        
        # add error margin
        #x_tensor = torch.clamp(x_tensor, min=-0.8, max=0.8)
        
        ones_tensor = torch.full(shape, 1).to(dtype)
        twos_tensor = torch.full(shape, 2).to(dtype)
        output_golden = torch.atanh(x_tensor).to(dtype)

        # create builder output
        num = builder.add(ones, x, unit_attrs=unit_attrs)
        denom = builder.subtract(ones, x, unit_attrs=unit_attrs)
        quotient = builder.div(num, denom, unit_attrs=unit_attrs)
        res = builder.div(builder.log(quotient, unit_attrs=unit_attrs), twos, unit_attrs=unit_attrs)

        # set goldens
        builder.set_goldens(
            {x: x_tensor, ones: ones_tensor, twos: twos_tensor}, {res: output_golden}
        )
        return res

    options = []
    compile_ttir_to_flatbuffer(
        atanh,
        [shape, shape, shape],
        [dtype, dtype, dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_cbrt(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def cbrt(
        x: Operand,
        power: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # compute torch inputs and outputs    
        x_tensor = torch.randn(shape).to(dtype)
        
        power_tensor = torch.full(shape, -1/3).to(dtype)
        output_golden = torch.pow(x_tensor, power_tensor).to(dtype)

        # create builder output
        res = builder.pow(x, power, unit_attrs=unit_attrs)

        # set goldens
        builder.set_goldens(
            {x: x_tensor, power: power_tensor}, {res: output_golden}
        )
        return res #builder.cbrt(x, unit_attrs=unit_attrs)

    options = []
    compile_ttir_to_flatbuffer(
        cbrt,
        [shape, shape],
        [dtype, dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_cosh(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def cosh(
        x: Operand,
        twos: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # compute torch inputs and outputs    
        x_tensor = torch.randn(shape).to(dtype)

        #twos_tensor = torch.full(shape, 2).to(dtype)
        output_golden = torch.cosh(x_tensor).to(dtype)

        #builder constants
        twos = builder.zeros([(0, 0)], dtype, unit_attrs=unit_attrs)
        twos_u = builder.pad(twos, shape, 2, unit_attrs=unit_attrs)

        # create builder output
        intermediate1 = builder.exp(x, unit_attrs=unit_attrs)
        intermediate2 = builder.exp(builder.neg(x, unit_attrs=unit_attrs), unit_attrs=unit_attrs)
        sum = builder.add(intermediate1, intermediate2, unit_attrs=unit_attrs)
        res = builder.div(sum, twos_u, unit_attrs=unit_attrs)

        # set goldens
        builder.set_goldens(
            {x: x_tensor}, {res: output_golden}
        )
        return res

    options = []
    compile_ttir_to_flatbuffer(
        cosh,
        [shape, shape],
        [dtype, dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )

# add margin so that its not close to 0???
@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_sinh(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def sinh(
        x: Operand,
        twos: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # compute torch inputs and outputs    
        x_tensor = torch.randn(shape).to(dtype)

        # add error margin
        error_margin = torch.full(x_tensor.shape, 0.05).to(dtype)
        error_margin[x_tensor < 0] = -0.05
        x_tensor = torch.add(x_tensor, error_margin)      

        twos_tensor = torch.full(shape, 2).to(dtype)
        output_golden = torch.sinh(x_tensor).to(dtype)

        # create builder output
        intermediate1 = builder.exp(x, unit_attrs=unit_attrs)
        intermediate2 = builder.exp(builder.neg(x, unit_attrs=unit_attrs), unit_attrs=unit_attrs)
        sum = builder.subtract(intermediate1, intermediate2, unit_attrs=unit_attrs)
        res = builder.div(sum, twos, unit_attrs=unit_attrs)

        # set goldens
        builder.set_goldens(
            {x: x_tensor, twos: twos_tensor}, {res: output_golden}
        )
        return res

    options = []
    compile_ttir_to_flatbuffer(
        sinh,
        [shape, shape],
        [dtype, dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )

# similar to compile_ttir_to_flatbuffer to fn can take list of Operand
def compile_ttir_to_flatbuffer_wrapper(fn, *args, **kwargs):
    # take Operands from inputs and make into a list then pass to fn
    def fn_expanded_inputs(*args):
        input_args = []
        ttir_builder_arg = None
        remaining_args = []
        for arg in args:
            if isinstance(arg, Operand):
                input_args.append(arg)
            elif isinstance(arg, TTIRBuilder):
                ttir_builder_arg = arg
            else:
               remaining_args.append(arg) 

        return fn(input_args, ttir_builder_arg, *remaining_args)

    compile_ttir_to_flatbuffer(
        fn_expanded_inputs,
        *args,
        **kwargs
    )


# add margin so that its not close to 0???
@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_sinh_v2(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def sinh(
        inputs: List[Operand],
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # compute torch inputs and outputs
        x = inputs[0]  
        x_tensor = torch.randn(shape).to(dtype)

        # add error margin
        error_margin = torch.full(x_tensor.shape, 0.05).to(dtype)
        error_margin[x_tensor < 0] = -0.05
        x_tensor = torch.add(x_tensor, error_margin)      

        constants = inputs[1:]
        constant_tensors = [
            torch.full(shape, 2).to(dtype)
        ]
        output_golden = torch.sinh(x_tensor).to(dtype)

        # create builder output
        intermediate1 = builder.exp(x, unit_attrs=unit_attrs)
        intermediate2 = builder.exp(builder.neg(x, unit_attrs=unit_attrs), unit_attrs=unit_attrs)
        sum = builder.subtract(intermediate1, intermediate2, unit_attrs=unit_attrs)
        res = builder.div(sum, constants[0], unit_attrs=unit_attrs)

        # set goldens
        input_goldens = {x: x_tensor}
        for constant, constant_tensor in zip(constants, constant_tensors):
            input_goldens[constant] = constant_tensor
        builder.set_goldens(
            input_goldens, {res: output_golden}
        )
        return res

    options = []
    compile_ttir_to_flatbuffer_wrapper(
        sinh,
        [shape, shape],
        [dtype, dtype],
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )

@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_digamma(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def digamma_subgraph(
        x: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # compute torch inputs and outputs    
        x_tensor = torch.randn(shape).to(dtype)
        output_golden = torch.digamma(x_tensor).to(dtype)

        # create builder output
        recip = builder.reciprocal(x, unit_attrs=unit_attrs)
        term1 = builder.multiply(recip, 0.5, unit_attrs=unit_attrs)

        recip_square = builder.square(recip, unit_attrs=unit_attrs)
        term2 = builder.multiply(recip_square, 0.083333333, unit_attrs=unit_attrs)
        interm2 = builder.subtract(term1, term2)

        recip_pow_4 = builder.multiply(recip_square, recip_square, unit_attrs=unit_attrs)
        term3 = builder.multiply(recip_pow_4, 0.008333333333333333, unit_attrs=unit_attrs)
        interm3 = builder.add(interm2, term3)

        recip_pow_6 = builder.multiply(recip_pow_4, recip_square, unit_attrs=unit_attrs)
        term4 = builder.multiply(recip_pow_6, 0.003968253968253968, unit_attrs=unit_attrs)
        interm4 = builder.subtract(interm3, term4)

        recip_pow_8 = builder.multiply(recip_pow_6, recip_square, unit_attrs=unit_attrs)
        term5 = builder.multiply(recip_pow_8, 0.004166666666666667, unit_attrs=unit_attrs)
        interm5 = builder.add(interm4, term5)

        recip_pow_10 = builder.multiply(recip_pow_8, recip_square, unit_attrs=unit_attrs)
        term6 = builder.multiply(recip_pow_10, 0.007575757575757576, unit_attrs=unit_attrs)
        interm6 = builder.subtract(interm5, term6)

        recip_pow_12 = builder.multiply(recip_pow_10, recip_square, unit_attrs=unit_attrs)
        term7 = builder.multiply(recip_pow_12, 0.021092796092796094, unit_attrs=unit_attrs)
        interm7 = builder.add(interm6, term7)

        recip_pow_14 = builder.multiply(recip_pow_12, recip_square, unit_attrs=unit_attrs)
        term8 = builder.multiply(recip_pow_14, 0.08333333333333333, unit_attrs=unit_attrs)
        interm8 = builder.subtract(interm7, term8)

        # set goldens
        builder.set_goldens(
            {x: x_tensor}, {interm8: output_golden}
        )

        return interm8

    options = []
    compile_ttir_to_flatbuffer(
        digamma_subgraph,
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


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_lgamma(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def lgamma_subgraph(
        x: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.lgamma(x, unit_attrs=unit_attrs)

    options = []
    compile_ttir_to_flatbuffer(
        lgamma_subgraph,
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


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_multigammaln(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def multigammaln_subgraph(
        x: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.multigammaln(x, unit_attrs=unit_attrs)

    options = []
    compile_ttir_to_flatbuffer(
        multigammaln_subgraph,
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


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_variance(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def variance_subgraph(
        x: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.variance(x, unit_attrs=unit_attrs)

    options = []
    compile_ttir_to_flatbuffer(
        variance_subgraph,
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


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_std_overload(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def std_overload_subgraph(
        x: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.std_overload(x, unit_attrs=unit_attrs)

    options = []
    compile_ttir_to_flatbuffer(
        std_overload_subgraph,
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


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_normalize(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def normalize_subgraph(
        x: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.normalize(x, unit_attrs=unit_attrs)

    options = []
    compile_ttir_to_flatbuffer(
        normalize_subgraph,
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


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(128, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_normalize_global(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def normalize_global_subgraph(
        x: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.normalize_global(x, unit_attrs=unit_attrs)

    options = []
    compile_ttir_to_flatbuffer(
        normalize_global_subgraph,
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