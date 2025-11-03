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

######################## Utility functions ###################################

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


##########################################################################

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



# error is signficant (in order of 10^-1) when use range (1, 10)
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
        inputs: List[Operand],
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # set torch inputs and constants
        x = inputs[0]  
        x_tensor = torch.rand(shape).to(dtype) * 1e5
        x_tensor = torch.clamp(x_tensor, min=1)
        constants = inputs[1:]
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

        # compute outputs
        output_golden = torch.digamma(x_tensor).to(dtype)

        # create builder output
        recip = builder.reciprocal(x, unit_attrs=unit_attrs)
        term1 = builder.multiply(recip, constants[0], unit_attrs=unit_attrs)

        recip_square = builder.multiply(recip, recip, unit_attrs=unit_attrs)
        term2 = builder.multiply(recip_square, constants[1], unit_attrs=unit_attrs)
        interm2 = builder.subtract(term1, term2)

        recip_pow_4 = builder.multiply(recip_square, recip_square, unit_attrs=unit_attrs)
        term3 = builder.multiply(recip_pow_4, constants[2], unit_attrs=unit_attrs)
        interm3 = builder.add(interm2, term3)

        recip_pow_6 = builder.multiply(recip_pow_4, recip_square, unit_attrs=unit_attrs)
        term4 = builder.multiply(recip_pow_6, constants[3], unit_attrs=unit_attrs)
        interm4 = builder.subtract(interm3, term4)

        recip_pow_8 = builder.multiply(recip_pow_6, recip_square, unit_attrs=unit_attrs)
        term5 = builder.multiply(recip_pow_8, constants[4], unit_attrs=unit_attrs)
        interm5 = builder.add(interm4, term5)

        recip_pow_10 = builder.multiply(recip_pow_8, recip_square, unit_attrs=unit_attrs)
        term6 = builder.multiply(recip_pow_10, constants[5], unit_attrs=unit_attrs)
        interm6 = builder.subtract(interm5, term6)

        recip_pow_12 = builder.multiply(recip_pow_10, recip_square, unit_attrs=unit_attrs)
        term7 = builder.multiply(recip_pow_12, constants[6], unit_attrs=unit_attrs)
        interm7 = builder.add(interm6, term7)

        recip_pow_14 = builder.multiply(recip_pow_12, recip_square, unit_attrs=unit_attrs)
        term8 = builder.multiply(recip_pow_14, constants[7], unit_attrs=unit_attrs)
        interm8 = builder.subtract(interm7, term8)

        log_x = builder.log(x)
        res = builder.subtract(log_x, interm8)

        # set goldens
        input_goldens = {x: x_tensor}
        for constant, constant_tensor in zip(constants, constant_tensors):
            input_goldens[constant] = constant_tensor
        builder.set_goldens(
            input_goldens, {res: output_golden}
        )
        return res

    options = []
    num_inputs = 9
    compile_ttir_to_flatbuffer_wrapper(
        digamma_subgraph,
        [shape] * num_inputs,
        [dtype] * num_inputs,
        target=target,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
