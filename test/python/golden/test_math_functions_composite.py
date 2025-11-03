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

    num_inputs = 2
    options = []
    compile_ttir_to_flatbuffer_wrapper(
        sinh,
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

# error is signficant (in order of 10^-1) when use range (1, 10)
@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(1024, 1024)])
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

# ttir.where marked illegal???
# passes pcc but not all close with Range [1, 10]
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
    def lgamma_composite(
        inputs: List[Operand],
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Set torch inputs and constants
        x = inputs[0]
        x_tensor = (torch.rand(shape) * 9 + 1).to(dtype)  # Range [1, 10]
        
        constants = inputs[1:]
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
            #torch.full(shape, 0.0).to(dtype),
        ]
        
        # Compute golden output
        output_golden = torch.lgamma(x_tensor).to(dtype)
        
        # Create builder output following C++ implementation
        # input = x - 1.0
        input_val = builder.subtract(x, constants[6], unit_attrs=unit_attrs)
        
        # Build temp accumulator
        # z1 = 1/(input + 1.0) * 76.18009172947146
        z1 = builder.multiply(
            builder.reciprocal(builder.add(input_val, constants[6], unit_attrs=unit_attrs), unit_attrs=unit_attrs),
            constants[0],
            unit_attrs=unit_attrs
        )
        temp = builder.add(z1, constants[6], unit_attrs=unit_attrs)
        
        # z1 = 1/(input + 2.0) * -86.50532032941677
        z1 = builder.multiply(
            builder.reciprocal(builder.add(input_val, constants[7], unit_attrs=unit_attrs), unit_attrs=unit_attrs),
            constants[1],
            unit_attrs=unit_attrs
        )
        temp = builder.add(temp, z1, unit_attrs=unit_attrs)
        
        # z1 = 1/(input + 3.0) * 24.01409824083091
        z1 = builder.multiply(
            builder.reciprocal(builder.add(input_val, constants[8], unit_attrs=unit_attrs), unit_attrs=unit_attrs),
            constants[2],
            unit_attrs=unit_attrs
        )
        temp = builder.add(temp, z1, unit_attrs=unit_attrs)
        
        # z1 = 1/(input + 4.0) * -1.231739572450155
        z1 = builder.multiply(
            builder.reciprocal(builder.add(input_val, constants[9], unit_attrs=unit_attrs), unit_attrs=unit_attrs),
            constants[3],
            unit_attrs=unit_attrs
        )
        temp = builder.add(temp, z1, unit_attrs=unit_attrs)
        
        # z1 = 1/(input + 5.0) * 0.1208650973866179e-2
        z1 = builder.multiply(
            builder.reciprocal(builder.add(input_val, constants[10], unit_attrs=unit_attrs), unit_attrs=unit_attrs),
            constants[4],
            unit_attrs=unit_attrs
        )
        temp = builder.add(temp, z1, unit_attrs=unit_attrs)
        
        # z1 = 1/(input + 6.0) * -0.5395239384953e-5
        z1 = builder.multiply(
            builder.reciprocal(builder.add(input_val, constants[11], unit_attrs=unit_attrs), unit_attrs=unit_attrs),
            constants[5],
            unit_attrs=unit_attrs
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
                unit_attrs=unit_attrs
            ),
            constants[14],
            unit_attrs=unit_attrs
        )
        
        # result = result + temp_log
        result = builder.add(result, temp_log, unit_attrs=unit_attrs)
        
        # result = result - t
        result = builder.subtract(result, t, unit_attrs=unit_attrs)

        # ttir.where marked illegal???
        #result = builder.where(builder.eq(x, constants[6]), constants[15], result)
        #result = builder.where(builder.eq(x, constants[7]), constants[15], result)
        
        # Set goldens
        input_goldens = {x: x_tensor}
        for constant, constant_tensor in zip(constants, constant_tensors):
            input_goldens[constant] = constant_tensor
        builder.set_goldens(input_goldens, {result: output_golden})
        
        return result

    options = []
    num_inputs = 16  # 1 input + 15 constants
    compile_ttir_to_flatbuffer_wrapper(
        lgamma_composite,
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

# to do: need to implement helper function lgamma
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
    def multigammaln_composite(
        inputs: List[Operand],
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Set torch inputs and constants
        x = inputs[0]
        x_tensor = (torch.rand(shape) * 9 + 2.5).to(dtype)  # Range [2.5, 11.5] (needs x > 2 for p=4)
        
        constants = inputs[1:]
        constant_tensors = [
            torch.full(shape, 0.5).to(dtype),
            torch.full(shape, 1.0).to(dtype),
            torch.full(shape, 1.5).to(dtype),
            torch.full(shape, 3.434189657547).to(dtype),
        ]
        
        # Compute golden output (p=4 for multigammaln)
        output_golden = torch.special.multigammaln(x_tensor, 4).to(dtype)
        
        # Create builder output following C++ implementation
        # result = lgamma(x) + lgamma(x - 0.5) + lgamma(x - 1.0) + lgamma(x - 1.5) + 3.434189657547
        # Note: We use torch.lgamma for the intermediate lgamma calculations
        lgamma_x = builder.lgamma(x, unit_attrs=unit_attrs)
        
        x_minus_0_5 = builder.subtract(x, constants[0], unit_attrs=unit_attrs)
        lgamma_x_0_5 = builder.lgamma(x_minus_0_5, unit_attrs=unit_attrs)
        
        x_minus_1_0 = builder.subtract(x, constants[1], unit_attrs=unit_attrs)
        lgamma_x_1_0 = builder.lgamma(x_minus_1_0, unit_attrs=unit_attrs)
        
        x_minus_1_5 = builder.subtract(x, constants[2], unit_attrs=unit_attrs)
        lgamma_x_1_5 = builder.lgamma(x_minus_1_5, unit_attrs=unit_attrs)
        
        result = builder.add(lgamma_x, lgamma_x_0_5, unit_attrs=unit_attrs)
        result = builder.add(result, lgamma_x_1_0, unit_attrs=unit_attrs)
        result = builder.add(result, lgamma_x_1_5, unit_attrs=unit_attrs)
        result = builder.add(result, constants[3], unit_attrs=unit_attrs)
        
        # Set goldens
        input_goldens = {x: x_tensor}
        for constant, constant_tensor in zip(constants, constant_tensors):
            input_goldens[constant] = constant_tensor
        builder.set_goldens(input_goldens, {result: output_golden})
        
        return result

    options = []
    num_inputs = 5  # 1 input + 4 constants
    compile_ttir_to_flatbuffer_wrapper(
        multigammaln_composite,
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

# correction unused (like ttnn implementation)
# ttir mean marked illegal so had to use sum
# keep_dim=False also causes a crash during compile_ttir_module_to_flatbuffer, not supported currently
#ones_ = builder.ones(shape, unit_attrs=unit_attrs) not supported, have to pass one as a constant
# bloat16 crashes?
# without broadcast or keep dim = false, cannot implement reduce-based variance/std/
@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(16*32, 8*32)])
@pytest.mark.parametrize("dtype", [torch.float32], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_variance(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def variance_composite(
        inputs: List[Operand],
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None
    ):
        dims = [1]
        correction = 0.0
        keep_dim = True

        # Set torch inputs
        y = inputs[0]
        y_tensor = torch.randn(shape).to(dtype)
        constants = inputs[1:]
        constant_tensors = [
            torch.full(shape, 1.0).to(dtype),
        ]

        # Compute golden output
        output_golden = torch.var(y_tensor, dim=dims, keepdim=keep_dim, correction=correction).to(dtype)

        # Create builder output following C++ implementation
        # mean_y = mean(y)
        count = builder.sum(constants[0], dims, keep_dim=True, unit_attrs=unit_attrs) # to do: better way to get this???\, using shape in ir???
        mean_y = builder.div(builder.sum(y, dims, keep_dim=True, unit_attrs=unit_attrs), count, unit_attrs=unit_attrs)
        # y_minus_mean_y = y - mean_y 
        y_minus_mean_y = builder.subtract(y, mean_y, unit_attrs=unit_attrs)
        # sqr_y_minus_mean_y = square(y_minus_mean_y)
        sqr_y_minus_mean_y = builder.multiply(y_minus_mean_y, y_minus_mean_y, unit_attrs=unit_attrs)
        # variance = mean(sqr_y_minus_mean_y)
        result = builder.div(builder.sum(sqr_y_minus_mean_y, dims, keep_dim, unit_attrs=unit_attrs), count, unit_attrs=unit_attrs)
        
        # Set goldens
        input_goldens = {y: y_tensor}
        for constant, constant_tensor in zip(constants, constant_tensors):
            input_goldens[constant] = constant_tensor
        builder.set_goldens(input_goldens, {result: output_golden})
        
        return result

    options = []
    num_inputs = 2
    compile_ttir_to_flatbuffer_wrapper(
        variance_composite,
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
    def std_composite(
        inputs: List[Operand],
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Set torch inputs
        y = inputs[0]
        y_tensor = torch.randn(shape).to(dtype)
        
        # Compute golden output (unbiased=False, correction=0)
        output_golden = torch.std(y_tensor, unbiased=False).to(dtype)
        
        # Create builder output following C++ implementation
        # std = sqrt(variance(y))
        
        # mean_y = mean(y)
        mean_y = builder.mean(y, unit_attrs=unit_attrs)
        
        # y_minus_mean_y = y - mean_y
        y_minus_mean_y = builder.subtract(y, mean_y, unit_attrs=unit_attrs)
        
        # sqr_y_minus_mean_y = square(y_minus_mean_y)
        sqr_y_minus_mean_y = builder.square(y_minus_mean_y, unit_attrs=unit_attrs)
        
        # variance = mean(sqr_y_minus_mean_y)
        variance = builder.mean(sqr_y_minus_mean_y, unit_attrs=unit_attrs)
        
        # std = sqrt(variance)
        result = builder.sqrt(variance, unit_attrs=unit_attrs)
        
        # Set goldens
        builder.set_goldens({y: y_tensor}, {result: output_golden})
        
        return result

    options = []
    num_inputs = 1
    compile_ttir_to_flatbuffer_wrapper(
        std_composite,
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
    def normalize_composite(
        inputs: List[Operand],
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Set torch inputs
        y = inputs[0]
        y_tensor = torch.randn(shape).to(dtype)
        
        # Compute golden output: (y - mean(y)) / std(y)
        mean_y_torch = y_tensor.mean()
        std_y_torch = y_tensor.std(unbiased=False)
        output_golden = ((y_tensor - mean_y_torch) / std_y_torch).to(dtype)
        
        # Create builder output following C++ implementation
        # normalize = (y - mean(y)) / std(y)
        
        # mean_y = mean(y)
        mean_y = builder.mean(y, unit_attrs=unit_attrs)
        
        # y_minus_mean_y = y - mean_y
        y_minus_mean_y = builder.subtract(y, mean_y, unit_attrs=unit_attrs)
        
        # std_y = std(y, mean_y, y_minus_mean_y)
        sqr_y_minus_mean_y = builder.square(y_minus_mean_y, unit_attrs=unit_attrs)
        variance = builder.mean(sqr_y_minus_mean_y, unit_attrs=unit_attrs)
        std_y = builder.sqrt(variance, unit_attrs=unit_attrs)
        
        # recip_std_y = 1 / std_y
        recip_std_y = builder.reciprocal(std_y, unit_attrs=unit_attrs)
        
        # result = y_minus_mean_y * recip_std_y
        result = builder.multiply(y_minus_mean_y, recip_std_y, unit_attrs=unit_attrs)
        
        # Set goldens
        builder.set_goldens({y: y_tensor}, {result: output_golden})
        
        return result

    options = []
    num_inputs = 1
    compile_ttir_to_flatbuffer_wrapper(
        normalize_composite,
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


@pytest.mark.fails_golden
@pytest.mark.parametrize("shape", [(2, 2, 32, 32)])  # 4D shape for global normalize
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_normalize_global(
    shape: Shape,
    dtype: torch.dtype,
    target: str,
    request,
):
    def normalize_global_composite(
        inputs: List[Operand],
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Set torch inputs
        y = inputs[0]
        y_tensor = torch.randn(shape).to(dtype)
        
        # Compute golden output: global normalize (normalize over all elements)
        mean_y_torch = y_tensor.mean()
        std_y_torch = y_tensor.std(unbiased=False)
        output_golden = ((y_tensor - mean_y_torch) / std_y_torch).to(dtype)
        
        # Create builder output following C++ implementation
        # normalize_global reshapes to [1,1,H,W*N*C], applies normalize, then reshapes back
        # For simplicity in TTIR, we can just apply normalize directly since it operates globally
        
        # mean_y = mean(y)
        mean_y = builder.mean(y, unit_attrs=unit_attrs)
        
        # y_minus_mean_y = y - mean_y
        y_minus_mean_y = builder.subtract(y, mean_y, unit_attrs=unit_attrs)
        
        # std_y = std(y, mean_y, y_minus_mean_y)
        sqr_y_minus_mean_y = builder.square(y_minus_mean_y, unit_attrs=unit_attrs)
        variance = builder.mean(sqr_y_minus_mean_y, unit_attrs=unit_attrs)
        std_y = builder.sqrt(variance, unit_attrs=unit_attrs)
        
        # recip_std_y = 1 / std_y
        recip_std_y = builder.reciprocal(std_y, unit_attrs=unit_attrs)
        
        # result = y_minus_mean_y * recip_std_y
        result = builder.multiply(y_minus_mean_y, recip_std_y, unit_attrs=unit_attrs)
        
        # Set goldens
        builder.set_goldens({y: y_tensor}, {result: output_golden})
        
        return result

    options = []
    num_inputs = 1
    compile_ttir_to_flatbuffer_wrapper(
        normalize_global_composite,
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

