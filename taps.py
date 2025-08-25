import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union

from builder.base.builder import Operand, Shape, TypeInfo
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer
from builder.base.builder_golden import BuilderGoldenTensor

shape = [(128, 128)]

def exp(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.exp(in0, unit_attrs=unit_attrs)

def expm1(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.expm1(in0, unit_attrs=unit_attrs)

def ceil(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.ceil(in0, unit_attrs=unit_attrs)

def floor(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.floor(in0, unit_attrs=unit_attrs)

def abs(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.abs(in0, unit_attrs=unit_attrs)

def neg(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.neg(in0, unit_attrs=unit_attrs)

def sign(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sign(in0, unit_attrs=unit_attrs)

def sin(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sin(in0, unit_attrs=unit_attrs)

def cos(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.cos(in0, unit_attrs=unit_attrs)

def atan(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.atan(in0, unit_attrs=unit_attrs)

def tanh(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.tanh(in0, unit_attrs=unit_attrs)

def relu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.relu(in0, unit_attrs=unit_attrs)

def gelu(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.gelu(in0, unit_attrs=unit_attrs)

def leaky_relu(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.leaky_relu(in0, unit_attrs=unit_attrs)

def cbrt(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.cbrt(in0, unit_attrs=unit_attrs)

def sigmoid(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sigmoid(in0, unit_attrs=unit_attrs)

def reciprocal(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.reciprocal(in0, unit_attrs=unit_attrs)

def is_finite(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.is_finite(in0, unit_attrs=unit_attrs)

def get_dimension_size(
    in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None
):
    return builder.get_dimension_size(in0, unit_attrs=unit_attrs)

def sum(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.sum(in0, unit_attrs=unit_attrs)

def mean(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.mean(in0, unit_attrs=unit_attrs)

def max(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.max(in0, unit_attrs=unit_attrs)

def min(in0: Operand, builder: TTIRBuilder, unit_attrs: Optional[List[str]] = None):
    return builder.min(in0, unit_attrs=unit_attrs)

def execute(some_callable, shape):
    compile_ttir_to_flatbuffer(
        some_callable,
        inputs_shapes=shape,
        inputs_types=[torch.float32],
        target="ttnn",
    )

#execute(exp, shape)
#execute(expm1, shape)
#execute(ceil, shape)
#execute(floor, shape)
#execute(abs, shape)
#execute(neg, shape)
#execute(sign, shape)
#execute(sin, shape)
#execute(cos, shape)
#execute(atan, shape)
#execute(tanh, shape)
#execute(relu, shape)
#execute(gelu, shape)
#execute(leaky_relu, shape)
#execute(cbrt, shape)
#execute(sigmoid, shape)
#execute(reciprocal, shape)
#execute(is_finite, shape)
execute(get_dimension_size, shape)
#execute(sum, shape)
#execute(mean, shape)
#execute(max, shape)
#execute(min, shape)