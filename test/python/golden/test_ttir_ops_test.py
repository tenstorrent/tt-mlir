# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator
from conftest import x86_only

from builder.base.builder import Operand, Shape, TypeInfo
from builder.base.builder_golden import BuilderGoldenTensor
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer
from ttmlir.ir import DenseI32ArrayAttr
from test_utils import (
    Marks,
    shape_str,
    make_shard_shape,
    shard_wrap_factory,
)
from wrapper_functions import *

pytestmark = pytest.mark.frontend("ttir")


ttir_func_names = [
    "permute",
    "permute2",
]

ttir_funcs = {
    permute: {"shapes": [(2, 3, 5), (2, 5, 3)], "args": {"permutation": [2, 0, 1]}},
}

"""
@pytest.fixture(params=ttir_func_names)
def fns(request):
   print("\n SETUP", request.param)
   return request.param
"""


@pytest.fixture(params=ttir_funcs.keys(), ids=lambda fn: fn.__name__)
def test_fn(request):
    print(request.param)
    return request.param


@pytest.mark.parametrize("shapes", ttir_funcs[test_fn]["shapes"], ids=shape_str)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_ttir_funcs(test_fn, shapes, target, request):
    # shapes = ttir_funcs[test_fn]["shapes"]
    dtypes = ttir_funcs[test_fn]["dtypes"] if "dtypes" in ttir_funcs[test_fn] else None
    args = ttir_funcs[test_fn]["args"].values() if "args" in ttir_funcs[test_fn] else []

    def wrapper(
        operands: List[Operand],
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return test_fn(
            *operands,
            **ttir_funcs[test_fn]["args"],
            builder=builder,
            unit_attrs=unit_attrs,
        )

    compile_ttir_to_flatbuffer(
        wrapper,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


"""
# Generate specific combinations
param_combinations = list(itertools.product([1, 2], ["A", "B"]))

@pytest.mark.parametrize("param1, param2", param_combinations)
def test_manual_combinations(param1, param2):
    assert True # Your test logic
    print(f"Testing with param1={param1}, param2={param2}")



@pytest.fixture(params=ttir_funcs.keys(), ids=lambda fn: fn.__name__)
def test_fn(request):
    return request.param


@pytest.fixture(params=[*])

@pytest.mark.parametrize("test_fn", ttir_funcs.keys())
@pytest.mark.parametrize("shapes", ttir_funcs[test_fn]["shapes"], ids=shape_str)
@pytest.mark.parametrize("target", ["ttnn", "ttmetal"])
def test_ttir_funcs(test_fn, shapes, target, request):
    #shapes = ttir_funcs[test_fn]["shapes"]
    dtypes = ttir_funcs[test_fn]["dtypes"] if "dtypes" in ttir_funcs[test_fn] else None
    args = ttir_funcs[test_fn]["args"].values() if "args" in ttir_funcs[test_fn] else []

    def wrapper(
        operands: List[Operand],
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return test_fn(
            *operands,
            **ttir_funcs[test_fn]["args"],
            builder=builder,
            unit_attrs=unit_attrs,
        )

    compile_ttir_to_flatbuffer(
        wrapper,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )



# Generate specific combinations
param_combinations = list(itertools.product([1, 2], ["A", "B"]))

@pytest.mark.parametrize("param1, param2", param_combinations)
def test_manual_combinations(param1, param2):
    assert True # Your test logic
    print(f"Testing with param1={param1}, param2={param2}")

"""
