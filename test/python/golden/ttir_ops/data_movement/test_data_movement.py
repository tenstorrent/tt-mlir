# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from typing import List, Optional
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir
from test_utils import (
    shapes_list_str,
    shape_str,
)

pytestmark = pytest.mark.frontend("ttir")

# Concat tests
@pytest.mark.parametrize(
    "shapes",
    [
        [
            (64, 128),
            (32, 128),
            (16, 128),
        ]
    ],
    ids=shapes_list_str,
)
@pytest.mark.parametrize("dim", [0])
def test_concat(shapes: List[Shape], dim: int, request, device):
    # Create a wrapper function that captures dim
    def concat_wrapper(
        in0: Operand,
        in1: Operand,
        in2: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.concat([in0, in1, in2], dim, unit_attrs)

    # Set the name for better test identification.
    concat_wrapper.__name__ = "concat"

    compile_and_execute_ttir(
        concat_wrapper,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Pad tests
@pytest.mark.parametrize("shape", [(1, 1, 5, 5)], ids=shape_str)
@pytest.mark.parametrize("padding", [[0, 1, 2, 3, 4, 5, 6, 7]])
@pytest.mark.parametrize("value", [0])
def test_pad(shape: Shape, padding: List[int], value: int, request, device):
    def pad(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.pad(in0, padding=padding, value=value, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        pad,
        inputs_shapes=[shape],
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


# Permute tests
def permute(
    in0: Operand,
    builder: TTIRBuilder,
    permutation: List[int],
    unit_attrs: Optional[List[str]] = None,
):
    return builder.permute(
        in0,
        permutation=permutation,
        unit_attrs=unit_attrs,
    )


@pytest.mark.parametrize("shapes", [[(2, 3, 4)]], ids=shapes_list_str)
@pytest.mark.parametrize("permutation", [[1, 2, 0]])
def test_permute(shapes: List[Shape], permutation: List[int], request, device):
    compile_and_execute_ttir(
        permute,
        shapes,
        test_base=request.node.name,
        device=device,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
