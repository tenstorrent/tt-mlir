# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import einops
import itertools
from typing import Callable, List

from ttmlir.dialects import ttir, ttcore
from ttmlir.ir import *

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


def _test_pattern_map(pattern, shape, pattern_map):
    print(pattern, ":", shape, ":", pattern_map)
    t = torch.randn(shape)
    golden = einops.rearrange(t, pattern)
    output = torch.zeros(golden.shape)
    for pos in itertools.product(*[range(dim) for dim in output.shape]):
        p = ttir.ir.affine_map_compose(pattern_map, pos)
        output[*pos] = t[*p]
    assert torch.allclose(output, golden)


@pytest.mark.parametrize("target", ["ttmetal"])
def test_rearrange(
    target: str,
    request,
    device,
):
    in_shape = (3, 32, 32)
    pattern = "z y x -> (y z) x"

    patterns = [
        ("z y x -> y z x", (3, 4, 32)),
        ("z y x -> x z y", (3, 4, 32)),
        ("z y x -> z (y x)", (3, 4, 32)),
        ("z y x -> (z y) x", (3, 4, 32)),
        ("z y x -> (z y x)", (3, 4, 32)),
        ("z y x -> (x z) y", (3, 4, 32)),
        ("z y x -> x (z y)", (3, 4, 32)),
    ]
    ctx = Context()
    for p, s in patterns:
        pattern_map = ttir.ir.rearrange_inv_pattern_map(ctx, p, s)
        _test_pattern_map(p, s, pattern_map)

    def rearrange(
        in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None
    ):
        return builder.rearrange(in0, pattern, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        rearrange,
        [in_shape],
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
