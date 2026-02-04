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

from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from test_utils import Marks
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


def _test_pattern_map(pattern, shape, pattern_map):
    print(pattern, ":", shape, ":", pattern_map)
    t = torch.randn(shape)
    golden = einops.rearrange(t, pattern)
    output = torch.zeros(golden.shape)
    for pos in itertools.product(*[range(dim) for dim in output.shape]):
        p = ttir.ir.affine_map_compose(pattern_map, pos)
        output[slice(*pos)] = t[slice(*p)]
    assert torch.allclose(output, golden)


# Currently any permute that involves the innermost dim is not supported
@pytest.mark.parametrize(
    "shape,pattern",
    [
        # Tilized: inner dims preserved
        ((3, 32, 64), "z y x -> z y x"),
        ((2, 3, 32, 64), "w z y x -> z w y x"),
        ((2, 3, 32, 64), "w z y x -> (w z) y x"),
        ((4, 3, 32, 64), "w z y x -> (z w) y x"),
        # Non-tilized: inner dims modified
        ((3, 32, 32), "z y x -> y z x"),
        ((3, 32, 32), "z y x -> (y z) x"),
        ((3, 32, 32), "z y x -> y (z x)"),
        # Unaligned
        ((3, 4, 5), "z y x -> y z x"),
        ((5, 7, 8), "z y x -> (y z) x"),
        ((5, 7, 8), "z y x -> y (z x)"),
        # Multicore
        ((2, 4, 250), "z y x -> y z x"),
        ((2, 7, 180), "z y x -> (y z) x"),
        ((25, 7, 8), "z y x -> y (z x)"),
        ((50, 7, 8), "z y x -> y (z x)"),
        ((50, 15, 8), "z y x -> z (y x)"),
        ((4, 85, 1055), "z y x -> (z y) x"),
        # 4d
        ((2, 3, 4, 32), "w z y x -> y w z x"),
        ((2, 3, 4, 32), "w z y x -> y (w z) x"),
        ((2, 3, 4, 32), "w z y x -> (y w z) x"),
        ((2, 3, 4, 32), "w z y x -> (y w) z x"),
        ((2, 3, 4, 32), "w z y x -> (y w) (z x)"),
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_rearrange(
    shape,
    pattern,
    target: str,
    request,
    device,
):
    # Enable for local debug of the pattern -> affine map conversion.
    test_pattern_map = False
    if test_pattern_map:
        pattern_map = ttir.ir.rearrange_inv_pattern_map(Context(), pattern, shape)
        _test_pattern_map(pattern, shape, pattern_map)

    def rearrange_module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def rearrange(in0, builder: TTIRBuilder, unit_attrs: List[str] = None):
            return builder.rearrange(in0, pattern, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        rearrange_module,
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline",
        **get_request_kwargs(request),
    )
