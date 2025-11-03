# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple
from conftest import x86_only
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base import get_golden_function
from builder.base.builder_utils import (
    compile_and_execute_ttir,
)
from test_utils import (
    Marks,
    shape_str,
    shapes_list_str,
)
from ttmlir.dialects import ttir

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "shapes", [((3, 128, 128), (3, 128, 128), (128,))], ids=shapes_list_str
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("transpose_a", [False, True])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.xfail(
    reason="Batched input not supported when bias exists (linear operation). https://github.com/tenstorrent/tt-metal/issues/31634"
)
def test_linear_without_workaround(
    shapes: List[Shape],
    dtype: torch.dtype,
    transpose_a: bool,
    transpose_b: bool,
    target: str,
    request,
    device,
):
    def linear_wrapper(
        in0: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        return builder.linear(
            in0, weight, bias, transpose_a, transpose_b, unit_attrs=unit_attrs
        )

    linear_wrapper.__name__ = "linear"

    compile_and_execute_ttir(
        linear_wrapper,
        inputs_shapes=shapes,
        inputs_types=[dtype, dtype, dtype],
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
        device=device,
        pipeline_options=["disable-workarounds=true"],
    )
