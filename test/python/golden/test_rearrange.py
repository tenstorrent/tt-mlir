# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List

from ttmlir.dialects import ttir, ttcore
from ttmlir.ir import *

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("target", ["ttmetal"])
def test_rearrange(
    target: str,
    request,
    device,
):
    in_shape = (3, 4, 32)
    pattern = "z y x -> y z x"

    def rearrange(
        in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None
    ):
        return builder.rearrange(in0, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        rearrange,
        [in_shape],
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        module_dump=True,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
