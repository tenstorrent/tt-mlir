# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""E2E test for the O-projection + residual-add pattern.

Pattern B of the activation-dtype-lowering pass: a single matmul whose output
flows through a CCL pair (reduce_scatter -> all_gather) and is then added to a
bf16 residual. The pass must lower the matmul output to bfp_bf8 and set the
residual add `dtype = bf16` so the block output stays bf16. PCC is compared
against a bf16 reference.

This test exists in scaffold form — to be fleshed out and validated on
hardware once a multi-device system is available.
"""
import pytest
import torch
from typing import List, Optional
from conftest import x86_only, get_request_kwargs
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.skip(
    reason="Scaffold — requires multi-device system + hardware validation."
)
@x86_only
@pytest.mark.parametrize("shape", [((32, 128), (128, 256), (32, 256))])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_o_proj_residual_dtype_lowering(
    shape: List[Shape],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """matmul -> reduce_scatter -> all_gather -> add(residual)."""
    act_shape, weight_shape, res_shape = shape

    def module(builder: TTIRBuilder):
        @builder.func(
            [act_shape, weight_shape, res_shape], [dtype] * 3
        )
        def o_proj_residual(
            act: Operand,
            weight: Operand,
            residual: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            mm = builder.matmul(act, weight, unit_attrs=unit_attrs)
            # TODO: thread cluster_axis / scatter_dim args once we know the
            # multi-device topology this test will run against.
            rs = builder.reduce_scatter(mm, cluster_axis=0, scatter_dim=1)
            ag = builder.all_gather(rs, cluster_axis=0, all_gather_dim=1)
            return builder.add(residual, ag, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        module,
        argument_types_string="o_proj_residual=input,parameter,input",
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["enable-activation-dtype-lowering=true"],
        pcc=0.99,
    )
