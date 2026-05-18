# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""E2E test for the LM-head + argmax pattern.

Pattern D of the activation-dtype-lowering pass: matmul -> all_gather -> sum
-> all_gather -> argmax. The matmul output is lowered to bfp_bf8; a typecast
back to bf16 is inserted before argmax.

Scaffold — requires multi-device system + hardware validation.
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
@pytest.mark.parametrize("shape", [((32, 128), (128, 512))])
@pytest.mark.parametrize("dtype", [torch.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("target", ["ttnn"])
def test_lm_head_argmax_dtype_lowering(
    shape: List[Shape],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """matmul -> all_gather -> sum -> all_gather -> argmax."""
    act_shape, weight_shape = shape

    def module(builder: TTIRBuilder):
        @builder.func([act_shape, weight_shape], [dtype, dtype])
        def lm_head(
            act: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            mm = builder.matmul(act, weight, unit_attrs=unit_attrs)
            ag1 = builder.all_gather(mm, cluster_axis=0, all_gather_dim=0)
            s = builder.sum(ag1, dim_arg=[0], keep_dim=False)
            ag2 = builder.all_gather(s, cluster_axis=1, all_gather_dim=1)
            return builder.argmax(ag2, dim=1, keep_dim=True, use_multicore=True)

    compile_and_execute_ttir(
        module,
        argument_types_string="lm_head=input,parameter",
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["enable-activation-dtype-lowering=true"],
        pcc=0.99,
    )
