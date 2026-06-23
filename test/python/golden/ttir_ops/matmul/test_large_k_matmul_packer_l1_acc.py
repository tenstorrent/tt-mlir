# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional

from builder.base.builder_apis import compile_and_execute_ttir
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import Operand
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


def _matmul_module(in_lhs: torch.Tensor, in_rhs: torch.Tensor) -> Callable:
    lhs_shape = tuple(in_lhs.shape)
    rhs_shape = tuple(in_rhs.shape)

    def module(builder: TTIRBuilder):
        @builder.func([lhs_shape, rhs_shape], [torch.bfloat16, torch.bfloat16])
        def matmul(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            builder.set_goldens(inputs={in0: in_lhs, in1: in_rhs})
            return builder.matmul(in0, in1, unit_attrs=unit_attrs)

    return module


def _make_bf16_small_inputs(k: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Small-magnitude bf16 activations."""
    torch.manual_seed(seed)
    scale = 0.001
    in_lhs = (torch.randn(32, k) * scale).bfloat16()
    in_rhs = (torch.randn(k, 32) * scale).bfloat16()
    return in_lhs, in_rhs


@pytest.mark.parametrize(
    "k,pcc",
    [
        pytest.param(4096, 0.99, id="small_k"),
        pytest.param(50176, 0.99, id="large_k"),
    ],
)
def test_bf16_matmul_compute_config_passes_pcc(k: int, pcc: float, request, device):
    """Large-K bf16 matmul gets fp32_dest_acc_en and packer_l1_acc; small-K skips the fix."""
    in_lhs, in_rhs = _make_bf16_small_inputs(k)

    kwargs = get_request_kwargs(request)
    kwargs["skip_exec"] = False

    compile_and_execute_ttir(
        _matmul_module(in_lhs, in_rhs),
        **kwargs,
        target="ttnn",
        device=device,
        # Needed as the default pipeline options set fp32_dest_acc_en=true.
        pipeline_options=["compute-cfg-fp32-dest-acc-en=false"],
        pcc=pcc,
    )
