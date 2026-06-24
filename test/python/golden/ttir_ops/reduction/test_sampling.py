# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
from conftest import get_request_kwargs
from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize(
    "candidates,vocab_size",
    [
        (128, 128256),  # Llama-3.x
        (64, 50272),  # OPT-125M
    ],
    ids=["llama", "opt"],
)
@pytest.mark.parametrize(
    "k_dtype",
    [torch.uint32, torch.int32],
    ids=["k_ui32", "k_si32"],
)
@pytest.mark.parametrize("target", ["ttnn", "emitc", "emitpy"])
def test_sampling(
    candidates: int,
    vocab_size: int,
    k_dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """TTIR builder test for ttir.sampling: fused top-k/p multinomial sampling.

    Exercises the full TTIR -> TTNN lowering (ttir.sampling -> ttnn.sampling,
    including the rank-2 -> rank-4 workaround). The k_si32 variant additionally
    exercises the workaround that retypes a non-uint32 k tensor to uint32
    before the kernel call.
    """
    batch = 32

    def module(builder: TTIRBuilder):
        @builder.func(
            [(batch, candidates), (batch, candidates), (batch,), (batch,), (batch,)],
            [torch.bfloat16, torch.int32, k_dtype, torch.bfloat16, torch.bfloat16],
        )
        def sampling_fn(
            vals: Operand,
            idx: Operand,
            k: Operand,
            p: Operand,
            temp: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Override index tensor with valid global vocab positions.
            valid_indices = torch.stack(
                [torch.randperm(vocab_size)[:candidates] for _ in range(batch)]
            ).to(torch.int32)
            valid_k = torch.full((batch,), candidates, dtype=k_dtype)
            valid_p = torch.ones(batch, dtype=torch.bfloat16)
            valid_temp = torch.full((batch,), 1.667, dtype=torch.bfloat16)
            builder.set_goldens(
                {idx: valid_indices, k: valid_k, p: valid_p, temp: valid_temp}, {}
            )
            return builder.sampling(vals, idx, k, p, temp, unit_attrs=unit_attrs)

    # Sampling is stochastic (multinomial); CPU golden cannot be matched
    # element-wise. Skip PCC to verify only compile+device-execute succeed.
    kwargs = get_request_kwargs(request)
    kwargs["check_pcc"] = False
    compile_and_execute_ttir(
        module,
        **kwargs,
        target=target,
        device=device,
    )
