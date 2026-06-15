# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs
from typing import Optional, List


pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "shape,k,dim",
    [
        pytest.param((32, 64), 16, -1, id="32x64_k16_dim1"),
        pytest.param((32, 128), 16, -1, id="32x128_k16_dim1"),
        pytest.param((32, 256), 16, -1, id="32x256_k16_dim1"),
        pytest.param((32, 64), 48, -1, id="32x64_k48_dim1"),
        pytest.param((32, 64), 64, -1, id="32x64_k64_dim1"),
        pytest.param((32, 256), 64, -1, id="32x256_k64_dim1"),
        pytest.param((64, 32), 16, 0, id="64x32_k16_dim0"),
        pytest.param((128, 32), 16, 0, id="128x32_k16_dim0"),
        pytest.param((256, 32), 16, 0, id="256x32_k16_dim0"),
        pytest.param((64, 32), 48, 0, id="64x32_k48_dim0"),
        pytest.param((64, 32), 64, 0, id="64x32_k64_dim0"),
        pytest.param((256, 32), 64, 0, id="256x32_k64_dim0"),
    ],
)
def test_topk(shape, k, dim, target, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def topk(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.topk(
                in0,
                k=k,
                dim=dim,
                largest=True,
                sorted=False,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["override-device-shape=1,1"],
        save_artifacts=True,
    )
