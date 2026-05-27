# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from conftest import get_request_kwargs

from builder.base.builder_apis import compile_and_execute_ttir
from d2m.test_matmul import create_matmul_constrained_inputs

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("target", ["ttmetal"])
def test_matmul_auto_mn_policy(
    target,
    request,
    device,
):
    options = [
        "test-buffer-size-policy=auto-mn",
        "num-stream-buffers=1",
        "use-tile-matmul=false",
    ]
    compile_and_execute_ttir(
        create_matmul_constrained_inputs((512, 32), (32, 2048), torch.bfloat16),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
    )
