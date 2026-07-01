# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir
from conftest import get_request_kwargs
from typing import List, Optional


pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "shape,start,step,dtype",
    [
        # Single tile (1x1 tile grid)
        pytest.param((1, 32), 0, 1, torch.float32, id="1x32_s0_step1_f32"),
        pytest.param((32, 32), 0, 1, torch.float32, id="32x32_s0_step1_f32"),
        # Non-zero start / step > 1
        pytest.param((1, 32), 4, 2, torch.float32, id="1x32_s4_step2_f32"),
        pytest.param((1, 32), 0, 1, torch.bfloat16, id="1x32_s0_step1_bf16"),
        # Multi-tile along reduction dim (stresses tile-grid layout)
        pytest.param((1, 64), 0, 1, torch.float32, id="1x64_s0_step1_f32"),
        pytest.param((1, 128), 0, 1, torch.float32, id="1x128_s0_step1_f32"),
        pytest.param((1, 256), 0, 1, torch.float32, id="1x256_s0_step1_f32"),
        pytest.param((1, 1024), 0, 1, torch.float32, id="1x1024_s0_step1_f32"),
        # Multi-tile non-reduction dim (ht > 1)
        pytest.param((64, 32), 0, 1, torch.float32, id="64x32_s0_step1_f32"),
        pytest.param((128, 32), 0, 1, torch.float32, id="128x32_s0_step1_f32"),
        # Non-power-of-2 tile counts (ragged)
        pytest.param((1, 96), 0, 1, torch.float32, id="1x96_s0_step1_f32"),
        pytest.param((1, 160), 0, 1, torch.float32, id="1x160_s0_step1_f32"),
        # Large multi-tile grid
        pytest.param((32, 256), 0, 1, torch.float32, id="32x256_s0_step1_f32"),
        pytest.param((32, 1024), 0, 1, torch.float32, id="32x1024_s0_step1_f32"),
    ],
)
def test_arange(shape, start, step, dtype, target, request, device):
    """Test D2M arange lowering with various shapes, dtypes, and strides.

    arange_dimension is always the last dim (dim=1 for 2-D tensors).
    end is derived from start/step/shape so the arange fills the tensor exactly.
    """
    if dtype == torch.int32:
        pytest.xfail(
            reason="No LLK for i32 scalar multiply; see https://github.com/tenstorrent/tt-mlir/issues/7946"
        )

    num_elements = shape[1]
    end = start + num_elements * step
    arange_dimension = 1

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def arange(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.arange(
                shape=list(shape),
                dtype=dtype,
                start=start,
                end=end,
                step=step,
                arange_dimension=arange_dimension,
                unit_attrs=unit_attrs,
            )

    compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        target=target,
        device=device,
        pipeline_options=["override-device-shape=1,1"],
        atol=1e-6,
        check_atol=True,
        save_artifacts=True,
    )
