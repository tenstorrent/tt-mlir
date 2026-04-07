# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Virtual grid DRAM transfer grid sweep.

Isolates the DRAM↔L1 transfer cost by sweeping device shapes for both
18x128 and 128x18 orientations.  The device shape controls the DRAM
writeback grid (collapsed 2D shape), which is the dominant cost.

Usage:
  pytest benchmark/sweep/test_vgrid_dram_grid_sweep.py \
      --save-artifacts --sys-desc <path> --path benchmark/sweep
"""

import pytest
import torch

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = [pytest.mark.frontend("ttir"), pytest.mark.perf_sweep]


def _pipeline(dev_shape):
    return f"ttir-to-ttmetal-pipeline{{collapse-tensors-2d=0 override-device-shape={dev_shape}}}"


SHAPES = [
    (4, 4, 18, 128),
    (4, 4, 128, 18),
]

DEV_SHAPES = [
    "8,8",
    "4,4",
    "2,8",
    "8,2",
    "2,4",
    "4,2",
]


def _shape_id(s):
    return "x".join(map(str, s))


def _dev_id(d):
    return f"dev{d.replace(',','x')}"


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
@pytest.mark.parametrize("dev_shape", DEV_SHAPES, ids=_dev_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_vgrid_dram_sweep(shape, dev_shape, target, request, device):
    """Eltwise multiply with varying device shape to change DRAM transfer grid."""

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [torch.bfloat16, torch.bfloat16])
        def multiply(in0: Operand, in1: Operand, builder: TTIRBuilder):
            return builder.multiply(in0, in1)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(dev_shape),
        **get_request_kwargs(request),
    )
