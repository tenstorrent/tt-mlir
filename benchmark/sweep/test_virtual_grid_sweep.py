# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Virtual grid dimension ordering sweep.

Tests the coworker-reported case: 32x32x18x128 vs 32x32x128x18.
Same total data, same compute, but the dimension ordering changes:
  - which axis gets tile-padding waste (18 pads to 32, 44% waste)
  - which axes participate in the virtual grid
  - per-core L1 footprint (3.7x difference observed at 32x32 batch)

We use smaller batch dims (4x4, 2x2) that fit in L1 and compare perf
across orientations and batch sizes.

At 4x4 batch:
  18x128 inner → grid 4x4x1x4 (64 virt cores, mapped to 8x8 physical)
  128x18 inner → grid 4x4x4x1 (64 virt cores, mapped to 8x8 physical)
Same core count, different shard shapes, different DMA patterns.

Usage:
  pytest benchmark/sweep/test_virtual_grid_sweep.py -m perf_sweep \
      --save-artifacts --sys-desc <path>
"""

import pytest
import torch

from conftest import get_request_kwargs
from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = [pytest.mark.frontend("ttir"), pytest.mark.perf_sweep]


def _shape_id(shape):
    return "x".join(str(d) for d in shape)


def _pipeline(extra_opts=""):
    opts = "collapse-tensors-2d=0"
    if extra_opts:
        opts = f"{opts} {extra_opts}"
    return f"ttir-to-ttmetal-pipeline{{{opts}}}"


# ---------------------------------------------------------------------------
# Case 1: Dimension ordering — 18 on penultimate vs last dim
# ---------------------------------------------------------------------------

VGRID_SHAPES = [
    # Batch 4x4: fits L1
    (4, 4, 18, 128),   # 18 on rows (padded tile), 4 tiles on cols
    (4, 4, 128, 18),   # 18 on cols (padded tile), 4 tiles on rows

    # Clean baseline (no padding waste)
    (4, 4, 64, 128),   # 2x4 inner tiles
    (4, 4, 128, 64),   # 4x2 inner tiles

    # Batch 2x2: fewer virtual cores, different mapping
    (2, 2, 18, 128),
    (2, 2, 128, 18),
    (2, 2, 64, 128),
    (2, 2, 128, 64),

    # Batch 8x8: more virtual cores, higher mapping pressure
    (8, 8, 18, 128),
    (8, 8, 128, 18),
]


@pytest.mark.parametrize("shape", VGRID_SHAPES, ids=_shape_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_vgrid_multiply(shape, target, request, device):
    """Eltwise multiply on ND tensor — profiles virtual grid mapping cost."""

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [torch.bfloat16, torch.bfloat16])
        def multiply(in0: Operand, in1: Operand, builder: TTIRBuilder):
            return builder.multiply(in0, in1)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(),
        **get_request_kwargs(request),
    )


# ---------------------------------------------------------------------------
# Case 2: Same shapes but with override-device-shape to force smaller grids
#
# TODO: Use per-op grid_override once multiply supports it.
# For now, constrain the device grid to force fewer virtual cores.
# ---------------------------------------------------------------------------

VGRID_DEVICE_OVERRIDES = [
    # (shape, device_shape, description)
    # 4x4x18x128: auto picks 64 virt cores on 8x8 phys
    ((4, 4, 18, 128), "4,4", "phys4x4"),
    ((4, 4, 18, 128), "2,2", "phys2x2"),
    # 4x4x128x18: same volume, different orientation
    ((4, 4, 128, 18), "4,4", "phys4x4"),
    ((4, 4, 128, 18), "2,2", "phys2x2"),
]


def _override_id(cfg):
    shape, dev, desc = cfg
    return f"{_shape_id(shape)}-{desc}"


@pytest.mark.parametrize("config", VGRID_DEVICE_OVERRIDES, ids=_override_id)
@pytest.mark.parametrize("target", ["ttmetal"])
def test_vgrid_multiply_constrained(config, target, request, device):
    """Eltwise multiply with constrained device grid to force fewer virt cores."""
    shape, dev_shape, _ = config

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [torch.bfloat16, torch.bfloat16])
        def multiply(in0: Operand, in1: Operand, builder: TTIRBuilder):
            return builder.multiply(in0, in1)

    compile_and_execute_ttir(
        module,
        target=target,
        device=device,
        custom_pipeline=_pipeline(f"override-device-shape={dev_shape}"),
        **get_request_kwargs(request),
    )
