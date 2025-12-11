# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for tensor manipulation ops (reshape, etc.) via D2M lowering.
"""

import pytest
import torch
from typing import List, Tuple

from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_and_execute_ttir

pytestmark = pytest.mark.frontend("ttir")

# Test shapes: (input_shape, output_shape)
# All shapes must have the same total number of elements.
RESHAPE_SHAPES: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = [
    # Identity reshapes (same shape)
    ((64, 64), (64, 64)),
    ((3, 32, 64), (3, 32, 64)),
    # 2D -> 2D reshapes
    ((64, 64), (32, 128)),
    ((128, 64), (64, 128)),
    ((32, 128), (64, 64)),
    # 2D -> 3D reshapes
    ((96, 64), (3, 32, 64)),
    ((128, 96), (4, 32, 96)),
    ((192, 64), (6, 32, 64)),
    # 2D -> 4D reshapes
    ((192, 64), (2, 3, 32, 64)),
    ((256, 96), (2, 4, 32, 96)),
    # 3D -> 2D reshapes
    ((3, 32, 64), (96, 64)),
    ((4, 64, 32), (256, 32)),
    ((5, 32, 64), (160, 64)),
    # 3D -> 3D reshapes
    ((2, 64, 64), (4, 32, 64)),
    ((3, 32, 96), (3, 96, 32)),
    ((6, 32, 64), (3, 64, 64)),
    # 3D -> 4D reshapes
    ((6, 32, 64), (2, 3, 32, 64)),
    ((12, 64, 32), (3, 4, 64, 32)),
    # 4D -> 2D reshapes
    ((2, 3, 32, 64), (192, 64)),
    ((2, 4, 64, 32), (512, 32)),
    # 4D -> 3D reshapes
    ((2, 3, 32, 64), (6, 32, 64)),
    ((2, 4, 32, 96), (8, 32, 96)),
    # 4D -> 4D reshapes
    ((2, 3, 32, 64), (3, 2, 32, 64)),
    ((2, 2, 64, 64), (4, 1, 64, 64)),
    # 5D -> 3D reshapes
    ((2, 3, 2, 32, 64), (12, 32, 64)),
    # 3D -> 5D reshapes
    ((12, 32, 64), (2, 3, 2, 32, 64)),
    # Inner dimension changes (more complex data movement)
    ((128, 32), (64, 64)),
    ((64, 128), (128, 64)),
    ((32, 192), (96, 64)),
    ((64, 96), (96, 64)),
    ((3, 64, 32), (3, 32, 64)),
    ((2, 128, 64), (2, 64, 128)),
]


def shapes_to_id(shapes: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> str:
    """Generate a readable test ID from input/output shapes."""
    input_shape, output_shape = shapes
    return f"{input_shape}->{output_shape}"


@pytest.mark.parametrize(
    "shapes", RESHAPE_SHAPES, ids=[shapes_to_id(s) for s in RESHAPE_SHAPES]
)
@pytest.mark.parametrize("dtype", [torch.float32], ids=["f32"])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_reshape(
    shapes: Tuple[Tuple[int, ...], Tuple[int, ...]],
    dtype: torch.dtype,
    target: str,
    request,
    device,
):
    """Test reshape operation with various input/output shape combinations."""
    input_shape, output_shape = shapes

    def reshape_module(builder: TTIRBuilder):
        @builder.func([input_shape], [dtype])
        def reshape(in0, builder: TTIRBuilder, unit_attrs: List[str] = None):
            return builder.reshape(in0, output_shape, unit_attrs=unit_attrs)

    compile_and_execute_ttir(
        reshape_module,
        target=target,
        device=device,
        custom_pipeline="ttir-to-ttmetal-pipeline",
        test_base=request.node.name,
        module_dump=True,
        print_ir=False,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
