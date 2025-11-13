# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from builder.base.builder_utils import compile_and_execute_ttir

# borrow currently constrained way to build matmul inputs:
from test_metal_matmul import create_matmul_constrained_inputs as create_matmul_inputs

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("m", [1])
@pytest.mark.parametrize("k", [4, 8])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_allocate_matmul(m: int, k: int, n: int, target: str, request, device):
    tile_size = 32
    lhs = (
        m * tile_size,
        k * tile_size,
    )
    rhs = (
        k * tile_size,
        n * tile_size,
    )

    options = [
        f"override-device-shape=1,1",
        f"num-stream-buffers=1",
        # request the allocator to attempt to minimize stream buffer sizes
        # and reblock streams accordingly:
        f"test-buffer-size-policy=min",
    ]

    compile_and_execute_ttir(
        create_matmul_inputs(lhs, rhs),
        [lhs, rhs],
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        test_base=request.node.name,
        print_ir=True,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
