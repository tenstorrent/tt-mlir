# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import get_request_kwargs

from builder.base.builder_apis import compile_and_execute_ttir

# borrow currently constrained way to build matmul inputs:
from test_metal_matmul import create_matmul_constrained_inputs as create_matmul_inputs

# borrow currently constrained way to build reduction inputs:
from test_metal_reductions import (
    create_reductions_constrained_inputs as create_reduction_inputs,
)

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
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
    )


@pytest.mark.parametrize("m", [8])
@pytest.mark.parametrize("n", [8])
@pytest.mark.parametrize("dim_arg", [[0], [1]])
@pytest.mark.parametrize("keep_dim", [True])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_allocate_max(
    m: int, n: int, dim_arg: int, keep_dim: bool, target: str, request, device
):
    tile_size = 32
    shape = (
        m * tile_size,
        n * tile_size,
    )

    options = [
        # request the allocator to attempt to minimize stream buffer sizes
        # and reblock streams accordingly:
        f"test-buffer-size-policy=min",
    ]

    compile_and_execute_ttir(
        create_reduction_inputs(shape, "max", dim_arg, keep_dim),
        target=target,
        device=device,
        custom_pipeline=f"ttir-to-ttmetal-pipeline{{{' '.join(options)}}}",
        **get_request_kwargs(request),
    )
