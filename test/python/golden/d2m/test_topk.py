# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import (
    compile_and_execute_ttir,
    compile_ttir_to_flatbuffer,
)
from builder.base.builder_runtime import execute_fb
from builder.base.builder_apis import get_artifact_dir
from golden.mapping import GoldenMapTensor
from conftest import get_request_kwargs
from typing import Optional, List, Tuple


pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "shape,k,dim",
    [
        # Single-tile target dim, dim=1 and dim=0
        pytest.param((32, 256), 16, -1, id="32x256_k16_dim1"),
        pytest.param((32, 256), 64, -1, id="32x256_k64_dim1"),
        pytest.param((256, 32), 16, 0, id="256x32_k16_dim0"),
        pytest.param((256, 32), 64, 0, id="256x32_k64_dim0"),
        # Large target dim (many tiles in reduction)
        pytest.param((32, 1024), 16, -1, id="32x1024_k16_dim1"),
        pytest.param((1024, 32), 64, 0, id="1024x32_k64_dim0"),
        # Ragged (non-power-of-2 tile count)
        pytest.param((32, 96), 16, -1, id="32x96_k16_dim1"),
        pytest.param((544, 32), 16, 0, id="544x32_k16_dim0"),
        # Multi-tile non-target dim (ht>1 for dim=1, wt>1 for dim=0)
        pytest.param((128, 384), 32, -1, id="128x384_k32_dim1"),
        pytest.param((384, 96), 64, 0, id="384x96_k64_dim0"),
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


def _build_tile_distribution_input(
    shape: Tuple[int, ...], k: int, dim: int, pattern: str
) -> torch.Tensor:
    """Build an input tensor where top-k values are concentrated in specific tiles.

    Each tile is 32 elements along the topk dim.  By placing large values only
    in certain tiles we force the merge tree to propagate them correctly.
    """
    # Normalize negative dim so index comparisons work correctly.
    if dim < 0:
        dim = len(shape) + dim
    tensor = torch.randn(shape) * 0.01  # near-zero baseline
    tile_size = 32
    num_tiles = shape[dim] // tile_size

    if pattern == "first_tiles":
        # Top values concentrated in the first 2 tiles (indices 0..63).
        slices = [slice(None)] * len(shape)
        slices[dim] = slice(0, min(2 * tile_size, shape[dim]))
        tensor[tuple(slices)] = (
            torch.randn(
                *[
                    s if i != dim else min(2 * tile_size, shape[dim])
                    for i, s in enumerate(shape)
                ]
            ).abs()
            + 10.0
        )

    elif pattern == "last_tiles":
        # Top values concentrated in the last 2 tiles.
        slices = [slice(None)] * len(shape)
        start = max(0, shape[dim] - 2 * tile_size)
        slices[dim] = slice(start, shape[dim])
        tensor[tuple(slices)] = (
            torch.randn(
                *[s if i != dim else shape[dim] - start for i, s in enumerate(shape)]
            ).abs()
            + 10.0
        )

    elif pattern == "strided":
        # Top values in every other tile — forces merge to pull from
        # non-adjacent tiles at every level of the reduction tree.
        for t in range(0, num_tiles, 2):
            slices = [slice(None)] * len(shape)
            slices[dim] = slice(t * tile_size, (t + 1) * tile_size)
            tile_shape = [s if i != dim else tile_size for i, s in enumerate(shape)]
            tensor[tuple(slices)] = torch.randn(*tile_shape).abs() + 10.0

    return tensor


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("pattern", ["first_tiles", "last_tiles", "strided"])
@pytest.mark.parametrize(
    "shape,k,dim",
    [
        # pow2 tile count, dim=1 and dim=0
        pytest.param((32, 256), 16, -1, id="32x256_k16_dim1"),
        pytest.param((256, 32), 64, 0, id="256x32_k64_dim0"),
        # Large reduction dim
        pytest.param((32, 1024), 64, -1, id="32x1024_k64_dim1"),
        # Ragged (non-power-of-2): odd tile count, even tile count
        pytest.param((32, 96), 16, -1, id="32x96_k16_dim1"),  # 3 tiles, odd
        pytest.param((544, 32), 16, 0, id="544x32_k16_dim0"),  # 17 tiles, odd
        # Multi-tile non-target dim
        pytest.param((64, 256), 64, -1, id="64x256_k64_dim1"),  # ht=2, large-k
    ],
)
def test_topk_tile_distribution(shape, k, dim, pattern, target, request, device):
    """Run topk with hand-crafted inputs that concentrate top values in
    specific tiles, stressing the merge-tree reduction logic."""

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

    kwargs = get_request_kwargs(request)
    artifact_dir = get_artifact_dir(
        kwargs["output_root"], "TTIRBuilder", kwargs["test_base"], make_dir=True
    )

    (
        builder,
        compiled_bin,
        io_goldens,
        intermediate_goldens,
    ) = compile_ttir_to_flatbuffer(
        module,
        system_desc_path=kwargs["system_desc_path"],
        artifact_dir=artifact_dir,
        target=target,
        pipeline_options=["override-device-shape=1,1"],
        save_artifacts=True,
    )

    # Replace the random input with our adversarial tensor and recompute the
    # expected output so the golden comparison is valid.
    adversarial_input = _build_tile_distribution_input(shape, k, dim, pattern)
    golden_topk = torch.topk(adversarial_input, k=k, dim=dim, largest=True)

    mesh_shape = (1, 1)
    io_goldens[0]["input_0"] = GoldenMapTensor(
        {0: adversarial_input}, mesh_shape=mesh_shape
    )
    io_goldens[0]["output_0"] = GoldenMapTensor(
        {0: golden_topk.values}, mesh_shape=mesh_shape
    )
    io_goldens[0]["output_1"] = GoldenMapTensor(
        {0: golden_topk.indices}, mesh_shape=mesh_shape
    )

    execute_fb(
        compiled_bin,
        input_output_goldens=io_goldens,
        intermediate_goldens=intermediate_goldens,
        device=device,
        pcc=0.99,
        check_pcc=True,
        save_artifacts=True,
        artifact_dir=artifact_dir,
    )
