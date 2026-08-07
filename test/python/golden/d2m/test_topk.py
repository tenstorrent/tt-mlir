# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_ttir_to_flatbuffer
from builder.base.builder_runtime import execute_fb, check_outputs
from builder.base.builder_apis import get_artifact_dir
from golden.mapping import GoldenMapTensor
from conftest import get_request_kwargs
from typing import Optional, List, Tuple


pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)


def _verify_topk_outputs(input_tensor, golden_topk, dim, output_tensors):
    """PCC-checks topk device outputs against the golden via check_outputs.

    Indices are checked order-robustly: gather the values they point to out of
    the input and PCC-compare those against the device's values.
    """
    prog = output_tensors["program_0"]
    device_values = prog["device_output_0"][0]

    check_outputs(
        golden_topk.values,
        device_values,
        "topk_values",
        0.99,
        1e-08,
        1e-05,
        check_pcc=True,
        check_atol=False,
        check_rtol=False,
    )

    device_indices = prog["device_output_1"][0].long()
    gathered_values = torch.gather(input_tensor.float(), dim, device_indices)
    check_outputs(
        device_values.float(),
        gathered_values,
        "topk_gathered_values",
        0.99,
        1e-08,
        1e-05,
        check_pcc=True,
        check_atol=False,
        check_rtol=False,
    )


def _random_input(shape: Tuple[int, ...], k: int, dim: int) -> torch.Tensor:
    return torch.randn(shape) * 50


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
    tensor = torch.randn(shape) * 0.01  # near-zero baseline.
    tile_size = 32
    num_tiles = shape[dim] // tile_size

    if pattern == "first_tiles":
        # Top values in the first 2 tiles.
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


SINGLE_CORE_TOPK_SHAPES = [
    # Single-tile non-target dim; the reduction dim still merges and rebuilds.
    pytest.param((32, 256), 64, -1, id="32x256_k64_dim1"),
    pytest.param((256, 32), 64, 0, id="256x32_k64_dim0"),
    # Large target dim (many tiles in reduction), k > 32.
    pytest.param((32, 1376), 64, -1, id="32x1376_k64_dim1"),
    pytest.param((1376, 32), 64, 0, id="1376x32_k64_dim0"),
    # Ragged (non-power-of-2 tile count)
    pytest.param((32, 96), 16, -1, id="32x96_k16_dim1"),
    pytest.param((1208, 32), 16, 0, id="1208x32_k16_dim0"),
    # Multi-tile non-target dim (ht>1 for dim=1, wt>1 for dim=0)
    pytest.param((96, 446), 32, -1, id="96x446_k32_dim1"),
    pytest.param((383, 96), 63, 0, id="383x96_k63_dim0"),
]

MULTI_CORE_TOPK_SHAPES = [
    # Non-target dim is a single tile (32), on dim 0; target dim (dim=1) is
    # any multiple of 32.
    pytest.param((32, 5504), 16, -1, id="32x5504_k16_dim1"),
    pytest.param((32, 96256), 16, -1, id="32x96256_k16_dim1"),
    pytest.param((35, 7639), 16, -1, id="35x7639_k16_dim1"),
    # Transposed equivalents: non-target dim is a single tile (32), on dim 1;
    # target dim (dim=0) is any multiple of 32.
    pytest.param((8192, 32), 16, 0, id="8192x32_k16_dim0"),
    pytest.param((96256, 32), 16, 0, id="96256x32_k16_dim0"),
    pytest.param((7639, 35), 16, 0, id="7639x35_k16_dim0"),
    # k > 32: each core's partial spans two reduction tiles.
    pytest.param((32, 8192), 48, -1, id="32x8192_k48_dim1"),
    pytest.param((32, 96256), 64, -1, id="32x96256_k64_dim1"),
    pytest.param((8192, 32), 48, 0, id="8192x32_k48_dim0"),
    pytest.param((96256, 32), 64, 0, id="96256x32_k64_dim0"),
    # data parallel
    pytest.param((32, 8192), 16, 0, id="32x8192_k16_dim0"),
    pytest.param((8192, 32), 16, -1, id="8192x32_k16_dim1"),
]

# Shapes worth running against inputs whose top values sit in specific tiles,
# which stresses the merge tree.
TILE_DIST_SHAPES = [
    # pow2 tile count, dim=0
    pytest.param((256, 32), 64, 0, id="256x32_k64_dim0"),
    # Large reduction dim, still <= 1024.
    pytest.param((32, 1024), 64, -1, id="32x1024_k64_dim1"),
    # Ragged (non-power-of-2): odd tile count
    pytest.param((32, 96), 16, -1, id="32x96_k16_dim1"),  # 3 tiles, odd
    # Multi-tile non-target dim
    pytest.param((64, 256), 64, -1, id="64x256_k64_dim1"),  # ht=2, large-k
    # Multi-core: non-target dim is a single tile (32), on dim 0.
    pytest.param((32, 2080), 16, -1, id="32x2080_k16_dim1"),  # 65 tiles, odd
]


def _cases(shapes, input_fn, id_suffix=""):
    return [
        pytest.param(
            *shape.values, input_fn, id=shape.id + id_suffix, marks=shape.marks
        )
        for shape in shapes
    ]


TOPK_CASES = [
    *_cases(SINGLE_CORE_TOPK_SHAPES, _random_input),
    *_cases(MULTI_CORE_TOPK_SHAPES, _random_input),
    *[
        case
        for pattern in ("first_tiles", "last_tiles", "strided")
        for case in _cases(
            TILE_DIST_SHAPES,
            lambda shape, k, dim, pattern=pattern: _build_tile_distribution_input(
                shape, k, dim, pattern
            ),
            f"_{pattern}",
        )
    ],
]


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("shape,k,dim,input_fn", TOPK_CASES)
def test_topk(shape, k, dim, input_fn, target, request, device):
    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def topk(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            values = builder.topk(
                in0,
                k=k,
                dim=dim,
                largest=True,
                sorted=False,
                unit_attrs=unit_attrs,
            )
            indices = builder.topk_indices(values)
            return values, indices

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
        save_artifacts=True,
        print_ir=kwargs.get("print_ir", False),
    )

    input_tensor = input_fn(shape, k, dim)
    golden_topk = torch.topk(input_tensor, k=k, dim=dim, largest=True)

    mesh_shape = (1, 1)
    io_goldens[0]["input_0"] = GoldenMapTensor({0: input_tensor}, mesh_shape=mesh_shape)
    io_goldens[0]["output_0"] = GoldenMapTensor(
        {0: golden_topk.values}, mesh_shape=mesh_shape
    )
    # Device index dtype must match, though the raw index is ignored below.
    io_goldens[0]["output_1"] = GoldenMapTensor(
        {0: golden_topk.indices.to(io_goldens[0]["output_1"].dtype)},
        mesh_shape=mesh_shape,
    )

    # Positional PCC is invalid for unsorted values and tie-unstable indices;
    # _verify_topk_outputs checks both order-robustly instead.
    _, output_tensors = execute_fb(
        compiled_bin,
        input_output_goldens=io_goldens,
        intermediate_goldens=intermediate_goldens,
        device=device,
        pcc=0.99,
        check_pcc=False,
        save_artifacts=True,
        artifact_dir=artifact_dir,
    )

    _verify_topk_outputs(input_tensor, golden_topk, dim, output_tensors)
