# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_ttir_to_flatbuffer
from builder.base.builder_runtime import execute_fb
from builder.base.builder_apis import get_artifact_dir
from golden import get_atol_rtol_pcc
from golden.mapping import GoldenMapTensor
from conftest import get_request_kwargs
from typing import Optional, List, Tuple


pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)


def _verify_topk_outputs(input_tensor, golden_topk, dim, output_tensors):
    """PCC-checks topk device outputs against the golden.

    Values are compared order-robustly. Indices are validated on the values
    they point to, gathered from the input, rather than raw positions.
    """
    d = dim % input_tensor.ndim
    prog = output_tensors["program_0"]
    device_values = prog["device_output_0"][0]
    device_indices = prog["device_output_1"][0].long()

    dv_sorted, _ = torch.sort(device_values, dim=d)
    gv_sorted, _ = torch.sort(golden_topk.values, dim=d)
    _, _, values_pcc = get_atol_rtol_pcc(gv_sorted, dv_sorted, 1e-08, 1e-05)
    assert values_pcc >= 0.99, f"values PCC {values_pcc} < 0.99"

    # The pointed-to values are compared in bf16.
    device_gathered = torch.gather(input_tensor, d, device_indices).to(torch.bfloat16)
    golden_gathered = torch.gather(input_tensor, d, golden_topk.indices).to(
        torch.bfloat16
    )
    dg_sorted, _ = torch.sort(device_gathered, dim=d)
    gg_sorted, _ = torch.sort(golden_gathered, dim=d)
    _, _, index_value_pcc = get_atol_rtol_pcc(gg_sorted, dg_sorted, 1e-08, 1e-05)
    assert index_value_pcc >= 0.99, f"index-value PCC {index_value_pcc} < 0.99"


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
        pytest.param((32, 1376), 16, -1, id="32x1376_k16_dim1"),
        pytest.param((1024, 32), 64, 0, id="1024x32_k64_dim0"),
        # Ragged (non-power-of-2 tile count)
        pytest.param((32, 96), 16, -1, id="32x96_k16_dim1"),
        pytest.param((1536, 32), 16, 0, id="1536x32_k16_dim0"),
        # Multi-tile non-target dim (ht>1 for dim=1, wt>1 for dim=0)
        pytest.param((96, 446), 32, -1, id="96x446_k32_dim1"),
        pytest.param((383, 96), 63, 0, id="383x96_k63_dim0"),
    ],
)
def test_topk(shape, k, dim, target, request, device):
    input_tensor = torch.randn(shape) * 50
    golden_topk = torch.topk(input_tensor, k=k, dim=dim, largest=True)

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
            builder.set_goldens(
                {in0: input_tensor},
                {
                    values: golden_topk.values,
                    indices: golden_topk.indices.to(torch.uint16),
                },
            )
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
        pipeline_options=["override-device-shape=1,1"],
        save_artifacts=True,
    )

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

    adversarial_input = _build_tile_distribution_input(shape, k, dim, pattern)
    golden_topk = torch.topk(adversarial_input, k=k, dim=dim, largest=True)

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
            builder.set_goldens(
                {in0: adversarial_input},
                {
                    values: golden_topk.values,
                    indices: golden_topk.indices.to(torch.uint16),
                },
            )
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
        pipeline_options=["override-device-shape=1,1"],
        save_artifacts=True,
    )

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

    _verify_topk_outputs(adversarial_input, golden_topk, dim, output_tensors)
