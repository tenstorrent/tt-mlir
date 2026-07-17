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
from builder.base.builder_runtime import execute_fb, check_outputs
from builder.base.builder_apis import get_artifact_dir
from golden import get_atol_rtol_pcc
from golden.mapping import GoldenMapTensor
from conftest import get_request_kwargs
from typing import Optional, List, Tuple


pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)


def _verify_topk_outputs(
    input_tensor,
    golden_topk,
    dim,
    output_tensors,
    pcc=0.99,
    check_gathered=True,
    check_indices=False,
):
    """PCC-checks topk device outputs against the golden via check_outputs.

    Values go through the same check_outputs() PCC engine that execute_fb
    uses, so a failure raises TTBuilderGoldenException.

    check_gathered=True additionally validates the device's indices by
    gathering the values they point to from the original input and
    PCC-comparing those against the device's topk values. This is
    order-robust (doesn't depend on positional/tie-break ordering). Used on
    single-core tests.

    check_indices=True instead PCC-compares the device's indices positionally
    against the golden indices. Used on multi-core tests.
    """
    d = dim % input_tensor.ndim
    prog = output_tensors["program_0"]
    device_values = prog["device_output_0"][0]

    check_outputs(
        golden_topk.values,
        device_values,
        "topk_values",
        pcc,
        1e-08,
        1e-05,
        check_pcc=True,
        check_atol=False,
        check_rtol=False,
    )

    device_indices = prog["device_output_1"][0].long()

    if check_gathered:
        gathered_values = torch.gather(input_tensor.float(), dim, device_indices)
        check_outputs(
            device_values.float(),
            gathered_values,
            "topk_gathered_values",
            pcc,
            1e-08,
            1e-05,
            check_pcc=True,
            check_atol=False,
            check_rtol=False,
        )

    if check_indices:
        check_outputs(
            golden_topk.indices.float(),
            device_indices.float(),
            "topk_indices",
            pcc,
            1e-08,
            1e-05,
            check_pcc=True,
            check_atol=False,
            check_rtol=False,
        )


SINGLE_CORE_TOPK_SHAPES = [
    # Single-tile reduction dim (exactly 32 elements): no merge/rebuild, just
    # local_sort.
    pytest.param((32, 28), 16, -1, id="32x28_k16_dim1"),
    pytest.param((32, 32), 32, -1, id="32x32_k32_dim1"),
    pytest.param((32, 32), 16, 0, id="32x32_k16_dim0"),
    # Single-tile target dim, dim=1 and dim=0
    pytest.param((32, 256), 16, -1, id="32x256_k16_dim1"),
    pytest.param((32, 256), 64, -1, id="32x256_k64_dim1"),
    pytest.param((256, 32), 16, 1, id="256x32_k16_dim1"),
    pytest.param((256, 32), 64, 0, id="256x32_k64_dim0"),
    # Large target dim (many tiles in reduction
    pytest.param((32, 1024), 64, -1, id="32x1024_k64_dim1"),
    pytest.param((1024, 32), 64, 0, id="1024x32_k64_dim0"),
    # Ragged (non-power-of-2 tile count)
    pytest.param((32, 96), 16, -1, id="32x96_k16_dim1"),
    # Multi-tile non-target dim (ht>1 for dim=1, wt>1 for dim=0)
    pytest.param((96, 446), 32, -1, id="96x446_k32_dim1"),
    pytest.param((383, 96), 63, 0, id="383x96_k63_dim0"),
]

MULTI_CORE_TOPK_SHAPES = [
    # Non-target dim is a single tile (32), on dim 0; target dim (dim=1) is
    # any multiple of 32.
    pytest.param((32, 8192), 16, -1, id="32x8192_k16_dim1"),
    pytest.param((32, 23552), 16, -1, id="32x23552_k16_dim1"),
    pytest.param((32, 32768), 16, -1, id="32x32768_k16_dim1"),
    pytest.param((32, 33792), 16, -1, id="32x33792_k16_dim1"),
    pytest.param((32, 65536), 16, -1, id="32x65536_k16_dim1"),
]


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("shape,k,dim", SINGLE_CORE_TOPK_SHAPES)
def test_topk_single_core(shape, k, dim, target, request, device):
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

    # Recompute golden from a fresh random input for a valid comparison.
    input_tensor = torch.randn(shape) * 50
    golden_topk = torch.topk(input_tensor, k=k, dim=dim, largest=True)

    mesh_shape = (1, 1)
    io_goldens[0]["input_0"] = GoldenMapTensor({0: input_tensor}, mesh_shape=mesh_shape)
    io_goldens[0]["output_0"] = GoldenMapTensor(
        {0: golden_topk.values}, mesh_shape=mesh_shape
    )
    # Match device index dtype to avoid execute_fb dtype mismatch; raw index is ignored.
    io_goldens[0]["output_1"] = GoldenMapTensor(
        {0: golden_topk.indices.to(torch.uint16)}, mesh_shape=mesh_shape
    )

    # execute_fb's positional PCC is invalid for unsorted values / tie-unstable
    # indices; both are PCC-checked order-robustly in _verify_topk_outputs.
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


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("shape,k,dim", MULTI_CORE_TOPK_SHAPES)
def test_topk_multi_core(shape, k, dim, target, request, device):
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
        # pipeline_options=["override-device-shape=1,1"],
        save_artifacts=True,
        print_ir=kwargs.get("print_ir", False),
    )

    # Recompute golden from a fresh random input for a valid comparison.
    input_tensor = torch.randn(shape) * 50
    golden_topk = torch.topk(input_tensor, k=k, dim=dim, largest=True)

    mesh_shape = (1, 1)
    io_goldens[0]["input_0"] = GoldenMapTensor({0: input_tensor}, mesh_shape=mesh_shape)
    io_goldens[0]["output_0"] = GoldenMapTensor(
        {0: golden_topk.values}, mesh_shape=mesh_shape
    )
    # Match device index dtype to avoid execute_fb dtype mismatch; raw index is ignored.
    io_goldens[0]["output_1"] = GoldenMapTensor(
        {0: golden_topk.indices.to(torch.uint16)}, mesh_shape=mesh_shape
    )

    # execute_fb's positional PCC is invalid for unsorted values / tie-unstable
    # indices; both are PCC-checked order-robustly in _verify_topk_outputs.
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

    _verify_topk_outputs(
        input_tensor,
        golden_topk,
        dim,
        output_tensors,
        check_gathered=True,
        check_indices=False,
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


SINGLE_CORE_TILE_DIST_SHAPES = [
    # pow2 tile count, dim=1 and dim=0
    pytest.param((32, 256), 16, -1, id="32x256_k16_dim1"),
    pytest.param((256, 32), 64, 0, id="256x32_k64_dim0"),
    # Large reduction dim, still <= 1024.
    pytest.param((32, 1024), 64, -1, id="32x1024_k64_dim1"),
    # Ragged (non-power-of-2): odd tile count
    pytest.param((32, 96), 16, -1, id="32x96_k16_dim1"),  # 3 tiles, odd
    # Multi-tile non-target dim
    pytest.param((64, 256), 64, -1, id="64x256_k64_dim1"),  # ht=2, large-k
]

MULTI_CORE_TILE_DIST_SHAPES = [
    # Non-target dim is a single tile (32), on dim 0; target dim (dim=1) is
    # any multiple of 32.
    pytest.param((32, 544), 16, -1, id="32x544_k16_dim1"),  # 17 tiles, odd
]


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("pattern", ["first_tiles", "last_tiles", "strided"])
@pytest.mark.parametrize("shape,k,dim", SINGLE_CORE_TILE_DIST_SHAPES)
def test_topk_tile_distribution_single_core(
    shape, k, dim, pattern, target, request, device
):
    """Run topk with hand-crafted inputs that concentrate top values in
    specific tiles, stressing the merge-tree reduction logic."""

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
    # Match device index dtype to avoid execute_fb dtype mismatch; raw index is ignored.
    io_goldens[0]["output_1"] = GoldenMapTensor(
        {0: golden_topk.indices.to(torch.uint16)}, mesh_shape=mesh_shape
    )

    # execute_fb's positional PCC is invalid for unsorted values / tie-unstable
    # indices; both are PCC-checked order-robustly in _verify_topk_outputs.
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


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("pattern", ["first_tiles", "last_tiles", "strided"])
@pytest.mark.parametrize("shape,k,dim", MULTI_CORE_TILE_DIST_SHAPES)
def test_topk_tile_distribution_multi_core(
    shape, k, dim, pattern, target, request, device
):
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
        save_artifacts=True,
        print_ir=kwargs.get("print_ir", False),
    )

    # Replace the random input with our adversarial tensor and recompute the
    # expected output so the golden comparison is valid.
    adversarial_input = _build_tile_distribution_input(shape, k, dim, pattern)
    golden_output = torch.topk(adversarial_input, k=k, dim=dim, largest=True).values

    mesh_shape = (1, 1)
    io_goldens[0]["input_0"] = GoldenMapTensor(
        {0: adversarial_input}, mesh_shape=mesh_shape
    )
    io_goldens[0]["output_0"] = GoldenMapTensor(
        {0: golden_output}, mesh_shape=mesh_shape
    )

    # execute_fb's positional PCC is invalid for unsorted values / tie-unstable
    # indices; both are PCC-checked order-robustly in _verify_topk_outputs.
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

    _verify_topk_outputs(
        adversarial_input,
        golden_topk,
        dim,
        output_tensors,
        check_gathered=True,
        check_indices=False,
    )
