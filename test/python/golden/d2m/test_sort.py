# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_ttir_to_flatbuffer, get_artifact_dir
from builder.base.builder_runtime import execute_fb, check_outputs
from golden.mapping import GoldenMapTensor
from conftest import get_request_kwargs
from typing import Optional, List


pytestmark = pytest.mark.frontend("ttir")
torch.manual_seed(0)

# builder.sort declares its indices result as int64, but TTIR's
# ElementTypeNormalization (which runs before TTIRToD2M) folds 64-bit integers
# down to Int32, so the device output is int32. Test-side goldens must match
# this or execute_fb dtype-mismatches the index output.
SORT_INDEX_DTYPE = torch.int32


def _verify_sort_outputs(input_tensor, golden_sort, dim, output_tensors, pcc=0.99):
    """PCC-checks sort device outputs against the golden via check_outputs.

    Values go through the same check_outputs() PCC engine that execute_fb uses,
    so a failure raises TTBuilderGoldenException.

    Indices are validated order-robustly: the values they point to are gathered
    from the original input and PCC-compared against the device's sorted
    values. Comparing indices positionally against the golden would spuriously
    fail on duplicate values, where any of the tied indices is correct.
    """
    prog = output_tensors["program_0"]
    device_values = prog["device_output_0"][0]

    check_outputs(
        golden_sort.values,
        device_values,
        "sort_values",
        pcc,
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
        "sort_gathered_values",
        pcc,
        1e-08,
        1e-05,
        check_pcc=True,
        check_atol=False,
        check_rtol=False,
    )


# The bitonic network needs a power-of-two tile count along the sort dim, and
# TTIRToD2M pads up to one. Ragged and single-tile cases exercise that padding
# (and the mask fill that keeps it behind the real data).
SINGLE_CORE_SORT_SHAPES = [
    # Exactly the two tiles topk_local_sort spans: no merge network at all.
    pytest.param((64, 32), 0, id="64x32_dim0"),
    pytest.param((32, 32), -1, id="32x32_dim1"),
    # 8 tiles: a real merge network
    pytest.param((32, 256), -1, id="32x256_dim1"),
    pytest.param((256, 32), 0, id="256x32_dim0"),
    # Ragged: 3 tiles padded to 4, plus a sub-tile remainder.
    pytest.param((32, 1024), -1, id="32x1024_dim1"),
    pytest.param((1024, 32), 0, id="1024x32_dim0"),
    # Multi-tile non-target dim on one core: several independent sorts.
    pytest.param((96, 446), -1, id="96x446_dim1"),
    pytest.param((383, 96), 0, id="383x96_dim0"),
]


def _run_sort(shape, dim, descending, target, request, device):
    # Computed before `module` so the nested `sort` closure can capture it.
    input_tensor = torch.randn(shape) * 50
    golden_sort = torch.sort(input_tensor, dim=dim, descending=descending)

    def module(builder: TTIRBuilder):
        @builder.func([shape], [torch.float32])
        def sort(
            in0: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            values, indices = builder.sort(
                in0,
                dim=dim,
                descending=descending,
                stable=False,
                unit_attrs=unit_attrs,
            )
            builder.set_goldens(
                {in0: input_tensor},
                {
                    values: golden_sort.values,
                    indices: golden_sort.indices.to(SORT_INDEX_DTYPE),
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

    mesh_shape = (1, 1)
    io_goldens[0]["input_0"] = GoldenMapTensor({0: input_tensor}, mesh_shape=mesh_shape)
    io_goldens[0]["output_0"] = GoldenMapTensor(
        {0: golden_sort.values}, mesh_shape=mesh_shape
    )
    # Match device index dtype to avoid an execute_fb dtype mismatch; the raw
    # index values are checked order-robustly below instead.
    io_goldens[0]["output_1"] = GoldenMapTensor(
        {0: golden_sort.indices.to(SORT_INDEX_DTYPE)}, mesh_shape=mesh_shape
    )

    # execute_fb's positional PCC is invalid for tie-unstable indices; both
    # outputs are PCC-checked order-robustly in _verify_sort_outputs.
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

    _verify_sort_outputs(input_tensor, golden_sort, dim, output_tensors)


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("descending", [False, True], ids=["asc", "desc"])
@pytest.mark.parametrize("shape,dim", SINGLE_CORE_SORT_SHAPES)
def test_sort_single_core(shape, dim, descending, target, request, device):
    _run_sort(shape, dim, descending, target, request, device)
