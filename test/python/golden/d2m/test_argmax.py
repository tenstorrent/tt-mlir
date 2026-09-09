# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttmlir.ir import *

from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_apis import compile_ttir_to_flatbuffer, get_artifact_dir
from builder.base.builder_runtime import execute_fb
from conftest import get_request_kwargs

pytestmark = pytest.mark.frontend("ttir")


def _verify_argmax_output(
    input_tensor, golden_indices, dim_arg, keep_dim, output_tensors
):
    """Validates the output of argmax based on the values only. That is, if the values at the golden and device indices are equal, then the test passes. Allows ties to happen."""
    device_indices = output_tensors["program_0"]["device_output_0"][0].long()

    if dim_arg is None:
        # Full reduction: indices are into the flattened tensor.
        flat = input_tensor.flatten()
        device_vals = flat[device_indices.flatten()]
        golden_vals = flat[golden_indices.flatten()]
    else:
        d = dim_arg[0] % input_tensor.ndim
        # gather needs the reduced axis kept so shapes line up with the input.
        dev_idx = device_indices if keep_dim else device_indices.unsqueeze(d)
        gold_idx = golden_indices if keep_dim else golden_indices.unsqueeze(d)
        device_vals = torch.gather(input_tensor, d, dev_idx)
        golden_vals = torch.gather(input_tensor, d, gold_idx)

    mismatches = (device_vals != golden_vals).nonzero()
    assert mismatches.numel() == 0, (
        f"{mismatches.shape[0]} position(s) where the device index does not "
        f"select a maximal value; first few "
        f"device={device_vals[device_vals != golden_vals][:8].tolist()} "
        f"golden={golden_vals[device_vals != golden_vals][:8].tolist()}"
    )


def _run_argmax(
    shape, dim_arg, keep_dim, dtype, target, request, device, pipeline_options
):
    input_tensor = torch.randn(shape, dtype=dtype)
    if dim_arg is None:
        golden_indices = torch.argmax(input_tensor, keepdim=keep_dim)
    else:
        golden_indices = torch.argmax(input_tensor, dim=dim_arg[0], keepdim=keep_dim)

    def module(builder: TTIRBuilder):
        @builder.func([shape], [dtype])
        def argmax_inputs(
            in0: Operand, builder: TTIRBuilder, unit_attrs: List[str] = None
        ):
            result = builder.argmax(in0, dim_arg=dim_arg, keep_dim=keep_dim)
            builder.set_goldens({in0: input_tensor}, {result: golden_indices})
            return result

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
        pipeline_options=pipeline_options,
        save_artifacts=True,
    )

    # check_pcc=False: the built-in check compares raw index positions, which
    # ties make ambiguous. _verify_argmax_output does the meaningful comparison.
    _, output_tensors = execute_fb(
        compiled_bin,
        input_output_goldens=io_goldens,
        intermediate_goldens=intermediate_goldens,
        device=device,
        check_pcc=False,
        save_artifacts=True,
        artifact_dir=artifact_dir,
    )

    _verify_argmax_output(
        input_tensor, golden_indices, dim_arg, keep_dim, output_tensors
    )


@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("dim_arg", [[1]])
@pytest.mark.parametrize("keep_dim", [True, False])
@pytest.mark.parametrize(
    "shape",
    [(32, 32), (32, 64), (32, 96), (32, 128), (32, 256)],
)
def test_argmax_base(
    shape: tuple[int, int],
    target: str,
    dim_arg: list[int] | None,
    keep_dim: bool,
    request,
    device,
):
    _run_argmax(
        shape,
        dim_arg=dim_arg,
        keep_dim=keep_dim,
        dtype=torch.bfloat16,
        target=target,
        request=request,
        device=device,
        pipeline_options=[],
    )


@pytest.mark.skip(reason="LLK version does not handle large reduction dims yet.")
@pytest.mark.parametrize(
    "shape,target,dim_arg,keep_dim",
    [
        pytest.param((32, 32768), "ttmetal", [1], False, id="phi_1"),
        pytest.param((32, 51200), "ttmetal", [1], False, id="mistral_7b"),
        pytest.param((32, 128256), "ttmetal", [1], False, id="llama_3_2_3b"),
        pytest.param((32, 131072), "ttmetal", [1], False, id="ministral_8b"),
        pytest.param(
            (32, 151936),
            "ttmetal",
            [1],
            False,
            id="qwen_2_5_0_5b",
        ),
        pytest.param(
            (32, 256000),
            "ttmetal",
            [1],
            False,
            id="gemma_1_1_2b",
        ),
    ],
)
def test_argmax_models(
    shape: tuple[int, int],
    target: str,
    dim_arg: list[int] | None,
    keep_dim: bool,
    request,
    device,
):
    _run_argmax(
        shape,
        dim_arg=dim_arg,
        keep_dim=keep_dim,
        dtype=torch.bfloat16,
        target=target,
        request=request,
        device=device,
        pipeline_options=[
            "allow-l1-output-spilling=true",
            "enable-eltwise-reduction-fusion",
        ],
    )
