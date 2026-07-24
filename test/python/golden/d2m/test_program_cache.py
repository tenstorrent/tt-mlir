# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import pytest
import torch

import _ttmlir_runtime as tt_runtime
from builder.base.builder_apis import compile_ttir_to_flatbuffer
from builder.base.builder_runtime import (
    convert_input_layouts,
    create_tensor,
    program_outputs_as_dict,
    runtime_dtype_to_torch_dtype,
    runtime_str_dtype_to_torch_dtype,
)
from builder.base.builder_utils import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder

pytestmark = pytest.mark.frontend("ttir")


def _execute_add(device, fbb, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    mesh_shape = device.get_mesh_shape()
    inputs = [create_tensor({0: tensor}, mesh_shape) for tensor in (lhs, rhs)]
    converted_inputs = convert_input_layouts(device, inputs, fbb=fbb, program_index=0)

    runtime_outputs = tt_runtime.runtime.submit(device, fbb, 0, converted_inputs)
    tt_runtime.runtime.wait(runtime_outputs)

    output_desc = program_outputs_as_dict(fbb, 0)[0]["desc"]
    output = torch.zeros(
        output_desc["shape"],
        dtype=runtime_str_dtype_to_torch_dtype(
            output_desc["layout"]["memory_desc"]["data_type"]
        ),
    )
    output_tensor = create_tensor({0: output}, mesh_shape)
    output_shards = tt_runtime.runtime.to_host(runtime_outputs[0], untilize=True)
    assert len(output_shards) == 1
    tt_runtime.runtime.memcpy(output_tensor, output_shards[0])

    data = torch.frombuffer(
        bytearray(output_tensor.get_data_buffer()),
        dtype=runtime_dtype_to_torch_dtype(output_tensor.get_dtype()),
    ).reshape(output_tensor.get_shape())
    tt_runtime.runtime.deallocate_tensor(runtime_outputs[0], force=True)
    return data.clone()


@pytest.mark.parametrize("target", ["ttmetal"])
def test_program_cache_refreshes_dynamic_inputs(target: str, request, device):
    shape: Shape = (64, 64)
    dtype = torch.float32

    def module(builder: TTIRBuilder):
        @builder.func([shape, shape], [dtype, dtype])
        def add(
            in0: Operand,
            in1: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            return builder.add(in0, in1, unit_attrs=unit_attrs)

    _, capsule, _, _ = compile_ttir_to_flatbuffer(
        module,
        system_desc_path=request.config.getoption("--sys-desc"),
        target=target,
    )
    fbb = tt_runtime.binary.load_binary_from_capsule(capsule)

    assert device.is_program_cache_enabled()
    device.clear_program_cache()

    cold_lhs = torch.full(shape, 0.25, dtype=dtype)
    cold_rhs = torch.full(shape, 0.5, dtype=dtype)
    cold_output = _execute_add(device, fbb, cold_lhs, cold_rhs)
    torch.testing.assert_close(cold_output, cold_lhs + cold_rhs, atol=0.005, rtol=0.01)

    # New allocations and values must replace cached buffer and CB bindings.
    warm_lhs = torch.linspace(-1.0, 1.0, shape[0] * shape[1], dtype=dtype).reshape(
        shape
    )
    warm_rhs = torch.linspace(1.0, 2.0, shape[0] * shape[1], dtype=dtype).reshape(shape)
    warm_output = _execute_add(device, fbb, warm_lhs, warm_rhs)
    torch.testing.assert_close(warm_output, warm_lhs + warm_rhs, atol=0.005, rtol=0.01)
