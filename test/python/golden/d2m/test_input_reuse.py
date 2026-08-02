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


def _read_output(fbb, device, runtime_output) -> torch.Tensor:
    output_desc = program_outputs_as_dict(fbb, 0)[0]["desc"]
    output = torch.empty(
        output_desc["shape"],
        dtype=runtime_str_dtype_to_torch_dtype(
            output_desc["layout"]["memory_desc"]["data_type"]
        ),
    )
    output_tensor = create_tensor({0: output}, device.get_mesh_shape())
    output_shards = tt_runtime.runtime.to_host(runtime_output, untilize=True)
    assert len(output_shards) == 1
    tt_runtime.runtime.memcpy(output_tensor, output_shards[0])
    data = torch.frombuffer(
        bytearray(output_tensor.get_data_buffer()),
        dtype=runtime_dtype_to_torch_dtype(output_tensor.get_dtype()),
    ).reshape(output_tensor.get_shape())
    tt_runtime.runtime.deallocate_tensor(runtime_output, force=True)
    return data.clone()


def _submit_add(fbb, device, lhs, rhs: torch.Tensor) -> torch.Tensor:
    rhs_runtime = create_tensor({0: rhs}, device.get_mesh_shape())
    rhs_layout = tt_runtime.runtime.get_layout(fbb, 0, 1)
    rhs_runtime = tt_runtime.runtime.to_layout(rhs_runtime, device, rhs_layout, True)
    outputs = tt_runtime.runtime.submit(device, fbb, 0, [lhs, rhs_runtime])
    tt_runtime.runtime.wait(outputs)
    assert len(outputs) == 1
    return _read_output(fbb, device, outputs[0])


@pytest.mark.parametrize("target", ["ttmetal"])
def test_reusable_input_is_uploaded_once_and_preserves_output(
    target: str, request, device
):
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
    tt_runtime.runtime.set_compatible_device_runtime(fbb)

    lhs = torch.linspace(-1.0, 1.0, shape[0] * shape[1], dtype=dtype).reshape(shape)
    lhs_runtime = create_tensor({0: lhs}, device.get_mesh_shape())
    lhs_runtime = convert_input_layouts(
        device, [lhs_runtime], fbb=fbb, program_index=0
    )[0]
    assert not lhs_runtime.get_reusable()
    assert lhs_runtime.get_reuse_stats() == {
        "cache_hits": 0,
        "cache_misses": 0,
        "uploaded_bytes": 0,
        "device_buffer_count": 0,
    }
    lhs_runtime.set_reusable(True)
    assert lhs_runtime.get_reusable()

    first_rhs = torch.full(shape, 0.25, dtype=dtype)
    first_output = _submit_add(fbb, device, lhs_runtime, first_rhs)
    torch.testing.assert_close(first_output, lhs + first_rhs, atol=0.005, rtol=0.01)
    assert lhs_runtime.get_reuse_stats() == {
        "cache_hits": 0,
        "cache_misses": 1,
        "uploaded_bytes": lhs.numel() * lhs.element_size(),
        "device_buffer_count": 1,
    }

    second_rhs = torch.full(shape, -0.5, dtype=dtype)
    second_output = _submit_add(fbb, device, lhs_runtime, second_rhs)
    torch.testing.assert_close(second_output, lhs + second_rhs, atol=0.005, rtol=0.01)
    assert lhs_runtime.get_reuse_stats()["cache_hits"] == 1

    # A separately loaded binary has its own compiled allocation plan. It must
    # populate a distinct cache entry even when the flatbuffer bytes and input
    # destination global IDs are identical.
    second_fbb = tt_runtime.binary.load_binary_from_capsule(capsule)
    binary_scoped_rhs = torch.full(shape, 0.75, dtype=dtype)
    binary_scoped_output = _submit_add(
        second_fbb, device, lhs_runtime, binary_scoped_rhs
    )
    torch.testing.assert_close(
        binary_scoped_output, lhs + binary_scoped_rhs, atol=0.005, rtol=0.01
    )
    assert lhs_runtime.get_reuse_stats() == {
        "cache_hits": 1,
        "cache_misses": 2,
        "uploaded_bytes": 2 * lhs.numel() * lhs.element_size(),
        "device_buffer_count": 2,
    }

    # Clearing reuse is explicit invalidation: all retained buffers and stats
    # are released. Re-enabling it causes the next submit to upload again.
    lhs_runtime.set_reusable(False)
    assert not lhs_runtime.get_reusable()
    lhs.add_(2.0)
    lhs_runtime.set_reusable(True)
    third_rhs = torch.full(shape, 1.0, dtype=dtype)
    third_output = _submit_add(fbb, device, lhs_runtime, third_rhs)
    torch.testing.assert_close(third_output, lhs + third_rhs, atol=0.005, rtol=0.01)
    assert lhs_runtime.get_reuse_stats() == {
        "cache_hits": 0,
        "cache_misses": 1,
        "uploaded_bytes": lhs.numel() * lhs.element_size(),
        "device_buffer_count": 1,
    }
