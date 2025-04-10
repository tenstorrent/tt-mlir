# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import threading
import ttrt
import ttrt.runtime
import torch
from ttrt.common.util import *
from .constants import FLATBUFFER_BASE_PATH

from ..utils import (
    Helper,
    DeviceContext,
    Storage,
    get_torch_inputs,
    get_runtime_tensor_from_torch,
    get_torch_output_container,
    assert_pcc,
)

from .constants import FLATBUFFER_BASE_PATH


def get_to_layout_inputs(device, runtime_inputs, binary, program_index):
    input_layouts = [
        ttrt.runtime.get_layout(binary.fbb, program_index, i)
        for i in range(len(runtime_inputs))
    ]
    runtime_inputs_with_layout = [
        ttrt.runtime.to_layout(runtime_input, device, layout)
        for runtime_input, layout in zip(runtime_inputs, input_layouts)
    ]
    return runtime_inputs_with_layout


def run_and_verify(helper: Helper, retain_flags, storage, enable_async):
    num_devices = ttrt.runtime.get_num_available_devices()
    assert num_devices > 1, "Test requires at least 2 devices to enable async mode"
    program: Binary.Program = helper.binary.get_program(0)

    torch_inputs = get_torch_inputs(program)
    if storage == Storage.Borrowed:
        runtime_inputs = [
            get_runtime_tensor_from_torch(torch_input, storage=storage)
            for torch_input in torch_inputs
        ]
    elif storage in [Storage.Owned, Storage.Device]:
        runtime_inputs = [
            get_runtime_tensor_from_torch(torch_input, storage=Storage.Owned)
            for torch_input in torch_inputs
        ]
    else:
        raise ValueError(f"Invalid storage type: {storage}")

    assert len(retain_flags) == len(
        runtime_inputs
    ), "Mismatch in retain flags and runtime inputs size"

    should_retain = (
        retain_flags if storage != Storage.Borrowed else [True] * len(retain_flags)
    )

    with DeviceContext(mesh_shape=[1, 2], enable_async=enable_async) as parent_mesh:
        runtime_inputs_with_layouts = get_to_layout_inputs(
            parent_mesh, runtime_inputs, helper.binary, 0
        )
        if storage == Storage.Device or storage == Storage.Owned:
            runtime_inputs = runtime_inputs_with_layouts

        for i, retain_flag in enumerate(retain_flags):
            runtime_inputs_with_layouts[i].set_retain(retain_flag)

        output = ttrt.runtime.submit(
            parent_mesh, helper.binary.fbb, 0, runtime_inputs_with_layouts
        )[0]
        output_host = ttrt.runtime.to_host(output, untilize=True)[0]
        for i, runtime_input in enumerate(runtime_inputs):
            assert (
                should_retain[i] == runtime_input.is_allocated()
            ), f"Retain flag and tensor allocation mismatch ({should_retain[i]} != {runtime_input.is_allocated()} at idx: {i})"

        torch_output = get_torch_output_container(program)
        ttrt.runtime.memcpy(torch_output.data_ptr(), output_host)
        golden = torch.add(torch_inputs[0], torch_inputs[1])
        assert_pcc(torch_output, golden, threshold=0.99)


@pytest.mark.parametrize("storage", [Storage.Borrowed, Storage.Owned, Storage.Device])
@pytest.mark.parametrize("enable_async", [False, True])
@pytest.mark.parametrize(
    "retain_flags",
    [[True, False], [False, True], [True, True], [False, False]],
    ids=lambda x: str(x),
)
def test_implicit_deallocate(
    helper: Helper, request, storage, enable_async, retain_flags
):
    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "add.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    run_and_verify(
        helper, retain_flags=retain_flags, storage=storage, enable_async=enable_async
    )

    helper.teardown()
