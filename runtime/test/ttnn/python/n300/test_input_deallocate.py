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
    get_to_layout_inputs,
)


def verify_to_layout_deallocation(helper: Helper, retain_flags, storage):
    """Test memory deallocation behavior after to_layout operation"""
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

    # Setup retention expectations based on storage type
    should_retain = retain_flags

    with DeviceContext(mesh_shape=[1, 2]) as parent_mesh:
        # Apply retain flags to original inputs
        for i, retain_flag in enumerate(retain_flags):
            runtime_inputs[i].set_retain(retain_flag)

        # Perform to_layout operation
        _ = get_to_layout_inputs(parent_mesh, runtime_inputs, helper.binary, 0)

        # Verify allocation status after to_layout
        for i, runtime_input in enumerate(runtime_inputs):
            # Deallocation of host tensor does nothing
            # https://github.com/tenstorrent/tt-mlir/issues/3488
            assert (
                True == runtime_input.is_allocated()
            ), f"After to_layout: Retain flag and tensor allocation mismatch ({True} != {runtime_input.is_allocated()} at idx: {i})"


def verify_submit_deallocation(helper: Helper, retain_flags, storage):
    """Test memory deallocation behavior after submit operation"""
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

    with DeviceContext(mesh_shape=[1, 2]) as parent_mesh:
        # First, get the laid-out inputs - we don't care about retain flags for original inputs here
        runtime_inputs_with_layouts = get_to_layout_inputs(
            parent_mesh, runtime_inputs, helper.binary, 0
        )

        # Apply retain flags to the laid-out inputs
        for i, retain_flag in enumerate(retain_flags):
            runtime_inputs_with_layouts[i].set_retain(retain_flag)

        # Perform submit operation
        output = ttrt.runtime.submit(
            parent_mesh, helper.binary.fbb, 0, runtime_inputs_with_layouts
        )[0]
        output_host = ttrt.runtime.to_host(output, untilize=True)[0]

        # Verify allocation status after submit
        for i, runtime_input in enumerate(runtime_inputs_with_layouts):
            assert (
                retain_flags[i] == runtime_input.is_allocated()
            ), f"After submit: Retain flag and tensor allocation mismatch ({retain_flags[i]} != {runtime_input.is_allocated()} at idx: {i})"

        # Verify output correctness
        torch_output = get_torch_output_container(program)
        ttrt.runtime.memcpy(torch_output.data_ptr(), output_host)
        golden = torch.add(torch_inputs[0], torch_inputs[1])
        assert_pcc(torch_output, golden, threshold=0.99)


@pytest.mark.parametrize("storage", [Storage.Borrowed, Storage.Owned, Storage.Device])
@pytest.mark.parametrize(
    "retain_flags",
    [[True, False], [False, True], [True, True], [False, False]],
    ids=lambda x: str(x),
)
def test_to_layout_deallocation(helper: Helper, request, storage, retain_flags):
    """Test that to_layout correctly deallocates input tensors based on retain flags"""
    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "add.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    verify_to_layout_deallocation(helper, retain_flags=retain_flags, storage=storage)

    helper.teardown()


@pytest.mark.parametrize("storage", [Storage.Borrowed, Storage.Owned, Storage.Device])
@pytest.mark.parametrize(
    "retain_flags",
    [[True, False], [False, True], [True, True], [False, False]],
    ids=lambda x: str(x),
)
def test_submit_deallocation(helper: Helper, request, storage, retain_flags):
    """Test that submit correctly deallocates input tensors based on retain flags"""
    binary_path = os.path.join(FLATBUFFER_BASE_PATH, "add.mlir.tmp.ttnn")
    assert os.path.exists(binary_path), f"Binary file not found: {binary_path}"
    helper.initialize(request.node.name, binary_path)
    helper.check_constraints()

    verify_submit_deallocation(helper, retain_flags=retain_flags, storage=storage)

    helper.teardown()
