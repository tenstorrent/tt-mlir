# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import ttrt
import ttrt.runtime
import torch
from ttrt.common.query import Query
from ttrt.common.util import *
from enum import Enum
from dataclasses import dataclass
from typing import Callable, List, Any

TT_MLIR_HOME = os.environ.get("TT_MLIR_HOME", "")
TT_METAL_RUNTIME_ROOT_EXTERNAL = os.environ.get("TT_METAL_RUNTIME_ROOT_EXTERNAL", "")


class Storage(Enum):
    Borrowed = "Borrowed"
    Owned = "Owned"
    Device = "Device"


class Helper:
    def __init__(self, logger=None):
        self.artifacts_dir = f"{os.getcwd()}/ttrt-artifacts"
        self.logger = logger if logger is not None else Logger()
        self.logging = self.logger.get_logger()
        self.file_manager = FileManager(self.logger)
        self.artifacts = Artifacts(
            self.logger, self.file_manager, artifacts_folder_path=self.artifacts_dir
        )
        self.query = Query({"--quiet": True}, self.logger, self.artifacts)
        self.query()
        self.test_name = None
        self.binary_path = None
        self.binary = None

    def initialize(self, test_name, binary_path=None):
        self.test_name = test_name
        if binary_path:
            self.binary_path = binary_path
            self.binary = Binary(self.logger, self.file_manager, binary_path)

    def teardown(self):
        self.test_name = None
        self.binary_path = None
        self.binary = None

    def check_constraints(self):
        if not self.binary:
            return
        self.binary.check_version()
        self.binary.check_system_desc(self.query)


@dataclass
class ProgramTestConfig:
    name: str
    expected_num_inputs: int
    compute_golden: Callable[[List[Any]], Any]
    description: str = ""


class ProgramTestRunner:
    def __init__(
        self, binary: Binary, program_index: int, config: ProgramTestConfig = None
    ):

        program = binary.get_program(program_index)
        assert not program.is_private()

        self.config = None
        self.binary = binary
        self.program = program
        self.program_index = program_index
        if config:
            assert (
                program.num_inputs() == config.expected_num_inputs
            ), f"Expected {config.expected_num_inputs} inputs, got {program.num_inputs()}"
            assert (
                program.num_outputs() == 1
            ), "Currently only single output is supported"
            self.config = config

    def get_inputs_and_golden(self, device, borrow=True):
        inputs_torch = get_torch_inputs(self.program)
        storage_type = Storage.Borrowed if borrow else Storage.Owned
        inputs_runtime = [
            get_runtime_tensor_from_torch(torch_input, storage_type)
            for torch_input in inputs_torch
        ]
        input_layouts = [
            ttrt.runtime.get_layout(
                executable=self.binary.fbb,
                program_index=self.program_index,
                input_index=i,
            )
            for i in range(len(inputs_runtime))
        ]
        inputs_runtime_with_layout = [
            ttrt.runtime.to_layout(rt_input, device, layout, True)
            for rt_input, layout in zip(inputs_runtime, input_layouts)
        ]

        golden = None
        if self.config and self.config.compute_golden:
            golden = self.config.compute_golden(inputs_torch)

        return inputs_runtime_with_layout, golden

    def submit_program(self, device, inputs):
        return ttrt.runtime.submit(device, self.binary.fbb, self.program_index, inputs)[
            0
        ]

    def run_program(self, device, inputs, blocking_to_host=True):
        output = self.submit_program(device, inputs)
        output = ttrt.runtime.to_host(output, untilize=True, blocking=blocking_to_host)[
            0
        ]
        return output

    def run_program_and_compare_golden(self, device, inputs, golden):
        output_torch = get_torch_output_container(self.program)
        output = self.run_program(device, inputs)
        ttrt.runtime.memcpy(output_torch.data_ptr(), output)
        assert_pcc(output_torch, golden)


class DeviceContext:
    def __init__(
        self,
        mesh_shape=None,
        mesh_offset=None,
        enable_program_cache=False,
        trace_region_size=0,
        num_hw_cqs=1,
    ):
        options = ttrt.runtime.MeshDeviceOptions()
        if mesh_shape is not None:
            options.mesh_shape = mesh_shape
        if mesh_offset is not None:
            options.mesh_offset = mesh_offset
        options.enable_program_cache = enable_program_cache
        options.trace_region_size = trace_region_size
        options.num_hw_cqs = num_hw_cqs
        self.device = ttrt.runtime.open_mesh_device(options)

    def __enter__(self):
        return self.device

    def __exit__(self, exc_type, exc_value, traceback):
        ttrt.runtime.close_mesh_device(self.device)


def subprocess_get_system_descriptor(request):
    import shutil
    import subprocess

    folder_name = "-".join([request.fspath.basename, request.node.name, "artifacts"])
    artifacts_dir = f"{os.getcwd()}/{folder_name}"

    result = subprocess.run(
        ["ttrt", "query", "--save-artifacts", "--artifact-dir", artifacts_dir],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to query system descriptor: {result.stderr}"
    system_desc = ttrt.binary.load_system_desc_from_path(
        f"{artifacts_dir}/system_desc.ttsys"
    )
    shutil.rmtree(artifacts_dir)

    return system_desc


def get_runtime_tensor_from_torch(torch_tensor, storage=Storage.Borrowed):
    if storage == Storage.Borrowed:
        creator_fn = ttrt.runtime.create_borrowed_host_tensor
    elif storage == Storage.Owned:
        creator_fn = ttrt.runtime.create_owned_host_tensor

    runtime_dtype = Binary.Program.to_data_type(torch_tensor.dtype)
    runtime_tensor = creator_fn(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        list(torch_tensor.stride()),
        torch_tensor.element_size(),
        runtime_dtype,
    )
    return runtime_tensor


def get_torch_inputs(program):
    inputs_torch = []
    for program_input in program.inputs:
        torch_tensor = torch.randn(
            program_input["desc"]["shape"],
            dtype=Binary.Program.from_data_type(
                program_input["desc"]["layout"]["memory_desc"]["data_type"]
            ),
        )
        inputs_torch.append(torch_tensor)
    return inputs_torch


def get_torch_output_container(program):
    torch_result_container = torch.randn(
        program.outputs[0]["desc"]["shape"],
        dtype=Binary.Program.from_data_type(
            program.outputs[0]["desc"]["layout"]["memory_desc"]["data_type"]
        ),
    )
    return torch_result_container


def assert_tensors_match(tensor1, tensor2):
    assert torch.allclose(tensor1, tensor2)


def assert_pcc(x, y, threshold=0.99):
    combined = torch.stack([x.flatten(), y.flatten()])
    pcc = torch.corrcoef(combined)[0, 1].item()
    assert pcc >= threshold, f"Expected pcc {pcc} >= {threshold}"


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
