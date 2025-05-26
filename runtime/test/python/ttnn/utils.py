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

TT_MLIR_HOME = os.environ.get("TT_MLIR_HOME", "")


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


class DeviceContext:
    def __init__(
        self,
        mesh_shape,
        mesh_offset=None,
        enable_program_cache=False,
        trace_region_size=0,
    ):
        options = ttrt.runtime.MeshDeviceOptions()
        if mesh_offset is not None:
            options.mesh_offset = mesh_offset
        options.enable_program_cache = enable_program_cache
        options.trace_region_size = trace_region_size
        self.device = ttrt.runtime.open_mesh_device(mesh_shape, options)

    def __enter__(self):
        return self.device

    def __exit__(self, exc_type, exc_value, traceback):
        ttrt.runtime.close_mesh_device(self.device)


def get_runtime_tensor_from_torch(torch_tensor, storage=Storage.Borrowed):
    if storage == Storage.Borrowed:
        creator_fn = ttrt.runtime.create_tensor
    elif storage == Storage.Owned:
        creator_fn = ttrt.runtime.create_owned_tensor

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
    for program_input in program.program["inputs"]:
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
        program.program["outputs"][0]["desc"]["shape"],
        dtype=Binary.Program.from_data_type(
            program.program["outputs"][0]["desc"]["layout"]["memory_desc"]["data_type"]
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
