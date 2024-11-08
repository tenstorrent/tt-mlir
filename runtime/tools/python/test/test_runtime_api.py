# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import ttrt
import torch
from ttrt.common.run import Run
from ttrt.common.query import Query
from ttrt.common.util import *
from ttrt.common.api import API
from util import *


class Helper:
    def __init__(self, logger=None):
        self.artifacts_dir = f"{os.getcwd()}/ttrt-artifacts"
        self.logger = logger if logger is not None else Logger()
        self.logging = self.logger.get_logger()
        self.file_manager = FileManager(self.logger)
        self.results = Results(self.logger, self.file_manager)
        self.artifacts = Artifacts(
            self.logger, self.file_manager, artifacts_folder_path=self.artifacts_dir
        )
        self.query = Query({"--quiet": True}, self.logger, self.artifacts)
        self.query()
        self.test_name = None
        self.binary_path = None
        self.binary = None
        self.results_path = None
        self.results = None

    def initialize(self, test_name, results_path, binary_path=None):
        self.test_name = test_name
        self.results_path = results_path
        self.results = Results(self.logger, self.file_manager)
        if binary_path:
            self.binary_path = binary_path
            self.binary = Binary(self.logger, self.file_manager, binary_path)

    def teardown(self):
        self.test_name = None
        self.results_path = None
        self.results = None
        self.binary_path = None
        self.binary = None

    def create_and_save_result(
        self, status: str, exception: str = "", program_index: int = 0
    ):
        test_result = {
            "file_path": self.test_name,
            "result": status,
            "exception": exception,
            "log_file": self.logger.file_name,
            "artifacts": self.artifacts.artifacts_folder_path,
            "program_index": program_index,
        }
        self.results.add_result(test_result)
        self.results.save_results(self.results_path)

    def assert_and_save_results(self, callback, program_index=0):
        try:
            callback()
            self.create_and_save_result("pass")
        except Exception as e:
            self.create_and_save_result("error", str(e), program_index)
            raise e

    def check_constraints(self):
        if not self.binary:
            return

        self.assert_and_save_results(self.binary.check_version)
        self.assert_and_save_results(lambda: self.binary.check_system_desc(self.query))


class DeviceContext:
    def __init__(self, device_ids):
        self.device = ttrt.runtime.open_device(device_ids)

    def __enter__(self):
        return self.device

    def __exit__(self, exc_type, exc_value, traceback):
        ttrt.runtime.close_device(self.device)


def assert_tensors_match(tensor1, tensor2):
    assert torch.allclose(tensor1, tensor2)


def assert_pcc(x, y, threshold=0.99):
    combined = torch.stack([x.flatten(), y.flatten()])
    pcc = torch.corrcoef(combined)[0, 1].item()
    assert pcc >= threshold, f"Expected pcc {pcc} >= {threshold}"


@pytest.fixture(scope="module")
def helper():
    API.initialize_apis()
    helper = Helper()
    yield helper


@pytest.mark.parametrize("shape", [(64, 128)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_to_device_to_host(helper: Helper, shape, dtype, request):
    results_path = f"ttrt-results/{request.node.name}.json"
    helper.initialize(request.node.name, results_path)
    helper.check_constraints()
    tensor = torch.randn(shape, dtype=dtype)
    result_tensor = torch.zeros(shape, dtype=dtype)
    runtime_input_tensor = ttrt.runtime.create_tensor(
        tensor.data_ptr(),
        list(tensor.shape),
        list(tensor.stride()),
        tensor.element_size(),
        Binary.Program.to_data_type(tensor.dtype),
    )
    with DeviceContext([helper.query.device_ids[0]]) as device:
        device_tensor = ttrt.runtime.to_device(runtime_input_tensor, device)
        host_tensor = ttrt.runtime.to_host(device_tensor)

    ttrt.runtime.memcpy(result_tensor.data_ptr(), host_tensor)
    helper.assert_and_save_results(lambda: assert_tensors_match(tensor, result_tensor))


def test_host_layout_conversion(helper: Helper, request):
    results_path = f"ttrt-results/{request.node.name}.json"
    binary_path = BINARY_FILE_PATH
    helper.initialize(request.node.name, results_path, binary_path)
    helper.check_constraints()
    program_indices = list(range(helper.binary.get_num_programs()))
    for program_index in program_indices:
        program: Binary.Program = helper.binary.get_program(program_index)
        program.populate_inputs(Run.TorchInitializer.get_initilizer("randn"))
        for input_index, input_tensor in enumerate(program.input_tensors):
            layout = ttrt.runtime.get_layout(
                helper.binary.fbb, program_index, input_index
            )
            input_tensor_runtime = ttrt.runtime.create_tensor(
                input_tensor.data_ptr(),
                list(input_tensor.shape),
                list(input_tensor.stride()),
                input_tensor.element_size(),
                Binary.Program.to_data_type(input_tensor.dtype),
            )
            converted_tensor = ttrt.runtime.to_layout(input_tensor_runtime, layout)
            result_tensor = torch.zeros(input_tensor.shape, dtype=input_tensor.dtype)
            ttrt.runtime.memcpy(result_tensor.data_ptr(), converted_tensor)
            helper.assert_and_save_results(
                lambda: assert_tensors_match(input_tensor, result_tensor), program_index
            )


def test_runtime_stitching_eltwise_binary_op_chain(helper: Helper, request):
    results_path = f"ttrt-results/{request.node.name}.json"
    binary_path = f"{TT_MLIR_HOME}/build/test/ttmlir/Runtime/TTNN/runtime_stitching/Output/eltwise_binary_op_chain.mlir.tmp.ttnn"
    helper.initialize(request.node.name, results_path, binary_path)
    helper.check_constraints()
    first_program: Binary.Program = helper.binary.get_program(0)
    assert first_program.num_inputs() == 2
    inputs_torch = []
    inputs_runtime = []
    for i in first_program.program["inputs"]:
        torch_tensor = torch.randn(
            i["desc"]["shape"],
            dtype=Binary.Program.from_data_type(
                i["desc"]["layout"]["memory_desc"]["data_type"]
            ),
        )
        inputs_torch.append(torch_tensor)
        runtime_tensor = ttrt.runtime.create_tensor(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            list(torch_tensor.stride()),
            torch_tensor.element_size(),
            Binary.Program.to_data_type(torch_tensor.dtype),
        )
        inputs_runtime.append(runtime_tensor)

    activations, weights = inputs_runtime
    with DeviceContext([helper.query.device_ids[0]]) as device:
        activations = ttrt.runtime.to_device(activations, device)
        weights = ttrt.runtime.to_device(weights, device)
        program_indices = list(range(helper.binary.get_num_programs()))
        for program_index in program_indices:
            program = helper.binary.get_program(program_index)
            assert program.num_inputs() == 2 and program.num_outputs() == 1
            outputs = ttrt.runtime.submit(
                device, helper.binary.fbb, program_index, [activations, weights]
            )
            activations = ttrt.runtime.to_device(outputs[0], device)
            ttrt.runtime.deallocate_tensor(outputs[0], force=True)
        activations = ttrt.runtime.to_host(activations)

    last_program: Binary.Program = helper.binary.get_program(program_indices[-1])
    output_torch = torch.randn(
        last_program.program["outputs"][0]["desc"]["shape"],
        dtype=Binary.Program.from_data_type(
            last_program.program["outputs"][0]["desc"]["layout"]["memory_desc"][
                "data_type"
            ]
        ),
    )
    ttrt.runtime.memcpy(output_torch.data_ptr(), activations)
    golden = (
        (inputs_torch[0] + inputs_torch[1]).mul(inputs_torch[1]).sub(inputs_torch[1])
    )
    helper.assert_and_save_results(
        lambda: assert_pcc(golden, output_torch, threshold=0.999), program_index
    )


# TODO: Add test for to_device with layout and on-device layout conversion
# once compiler supports non-host input tensor layouts
