# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json

import torch

import runner


class FakeBinary:
    def get_program_inputs_as_json(self, program_index):
        assert program_index == 0
        return json.dumps(
            [{"desc": {"layout": {"memory_desc": {"data_type": "Int32"}}}}]
        )


class FakeRuntime:
    def __init__(self):
        self.borrowed_pointer = None

    def create_borrowed_host_tensor(
        self, pointer, shape, stride, element_size, runtime_dtype
    ):
        self.borrowed_pointer = pointer
        return (pointer, shape, stride, element_size, runtime_dtype)

    def get_layout(self, fbb, program_index, input_index):
        assert fbb is not None
        assert (program_index, input_index) == (0, 0)
        return "layout"

    def to_layout(self, tensor, device, layout, blocking):
        assert device == "device"
        assert layout == "layout"
        assert blocking is True
        return tensor


def test_prepare_runtime_inputs_converts_and_retains_borrowed_storage(monkeypatch):
    monkeypatch.setitem(runner._RT_STR_TO_TORCH, "Int32", torch.int32)
    monkeypatch.setitem(runner._TORCH_TO_RT, torch.int32, "runtime-int32")
    runtime = FakeRuntime()
    input_tensor = torch.arange(8, dtype=torch.int64)

    rt_inputs, host_inputs = runner._prepare_runtime_inputs(
        runtime, FakeBinary(), [input_tensor], "device", 0
    )

    assert len(rt_inputs) == 1
    assert len(host_inputs) == 1
    assert host_inputs[0].dtype == torch.int32
    assert host_inputs[0].tolist() == input_tensor.tolist()
    assert runtime.borrowed_pointer == host_inputs[0].data_ptr()


def test_parse_func_io_accepts_integer_arguments():
    ttir = """
    module {
      func.func @forward(%indices: tensor<8xi64>) -> tensor<8xi64> {
        return %indices : tensor<8xi64>
      }
    }
    """

    assert runner.parse_func_io(ttir) == [((8,), torch.int64)]
