# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from functools import partial
import torch

import golden
import ttrt.runtime
from ttrt.common.util import ttrt_datatype_to_torch_dtype, get_atol_rtol_pcc


class CallbackRuntimeConfig:
    def __init__(self):
        self.input_tensors = {}
        self.op_counter = 0


def pre_op_callback(callback_runtime_config, binary, program_context, op_context):
    print("PRE OP CALLBACK CALLED")
    runtime_inputs = []
    input_refs = ttrt.runtime.get_op_input_refs(op_context, program_context)
    for input_ref in input_refs:
        runtime_input = ttrt.runtime.retrieve_tensor_from_pool(
            program_context, input_ref
        )
        rt_buffer = runtime_input.get_data_buffer()
        dtype = ttrt_datatype_to_torch_dtype(runtime_input.get_dtype())
        runtime_input_torch = torch.frombuffer(rt_buffer, dtype=dtype).flatten()
        runtime_inputs.append(runtime_input_torch)

    callback_runtime_config.input_tensors[
        callback_runtime_config.op_counter
    ] = runtime_inputs


def post_op_callback(callback_runtime_config, binary, program_context, op_context):
    print("POST OP CALLBACK CALLED")
    # In the future, make a specific binary nanobind func for op name
    debug_str = ttrt.runtime.get_op_debug_str(op_context)
    parts = debug_str.split('"')
    op_function_str = parts[1] if len(parts) >= 2 else ""

    golden_fn = golden.get_golden_by_op_function_str(op_function_str)
    if not golden_fn:
        print(f"No golden mapping for operation: {op_function_str}")
        callback_runtime_config.op_counter += 1
        return

    op_output_tensor_map = ttrt.runtime.get_op_output_tensor(
        op_context, program_context
    )
    if len(op_output_tensor_map) == 0:
        callback_runtime_config.op_counter += 1
        print("Output tensor is empty - skipping golden comparison")
        return

    runtime_inputs = callback_runtime_config.input_tensors[
        callback_runtime_config.op_counter
    ]

    for device_id, op_output_tensor in op_output_tensor_map.items():
        rt_buffer = op_output_tensor.get_data_buffer()
        dtype = ttrt_datatype_to_torch_dtype(op_output_tensor.get_dtype())
        output_tensor_torch = torch.frombuffer(rt_buffer, dtype=dtype).flatten()

    golden_tensor_torch = golden_fn(*runtime_inputs)

    a, b, cal_pcc, output_str = get_atol_rtol_pcc(
        golden_tensor_torch,
        output_tensor_torch,
        1e-08,
        1e-05,
    )
    print(a, b, cal_pcc, output_str)

    callback_runtime_config.op_counter += 1


def pre_op_get_callback_fn(callback_runtime_config):
    return partial(pre_op_callback, callback_runtime_config)


def post_op_get_callback_fn(callback_runtime_config):
    return partial(post_op_callback, callback_runtime_config)


def register(message: str):
    print("REGISTER CALLED")
    callback_runtime_config = CallbackRuntimeConfig()
    # post_op_callback_runtime_config = CallbackRuntimeConfig()
    callback_env = ttrt.runtime.GoldenHooks.get(
        pre_op_get_callback_fn(callback_runtime_config),
        post_op_get_callback_fn(callback_runtime_config),
    )
