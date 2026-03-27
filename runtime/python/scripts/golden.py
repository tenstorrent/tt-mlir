# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from functools import partial
import torch

import golden as golden_module
import _ttmlir_runtime as tt_runtime
from builder.base.builder_runtime import *


class CallbackRuntimeConfig:
    def __init__(self):
        self.input_tensors = []


def pre_op_callback(callback_runtime_config, binary, program_context, op_context):
    runtime_inputs = []
    input_refs = tt_runtime.runtime.get_op_input_refs(op_context, program_context)
    for input_ref in input_refs:
        runtime_input = tt_runtime.runtime.retrieve_tensor_from_pool(
            program_context, input_ref
        )
        rt_buffer = runtime_input.get_data_buffer()
        dtype = runtime_dtype_to_torch_dtype(runtime_input.get_dtype())
        runtime_input_torch = torch.frombuffer(rt_buffer, dtype=dtype).flatten()
        runtime_inputs.append(runtime_input_torch)

    callback_runtime_config.input_tensors.extend(runtime_inputs)


def post_op_callback(callback_runtime_config, binary, program_context, op_context):
    # In the future, make a specific binary nanobind func for op name
    debug_str = tt_runtime.runtime.get_op_debug_str(op_context)
    parts = debug_str.split('"')
    op_function_str = parts[1] if len(parts) >= 2 else ""

    golden_fn = golden_module.get_golden_by_op_function_str(op_function_str)
    if not golden_fn:
        callback_runtime_config.input_tensors = []
        print(f"No golden mapping for operation: {op_function_str}")
        return

    op_output_tensor_map = tt_runtime.runtime.get_op_output_tensor(
        op_context, program_context
    )
    if len(op_output_tensor_map) == 0:
        callback_runtime_config.input_tensors = []
        print("Output tensor is empty - skipping golden comparison")
        return

    for device_id, op_output_tensor in op_output_tensor_map.items():
        rt_buffer = op_output_tensor.get_data_buffer()
        dtype = runtime_dtype_to_torch_dtype(op_output_tensor.get_dtype())
        output_tensor_torch = torch.frombuffer(rt_buffer, dtype=dtype).flatten()

    op_attrs = tt_runtime.runtime.get_op_attrs(op_context, program_context)
    reformatted_attrs = {}
    for key, value in op_attrs.items():
        reformatted_attrs[key + "_attr"] = value
    # May prove to be an issue for multi-output ops if they have different output types
    reformatted_attrs["output_type_mlir"] = runtime_dtype_to_mlir_type(
        op_output_tensor_map[0].get_dtype()
    )

    golden_tensor_torch = golden_fn(
        *callback_runtime_config.input_tensors, **reformatted_attrs
    ).flatten()

    a, b, cal_pcc = get_atol_rtol_pcc(
        golden_tensor_torch,
        output_tensor_torch,
        1e-08,
        1e-05,
    )
    print("Runtime PCC:", cal_pcc)

    callback_runtime_config.input_tensors = []


def pre_op_get_callback_fn(callback_runtime_config):
    return partial(pre_op_callback, callback_runtime_config)


def post_op_get_callback_fn(callback_runtime_config):
    return partial(post_op_callback, callback_runtime_config)


def register(message: str):
    callback_runtime_config = CallbackRuntimeConfig()
    callback_env = tt_runtime.runtime.DebugHooks.get(
        pre_op_get_callback_fn(callback_runtime_config),
        post_op_get_callback_fn(callback_runtime_config),
    )
