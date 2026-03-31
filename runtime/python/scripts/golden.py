# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from functools import partial
import torch
import json
import os
from pathlib import Path

import golden as golden_module
import _ttmlir_runtime as tt_runtime
from builder.base.builder_runtime import *


class CallbackRuntimeConfig:
    def __init__(self, artifact_dir=None):
        self.input_tensors = []
        self.artifact_dir = artifact_dir
        self.golden_results = []

    def add_result(
        self,
        op_type,
        pcc_value=None,
        atol=None,
        rtol=None,
        status="success",
        error_msg=None,
    ):
        """Add a golden comparison result."""
        result = {
            "op_type": op_type,
            "status": status,
        }
        if pcc_value is not None:
            result["pcc"] = float(pcc_value)
        if atol is not None:
            result["atol"] = float(atol)
        if rtol is not None:
            result["rtol"] = float(rtol)
        if error_msg:
            result["error"] = error_msg
        self.golden_results.append(result)

    def save_results(self):
        """Save golden results to JSON file."""
        if self.artifact_dir is None:
            return

        # Create artifact directory if it doesn't exist
        Path(self.artifact_dir).mkdir(parents=True, exist_ok=True)

        # Save results to JSON file
        output_path = os.path.join(self.artifact_dir, "golden_results.json")
        with open(output_path, "w") as f:
            json.dump(self.golden_results, f, indent=4)
        print(f"Golden results saved to: {output_path}")


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
    from ttmlir.dialects import ttnn

    op_type_str = tt_runtime.runtime.get_op_type(op_context)
    op_type = getattr(ttnn, op_type_str[5:])
    try:
        golden_fn = golden_module.get_golden_function(op_type)
    except:
        callback_runtime_config.input_tensors = []
        callback_runtime_config.add_result(
            op_type_str,
            status="skipped",
            error_msg="No golden mapping for operation",
        )
        print(f"No golden mapping for operation: {op_type_str}")
        return

    op_output_tensor_map = tt_runtime.runtime.get_op_output_tensor(
        op_context, program_context
    )
    if len(op_output_tensor_map) == 0:
        callback_runtime_config.input_tensors = []
        callback_runtime_config.add_result(
            op_type_str,
            status="skipped",
            error_msg="Output tensor is empty",
        )
        print("Output tensor is empty - skipping golden comparison")
        return

    try:
        for device_id, op_output_tensor in op_output_tensor_map.items():
            rt_buffer = op_output_tensor.get_data_buffer()
            dtype = runtime_dtype_to_torch_dtype(op_output_tensor.get_dtype())
            output_tensor_torch = torch.frombuffer(rt_buffer, dtype=dtype).flatten()

        op_attrs = tt_runtime.runtime.get_op_attrs(op_context)
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

        atol = 1e-08
        rtol = 1e-05
        a, b, cal_pcc = get_atol_rtol_pcc(
            golden_tensor_torch,
            output_tensor_torch,
            atol,
            rtol,
        )
        print("Runtime PCC:", cal_pcc)

        # Store the result
        callback_runtime_config.add_result(
            op_type_str, cal_pcc, atol, rtol, status="success"
        )  # CHECK RESULTS INSTEAD

    except Exception as e:
        print(f"Error in golden comparison for {op_type_str}: {str(e)}")
        callback_runtime_config.add_result(
            op_type_str, status="error", error_msg=str(e)
        )

    callback_runtime_config.input_tensors = []


def pre_op_get_callback_fn(callback_runtime_config):
    return partial(pre_op_callback, callback_runtime_config)


def post_op_get_callback_fn(callback_runtime_config):
    return partial(post_op_callback, callback_runtime_config)


def post_execution_callback(
    callback_runtime_config, binary, program_context, op_context
):
    callback_runtime_config.save_results()


def post_execution_get_callback_fn(callback_runtime_config):
    return partial(post_execution_callback, callback_runtime_config)


def register(artifact_dir=None):
    callback_runtime_config = CallbackRuntimeConfig(artifact_dir)
    callback_env = tt_runtime.runtime.DebugHooks.get(
        pre_op_get_callback_fn(callback_runtime_config),
        post_op_get_callback_fn(callback_runtime_config),
        post_execution_get_callback_fn(callback_runtime_config),
    )
