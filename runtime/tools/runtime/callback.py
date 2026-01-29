# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import ast
import time
import torch
import numpy as np
from functools import reduce
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict
from collections import OrderedDict
import json
from functools import partial

from builder.base.builder import *
from builder.base.builder_utils import *

import _ttmlir_runtime as tt_runtime


class CallbackRuntimeConfig:
    def __init__(
        self,
        device=None,
        artifact_dir: str = ".",
        logging=None,
        goldens={},
        save_artifacts: bool = False,
        enable_golden: bool = False,
        enable_memory: bool = False,
        pcc=0.99,
        atol=1e-08,
        rtol=1e-05,
        check_pcc: bool = True,
        check_atol: bool = True,
        check_rtol: bool = True,
        bypass_ops=None,
    ):
        self.device = device
        self.pcc = pcc
        self.atol = atol
        self.rtol = rtol
        self.check_pcc = check_pcc
        self.check_atol = check_atol
        self.check_rtol = check_rtol
        self.goldens = goldens
        self.bypass_ops = bypass_ops if bypass_ops else []
        self.save_artifacts = save_artifacts
        self.artifact_dir = artifact_dir
        self.enable_golden = enable_golden
        self.enable_memory = enable_memory
        self.golden_report = {}
        self.memory_report = []

    def start_new_program(self, artifact_dir):
        self.artifact_dir = artifact_dir
        self.golden_report = {}
        self.memory_report = []


def golden(callback_runtime_config, binary, program_context, op_context):
    loc = tt_runtime.runtime.get_op_loc_info(op_context)
    op_output_tensor_map = tt_runtime.runtime.get_op_output_tensor(
        op_context, program_context
    )

    if len(op_output_tensor_map) == 0:
        return

    if loc not in callback_runtime_config.goldens.keys():
        original_op_loc = get_original_op_loc(loc)
        if original_op_loc not in callback_runtime_config.goldens.keys():
            return
        else:
            op_golden_tensor_map = callback_runtime_config.goldens[original_op_loc]
            loc = original_op_loc
    else:
        op_golden_tensor_map = callback_runtime_config.goldens[loc]

    if len(op_golden_tensor_map) == 0:
        return

    device_results = {}
    for device_id, golden_tensor_torch in op_golden_tensor_map.items():
        if device_id not in op_output_tensor_map.keys():
            continue

        try:
            op_output_tensor = op_output_tensor_map[device_id]
            rt_buffer = op_output_tensor.get_data_buffer()
            golden_tensor_torch = golden_tensor_torch.flatten()
            output_tensor_torch = torch.frombuffer(
                rt_buffer, dtype=golden_tensor_torch.dtype
            ).flatten()
        except Exception as e:
            return

        if golden_tensor_torch.shape != output_tensor_torch.shape:
            return

        if callback_runtime_config.save_artifacts:
            golden_tensor_torch_name = get_sanitized_filename(
                f"{loc}_{device_id}_golden.pt"
            )
            device_tensor_torch_name = get_sanitized_filename(
                f"{loc}_{device_id}_device.pt"
            )
            save_torch_tensor(
                golden_tensor_torch,
                callback_runtime_config.artifact_dir,
                golden_tensor_torch_name,
            )
            save_torch_tensor(
                output_tensor_torch,
                callback_runtime_config.artifact_dir,
                device_tensor_torch_name,
            )

        try:
            results = check_outputs(
                golden_tensor_torch,
                output_tensor_torch,
                f"{loc}_{device_id}",
                callback_runtime_config.pcc,
                callback_runtime_config.atol,
                callback_runtime_config.rtol,
                callback_runtime_config.check_pcc,
                callback_runtime_config.check_atol,
                callback_runtime_config.check_rtol,
                raise_exception=False,
            )
            results["debug_info"] = tt_runtime.runtime.get_op_debug_str(op_context)

            # Bypass the runtime tensor by replacing it with the intermediate golden tensor if the op is in the bypass list
            if loc in callback_runtime_config.bypass_ops:
                output_tensor_ref = tt_runtime.runtime.get_op_output_ref(
                    op_context, program_context
                )
                tensor = tt_runtime.runtime.retrieve_tensor_from_pool(
                    program_context, output_tensor_ref
                )
                update_device_tensor(
                    program_context, output_tensor_ref, tensor, golden_tensor_torch
                )
                results["bypassed"] = "True"

            device_results[device_id] = results
        except Exception as e:
            print(e)
            return

    callback_runtime_config.golden_report[loc] = device_results


def create_memory_dictionary(memory_view):
    memory_dict = {}
    memory_dict["num_banks"] = memory_view.num_banks
    memory_dict["total_bytes_per_bank"] = memory_view.total_bytes_per_bank
    memory_dict[
        "total_bytes_allocated_per_bank"
    ] = memory_view.total_bytes_allocated_per_bank
    memory_dict["total_bytes_free_per_bank"] = memory_view.total_bytes_free_per_bank
    memory_dict[
        "largest_contiguous_bytes_free_per_bank"
    ] = memory_view.largest_contiguous_bytes_free_per_bank
    memory_dict["block_table"] = memory_view.block_table

    return memory_dict


def memory(callback_runtime_config, binary, program_context, op_context):
    device = callback_runtime_config.device
    loc = tt_runtime.runtime.get_op_loc_info(op_context)
    debug_str = tt_runtime.runtime.get_op_debug_str(op_context)

    memory_views = device.get_memory_view()
    dram_memory_view = memory_views[tt_runtime.runtime.MemoryBufferType.DRAM]
    l1_memory_view = memory_views[tt_runtime.runtime.MemoryBufferType.L1]
    l1_small_memory_view = memory_views[tt_runtime.runtime.MemoryBufferType.L1_SMALL]
    trace_memory_view = memory_views[tt_runtime.runtime.MemoryBufferType.TRACE]

    op_memory_report = {}
    op_memory_report["loc"] = loc
    op_memory_report["debug_str"] = debug_str
    op_memory_report["dram"] = create_memory_dictionary(dram_memory_view)
    op_memory_report["l1"] = create_memory_dictionary(l1_memory_view)
    op_memory_report["l1_small"] = create_memory_dictionary(l1_small_memory_view)
    op_memory_report["trace"] = create_memory_dictionary(trace_memory_view)

    callback_runtime_config.memory_report.append(op_memory_report)


def pre_op_callback(callback_runtime_config, binary, program_context, op_context):
    pass


def pre_op_get_callback_fn(callback_runtime_config):
    return partial(pre_op_callback, callback_runtime_config)


def post_op_callback(callback_runtime_config, binary, program_context, op_context):
    if callback_runtime_config.enable_golden:
        golden(callback_runtime_config, binary, program_context, op_context)

    if callback_runtime_config.enable_memory:
        memory(callback_runtime_config, binary, program_context, op_context)


def post_op_get_callback_fn(callback_runtime_config):
    return partial(post_op_callback, callback_runtime_config)
