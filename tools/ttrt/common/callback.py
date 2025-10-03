# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from functools import partial
import csv
import json

from ttrt.common.util import *


class CallbackRuntimeConfig:
    def __init__(
        self,
        device=None,
        artifact_dir="",
        pcc=0.99,
        atol=1e-08,
        rtol=1e-05,
        save_golden_tensors=False,
        logging=None,
        enable_golden=False,
        enable_memory=False,
        enable_debugger=False,
        golden_report={},
        memory_report={},
    ):
        self.device = device
        self.artifact_dir = artifact_dir
        self.pcc = pcc
        self.atol = atol
        self.rtol = rtol
        self.save_golden_tensors = save_golden_tensors
        self.logging = logging
        self.enable_golden = enable_golden
        self.enable_memory = enable_memory
        self.enable_debugger = enable_debugger
        self.golden_report = golden_report
        self.memory_report = memory_report
        self.counter = -1

    def start_new_callback(self, artifact_dir):
        self.artifact_dir = artifact_dir
        self.counter = -1
        self.golden_report = {}
        self.memory_report = {}

    def callback_counter(self):
        self.counter = self.counter + 1
        return self.counter

    def save_golden_report(self, golden_report_path):
        with open(golden_report_path, "w") as json_file:
            json.dump(self.golden_report, json_file, indent=4)

        self.logging.debug(f"Saved golden report to={golden_report_path}")

    def save_memory_report(self, memory_report_path):
        with open(memory_report_path, "w") as json_file:
            json.dump(self.memory_report, json_file, indent=4)

        self.logging.debug(f"Saved memory report to={memory_report_path}")

    def check_pcc(self):
        for loc, device_data in self.golden_report.items():
            for device_id, golden_data in device_data.items():
                if golden_data["actual_pcc"] < golden_data["expected_pcc"]:
                    raise PCCErrorException(
                        f"Failed: golden comparison failed at loc={loc} for device={device_id}, actual_pcc={golden_data['actual_pcc']} < expected_pcc={golden_data['expected_pcc']}"
                    )

    def check_memory_leak(self):
        num_items = 0
        for key, value in self.memory_report.items():
            num_items += 1

        if num_items == 0:
            self.logging.warning(f"No memory data found")
        else:
            # query initial memory usage
            dram_initial_size_per_device = self.memory_report[0]["dram"]
            l1_initial_size_per_device = self.memory_report[0]["l1"]

            # query final memory usage and ensure no memory leaks
            dram_final_size_per_device = self.memory_report[num_items - 1]["dram"]
            l1_final_size_per_device = self.memory_report[num_items - 1]["l1"]

            for key, value in dram_initial_size_per_device.items():
                dram_initial_size = value["total_bytes_allocated_per_bank"]
                dram_final_size = dram_final_size_per_device[key][
                    "total_bytes_allocated_per_bank"
                ]

                if dram_final_size > dram_initial_size:
                    raise Exception(f"Memory leak detected in DRAM for device={key}")

            for key, value in l1_initial_size_per_device.items():
                l1_initial_size = value["total_bytes_allocated_per_bank"]
                l1_final_size = l1_final_size_per_device[key][
                    "total_bytes_allocated_per_bank"
                ]

                if l1_final_size > l1_initial_size:
                    raise Exception(
                        f"Memory leak detected in L1 cache for device={key}"
                    )


"""
-----------------------GOLDEN CALLBACK-----------------------
"""


def golden(callback_runtime_config, binary, program_context, op_context):
    import torch
    import ttrt.runtime

    logging = callback_runtime_config.logging
    logging.debug("executing golden comparison")

    loc = ttrt.runtime.get_op_loc_info(op_context)

    op_golden_tensor_map = binary.get_debug_info_golden(loc)
    if len(op_golden_tensor_map) == 0:
        logging.debug("Golden tensor is None - skipping golden comparison")
        return

    op_output_tensor_map = ttrt.runtime.get_op_output_tensor(
        op_context, program_context
    )
    if len(op_output_tensor_map) == 0:
        logging.debug("Output tensor is empty - skipping golden comparison")
        return

    # loop through all devices and compare golden tensors
    device_results = {}
    for device_id, op_golden_tensor in op_golden_tensor_map.items():
        if device_id not in op_output_tensor_map.keys():
            logging.debug(
                f"Device {device_id} does not have an output tensor - skipping golden comparison"
            )
            continue

        op_output_tensor = op_output_tensor_map[device_id]
        rt_buffer = op_output_tensor.get_data_buffer()
        dtype = ttrt_datatype_to_torch_dtype(op_golden_tensor.dtype)
        golden_tensor_torch = golden_tensor_to_torch(op_golden_tensor)

        output_tensor_torch = torch.frombuffer(rt_buffer, dtype=dtype).reshape(
            op_output_tensor.get_shape()
        )
        if callback_runtime_config.save_golden_tensors:
            golden_tensor_torch_name = get_sanitized_filename(
                f"{loc}_{device_id}_golden.pt"
            )
            device_tensor_torch_name = get_sanitized_filename(
                f"{loc}_{device_id}_device.pt"
            )
            torch.save(
                golden_tensor_torch,
                f"{callback_runtime_config.artifact_dir}/{golden_tensor_torch_name}",
            )
            torch.save(
                output_tensor_torch,
                f"{callback_runtime_config.artifact_dir}/{device_tensor_torch_name}",
            )

        if golden_tensor_torch.shape != output_tensor_torch.shape:
            logging.debug(
                "Golden and output tensor shapes do not match - skipping golden comparison"
            )
            return

        _, _, cal_pcc, output_str = get_atol_rtol_pcc(
            golden_tensor_torch, output_tensor_torch, logging
        )

        # Handle case where tensor has only one element.
        if golden_tensor_torch.numel() == 1:
            cal_pcc = (
                1.0
                if torch.nn.functional.cosine_similarity(
                    golden_tensor_torch.float().unsqueeze(0),
                    output_tensor_torch.float().unsqueeze(0),
                ).item()
                else 0.0
            )

        logging.debug(f"For device {device_id} at loc={loc}, PCC={cal_pcc}")
        logging.debug(output_str)

        results = {}
        results["expected_pcc"] = callback_runtime_config.pcc
        results["actual_pcc"] = cal_pcc
        results["atol"] = callback_runtime_config.atol
        results["rtol"] = callback_runtime_config.rtol
        results["allclose"] = torch.allclose(
            golden_tensor_torch,
            output_tensor_torch,
            atol=callback_runtime_config.atol,
            rtol=callback_runtime_config.rtol,
        )
        results["max"] = torch.max(
            torch.abs(golden_tensor_torch - output_tensor_torch)
        ).item()
        results["mean_absolute_error"] = torch.mean(
            torch.abs(golden_tensor_torch.float() - output_tensor_torch.float())
        ).item()
        results["root_mean_square_error"] = torch.sqrt(
            torch.mean((golden_tensor_torch.float() - output_tensor_torch.float()) ** 2)
        ).item()
        results["cosine_similarity"] = torch.nn.functional.cosine_similarity(
            golden_tensor_torch.float().flatten(),
            output_tensor_torch.float().flatten(),
            dim=0,
        ).item()

        device_results[device_id] = results

    callback_runtime_config.golden_report[loc] = device_results


"""
-----------------------MEMORY CALLBACK-----------------------
"""


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
    import ttrt.runtime

    device = callback_runtime_config.device
    logging = callback_runtime_config.logging
    logging.debug("executing memory dump")
    loc = ttrt.runtime.get_op_loc_info(op_context)
    debug_str = ttrt.runtime.get_op_debug_str(op_context)

    memory_views = device.get_memory_view()
    dram_memory_view = memory_views[ttrt.runtime.MemoryBufferType.DRAM]
    l1_memory_view = memory_views[ttrt.runtime.MemoryBufferType.L1]
    l1_small_memory_view = memory_views[ttrt.runtime.MemoryBufferType.L1_SMALL]
    trace_memory_view = memory_views[ttrt.runtime.MemoryBufferType.TRACE]

    op_memory_report = {}
    op_memory_report["loc"] = loc
    op_memory_report["debug_str"] = debug_str
    op_memory_report["dram"] = create_memory_dictionary(dram_memory_view)
    op_memory_report["l1"] = create_memory_dictionary(l1_memory_view)
    op_memory_report["l1_small"] = create_memory_dictionary(l1_small_memory_view)
    op_memory_report["trace"] = create_memory_dictionary(trace_memory_view)

    callback_runtime_config.memory_report[
        callback_runtime_config.callback_counter()
    ] = op_memory_report


"""
-----------------------DEBUGGER CALLBACK-----------------------
"""


def debugger(callback_runtime_config, binary, program_context, op_context):
    import pdb
    import ttrt.runtime

    device = callback_runtime_config.device
    logging = callback_runtime_config.logging
    logging.debug("starting pdb debugger")
    pdb.set_trace()


def pre_op_callback(callback_runtime_config, binary, program_context, op_context):
    # Pre_callback logic to be implemented here
    pass


def pre_op_get_callback_fn(callback_runtime_config):
    return partial(pre_op_callback, callback_runtime_config)


def post_op_callback(callback_runtime_config, binary, program_context, op_context):

    if callback_runtime_config.enable_golden:
        golden(callback_runtime_config, binary, program_context, op_context)

    if callback_runtime_config.enable_memory:
        memory(callback_runtime_config, binary, program_context, op_context)

    if callback_runtime_config.enable_debugger:
        debugger(callback_runtime_config, binary, program_context, op_context)


def post_op_get_callback_fn(callback_runtime_config):
    return partial(post_op_callback, callback_runtime_config)
