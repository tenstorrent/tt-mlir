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
        for loc, golden_data in self.golden_report.items():
            if golden_data["actual_pcc"] < golden_data["expected_pcc"]:
                raise PCCErrorException(
                    f"Failed: golden comparison failed, actual_pcc={golden_data['actual_pcc']} < expected_pcc={golden_data['expected_pcc']}"
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


def get_atol_rtol_pcc(golden, calculated):
    import numpy as np
    import torch

    # Calculate atol and rtol
    cal_atol = torch.max(torch.abs(golden - calculated)).item()
    cal_rtol = torch.max(torch.abs(golden - calculated) / torch.abs(calculated)).item()

    # Calculate PCC
    def get_pcc(golden, calculated):
        # Both tensors are nan
        if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
            logging.debug("Both tensors are 'nan'")
            return 1.0
        # Test if either is completely zero
        elif torch.any(golden.bool()) != torch.any(calculated.bool()):
            return 0.0
        # One tensor is all nan, the other is not
        elif torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
            logging.debug("One tensor is all nan, the other is not.")
            return 0.0
        else:
            # For now, mask all infs and nans so that we check the rest... TODO
            golden = golden.clone()
            golden[
                torch.logical_or(
                    torch.isnan(golden),
                    torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
                )
            ] = 0
            calculated = calculated.clone()
            calculated[
                torch.logical_or(
                    torch.isnan(calculated),
                    torch.logical_or(
                        torch.isinf(calculated), torch.isneginf(calculated)
                    ),
                )
            ] = 0

            if torch.equal(golden, calculated):
                return 1.0

            if golden.dtype == torch.bfloat16:
                golden = golden.type(torch.float32)
                calculated = calculated.type(torch.float32)

            # Single element case
            if golden.numel() == 1:
                return float(torch.equal(golden, calculated))

            # If both tensors are contant
            if torch.max(golden) == torch.min(golden) and torch.max(
                calculated
            ) == torch.min(calculated):
                return torch.isclose(torch.max(golden), torch.max(calculated)).item()

            cal_pcc = np.ma.corrcoef(
                np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
                np.ma.masked_invalid(
                    torch.squeeze(calculated).detach().numpy()
                ).flatten(),
            )
            # Remove correlation coefficient with self (typically always 1.0)
            mask = np.ones(cal_pcc.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            cal_pcc = np.min(cal_pcc[mask])

            if isinstance(cal_pcc, np.ma.core.MaskedConstant):
                return 1.0

            return cal_pcc

    cal_pcc = get_pcc(golden, calculated)

    return (
        cal_atol,
        cal_rtol,
        cal_pcc,
        f"Max ATOL Delta: {cal_atol}, Max RTOL Delta: {cal_rtol}, PCC: {cal_pcc}",
    )


def golden(callback_runtime_config, binary, program_context, op_context):
    import torch
    import ttrt.runtime
    import ttrt.binary

    logging = callback_runtime_config.logging
    logging.debug("executing golden comparison")

    loc = ttrt.runtime.get_op_loc_info(op_context)

    op_golden_tensor = binary.get_debug_info_golden(loc)

    if op_golden_tensor is None:
        logging.debug("Golden tensor is None - skipping golden comparison")
        return

    op_output_tensor = ttrt.runtime.get_op_output_tensor(op_context, program_context)

    if len(op_output_tensor) == 0:
        logging.debug("Output tensor is empty - skipping golden comparison")
        return

    dtype = ttrt_datatype_to_torch_dtype(op_golden_tensor.dtype)

    golden_tensor_torch = torch.frombuffer(op_golden_tensor, dtype=dtype).flatten()

    output_tensor_torch = torch.tensor(op_output_tensor, dtype=dtype).flatten()

    if callback_runtime_config.save_golden_tensors:
        torch.save(
            golden_tensor_torch,
            f"{callback_runtime_config.artifact_dir}/{loc}_golden.pt",
        )
        torch.save(
            output_tensor_torch,
            f"{callback_runtime_config.artifact_dir}/{loc}_device.pt",
        )

    if golden_tensor_torch.shape != output_tensor_torch.shape:
        logging.debug(
            "Golden and output tensor shapes do not match - skipping golden comparison"
        )
        return

    _, _, cal_pcc, output_str = get_atol_rtol_pcc(
        golden_tensor_torch, output_tensor_torch
    )

    logging.debug(f"PCC={cal_pcc}")
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
        torch.abs(golden_tensor_torch - output_tensor_torch)
    ).item()
    results["root_mean_square_error"] = torch.sqrt(
        torch.mean((golden_tensor_torch - output_tensor_torch) ** 2)
    ).item()
    results["cosine_similarity"] = torch.nn.functional.cosine_similarity(
        golden_tensor_torch.unsqueeze(0), output_tensor_torch.unsqueeze(0)
    ).item()

    callback_runtime_config.golden_report[loc] = results


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
    import ttrt.binary

    device = callback_runtime_config.device
    logging = callback_runtime_config.logging
    logging.debug("executing memory dump")
    loc = ttrt.runtime.get_op_loc_info(op_context)
    debug_str = ttrt.runtime.get_op_debug_str(op_context)
    device_id = 0

    memory_views = device.get_memory_view(device_id)
    dram_memory_view = memory_views[ttrt.runtime.MemoryBufferType.DRAM]
    l1_memory_view = memory_views[ttrt.runtime.MemoryBufferType.L1]
    l1_small_memory_view = memory_views[ttrt.runtime.MemoryBufferType.L1_SMALL]
    trace_memory_view = memory_views[ttrt.runtime.MemoryBufferType.TRACE]

    op_memory_report = {}
    op_memory_report["loc"] = loc
    op_memory_report["debug_str"] = debug_str

    dram_op_device_memory_report = {}
    dram_op_device_memory_report["device_" + str(device_id)] = create_memory_dictionary(
        dram_memory_view
    )

    l1_op_device_memory_report = {}
    l1_op_device_memory_report["device_" + str(device_id)] = create_memory_dictionary(
        l1_memory_view
    )

    l1_small_op_device_memory_report = {}
    l1_small_op_device_memory_report[
        "device_" + str(device_id)
    ] = create_memory_dictionary(l1_small_memory_view)

    trace_op_device_memory_report = {}
    trace_op_device_memory_report[
        "device_" + str(device_id)
    ] = create_memory_dictionary(trace_memory_view)

    op_memory_report["dram"] = dram_op_device_memory_report
    op_memory_report["l1"] = l1_op_device_memory_report
    op_memory_report["l1_small"] = l1_small_op_device_memory_report
    op_memory_report["trace"] = trace_op_device_memory_report

    callback_runtime_config.memory_report[
        callback_runtime_config.callback_counter()
    ] = op_memory_report


"""
-----------------------DEBUGGER CALLBACK-----------------------
"""


def debugger(callback_runtime_config, binary, program_context, op_context):
    import pdb
    import ttrt.runtime
    import ttrt.binary

    device = callback_runtime_config.device
    logging = callback_runtime_config.logging
    logging.debug("starting pdb debugger")
    pdb.set_trace()


def callback(callback_runtime_config, binary, program_context, op_context):

    if callback_runtime_config.enable_golden:
        golden(callback_runtime_config, binary, program_context, op_context)

    if callback_runtime_config.enable_memory:
        memory(callback_runtime_config, binary, program_context, op_context)

    if callback_runtime_config.enable_debugger:
        debugger(callback_runtime_config, binary, program_context, op_context)


def get_callback_fn(callback_runtime_config):
    return partial(callback, callback_runtime_config)
