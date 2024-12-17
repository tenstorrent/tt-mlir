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
    op_output_tensor = ttrt.runtime.get_op_output_tensor(op_context, program_context)

    if op_golden_tensor is None:
        logging.debug("Golden tensor is None - skipping golden comparison")
        return

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


def add_key(dram_memory_usage, l1_memory_usage, current_section, key, value):
    if current_section == "DRAM":
        dram_memory_usage[key] = value
    elif current_section == "L1":
        l1_memory_usage[key] = value


def parse_detailed_memory_usage_file(dram_memory_usage, l1_memory_usage, file_path):
    current_section = None

    with open(file_path, "r") as file:
        reader = csv.reader(file)
        blocks = []

        for row in reader:
            if not any(row):
                continue

            if row[1].strip() == "DRAM":
                current_section = "DRAM"
            elif row[1].strip() == "L1":
                current_section = "L1"
            elif "Total" in row[1]:
                if row[1].strip() == "Total allocatable (B):":
                    add_key(
                        dram_memory_usage,
                        l1_memory_usage,
                        current_section,
                        "total_allocatable (bytes) : total_allocatable/bank * num_banks",
                        row[2].strip(),
                    )
                elif row[1].strip() == "Total allocated (B):":
                    add_key(
                        dram_memory_usage,
                        l1_memory_usage,
                        current_section,
                        "total_allocated (bytes) : total_allocated/bank * num_banks",
                        row[2].strip(),
                    )
                elif row[1].strip() == "Total free (B):":
                    add_key(
                        dram_memory_usage,
                        l1_memory_usage,
                        current_section,
                        "total_free (bytes) : total_allocatable - total_allocated",
                        row[2].strip(),
                    )
            elif "Blocks" in row[2]:
                blocks = []
            else:
                block = {}
                block["address (bytes)"] = row[3].strip()
                block["size (bytes)"] = row[4].strip()
                block["allocated (y/n)"] = row[5].strip()

                blocks.append(block)
                add_key(
                    dram_memory_usage,
                    l1_memory_usage,
                    current_section,
                    "blocks",
                    blocks,
                )


def parse_memory_usage_summary_file(dram_memory_usage, l1_memory_usage, file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        current_section = "DRAM"

        for row in reader:
            if not any(row):
                continue

            if "Total Allocatable Size" in row[1]:
                continue

            add_key(
                dram_memory_usage,
                l1_memory_usage,
                current_section,
                "total_allocatable (bytes) : per bank",
                row[1].strip(),
            )
            add_key(
                dram_memory_usage,
                l1_memory_usage,
                current_section,
                "total_allocated (bytes): per bank",
                row[2].strip(),
            )
            add_key(
                dram_memory_usage,
                l1_memory_usage,
                current_section,
                "total_free (bytes) : per bank",
                row[3].strip(),
            )
            add_key(
                dram_memory_usage,
                l1_memory_usage,
                current_section,
                "largest_free_block (bytes) : per bank",
                row[4].strip(),
            )

            if current_section == "DRAM":
                current_section = "L1"


def parse_l1_usage_summary_file(dram_memory_usage, l1_memory_usage, file_path):
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        dram_row = True

        for index, row in enumerate(reader):
            if index == 2:
                add_key(
                    dram_memory_usage,
                    l1_memory_usage,
                    "L1",
                    "largest_contiguous_free_block (bytes) : per bank",
                    row[1].strip(),
                )


def parse_memory_csv_files(
    detailed_memory_usage_file_path,
    memory_usage_summary_file_path,
    l1_usage_summary_file_path,
):
    dram_memory_usage = {}
    l1_memory_usage = {}

    parse_detailed_memory_usage_file(
        dram_memory_usage, l1_memory_usage, detailed_memory_usage_file_path
    )
    parse_memory_usage_summary_file(
        dram_memory_usage, l1_memory_usage, memory_usage_summary_file_path
    )
    parse_l1_usage_summary_file(
        dram_memory_usage, l1_memory_usage, l1_usage_summary_file_path
    )

    return dram_memory_usage, l1_memory_usage


def memory(callback_runtime_config, binary, program_context, op_context):
    import ttrt.runtime
    import ttrt.binary

    device = callback_runtime_config.device
    logging = callback_runtime_config.logging
    logging.debug("executing memory dump")
    loc = ttrt.runtime.get_op_loc_info(op_context)
    debug_str = ttrt.runtime.get_op_debug_str(op_context)

    device.dump_memory_report()
    memory_dump_dir_path = f"{get_ttrt_metal_home_path()}/generated/reports"

    # read generated memory reports and store in condensed memory_report
    dram_memory_usage, l1_memory_usage = parse_memory_csv_files(
        f"{memory_dump_dir_path}/detailed_memory_usage.csv",
        f"{memory_dump_dir_path}/memory_usage_summary.csv",
        f"{memory_dump_dir_path}/l1_usage_summary.csv",
    )

    op_memory_report = {}
    op_memory_report["loc"] = loc
    op_memory_report["debug_str"] = debug_str
    op_memory_report["dram"] = dram_memory_usage
    op_memory_report["l1"] = l1_memory_usage
    callback_runtime_config.memory_report[
        callback_runtime_config.callback_counter()
    ] = op_memory_report


"""
-----------------------DEBUGGER CALLBACK-----------------------
"""


def debugger(callback_runtime_config, binary, program_context, op_context):
    import pdb

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
