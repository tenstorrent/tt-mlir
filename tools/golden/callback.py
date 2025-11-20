# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from functools import partial
import csv
import json
import torch

from ttrt.common.util import get_atol_rtol_pcc


def get_original_op_loc(text: str) -> str:
    try:
        # Get the original location string before it was modified by passes
        segments = re.findall(r'"([^"]*)"', text)
        loc_str = f'"{segments[1]}"' if len(segments) >= 2 else ""
        return "loc(" + loc_str + ")"
    except Exception:
        return ""


class CallbackRuntimeConfig:
    """Runtime config for intermediate golden comparison callbacks using a golden_map of torch tensors"""

    def __init__(
        self,
        device=None,
        pcc=0.99,
        atol=1e-08,
        rtol=1e-05,
        check_atol: bool = True,
        check_rtol: bool = True,
        goldens={},
    ):
        self.device = device
        self.pcc = pcc
        self.atol = atol
        self.rtol = rtol
        self.check_atol = check_atol
        self.check_rtol = check_rtol
        self.goldens = goldens
        self.golden_report = {}
        self.counter = -1

    def callback_counter(self):
        self.counter = self.counter + 1
        return self.counter

    def save_golden_report(self, golden_report_path):
        with open(golden_report_path, "w") as json_file:
            json.dump(self.golden_report, json_file, indent=4)

        self.print(f"Saved golden report to={golden_report_path}")


def pre_op_callback(callback_runtime_config, binary, program_context, op_context):
    # Pre_callback logic to be implemented here
    pass


def pre_op_get_callback_fn(callback_runtime_config):
    return partial(pre_op_callback, callback_runtime_config)


def post_op_callback(callback_runtime_config, binary, program_context, op_context):
    import ttrt.runtime

    loc = ttrt.runtime.get_op_loc_info(op_context)
    op_output_tensor_map = ttrt.runtime.get_op_output_tensor(
        op_context, program_context
    )
    if len(op_output_tensor_map) == 0:
        print("Output tensor is empty - skipping golden comparison")
        return

    if loc not in callback_runtime_config.goldens.keys():
        # try getting golden tensor using the loc before it was modified by passes
        original_op_loc = get_original_op_loc(loc)
        if original_op_loc not in callback_runtime_config.goldens.keys():
            print(f"Loc {loc} not found in golden map - skipping golden comparison")
            return
        else:
            op_golden_tensor_map = callback_runtime_config.goldens[original_op_loc]
            loc = original_op_loc
    else:
        op_golden_tensor_map = callback_runtime_config.goldens[loc]
    if len(op_golden_tensor_map) == 0:
        if len(op_golden_tensor_map) == 0:
            print("Golden tensor is None - skipping golden comparison")
            return

    # loop through all devices and compare golden tensors
    device_results = {}
    for device_id, golden_tensor_torch in op_golden_tensor_map.items():
        if device_id not in op_output_tensor_map.keys():
            print(
                f"Device {device_id} does not have an output tensor - skipping golden comparison"
            )
            continue

        op_output_tensor = op_output_tensor_map[device_id]
        rt_buffer = op_output_tensor.get_data_buffer()
        golden_tensor_torch = golden_tensor_torch.flatten()
        output_tensor_torch = torch.frombuffer(
            rt_buffer, dtype=golden_tensor_torch.dtype
        ).flatten()

        if golden_tensor_torch.shape != output_tensor_torch.shape:
            print(
                "Golden and output tensor shapes do not match - skipping golden comparison",
                golden_tensor_torch.shape,
                output_tensor_torch.shape,
            )
            return

        cal_atol, cal_rtol, cal_pcc, output_str = get_atol_rtol_pcc(
            golden_tensor_torch,
            output_tensor_torch,
            callback_runtime_config.atol,
            callback_runtime_config.rtol,
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

        print(
            f"For device {device_id} at loc={loc}, golden tensor comparrison: {output_str}"
        )

        result = "pass"
        if cal_pcc < callback_runtime_config.pcc:
            result = "fail"
        if (
            callback_runtime_config.check_atol
            and cal_atol > callback_runtime_config.atol
        ):
            result = "fail"
        if (
            callback_runtime_config.check_rtol
            and cal_rtol > callback_runtime_config.rtol
        ):
            result = "fail"

        results = {}
        results["result"] = result
        results["expected_pcc"] = callback_runtime_config.pcc
        results["actual_pcc"] = cal_pcc
        if callback_runtime_config.check_atol:
            results["expected_atol"] = callback_runtime_config.atol
            results["actual_atol"] = cal_atol
        if callback_runtime_config.check_rtol:
            results["expected_rtol"] = callback_runtime_config.rtol
            results["actual_rtol"] = cal_rtol
        results["allclose"] = torch.allclose(
            golden_tensor_torch,
            output_tensor_torch,
            atol=callback_runtime_config.atol,
            rtol=callback_runtime_config.rtol,
        )
        if (
            golden_tensor_torch.dtype == torch.uint16
            or golden_tensor_torch.dtype == torch.uint32
        ):
            print(
                "Skipping max metric for uint16 or uint32 tensors, not supported in pytorch"
            )
        else:
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
            golden_tensor_torch.float().unsqueeze(0),
            output_tensor_torch.float().unsqueeze(0),
        ).item()

        device_results[device_id] = results

    callback_runtime_config.golden_report[loc] = device_results


def post_op_get_callback_fn(callback_runtime_config):
    return partial(post_op_callback, callback_runtime_config)
