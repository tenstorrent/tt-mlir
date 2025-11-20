# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from functools import partial
import csv
import json

from ttrt.common.util import *


def passes_golden_tensor_to_torch(golden_tensor):
    """
    Convert a GoldenTensor originating from ttmlir.passes (python/Passes.cpp) into a torch tensor.
    That binding exposes fields: name, shape, strides, dtype, data (std::vector[uint8_t]).
    """
    import torch

    print("TTTTTTT", type(golden_tensor))
    try:
        from ttmlir.passes import DataType as PassDataType

        dtype_enum = getattr(golden_tensor, "dtype", None)
        if dtype_enum == PassDataType.Float32:
            torch_dtype = torch.float32
        elif dtype_enum == PassDataType.BFloat16:
            torch_dtype = torch.bfloat16
        elif dtype_enum == PassDataType.Float16:
            torch_dtype = torch.float16
        elif dtype_enum == PassDataType.UInt32:
            torch_dtype = torch.uint32
        elif dtype_enum == PassDataType.UInt16:
            torch_dtype = torch.uint16
        elif dtype_enum == PassDataType.UInt8:
            torch_dtype = torch.uint8
        elif dtype_enum == PassDataType.Int32:
            torch_dtype = torch.int32
        else:
            # Fallback to float32 if enum is unrecognized
            torch_dtype = torch.float32
    except Exception:
        torch_dtype = torch.float32

    shape = getattr(golden_tensor, "shape", [])
    if any(dim == 0 for dim in shape):
        return torch.empty(shape, dtype=torch_dtype)

    # Prefer runtime-style buffer if present, otherwise convert vector<uint8_t> to bytes
    if hasattr(golden_tensor, "get_data_buffer"):
        buffer_obj = golden_tensor.get_data_buffer()
    else:
        # 'data' is a std::vector<uint8_t> from nanobind; convert to a bytes-like buffer
        buffer_obj = bytes(getattr(golden_tensor, "data", []))

    return torch.frombuffer(buffer_obj, dtype=torch_dtype).reshape(shape)


def passes_ttrt_datatype_to_torch_dtype(dtype) -> torch.dtype:
    """Converts a PyBound `::tt::target::DataType` into a `torch.dtype`.

    Currently, only `float32`, `uint32`, `uint16`, & `uint8` are supported for
    this conversion

    Arguments
    ---------

    dtype : DataType
        A datatype from the PyBound `DataType` enum from ttrt

    Returns
    -------

    A `torch.dtype` corresponding to `dtype`

    Throws
    ------

    A `ValueError` if `dtype` is not one of `Float32`, `UInt32`, `UInt16`, or `UInt8`

    """
    from ttmlir.passes import DataType as PassDataType

    print("1.0")

    return torch.float32

    print("1")
    print(dtype, type(dtype))

    if dtype == PassDataType.Float32:
        return torch.float32
    elif dtype == PassDataType.UInt32:
        return torch.uint32
    elif dtype == PassDataType.UInt16:
        return torch.uint16
    elif dtype == PassDataType.UInt8:
        return torch.uint8
    elif dtype == PassDataType.BFloat16:
        return torch.bfloat16
    elif dtype == PassDataType.Int32:
        return torch.int32
    else:
        raise ValueError(
            "Only F32, BF16, and unsigned integers are supported in the runtime"
        )


def get_original_op_loc(text: str) -> str:
    try:
        # Get the original location string before it was modified by passes
        segments = re.findall(r'"([^"]*)"', text)
        loc_str = f'"{segments[1]}"' if len(segments) >= 2 else ""
        return "loc(" + loc_str + ")"
    except Exception:
        return ""


class CallbackRuntimeConfig:
    def __init__(
        self,
        device=None,
        artifact_dir="",
        pcc=0.99,
        atol=1e-08,
        rtol=1e-05,
        check_atol: bool = True,
        check_rtol: bool = True,
        save_golden_tensors=False,
        logging=None,
        golden_report={},
        goldens={},
    ):
        self.device = device
        self.artifact_dir = artifact_dir
        self.pcc = pcc
        self.atol = atol
        self.rtol = rtol
        self.check_atol = check_atol
        self.check_rtol = check_rtol
        self.save_golden_tensors = save_golden_tensors
        self.logging = logging
        self.golden_report = golden_report
        self.counter = -1
        self.goldens = goldens

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

        self.print(f"Saved golden report to={golden_report_path}")

    def save_memory_report(self, memory_report_path):
        with open(memory_report_path, "w") as json_file:
            json.dump(self.memory_report, json_file, indent=4)

        self.print(f"Saved memory report to={memory_report_path}")

    def check_pcc(self):
        for loc, device_data in self.golden_report.items():
            for device_id, golden_data in device_data.items():
                if golden_data["actual_pcc"] < golden_data["expected_pcc"]:
                    raise PCCErrorException(
                        f"Failed: golden comparison failed at loc={loc} for device={device_id}, actual_pcc={golden_data['actual_pcc']} < expected_pcc={golden_data['expected_pcc']}"
                    )


def pre_op_callback(callback_runtime_config, binary, program_context, op_context):
    # Pre_callback logic to be implemented here
    pass


def pre_op_get_callback_fn(callback_runtime_config):
    return partial(pre_op_callback, callback_runtime_config)


def post_op_callback(callback_runtime_config, binary, program_context, op_context):
    import torch
    import ttrt.runtime

    logging = callback_runtime_config.logging
    loc = ttrt.runtime.get_op_loc_info(op_context)

    op_output_tensor_map = ttrt.runtime.get_op_output_tensor(
        op_context, program_context
    )
    if len(op_output_tensor_map) == 0:
        print("Output tensor is empty - skipping golden comparison")
        return

    op_golden_tensor_map = callback_runtime_config.goldens[
        loc
    ]  # binary.get_debug_info_golden(loc)
    if len(op_golden_tensor_map) == 0:
        # try getting golden tensor using the loc before it was modified by passes
        loc = get_original_op_loc(loc)
        op_golden_tensor_map = binary.get_debug_info_golden(loc)
        if len(op_golden_tensor_map) == 0:
            print("Golden tensor is None - skipping golden comparison")
            return

    # loop through all devices and compare golden tensors
    device_results = {}
    for device_id, op_golden_tensor in op_golden_tensor_map.items():
        if device_id not in op_output_tensor_map.keys():
            print(
                f"Device {device_id} does not have an output tensor - skipping golden comparison"
            )
            continue

        op_output_tensor = op_output_tensor_map[device_id]
        rt_buffer = op_output_tensor.get_data_buffer()
        print("3")
        dtype = passes_ttrt_datatype_to_torch_dtype(op_golden_tensor.dtype)
        print("4")
        golden_tensor_torch = passes_golden_tensor_to_torch(op_golden_tensor).flatten()

        output_tensor_torch = torch.frombuffer(rt_buffer, dtype=dtype).flatten()
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
            print(
                "Golden and output tensor shapes do not match - skipping golden comparison"
            )
            return

        cal_atol, cal_rtol, cal_pcc, output_str = get_atol_rtol_pcc(
            golden_tensor_torch,
            output_tensor_torch,
            callback_runtime_config.atol,
            callback_runtime_config.rtol,
            logging,
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

        print(f"For device {device_id} at loc={loc}, PCC={cal_pcc}")
        print(output_str)

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
