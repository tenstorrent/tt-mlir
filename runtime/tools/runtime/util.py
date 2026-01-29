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


def runtime_dtype_to_torch_dtype(dtype) -> torch.dtype:
    if dtype == tt_runtime.runtime.DataType.Float32:
        return torch.float32
    elif dtype == tt_runtime.runtime.DataType.UInt32:
        return torch.uint32
    elif dtype == tt_runtime.runtime.DataType.UInt16:
        return torch.uint16
    elif dtype == tt_runtime.runtime.DataType.UInt8:
        return torch.uint8
    elif dtype == tt_runtime.runtime.DataType.BFloat16:
        return torch.bfloat16
    elif dtype == tt_runtime.runtime.DataType.Int32:
        return torch.int32


def torch_dtype_to_runtime_dtype(dtype):
    if dtype == torch.float32:
        return tt_runtime.runtime.DataType.Float32
    if dtype == torch.float16:
        return tt_runtime.runtime.DataType.Float16
    if dtype == torch.bfloat16:
        return tt_runtime.runtime.DataType.BFloat16
    if dtype == torch.uint32:
        return tt_runtime.runtime.DataType.UInt32
    if dtype == torch.uint16:
        return tt_runtime.runtime.DataType.UInt16
    if dtype == torch.uint8:
        return tt_runtime.runtime.DataType.UInt8
    if dtype == torch.int32:
        return tt_runtime.runtime.DataType.Int32
    if dtype == torch.float64:
        return tt_runtime.runtime.DataType.Float64
    if dtype == torch.int64:
        return tt_runtime.runtime.DataType.Int64
    if dtype == torch.uint64:
        return tt_runtime.runtime.DataType.UInt64
    if dtype == torch.int16:
        return tt_runtime.runtime.DataType.Int16
    if dtype == torch.int8:
        return tt_runtime.runtime.DataType.Int8
    if dtype == torch.bool:
        return tt_runtime.runtime.DataType.Bool
    raise ValueError(f"Torch dtype: {dtype} has no runtime DataType equivalent")


def runtime_str_dtype_to_torch_dtype(dtype):
    if dtype == "Float32":
        return torch.float32
    if dtype == "Float16":
        return torch.float16
    if dtype == "BFloat16":
        return torch.bfloat16
    if dtype == "UInt32":
        return torch.uint32
    if dtype == "UInt16":
        return torch.uint16
    if dtype == "UInt8":
        return torch.uint8
    if dtype == "Int32":
        return torch.int32
    if dtype == "Float64":
        return torch.float64
    if dtype == "Int64":
        return torch.int64
    if dtype == "UInt64":
        return torch.uint64
    if dtype == "Int16":
        return torch.int16
    if dtype == "Int8":
        return torch.int8
    if dtype == "Bool":
        return torch.bool

    raise ValueError(f"unsupported dtype: {dtype}")


def create_tensor(tensor):
    isEmptyTensor = not all(tensor.shape)
    if isEmptyTensor:
        return tt_runtime.runtime.create_owned_host_tensor(
            tensor.data_ptr(),
            list(tensor.shape),
            list(tensor.stride()),
            tensor.element_size(),
            torch_dtype_to_runtime_dtype(tensor.dtype),
        )

    return tt_runtime.runtime.create_borrowed_host_tensor(
        tensor.data_ptr(),
        list(tensor.shape),
        list(tensor.stride()),
        tensor.element_size(),
        torch_dtype_to_runtime_dtype(tensor.dtype),
    )


def convert_input_layouts(
    device: tt_runtime.runtime.Device,
    inputs: List[tt_runtime.runtime.Tensor],
    template_inputs: List[tt_runtime.runtime.Tensor] = None,
    fbb: tt_runtime.binary.Binary = None,
    program_index: int = None,
):
    inputs_converted = []
    for input_index in range(len(inputs)):
        if template_inputs:
            input_layout = template_inputs[input_index].get_layout()
        elif fbb and program_index is not None:
            input_layout = tt_runtime.runtime.get_layout(
                fbb, program_index, input_index
            )
        else:
            raise ValueError(
                "Either template_inputs or fbb and program_index must be provided"
            )
        inputs_converted.append(
            tt_runtime.runtime.to_layout(
                inputs[input_index], device, input_layout, True
            )
        )
    return inputs_converted


def json_string_as_dict(json_string):
    if json_string == "":
        return {}
    json_string = re.sub(r"\bnan\b", "NaN", json_string)
    json_string = re.sub(r"\binf\b", "Infinity", json_string)
    return json.loads(json_string)


def program_inputs_as_dict(bin, index):
    return json_string_as_dict(bin.get_program_inputs_as_json(index))


def program_outputs_as_dict(bin, index):
    return json_string_as_dict(bin.get_program_outputs_as_json(index))


def mask_torch_inf_nan(tensor):
    tensor[
        torch.logical_or(
            torch.isnan(tensor),
            torch.logical_or(torch.isinf(tensor), torch.isneginf(tensor)),
        )
    ] = 0
    return tensor


def get_atol_rtol_pcc(golden, calculated, atol, rtol):
    # Clone tensors to avoid modifying the originals
    golden = golden.clone()
    calculated = calculated.clone()
    if not torch.is_floating_point(golden):
        golden = golden.to(torch.float64)
    if not torch.is_floating_point(calculated):
        calculated = calculated.to(torch.float64)

    if golden.numel() == 0 or calculated.numel() == 0:
        cal_atol = 0.0
        cal_rtol = 0.0
    else:
        cal_atol = torch.max(torch.abs(golden - calculated)).item()
        cal_rtol = torch.max(torch.abs((golden - calculated) / calculated)).item()

    def get_pcc(golden, calculated):
        if golden.numel() == 0 and calculated.numel() == 0:
            if golden.shape == calculated.shape:
                return 1.0
            else:
                return 0.0
        elif golden.numel() == 0 or calculated.numel() == 0:
            return 0.0
        if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
            return 1.0
        elif torch.any(golden.bool()) != torch.any(calculated.bool()):
            return 0.0
        elif torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
            return 0.0
        else:
            golden = mask_torch_inf_nan(golden)
            calculated = mask_torch_inf_nan(calculated)

            if torch.equal(golden, calculated):
                return 1.0

            if golden.dtype == torch.bfloat16:
                golden = golden.type(torch.float32)
            if calculated.dtype == torch.bfloat16:
                calculated = calculated.type(torch.float32)

            if golden.numel() == 1:
                return float(torch.isclose(golden, calculated, atol=atol, rtol=rtol))

            if torch.max(golden) == torch.min(golden) and torch.max(
                calculated
            ) == torch.min(calculated):
                return float(
                    torch.isclose(
                        torch.max(golden), torch.max(calculated), atol=atol, rtol=rtol
                    ).item()
                )

            cal_pcc = np.ma.corrcoef(
                np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
                np.ma.masked_invalid(
                    torch.squeeze(calculated).detach().numpy()
                ).flatten(),
            )
            mask = np.ones(cal_pcc.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            cal_pcc = np.min(cal_pcc[mask])

            if isinstance(cal_pcc, np.ma.core.MaskedConstant):
                return 1.0

            return cal_pcc

    if golden.numel() == 1 and golden.item() != 0:
        cal_pcc = (
            1.0
            if torch.nn.functional.cosine_similarity(
                golden.float().unsqueeze(0),
                calculated.float().unsqueeze(0),
                dim=0,
            ).item()
            else 0.0
        )
    else:
        cal_pcc = get_pcc(golden, calculated)

    return (
        cal_atol,
        cal_rtol,
        cal_pcc,
    )


def check_outputs(
    golden_tensor,
    output_tensor,
    tensor_name,
    pcc,
    atol,
    rtol,
    check_pcc,
    check_atol,
    check_rtol,
    raise_exception=True,
):
    cal_atol, cal_rtol, cal_pcc, = get_atol_rtol_pcc(
        golden_tensor,
        output_tensor,
        atol,
        rtol,
    )

    if raise_exception:
        if check_pcc:
            if cal_pcc < pcc:
                raise TTBuilderGoldenException(
                    f"Failed: program-level output golden comparison failed, actual_pcc={cal_pcc} < expected_pcc={pcc}"
                )
            else:
                print(f"Program level golden for {tensor_name} matched. pcc={cal_pcc}")

        if check_atol:
            if cal_atol > atol:
                raise TTBuilderGoldenException(
                    f"Failed: program-level output atol check failed, actual_atol={cal_atol} > expected_atol={atol}"
                )
            else:
                print(
                    f"Program level atol check for {tensor_name} passed. atol={cal_atol}"
                )

        if check_rtol:
            if cal_rtol > rtol:
                raise TTBuilderGoldenException(
                    f"Failed: program-level output rtol check failed, actual_rtol={cal_rtol} > expected_rtol={rtol}"
                )
            else:
                print(
                    f"Program level rtol check for {tensor_name} passed. rtol={cal_rtol}"
                )

    result = "pass"
    if (
        (check_pcc and cal_pcc < pcc)
        or (check_atol and cal_atol > atol)
        or (check_rtol and cal_rtol > rtol)
    ):
        result = "fail"

    results = {}
    results["result"] = result
    results["expected_pcc"] = pcc
    results["actual_pcc"] = cal_pcc
    results["expected_atol"] = atol
    results["actual_atol"] = cal_atol
    results["expected_rtol"] = rtol
    results["actual_rtol"] = cal_rtol
    results["allclose"] = torch.allclose(
        golden_tensor,
        output_tensor,
        atol=atol,
        rtol=rtol,
    )
    if (
        golden_tensor.dtype not in (torch.uint16, torch.uint32)
        and golden_tensor.numel() > 0
    ):
        results["max"] = torch.max(torch.abs(golden_tensor - output_tensor)).item()
    results["mean_absolute_error"] = torch.mean(
        torch.abs(golden_tensor.float() - output_tensor.float())
    ).item()
    results["root_mean_square_error"] = torch.sqrt(
        torch.mean((golden_tensor.float() - output_tensor.float()) ** 2)
    ).item()
    golden_tensor = golden_tensor.flatten()
    output_tensor = output_tensor.flatten()
    results["cosine_similarity"] = torch.nn.functional.cosine_similarity(
        golden_tensor.float().unsqueeze(0),
        output_tensor.float().unsqueeze(0),
    ).item()

    return results


def get_sanitized_filename(name: str, replacement: str = "_") -> str:
    # make string safe for file name
    forbidden = ':"/\\|?*\0'
    s = re.sub(f"[{re.escape(forbidden)}\\x00-\\x1F]", replacement, name)
    if not s:
        s = "untitled"
    return s.strip()


def save_torch_tensor(torch_tensor, folder_path, torch_tensor_name):
    torch_tensor_name = get_sanitized_filename(torch_tensor_name)
    torch.save(torch_tensor, f"{folder_path}/{torch_tensor_name}")


def get_original_op_loc(text: str) -> str:
    try:
        segments = re.findall(r'"([^"]*)"', text)
        loc_str = f'"{segments[1]}"' if len(segments) >= 2 else ""
        return "loc(" + loc_str + ")"
    except Exception:
        return ""


def update_device_tensor(program_context, tensor_ref, dst_tensor, src_tensor):
    data_ptr = src_tensor.data_ptr()
    shape = dst_tensor.get_shape()
    stride = dst_tensor.get_stride()
    dtype = dst_tensor.get_dtype()
    size = torch.numel(src_tensor)
    tensor = tt_runtime.runtime.create_owned_host_tensor(
        data_ptr, shape, stride, size, dtype
    )
    tt_runtime.runtime.update_tensor_in_pool(program_context, tensor_ref, tensor)
