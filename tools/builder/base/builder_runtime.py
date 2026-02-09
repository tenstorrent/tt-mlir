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

import ttrt.runtime._ttmlir_runtime as tt_runtime


class TTBuilderCompileException(Exception):
    pass


class TTBuilderRuntimeException(Exception):
    pass


class TTBuilderGoldenException(Exception):
    pass


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


class CallbackRuntimeConfig:
    def __init__(
        self,
        device=None,
        pcc=0.99,
        atol=1e-08,
        rtol=1e-05,
        check_pcc: bool = True,
        check_atol: bool = True,
        check_rtol: bool = True,
        goldens={},
        bypass_ops=None,
        save_artifacts: bool = False,
        artifact_dir: str = ".",
        verify_intermediates: bool = False,
        save_memory: bool = False,
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
        self.verify_intermediates = verify_intermediates
        self.save_memory = save_memory
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
    if callback_runtime_config.verify_intermediates:
        golden(callback_runtime_config, binary, program_context, op_context)

    if callback_runtime_config.save_memory:
        memory(callback_runtime_config, binary, program_context, op_context)


def post_op_get_callback_fn(callback_runtime_config):
    return partial(post_op_callback, callback_runtime_config)


def convert_golden_intermediates_to_torch(
    goldens: Dict[str, Dict[int, GoldenMapTensor]],
) -> Dict[str, Dict[int, torch.Tensor]]:
    golden_torch_tensors = {}

    for loc, golden in goldens.items():
        golden_torch_tensors[loc] = golden.golden_map_tensor_as_torch_tensors()

    return golden_torch_tensors


def convert_golden_input_output_to_torch(
    goldens: Dict[int, Dict[str, Dict[int, GoldenMapTensor]]],
) -> Dict[int, Dict[str, Dict[int, torch.Tensor]]]:
    golden_torch_tensors = {}

    for program_index, loc_map in goldens.items():
        golden_torch_tensors[program_index] = {}
        for loc, golden in loc_map.items():
            golden_torch_tensors[program_index][
                loc
            ] = golden.golden_map_tensor_as_torch_tensors()

    return golden_torch_tensors


def execute_fb(
    compiled_bin,
    input_output_goldens: Dict[int, Dict[str, Dict[int, GoldenMapTensor]]] = None,
    intermediate_goldens: Dict[str, Dict[int, GoldenMapTensor]] = None,
    pcc: float = 0.99,
    atol: float = 1e-08,
    rtol: float = 1e-05,
    disable_golden: bool = False,
    device=None,
    check_pcc: bool = False,
    check_atol: bool = False,
    check_rtol: bool = False,
    enable_intermediate_verification: bool = False,
    bypass_ops: List[str] = None,
    save_artifacts: bool = False,
    artifact_dir: str = ".",
    dump_memory: bool = False,
):
    """
    Execute a flatbuffer binary on device and compare device outputs against goldens.

    Parameters
    ----------
    compiled_bin : Any
        The compiled flatbuffer capsule/binary for TTNN/TTMetal runtime.
    input_output_goldens : Dict[int, Dict[str, Dict[int, GoldenMapTensor]]]
        Per-program map of input/output goldens from the builder.
    intermediate_goldens : Dict[str, Dict[int, GoldenMapTensor]]
        Map of intermediate op-location goldens for debug hooks.
    pcc : float
        Threshold for PCC comparison.
    atol : float
        Absolute tolerance for comparisons.
    rtol : float
        Relative tolerance for comparisons.
    disable_golden : bool
        When True, skips golden comparison and uses random inputs.
    device : Optional
        tt_runtime device handle to execute on.
    check_pcc : bool
        Enable PCC check. TTBuilderGoldenException will be raised if PCC is below threshold.
    check_atol : bool
        Enable absolute tolerance check. TTBuilderGoldenException will be raised if absolute tolerance is above threshold.
    check_rtol : bool
        Enable relative tolerance check. TTBuilderGoldenException will be raised if relative tolerance is above threshold.
    enable_intermediate_verification : bool
        Enable runtime callbacks to verify intermediate device outputs match intermediate golden outputs.
    bypass_ops : List[str]
        List of op locations to bypass. Runtime outputs will be replaced on device with intermediate golden tensors to allow for continued intermediate golden verification.
    save_artifacts : bool
        Save output tensors (and intermediate tensors if intermediate verification is enabled) and golden reports to `artifact_dir`.
    artifact_dir : str
        Root directory for artifacts.
    dump_memory : bool
        Dump a per-op memory report into the artifact_dir.

    Returns
    -------
    Tuple[Dict[str, Dict], Dict[str, Dict]]
        golden_report, output_tensors
    """
    fbb = tt_runtime.binary.load_binary_from_capsule(compiled_bin)
    program_indices = range(fbb.get_num_programs())
    golden_input_output_tensors = convert_golden_input_output_to_torch(
        input_output_goldens
    )
    golden_intermediate_torch_tensors = convert_golden_intermediates_to_torch(
        intermediate_goldens
    )
    output_tensors = {}
    golden_report = {}
    if bypass_ops is None:
        bypass_ops = []
    verify_intermediates = enable_intermediate_verification or len(bypass_ops) > 0
    if input_output_goldens is None:
        disable_golden = True

    callback_runtime_config = CallbackRuntimeConfig(
        device=device,
        pcc=pcc,
        atol=atol,
        rtol=rtol,
        check_pcc=check_pcc,
        check_atol=check_atol,
        check_rtol=check_rtol,
        goldens=golden_intermediate_torch_tensors,
        bypass_ops=bypass_ops,
        save_artifacts=save_artifacts,
        artifact_dir=artifact_dir,
        verify_intermediates=verify_intermediates,
        save_memory=dump_memory,
    )

    if verify_intermediates or dump_memory:
        tt_runtime.runtime.DebugHooks.get(
            pre_op_get_callback_fn(callback_runtime_config),
            post_op_get_callback_fn(callback_runtime_config),
        )

    for program_index in program_indices:
        if fbb.is_program_private(program_index):
            continue

        program_artifact_dir = os.path.join(artifact_dir, f"program_{program_index}")
        if save_artifacts or dump_memory:
            os.makedirs(program_artifact_dir, exist_ok=True)

        callback_runtime_config.start_new_program(program_artifact_dir)
        program_golden_report = {}
        program_output_tensors = {}

        input_dict = program_inputs_as_dict(fbb, program_index)
        output_dict = program_outputs_as_dict(fbb, program_index)

        golden_inputs_torch = []
        for i, i_dict in enumerate(input_dict):
            if not disable_golden:
                golden_inputs_torch.append(
                    golden_input_output_tensors[program_index][f"input_{i}"][0]
                )
            else:
                torch_tensor = torch.randn(
                    i_dict["desc"]["shape"],
                    dtype=runtime_str_dtype_to_torch_dtype(
                        i_dict["desc"]["layout"]["memory_desc"]["data_type"]
                    ),
                )
                golden_inputs_torch.append(torch_tensor)

        golden_outputs_torch = []
        outputs_torch = []
        for i, o_dict in enumerate(output_dict):
            if not disable_golden:
                golden_outputs_torch.append(
                    golden_input_output_tensors[program_index][f"output_{i}"][0]
                )

            torch_tensor = torch.zeros(
                o_dict["desc"]["shape"],
                dtype=runtime_str_dtype_to_torch_dtype(
                    o_dict["desc"]["layout"]["memory_desc"]["data_type"]
                ),
            )
            outputs_torch.append(torch_tensor)

        inputs = []
        outputs = []
        for i in golden_inputs_torch:
            new_input = create_tensor(i)
            inputs.append(new_input)
        converted_inputs = convert_input_layouts(
            device,
            inputs,
            fbb=fbb,
            program_index=program_index,
        )

        for i in outputs_torch:
            new_output = create_tensor(i)
            outputs.append(new_output)

        start_submit = time.perf_counter_ns()
        try:
            runtime_outputs = tt_runtime.runtime.submit(
                device,
                fbb,
                program_index,
                converted_inputs,
            )
            tt_runtime.runtime.wait(runtime_outputs)
        except Exception as e:
            raise TTBuilderRuntimeException(e)
        finally:
            tt_runtime.runtime.unregister_hooks()
        end_submit = time.perf_counter_ns()
        e2e_duration_nanoseconds_submit = end_submit - start_submit

        e2e_duration_nanoseconds_output = 0
        for i, runtime_output_tensor in enumerate(runtime_outputs):
            start_get_output = time.perf_counter_ns()
            output_host = tt_runtime.runtime.to_host(
                runtime_output_tensor, untilize=True
            )[0]
            end_get_output = time.perf_counter_ns()
            e2e_duration_nanoseconds_output += end_get_output - start_get_output

            if disable_golden:
                continue

            tt_runtime.runtime.memcpy(
                outputs[i],
                output_host,
            )
            tt_runtime.runtime.deallocate_tensor(runtime_output_tensor, force=True)

            data_buffer = bytearray(outputs[i].get_data_buffer())

            if len(data_buffer) == 0:
                output_tensor_torch = torch.empty(
                    outputs[i].get_shape(),
                    dtype=runtime_dtype_to_torch_dtype(outputs[i].get_dtype()),
                )
            else:
                output_tensor_torch = torch.frombuffer(
                    data_buffer,
                    dtype=runtime_dtype_to_torch_dtype(outputs[i].get_dtype()),
                ).reshape(outputs[i].get_shape())

            golden_tensor_torch = golden_outputs_torch[i]
            results = check_outputs(
                golden_tensor_torch,
                output_tensor_torch,
                f"output_{i}",
                pcc,
                atol,
                rtol,
                check_pcc,
                check_atol,
                check_rtol,
            )

            program_golden_report[f"output_{i}"] = {0: results}
            program_output_tensors[f"device_output_{i}"] = output_tensor_torch
            program_output_tensors[f"golden_output_{i}"] = golden_tensor_torch

            if save_artifacts:
                save_torch_tensor(
                    output_tensor_torch,
                    program_artifact_dir,
                    f"device_output_{i}.pt",
                )
                save_torch_tensor(
                    golden_tensor_torch,
                    program_artifact_dir,
                    f"golden_output_{i}.pt",
                )

            for loc, device_results in callback_runtime_config.golden_report.items():
                program_golden_report[loc] = device_results

            if save_artifacts:
                golden_file = os.path.join(program_artifact_dir, "golden_report.json")
                with open(golden_file, "w") as f:
                    json.dump(program_golden_report, f, indent=4)

            if dump_memory:
                memory_file = os.path.join(
                    program_artifact_dir,
                    "memory_report.json",
                )
                with open(memory_file, "w") as f:
                    json.dump(callback_runtime_config.memory_report, f, indent=4)

            golden_report[f"program_{program_index}"] = program_golden_report
            output_tensors[f"program_{program_index}"] = program_output_tensors

    return golden_report, output_tensors


def execute_py(
    compiled_bin,
    input_output_goldens: Dict[int, Dict[str, Dict[int, GoldenMapTensor]]] = None,
    pcc: float = 0.99,
    atol: float = 1e-08,
    rtol: float = 1e-05,
    disable_golden: bool = False,
    check_pcc: bool = False,
    check_atol: bool = False,
    check_rtol: bool = False,
    save_artifacts: bool = False,
    artifact_dir: str = ".",
):
    """
    Execute an EmitPy Dylib and compare device outputs against goldens.

    Parameters
    ----------
    compiled_bin : str
        The compiled Python source string (EmitPy) containing program functions.
    input_output_goldens : Dict[int, Dict[str, Dict[int, GoldenMapTensor]]]
        Per-program input/output goldens for comparison.
    pcc : float
        Threshold for PCC comparison.
    atol : float
        Absolute tolerance for comparisons.
    rtol : float
        Relative tolerance for comparisons.
    disable_golden : bool
        When True, skips golden comparison.
    check_pcc : bool
        Enable PCC check. TTBuilderGoldenException will be raised if PCC is below threshold.
    check_atol : bool
        Enable absolute tolerance check. TTBuilderGoldenException will be raised if absolute tolerance is above threshold.
    check_rtol : bool
        Enable relative tolerance check. TTBuilderGoldenException will be raised if relative tolerance is above threshold.
    save_artifacts : bool
        Save output tensors and golden reports to `artifact_dir`.
    artifact_dir : str
        Root directory for artifacts.

    Returns
    -------
    Tuple[Dict[str, Dict], Dict[str, Dict]]
        golden_report, output_tensors
    """
    import importlib.util
    import types

    # Add tt-alchemist utils.py to path for EmitPy tests
    TT_MLIR_HOME = Path(os.environ.get("TT_MLIR_HOME", os.getcwd())).resolve()
    utils_path = os.path.join(TT_MLIR_HOME, "tools/tt-alchemist/templates/python/local")
    if utils_path not in sys.path:
        sys.path.append(utils_path)

    # Add tt-metal ttnn package to path
    TT_METAL_RUNTIME_ROOT = Path(
        os.environ.get("TT_METAL_RUNTIME_ROOT", os.getcwd())
    ).resolve()
    sys.path.append(os.path.join(TT_METAL_RUNTIME_ROOT, "ttnn"))

    import ttnn

    if input_output_goldens is None:
        disable_golden = True
    golden_input_output_tensors = convert_golden_input_output_to_torch(
        input_output_goldens
    )
    output_tensors = {}
    golden_report = {}

    try:
        # Parse the AST to find function names from the compiled source
        tree = ast.parse(compiled_bin)
        program_names = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name != "main"
                and node.name[0:18] != "create_inputs_for_"
                and not node.name.__contains__("_const_eval_")
                # TODO(dmilinkovic): this is getting out of hand, issue #6386.
                and not node.name.__contains__("hoisted_")
            ):
                program_names.append(node.name)

        module_name = program_names[0] if program_names else "emitpy_module"
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module
        exec(compile(compiled_bin, filename=module_name, mode="exec"), module.__dict__)

        for program_index, program_name in enumerate(program_names):
            program_golden_report = {}
            program_output_tensors = {}
            create_program_inputs = "create_inputs_for_" + program_name
            create_inputs_func = getattr(module, create_program_inputs)
            inputs = create_inputs_func()

            if not disable_golden:
                corrected_inputs = []
                golden_input_outputs = golden_input_output_tensors[program_index]

                for input_index, template_input in enumerate(inputs):
                    # Use the layout and device from the template_input
                    golden_input = golden_input_outputs[f"input_{input_index}"][0]
                    corrected_inputs.append(
                        ttnn.as_tensor(
                            golden_input,
                            dtype=template_input.dtype,
                            layout=template_input.layout,
                            device=template_input.device(),
                            memory_config=template_input.memory_config(),
                        )
                    )
                    # Deallocate template_input tensor
                    ttnn.deallocate(template_input)
                inputs = corrected_inputs

            program_func = getattr(module, program_name)
            outputs = program_func(inputs)

            if not disable_golden:
                for i, output in enumerate(outputs):
                    output_host = ttnn.from_device(output)
                    output_tensor_torch = output_host.to_torch()
                    golden_tensor_torch = golden_input_outputs[f"output_{i}"][0]

                    results = check_outputs(
                        golden_tensor_torch,
                        output_tensor_torch,
                        f"output_{i}",
                        pcc,
                        atol,
                        rtol,
                        check_pcc,
                        check_atol,
                        check_rtol,
                    )

                    program_golden_report[f"output_{i}"] = {0: results}
                    program_output_tensors[f"device_output_{i}"] = output_tensor_torch
                    program_output_tensors[f"golden_output_{i}"] = golden_tensor_torch

                    if save_artifacts:
                        program_artifact_dir = os.path.join(
                            artifact_dir, f"program_{program_index}"
                        )
                        os.makedirs(program_artifact_dir, exist_ok=True)
                        save_torch_tensor(
                            output_tensor_torch,
                            program_artifact_dir,
                            f"device_output_{i}.pt",
                        )
                        save_torch_tensor(
                            golden_tensor_torch,
                            program_artifact_dir,
                            f"golden_output_{i}.pt",
                        )

                if save_artifacts:
                    artifact_file = os.path.join(
                        artifact_dir, f"program_{program_index}", "golden_report.json"
                    )
                    with open(artifact_file, "w") as f:
                        json.dump(program_golden_report, f, indent=4)

                golden_report[f"program_{program_index}"] = program_golden_report
                output_tensors[f"program_{program_index}"] = program_output_tensors

    except Exception as e:
        raise TTBuilderRuntimeException(e) from e

    return golden_report, output_tensors


def execute_cpp(
    cpp_path: str,
    input_output_goldens: Dict[int, Dict[str, Dict[int, GoldenMapTensor]]] = None,
    pcc: float = 0.99,
    atol: float = 1e-08,
    rtol: float = 1e-05,
    disable_golden: bool = False,
    device=None,
    check_pcc: bool = False,
    check_atol: bool = False,
    check_rtol: bool = False,
    save_artifacts: bool = False,
    artifact_dir: str = ".",
):
    """
    Compile EmitC C++ file to a shared object, execute, and compare outputs.

    Parameters
    ----------
    cpp_path : str
        Path to the generated EmitC C++ source.
    input_output_goldens : Dict[int, Dict[str, Dict[int, GoldenMapTensor]]]
        Per-program input/output goldens for comparison.
    pcc : float
        Threshold for PCC comparison.
    atol : float
        Absolute tolerance for comparisons.
    rtol : float
        Relative tolerance for comparisons.
    disable_golden : bool
        When True, skips golden comparison.
    device : Optional
        tt_runtime device handle to execute on.
    check_pcc : bool
        Enable PCC check. TTBuilderGoldenException will be raised if PCC is below threshold.
    check_atol : bool
        Enable absolute tolerance check. TTBuilderGoldenException will be raised if absolute tolerance is above threshold.
    check_rtol : bool
        Enable relative tolerance check. TTBuilderGoldenException will be raised if relative tolerance is above threshold.
    save_artifacts : bool
        Save output tensors and golden reports to `artifact_dir`.
    artifact_dir : str
        Root directory for artifacts.

    Returns
    -------
    Tuple[Dict[str, Dict], Dict[str, Dict]]
        golden_report, output_tensors
    """
    # Add ttnn-standalone to sys.path for emitc compilation
    TT_MLIR_HOME = Path(os.environ.get("TT_MLIR_HOME", os.getcwd())).resolve()
    ttnn_standalone_path = os.path.join(TT_MLIR_HOME, "tools/ttnn-standalone")
    if ttnn_standalone_path not in sys.path:
        sys.path.append(ttnn_standalone_path)

    from emitc_compiler import compile_emitc_to_so

    metal_lib_dir = os.environ.get("TT_METAL_LIB")
    if metal_lib_dir is None:
        TT_METAL_RUNTIME_ROOT = Path(
            os.environ.get("TT_METAL_RUNTIME_ROOT", os.getcwd())
        ).resolve()
        metal_lib_candidates = [
            p for p in TT_METAL_RUNTIME_ROOT.glob("build*/lib") if p.is_dir()
        ]
        # if len(metal_lib_candidates) != 1:
        #    found = "\n".join(f"- {p}" for p in metal_lib_candidates) or "- <none>"
        #    raise TTBuilderRuntimeException(
        #        "Expected exactly one TT-Metal build lib directory matching "
        #        f"`{TT_METAL_RUNTIME_ROOT}/build*/lib`, but found {len(metal_lib_candidates)}:\n"
        #        f"{found}"
        #    )
        metal_lib_dir = str(metal_lib_candidates[0])

    output_dir = os.path.dirname(cpp_path)
    compile_emitc_to_so(
        cpp_path,
        output_dir,
        metal_lib_dir=metal_lib_dir,
    )
    so_path = cpp_path.replace(".cpp", ".so")

    if input_output_goldens is None:
        disable_golden = True
    golden_input_output_tensors = convert_golden_input_output_to_torch(
        input_output_goldens
    )
    output_tensors = {}
    golden_report = {}

    try:
        emitc_dylib_handle = tt_runtime.runtime.test.open_so(so_path)
        program_names = tt_runtime.runtime.test.get_so_programs(
            emitc_dylib_handle, so_path
        )

        for program_index, program_name in enumerate(program_names):
            program_golden_report = {}
            program_output_tensors = {}

            inputs = tt_runtime.runtime.test.create_inputs(
                emitc_dylib_handle,
                program_name,
                device,
                so_path,
            )
            if not disable_golden:
                corrected_inputs = []
                golden_input_outputs = golden_input_output_tensors[program_index]

                for input_index, template_input in enumerate(inputs):
                    # Use the layout from the template_input to convert the golden input
                    golden_input = golden_input_outputs[f"input_{input_index}"][0]
                    new_input = create_tensor(golden_input)
                    corrected_inputs.append(new_input)

                inputs = convert_input_layouts(
                    device,
                    corrected_inputs,
                    template_inputs=inputs,
                )

            outputs = tt_runtime.runtime.test.run_so_program(
                emitc_dylib_handle,
                program_name,
                inputs,
                device,
            )
            outputs = [
                tt_runtime.runtime.to_host(out, untilize=True)[0] for out in outputs
            ]

            if not disable_golden:
                for i, output in enumerate(outputs):
                    golden_tensor_torch = golden_input_outputs[f"output_{i}"][0]
                    data_buffer = bytearray(output.get_data_buffer())

                    if len(data_buffer) == 0:
                        output_tensor_torch = torch.empty(
                            output.get_shape(),
                            dtype=runtime_dtype_to_torch_dtype(output.get_dtype()),
                        )
                    else:
                        output_tensor_torch = torch.frombuffer(
                            data_buffer,
                            dtype=runtime_dtype_to_torch_dtype(output.get_dtype()),
                        ).reshape(output.get_shape())

                    results = check_outputs(
                        golden_tensor_torch,
                        output_tensor_torch,
                        f"output_{i}",
                        pcc,
                        atol,
                        rtol,
                        check_pcc,
                        check_atol,
                        check_rtol,
                    )

                    program_golden_report[f"output_{i}"] = {0: results}
                    program_output_tensors[f"device_output_{i}"] = output_tensor_torch
                    program_output_tensors[f"golden_output_{i}"] = golden_tensor_torch

                    if save_artifacts:
                        program_artifact_dir = os.path.join(
                            artifact_dir, f"program_{program_index}"
                        )
                        os.makedirs(program_artifact_dir, exist_ok=True)
                        save_torch_tensor(
                            output_tensor_torch,
                            program_artifact_dir,
                            f"device_output_{i}.pt",
                        )
                        save_torch_tensor(
                            golden_tensor_torch,
                            program_artifact_dir,
                            f"golden_output_{i}.pt",
                        )

                if save_artifacts:
                    artifact_file = os.path.join(
                        artifact_dir, f"program_{program_index}", "golden_report.json"
                    )
                    with open(artifact_file, "w") as f:
                        json.dump(program_golden_report, f, indent=4)

                golden_report[f"program_{program_index}"] = program_golden_report
                output_tensors[f"program_{program_index}"] = program_output_tensors

    except Exception as e:
        raise TTBuilderRuntimeException(e) from e

    return golden_report, output_tensors
