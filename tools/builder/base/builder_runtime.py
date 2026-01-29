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


class TTBuilderCompileException(Exception):
    pass


class TTBuilderRuntimeException(Exception):
    pass


class TTBuilderGoldenException(Exception):
    pass


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
        enable_golden=verify_intermediates,
        enable_memory=dump_memory,
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
