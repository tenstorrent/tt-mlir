# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
This file exposes hooks into op by op infra, aka workflows for frontends to use.
It is meant to simplify and abstract away all complexities of the infra.
"""

import os
import re
import subprocess
import sys
import tempfile
from typing import List, Optional

from ttmlir.ir import Module

from . import workflow_internal
from .mlir_module_splitter import MLIRModuleSplitter
from .pydantic_models import OpTest
from .utils import OpWrapper

_SYSTEM_DESC_PATH: Optional[str] = None


def _ensure_system_desc() -> None:
    """
    Generates the system descriptor via `_ttmlir_runtime` in a subprocess and
    points `SYSTEM_DESC_PATH` at it so the compiler pipeline can consume it.

    The subprocess is used so that device resources opened during descriptor
    generation are released cleanly when the subprocess exits, freeing the
    hardware for per-op flatbuffer runs later in the session.

    Generates once per Python process; subsequent calls are no-ops. Unconditionally
    overrides any pre-existing `SYSTEM_DESC_PATH`.
    """
    global _SYSTEM_DESC_PATH

    if _SYSTEM_DESC_PATH is not None and os.path.exists(_SYSTEM_DESC_PATH):
        os.environ["SYSTEM_DESC_PATH"] = _SYSTEM_DESC_PATH
        return

    path = os.path.join(
        tempfile.gettempdir(), f"op_by_op_system_desc_{os.getpid()}.ttsys"
    )

    from ttmlir import compile_and_run

    worker_path = os.path.join(
        os.path.dirname(compile_and_run.__file__), "_system_desc_worker.py"
    )

    try:
        subprocess.run(
            [sys.executable, worker_path, path],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Could not obtain system descriptor from _ttmlir_runtime "
            f"(subprocess exit {e.returncode}): {e.stderr}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"System descriptor generation timed out: {e}") from e

    if not os.path.exists(path):
        raise RuntimeError(
            f"System descriptor subprocess returned successfully but no file "
            f"was written at {path}"
        )

    _SYSTEM_DESC_PATH = path
    os.environ["SYSTEM_DESC_PATH"] = path


def run_op_by_op_workflow(
    module: Module | str,
    compile_before_split: bool = False,
    compile_each_submodule_after_split: bool = False,
    *,
    frontend: Optional[str] = None,
    model_name: Optional[str] = None,
) -> List[OpTest]:
    """
    Unified function to process a module based on selected compilation and splitting
    strategy.

    To enable showing progress of the workflow, set env var `SHOW_WORKFLOW_PROGRESS=ON`.

    Parameters
    ----------
    module: Module | str
        Original MLIR module (or module str) processed by the workflow.

    compile_before_split: bool
        If True, compiles the module before splitting.
        NOTE if True `compile_each_submodule_after_split` cannot be True.

    compile_each_submodule_after_split: bool
        If True, compiles each submodule after splitting.
        NOTE if True `compile_before_split` cannot be True.

    frontend: Optional[str]
        Name of the frontend using op by op infra.

    model_name: Optional[str]
        Name of the ML model which was passed as original MLIR module to the workflow.

    Returns
    -------
    List[OpTest]
        List of `OpTest` pydantic models
    """
    _ensure_system_desc()

    if compile_before_split:
        assert compile_each_submodule_after_split is not True, (
            f"Invalid workflow strategy. Cannot have `compile_before_split` and "
            f"`compile_each_submodule_after_split` both True at the same time."
        )
        execution_results = workflow_internal.compile_split_and_execute(module)

    if not compile_before_split:
        if not compile_each_submodule_after_split:
            execution_results = workflow_internal.split_and_execute(module)
        else:
            execution_results = workflow_internal.split_compile_split_and_execute(
                module
            )

    return workflow_internal.convert_results_to_pydantic_models(
        execution_results,
        frontend=frontend,
        model_name=model_name,
    )


def extract_ops_from_module(
    module: Module | str, *, origin_model: str = ""
) -> List[OpWrapper]:
    """
    Extracts operations from a module without executing them.

    Parameters
    ----------
    module : Module | str
        MLIR module (or module string) to extract ops from
    origin_model : str
        Name of the model this module originated from

    Returns
    -------
    List[OpWrapper]
        List of wrapped operations extracted from the module
    """
    splitter = MLIRModuleSplitter()
    sub_ops = splitter.split(module, origin_model=origin_model)
    return sub_ops


def _sanitize_filename(name: str) -> str:
    """Sanitizes a string to be safe for use as a filename."""
    # Replace characters that are invalid in filenames
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Replace dots (common in op names like "stablehlo.add") with underscores
    sanitized = sanitized.replace(".", "_")
    return sanitized


def _save_failed_ops(execution_results: list, folder_path: str) -> None:
    """Saves the last_generated_module for each failed op to a file in the specified folder."""
    failed_results = [
        (i, result)
        for i, result in enumerate(execution_results)
        if not result.device_run_passed
    ]

    if not failed_results:
        return

    os.makedirs(folder_path, exist_ok=True)

    for index, result in failed_results:
        op_name = result.last_generated_module.origin_op_name or "unknown_op"
        sanitized_name = _sanitize_filename(op_name)
        filename = f"{index:04d}_{sanitized_name}.mlir"
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "w") as f:
            f.write(str(result.last_generated_module.module))


def execute_extracted_ops(
    ops: List[OpWrapper],
    *,
    compile_only: bool = False,
    frontend: Optional[str] = None,
    debug_print: bool = False,
    failed_ops_folder: Optional[str] = None,
) -> List[OpTest]:
    """
    Takes a list of OpWrappers, makes a submodule out of each, compiles and executes them.

    Behavior is identical to split_and_execute but operates on pre-extracted ops.

    Parameters
    ----------
    ops : List[OpWrapper]
        List of wrapped operations to execute
    compile_only : bool
        If True, only compiles without executing on device
    frontend : Optional[str]
        Name of the frontend using op by op infra
    debug_print : bool
        If True, prints module at each compilation step (stablehlo -> ttir -> ttnn)
    failed_ops_folder : Optional[str]
        If provided, creates a folder at this path and saves the last_generated_module
        for each failed op (where device_run_passed is False) to a file in that folder

    Returns
    -------
    List[OpTest]
        List of OpTest pydantic models with execution results
    """
    _ensure_system_desc()

    executor = workflow_internal.MLIRModuleExecutor(
        compile_only, debug_print=debug_print
    )
    execution_results = []

    for op in workflow_internal.progress_bar(ops, desc="Executing submodules..."):
        try:
            sub_module = op.as_module()
        except Exception as e:
            print(f"ERROR: Failed to create module from op")
            print(f"Origin model: {op.origin_model}")
            print(f"Module string:\n{op.as_module_str()}")
            print(f"Exception: {e}")
            continue

        execution_result = executor.execute(sub_module)
        execution_results.append(execution_result)

    if failed_ops_folder is not None:
        _save_failed_ops(execution_results, failed_ops_folder)

    return workflow_internal.convert_results_to_pydantic_models(
        execution_results, frontend=frontend
    )
