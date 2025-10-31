# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Example script that runs the op-by-op workflow.

This example allows the user to:
1. Read input MLIR module from a file
2. Choose between different op-by-op workflow types

Requires:
    # Building with TTMLIR_ENABLE_RUNTIME=ON and TTMLIR_ENABLE_STABLEHLO=ON
    cmake -G Ninja -B build -DTTMLIR_ENABLE_RUNTIME=ON -DTTMLIR_ENABLE_STABLEHLO=ON
    cmake --build build

    # Having system descriptor saved
    ttrt query --save-artifacts

Usage Examples:
    # Generate JSON report with pytest
    pytest -svv test/python/op_by_op/op_by_op_read_ir_from_file.py::test_op_by_op_inference_from_file --json-report --json-report-file=report.json

    # Use a custom MLIR file path
    export OP_BY_OP_MLIR_FILE_PATH=/path/to/your/model.mlir
    pytest -svv test/python/op_by_op/op_by_op_read_ir_from_file.py::test_op_by_op_inference_from_file --json-report --json-report-file=report.json

    # Try different workflow types (currently unsupported)
    export OP_BY_OP_WORKFLOW_TYPE=compile_split_and_execute
    pytest -svv test/python/op_by_op/op_by_op_read_ir_from_file.py::test_op_by_op_inference_from_file --json-report --json-report-file=report.json

    # Enable compile-only mode
    export OP_BY_OP_COMPILE_ONLY=true
    pytest -svv test/python/op_by_op/op_by_op_read_ir_from_file.py::test_op_by_op_inference_from_file --json-report --json-report-file=report.json

Workflow Types:
    - split_and_execute: Split module into individual ops, then execute (default)
    - compile_split_and_execute: Compile the full module first, then split and execute
    - split_compile_split_and_execute: Split first, compile each op individually, split then execute

Environment Variables:
    - OP_BY_OP_MLIR_FILE_PATH: Path to IR file (default: test/python/op_by_op/example_shlo_ir.mlir)
    - OP_BY_OP_WORKFLOW_TYPE: Workflow execution strategy (default: split_and_execute)
    - OP_BY_OP_COMPILE_ONLY: Only compile operations without executing them (default: false)

Note:
    - File should contain one MLIR module in StableHLO dialect
"""

import os
from pathlib import Path

from op_by_op_infra import workflow_internal
from op_by_op_infra.pydantic_models import OpTest, model_to_dict


def run_op_by_op_workflow(
    workflow_type: str = "split_and_execute",
    compile_only: bool = False,
):
    file_path = _get_mlir_file_path()

    module = _read_mlir_file(file_path)
    if module is None:
        raise AssertionError("Failed to read MLIR module")

    print(f"INFO: Running {workflow_type} workflow...")
    if compile_only:
        print("INFO: Compile-only mode enabled")

    if workflow_type == "split_and_execute":
        execution_results = workflow_internal.split_and_execute(
            module, compile_only=compile_only
        )
    elif workflow_type == "compile_split_and_execute":
        raise ValueError(f"Currently unsupported: {workflow_type}")
        # execution_results = workflow_internal.compile_split_and_execute(module)
    elif workflow_type == "split_compile_split_and_execute":
        raise ValueError(f"Currently unsupported: {workflow_type}")
        # execution_results = workflow_internal.split_compile_split_and_execute(module)
    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}")

    return workflow_internal.convert_results_to_pydantic_models(execution_results)


def _get_mlir_file_path() -> Path:
    env_path = os.getenv("OP_BY_OP_MLIR_FILE_PATH")

    if env_path:
        user_path = Path(env_path)
        print(f"INFO: Using user-specified MLIR file: {user_path}")
        return user_path

    default_path = Path(__file__).parent / "example_shlo_ir.mlir"
    print(f"INFO: Using default MLIR file: {default_path}")
    print(
        "INFO: To use a different file, set: export OP_BY_OP_MLIR_FILE_PATH=/path/to/your/file.mlir"
    )

    return default_path


def _read_mlir_file(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if content == "":
            raise ValueError(f"IR file is empty: {file_path}")

        print(f"INFO: Successfully read IR file ({len(content)} characters)")
        return content

    except Exception as e:
        raise RuntimeError(f"Error reading IR file {file_path}: {e}")


def test_op_by_op_inference_from_file(record_property):
    """
    Test function that generates JSON report with operation information.

    This test function runs the op-by-op workflow and records properties
    for each operation result, similar to the frontend workflow approach.
    When run with pytest --json-report --json-report-file=report.json,
    it will generate a JSON file with detailed operation information.
    """
    # Read all environment variables in the test function
    workflow_type = os.getenv("OP_BY_OP_WORKFLOW_TYPE", "split_and_execute")
    compile_only = os.getenv("OP_BY_OP_COMPILE_ONLY", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    # Execute the workflow
    results = run_op_by_op_workflow(workflow_type, compile_only)

    # Record properties for each operation result
    for result in results:
        record_property(f"OpTest model for: {result.op_name}", model_to_dict(result))

    # Also record summary information
    record_property("total_operations", len(results))
    successful_operations = sum(1 for r in results if r.success)
    failed_operations = sum(1 for r in results if not r.success)
    record_property("successful_operations", successful_operations)
    record_property("failed_operations", failed_operations)
    record_property("workflow_type", workflow_type)
    record_property("compile_only", compile_only)

    # Fail the test if there are any failed operations
    assert (
        failed_operations == 0
    ), f"Test failed: {failed_operations} operation(s) failed out of {len(results)} total operations"
