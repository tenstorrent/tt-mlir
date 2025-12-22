# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
This file exposes hooks into op by op infra, aka workflows for frontends to use.
It is meant to simplify and abstract away all complexities of the infra.
"""

from typing import List, Optional

from ttmlir.ir import Module

from . import workflow_internal
from .mlir_module_splitter import MLIRModuleSplitter
from .pydantic_models import OpTest
from .utils import OpWrapper


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
    module: Module | str, origin_model: str = ""
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
    splitter.split(module, origin_model=origin_model)
    return splitter.sub_ops
