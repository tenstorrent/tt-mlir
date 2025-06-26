# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from op_by_op_infra.mlir_module_executor import (
    ExecutionPhase,
    ExecutionResult,
    MLIRModuleExecutor,
)
from op_by_op_infra.utils import ModuleWrapper
from ttmlir.compile_and_run_utils import ModuleDialect

from .fixtures import *


def test_shlo_compile(shlo_module_str: str):
    ex = MLIRModuleExecutor()
    result: ModuleWrapper = ex.compile(shlo_module_str)

    assert result.dialect == ModuleDialect.TTNN


def test_shlo_execute(shlo_module_str: str):
    ex = MLIRModuleExecutor()
    result: ExecutionResult = ex.execute(shlo_module_str)

    assert result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER


def test_ttir_compile(ttir_module_str: str):
    ex = MLIRModuleExecutor()
    result: ModuleWrapper = ex.compile(ttir_module_str)

    assert result.dialect == ModuleDialect.TTNN


def test_ttir_execute(ttir_module_str: str):
    ex = MLIRModuleExecutor()
    result: ExecutionResult = ex.execute(ttir_module_str)

    assert result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER


def test_compile(ttnn_module_str: str):
    ex = MLIRModuleExecutor()
    result: ModuleWrapper = ex.compile(ttnn_module_str)

    assert result.dialect == ModuleDialect.TTNN


def test_execute(ttnn_module_str: str):
    ex = MLIRModuleExecutor()
    result: ExecutionResult = ex.execute(ttnn_module_str)

    assert result.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
