# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import time

from op_by_op_infra.mlir_module_executor import (
    ExecutionPhase,
    ExecutionResult,
    MLIRModuleExecutor,
)
from op_by_op_infra.utils import ModuleWrapper
from ttmlir.compile_and_run_utils import ModuleDialect

from .fixtures import *


def test_execution_result_timestamps_are_per_instance():
    """Each ExecutionResult must timestamp itself, not share an import-time value.

    Regression test: these fields used to default to a bare ``datetime.now()``,
    which a dataclass evaluates once at class creation. Every instance then
    reported the same start time, making per-op durations meaningless.
    """
    first = ExecutionResult(ExecutionPhase.GENERATED_STABLE_HLO, None)
    time.sleep(0.01)
    second = ExecutionResult(ExecutionPhase.GENERATED_STABLE_HLO, None)

    assert first.execution_started != second.execution_started
    assert second.execution_started > first.execution_started


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
