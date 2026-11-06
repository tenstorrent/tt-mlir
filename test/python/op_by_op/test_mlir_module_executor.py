# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from datetime import datetime

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

    Asserted on the field definitions rather than by sleeping and comparing
    wall-clock values: `default_factory` *is* the fix, and checking it directly
    keeps the test independent of clock resolution and of the clock moving
    backwards.
    """
    fields = {f.name: f for f in dataclasses.fields(ExecutionResult)}

    for name in ("execution_started", "last_update"):
        field = fields[name]
        assert field.default is dataclasses.MISSING, (
            f"{name} must not carry a fixed default -- a bare datetime.now() is "
            f"evaluated once at class creation and shared by every instance"
        )
        # `==`, not `is`: `datetime.now` is a builtin method, and every attribute
        # access returns a fresh bound object, so identity never holds.
        assert field.default_factory == datetime.now

    # The factory therefore runs per instance rather than once at import.
    result = ExecutionResult(ExecutionPhase.GENERATED_STABLE_HLO, None)
    assert isinstance(result.execution_started, datetime)
    assert isinstance(result.last_update, datetime)


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
