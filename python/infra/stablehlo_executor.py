# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

from ttmlir.ir import Module
from ttrt.common.util import Binary

from .compile_and_run import (
    run_flatbuffer,
    stablehlo_to_ttir,
    ttir_to_ttnn,
    ttnn_to_flatbuffer,
)
from .mlir_module_executor import ExecutionPhase, ExecutionResult, MLIRModuleExecutor
from .stablehlo_splitter import StableHLOSplitter


class StableHLOExecutor(MLIRModuleExecutor):
    """Executor for StableHLO MLIR modules."""

    # ----- Public methods -----

    @staticmethod
    def create_from_module(module: Module) -> StableHLOExecutor:
        module_splitter = StableHLOSplitter.create_from_module(module)
        return StableHLOExecutor(module, module_splitter)

    @staticmethod
    def create_from_module_str(module_str: str) -> StableHLOExecutor:
        module_splitter = StableHLOSplitter.create_from_module_str(module_str)
        return StableHLOExecutor(module_splitter.get_module(), module_splitter)

    # ----- Private methods -----

    def __init__(self, module: Module, module_splitter: StableHLOSplitter) -> None:
        super().__init__(module, module_splitter, ExecutionPhase.GENERATED_STABLE_HLO)

    def _compile(self, module: Module) -> Module:
        try:
            ttir = stablehlo_to_ttir(module)
            self._mark_execution_step(ExecutionPhase.GENERATED_TTIR, ttir)

            ttnn = ttir_to_ttnn(ttir)
            self._mark_execution_step(ExecutionPhase.GENERATED_TTNN, ttnn)
        finally:
            return self._execution_result.last_generated_module

    def _generate_flatbuffer(
        self, ttnn_module: Module, flatbuffer_name: str = "ttnn_fb.ttnn"
    ) -> Optional[Binary]:
        try:
            flatbuffer = ttnn_to_flatbuffer(ttnn_module, flatbuffer_name)
            self._mark_execution_step(
                ExecutionPhase.GENERATED_FLATBUFFER, generated_flatbuffer=flatbuffer
            )
        finally:
            return self._execution_result.flatbuffer

    def _run(self, flatbuffer: Binary) -> bool:
        try:
            return_code = run_flatbuffer(flatbuffer)
            run_passed = return_code == 0
            self._mark_execution_step(
                ExecutionPhase.EXECUTED_FLATBUFFER, run_passed=run_passed
            )
        finally:
            return self._execution_result.device_run_passed

    def _compile_and_run(
        self, module: Module, flatbuffer_name: str = "ttnn_fb.ttnn"
    ) -> ExecutionResult:
        result = self._compile(module, flatbuffer_name)

        if not self._flatbuffer_generated:
            return result

        return self._run(result.flatbuffer)
