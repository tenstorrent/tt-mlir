# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ttmlir.ir import Module

from .compile_and_run import stablehlo_to_ttir, ttir_to_ttnn
from .mlir_module_executor import ExecutionPhase, MLIRModuleExecutor


class StableHLOExecutor(MLIRModuleExecutor):
    """Executor for StableHLO MLIR modules."""

    # ----- Public methods -----

    def __init__(self) -> None:
        super().__init__(ExecutionPhase.GENERATED_STABLE_HLO)

    # ----- Private methods -----

    # @override
    def _compile(self) -> Module:
        try:
            print(self._module)
            ttir = stablehlo_to_ttir(self._module)
            print(ttir)
            self._mark_execution_step(ExecutionPhase.GENERATED_TTIR, ttir)

            ttnn = ttir_to_ttnn(ttir)
            self._mark_execution_step(ExecutionPhase.GENERATED_TTNN, ttnn)
        finally:
            return self._execution_result.last_generated_module
