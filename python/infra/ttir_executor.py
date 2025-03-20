# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ttmlir.ir import Module

from .compile_and_run import ttir_to_ttnn
from .mlir_module_executor import ExecutionPhase, MLIRModuleExecutor


class TTIRExecutor(MLIRModuleExecutor):
    """Executor for TTIR MLIR modules."""

    # ----- Public methods -----

    def __init__(self) -> None:
        super().__init__(ExecutionPhase.GENERATED_TTIR)

    # ----- Private methods -----

    # @override
    def _compile(self) -> Module:
        try:
            ttnn = ttir_to_ttnn(self._module)
            self._mark_execution_step(ExecutionPhase.GENERATED_TTNN, ttnn)
        finally:
            return self._execution_result.last_generated_module
