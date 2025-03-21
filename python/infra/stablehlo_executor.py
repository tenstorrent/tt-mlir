# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .compile_and_run import stablehlo_to_ttir, ttir_to_ttnn
from .mlir_module_executor import ExecutionPhase, MLIRModuleExecutor
from .utils import ModuleWrapper


class StableHLOExecutor(MLIRModuleExecutor):
    """Executor for StableHLO MLIR modules."""

    # ----- Public methods -----

    def __init__(self) -> None:
        super().__init__(ExecutionPhase.GENERATED_STABLE_HLO)

    # ----- Private methods -----

    # @override
    def _compile(self) -> ModuleWrapper:
        # During compilation steps, keep in mind that compilation API uses MLIR `Module`.
        # Also, don't lose track of the possible originating op.
        try:
            shlo = self._module.module

            ttir = stablehlo_to_ttir(shlo)
            self._mark_execution_step(
                ExecutionPhase.GENERATED_TTIR,
                ModuleWrapper(ttir, generated_from_op=self._module.generated_from_op),
            )

            ttnn = ttir_to_ttnn(ttir)
            self._mark_execution_step(
                ExecutionPhase.GENERATED_TTNN,
                ModuleWrapper(ttnn, generated_from_op=self._module.generated_from_op),
            )
        finally:
            return self._execution_result.last_generated_module
