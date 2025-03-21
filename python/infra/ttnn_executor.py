# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from .mlir_module_executor import ExecutionPhase, MLIRModuleExecutor
from .utils import ModuleWrapper


class TTNNExecutor(MLIRModuleExecutor):
    """Executor for TTNN MLIR modules."""

    # ----- Public methods -----

    def __init__(self) -> None:
        super().__init__(ExecutionPhase.GENERATED_TTNN)

    # ----- Private methods -----

    # @override
    def _compile(self) -> ModuleWrapper:
        # Trivial, original module was already a TTNN module.
        return self._module
