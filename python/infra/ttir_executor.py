# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ttmlir.ir import Module
from ttrt.common.util import Binary

from .compile_and_run import ttir_to_ttnn, ttnn_to_flatbuffer
from .mlir_module_executor import ExecutionStatus, MLIRModuleExecutor
from .ttir_splitter import TTIRSplitter


class TTIRExecutor(MLIRModuleExecutor):
    """Executor for TTIR MLIR modules."""

    # ----- Public methods -----

    @staticmethod
    def create_from_module(module: Module) -> TTIRExecutor:
        module_splitter = TTIRSplitter.create_from_module(module)
        return TTIRExecutor(module_splitter.get_module(), module_splitter)

    @staticmethod
    def create_from_module_str(module_str: str) -> TTIRExecutor:
        module_splitter = TTIRSplitter.create_from_module_str(module_str)
        return TTIRExecutor(module_splitter.get_module(), module_splitter)

    # ----- Private methods -----

    def __init__(self, module: Module, module_splitter: TTIRSplitter) -> None:
        super().__init__(module, module_splitter, ExecutionStatus.GENERATED_TTIR)

    # @override
    def _compile(self, module: Module, flatbuffer_name: str = "ttnn_fb.ttnn") -> Binary:
        ttnn = ttir_to_ttnn(module)
        self._mark_execution_status(ExecutionStatus.GENERATED_TTNN)

        flatbuffer = ttnn_to_flatbuffer(ttnn, flatbuffer_name)
        self._mark_execution_status(ExecutionStatus.GENERATED_FLATBUFFER)

        return flatbuffer
