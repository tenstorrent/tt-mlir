# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ttmlir.ir import Module
from ttrt.common.util import Binary

from .compile_and_run import ttnn_to_flatbuffer
from .mlir_module_executor import CompileStep, MLIRModuleExecutor
from .ttnn_splitter import TTNNSplitter


class TTNNExecutor(MLIRModuleExecutor):
    """Compiler for TTNN MLIR modules."""

    # ----- Public methods -----

    @staticmethod
    def create_from_module(module: Module) -> TTNNExecutor:
        module_splitter = TTNNSplitter.create_from_module(module)
        return TTNNExecutor(module_splitter.get_module(), module_splitter)

    @staticmethod
    def create_from_module_str(module_str: str) -> TTNNExecutor:
        module_splitter = TTNNSplitter.create_from_module_str(module_str)
        return TTNNExecutor(module_splitter.get_module(), module_splitter)

    # ----- Private methods -----

    def __init__(self, module: Module, module_splitter: TTNNSplitter) -> None:
        super().__init__(module, module_splitter, CompileStep.GENERATED_TTNN)

    # @override
    def _compile(self, module: Module, flatbuffer_name: str = "ttnn_fb.ttnn") -> Binary:
        flatbuffer = ttnn_to_flatbuffer(module, flatbuffer_name)
        self._mark_compile_step(CompileStep.GENERATED_FLATBUFFER)

        return flatbuffer
