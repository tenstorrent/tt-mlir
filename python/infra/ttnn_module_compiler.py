# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ttmlir.compiler_passes import ttnn_to_flatbuffer
from ttmlir.ir import Module
from ttmlir.ttnn_module_splitter import TTNNModuleSplitter
from ttrt.common.util import Binary

from .module_compiler import CompileStep, ModuleCompiler


class TTNNModuleCompiler(ModuleCompiler):
    """Compiler for TTNN MLIR modules."""

    # ----- Public methods -----

    @staticmethod
    def create_from_module(module: Module) -> TTNNModuleCompiler:
        module_splitter = TTNNModuleSplitter.create_from_module(module)
        return TTNNModuleCompiler(module_splitter.get_module(), module_splitter)

    @staticmethod
    def create_from_module_str(module_str: str) -> TTNNModuleCompiler:
        module_splitter = TTNNModuleSplitter.create_from_module_str(module_str)
        return TTNNModuleCompiler(module_splitter.get_module(), module_splitter)

    # ----- Private methods -----

    def __init__(self, module: Module, module_splitter: TTNNModuleSplitter) -> None:
        super().__init__(module, module_splitter, CompileStep.TTNN)

    # @override
    def _compile(self, module: Module, flatbuffer_name: str = "ttnn_fb.ttnn") -> Binary:
        flatbuffer = ttnn_to_flatbuffer(module, flatbuffer_name)
        self._mark_compile_step(CompileStep.FLATBUFFER)

        return flatbuffer
