# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from mlir.ir import *
from module_splitter.ttnn_module_splitter import TTNNModuleSplitter

from .module_compiler import ModuleCompiler
from .ttmlir import Binary, ttnn_to_flatbuffer
from .utils import CompileStep


class TTNNModuleCompiler(ModuleCompiler):
    """Compiler for TTNN MLIR modules."""

    # ----- Public methods -----

    def create_from_module(module: Module) -> TTNNModuleCompiler:
        module_splitter = TTNNModuleSplitter.create_from_module(module)
        return TTNNModuleCompiler(module_splitter.get_module(), module_splitter)

    def create_from_module_str(module_str: str) -> TTNNModuleCompiler:
        module_splitter = TTNNModuleSplitter.create_from_module_str(module_str)
        return TTNNModuleCompiler(module_splitter.get_module(), module_splitter)

    # ----- Private methods -----

    def __init__(self, module: Module, module_splitter: TTNNModuleSplitter) -> None:
        super().__init__(module, module_splitter, CompileStep.TTNN)

    # @override
    def _compile(self, module: Module) -> Binary:
        flatbuffer = ttnn_to_flatbuffer(module)
        self._mark_compile_step(CompileStep.FLATBUFFER)

        return flatbuffer


if __name__ == "__main__":
    # TODO find a convenient TTNN graph
    ttnn_module_str = ""

    compiler: TTNNModuleCompiler = TTNNModuleCompiler.create_from_module_str(
        ttnn_module_str
    )
    compiler.compile_full_module()
    compiler.compile_op_by_op()
