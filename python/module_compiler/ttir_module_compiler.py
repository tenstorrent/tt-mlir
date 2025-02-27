# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from mlir.ir import *
from module_splitter.ttir_module_splitter import TTIRModuleSplitter

from .module_compiler import ModuleCompiler
from .ttmlir import Binary, ttir_to_ttnn, ttnn_to_flatbuffer
from .utils import CompileStep


class TTIRModuleCompiler(ModuleCompiler):
    """Compiler for TTIR MLIR modules."""

    # ----- Public methods -----

    def create_from_module(module: Module) -> TTIRModuleCompiler:
        module_splitter = TTIRModuleSplitter.create_from_module(module)
        return TTIRModuleCompiler(module_splitter.get_module(), module_splitter)

    def create_from_module_str(module_str: str) -> TTIRModuleCompiler:
        module_splitter = TTIRModuleSplitter.create_from_module_str(module_str)
        return TTIRModuleCompiler(module_splitter.get_module(), module_splitter)

    # ----- Private methods -----

    def __init__(self, module: Module, module_splitter: TTIRModuleSplitter) -> None:
        super().__init__(module, module_splitter, CompileStep.TTIR)

    # @override
    def _compile(self, module: Module) -> Binary:
        ttnn = ttir_to_ttnn(module)
        self._mark_compile_step(CompileStep.TTNN)

        flatbuffer = ttnn_to_flatbuffer(ttnn)
        self._mark_compile_step(CompileStep.FLATBUFFER)

        return flatbuffer


if __name__ == "__main__":
    # TODO find a convenient TTIR graph
    ttir_module_str = ""

    compiler: TTIRModuleCompiler = TTIRModuleCompiler.create_from_module_str(
        ttir_module_str
    )
    compiler.compile_full_module()
    compiler.compile_op_by_op()
