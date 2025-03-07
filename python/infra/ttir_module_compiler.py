# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ttmlir.compiler_passes import ttir_to_ttnn, ttnn_to_flatbuffer
from ttmlir.ir import Module
from ttmlir.ttir_module_splitter import TTIRModuleSplitter
from ttrt.common.util import Binary

from .module_compiler import CompileStep, ModuleCompiler


class TTIRModuleCompiler(ModuleCompiler):
    """Compiler for TTIR MLIR modules."""

    # ----- Public methods -----

    @staticmethod
    def create_from_module(module: Module) -> TTIRModuleCompiler:
        module_splitter = TTIRModuleSplitter.create_from_module(module)
        return TTIRModuleCompiler(module_splitter.get_module(), module_splitter)

    @staticmethod
    def create_from_module_str(module_str: str) -> TTIRModuleCompiler:
        module_splitter = TTIRModuleSplitter.create_from_module_str(module_str)
        return TTIRModuleCompiler(module_splitter.get_module(), module_splitter)

    # ----- Private methods -----

    def __init__(self, module: Module, module_splitter: TTIRModuleSplitter) -> None:
        super().__init__(module, module_splitter, CompileStep.TTIR)

    # @override
    def _compile(self, module: Module, flatbuffer_name: str = "ttnn_fb.ttnn") -> Binary:
        ttnn = ttir_to_ttnn(module)
        self._mark_compile_step(CompileStep.TTNN)

        flatbuffer = ttnn_to_flatbuffer(ttnn, flatbuffer_name)
        self._mark_compile_step(CompileStep.FLATBUFFER)

        return flatbuffer
