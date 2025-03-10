# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ttmlir.compiler_passes import stablehlo_to_ttir, ttir_to_ttnn, ttnn_to_flatbuffer
from ttmlir.ir import Module
from ttmlir.stablehlo_module_splitter import StableHLOModuleSplitter
from ttrt.common.util import Binary

from .module_compiler import CompileStep, ModuleCompiler


class StableHLOModuleCompiler(ModuleCompiler):
    """Compiler for StableHLO MLIR modules."""

    # ----- Public methods -----

    @staticmethod
    def create_from_module(module: Module) -> StableHLOModuleCompiler:
        module_splitter = StableHLOModuleSplitter.create_from_module(module)
        return StableHLOModuleCompiler(module_splitter.get_module(), module_splitter)

    @staticmethod
    def create_from_module_str(module_str: str) -> StableHLOModuleCompiler:
        module_splitter = StableHLOModuleSplitter.create_from_module_str(module_str)
        return StableHLOModuleCompiler(module_splitter.get_module(), module_splitter)

    # ----- Private methods -----

    def __init__(
        self, module: Module, module_splitter: StableHLOModuleSplitter
    ) -> None:
        super().__init__(module, module_splitter, CompileStep.STABLE_HLO)

    # @override
    def _compile(self, module: Module, flatbuffer_name: str = "ttnn_fb.ttnn") -> Binary:
        ttir = stablehlo_to_ttir(module)
        self._mark_compile_step(CompileStep.TTIR)

        ttnn = ttir_to_ttnn(ttir)
        self._mark_compile_step(CompileStep.TTNN)

        flatbuffer = ttnn_to_flatbuffer(ttnn, flatbuffer_name)
        self._mark_compile_step(CompileStep.FLATBUFFER)

        return flatbuffer
