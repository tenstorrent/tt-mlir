# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from mlir.ir import *
from module_splitter.stablehlo_module_splitter import StableHLOModuleSplitter

from .module_compiler import ModuleCompiler
from .ttmlir import Binary, stablehlo_to_ttir, ttir_to_ttnn, ttnn_to_flatbuffer
from .utils import CompileStep


class StableHLOModuleCompiler(ModuleCompiler):
    """Compiler for StableHLO MLIR modules."""

    # ----- Public methods -----

    def create_from_module(module: Module) -> StableHLOModuleCompiler:
        module_splitter = StableHLOModuleSplitter.create_from_module(module)
        return StableHLOModuleCompiler(module_splitter.get_module(), module_splitter)

    def create_from_module_str(module_str: str) -> StableHLOModuleCompiler:
        module_splitter = StableHLOModuleSplitter.create_from_module_str(module_str)
        return StableHLOModuleCompiler(module_splitter.get_module(), module_splitter)

    # ----- Private methods -----

    def __init__(
        self, module: Module, module_splitter: StableHLOModuleSplitter
    ) -> None:
        super().__init__(module, module_splitter, CompileStep.STABLE_HLO)

    # @override
    def _compile(self, module: Module) -> Binary:
        ttir = stablehlo_to_ttir(module)
        self._mark_compile_step(CompileStep.TTIR)

        ttnn = ttir_to_ttnn(ttir)
        self._mark_compile_step(CompileStep.TTNN)

        flatbuffer = ttnn_to_flatbuffer(ttnn)
        self._mark_compile_step(CompileStep.FLATBUFFER)

        return flatbuffer


if __name__ == "__main__":
    shlo_module_str = """
        module {
        func.func @main(%arg0: tensor<1x128xf32>, %arg1: tensor<128xf32>) -> tensor<1x128xf32> {
            %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<1x128xf32>
            %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
            %2 = stablehlo.add %0, %1 : tensor<1x128xf32>
            return %2 : tensor<1x128xf32>
        }
        }
    """

    compiler: StableHLOModuleCompiler = StableHLOModuleCompiler.create_from_module_str(
        shlo_module_str
    )
    compiler.compile_full_module()
    compiler.compile_op_by_op()
