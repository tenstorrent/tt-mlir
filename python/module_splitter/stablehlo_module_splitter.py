# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

from mlir.dialects import stablehlo
from mlir.ir import *

from .module_splitter import ModuleSplitter
from .utils import parse_module_str


class StableHLOModuleSplitter(ModuleSplitter):
    """Splits stablehlo MLIR module into constituent ops."""

    # ----- Public methods -----

    @staticmethod
    def create_from_module(module: Module) -> StableHLOModuleSplitter:
        return StableHLOModuleSplitter(module)

    @staticmethod
    def create_from_module_str(module_str: str) -> StableHLOModuleSplitter:
        module = parse_module_str(
            module_str,
            StableHLOModuleSplitter._get_required_dialects(),
        )
        return StableHLOModuleSplitter(module)

    # ----- Private methods -----

    def __init__(self, module: Module) -> None:
        super().__init__(module)

    # @override
    @staticmethod
    def _get_required_dialects() -> List[Dialect]:
        return [stablehlo]


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

    splitter: StableHLOModuleSplitter = StableHLOModuleSplitter.create_from_module_str(
        shlo_module_str
    )
    print(splitter.get_sub_ops())
    print(splitter.get_sub_modules())
