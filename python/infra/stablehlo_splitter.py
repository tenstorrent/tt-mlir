# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ttmlir.dialects import stablehlo
from ttmlir.ir import Context, Module

from .mlir_module_splitter import MLIRModuleSplitter
from .utils import parse_module_str


class StableHLOSplitter(MLIRModuleSplitter):
    """Splits stablehlo MLIR module into constituent ops."""

    # ----- Public methods -----

    @staticmethod
    def create_from_module(module: Module) -> StableHLOSplitter:
        return StableHLOSplitter(module)

    @staticmethod
    def create_from_module_str(module_str: str) -> StableHLOSplitter:
        return StableHLOSplitter(StableHLOSplitter._parse_module_str(module_str))

    # ----- Private methods -----

    def __init__(self, module: Module) -> None:
        super().__init__(module)

    # @override
    @staticmethod
    def _parse_module_str(module_str: str) -> Module:
        with Context() as ctx:
            stablehlo.register_dialect(ctx)
            return parse_module_str(module_str, ctx)
