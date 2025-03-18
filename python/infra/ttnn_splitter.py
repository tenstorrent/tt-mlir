# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ttmlir.dialects import ttnn
from ttmlir.ir import Context, Module

from .mlir_module_splitter import MLIRModuleSplitter
from .utils import parse_module_str


class TTNNSplitter(MLIRModuleSplitter):
    """Splits TTNN MLIR module into constituent ops."""

    # ----- Public methods -----

    @staticmethod
    def create_from_module(module: Module) -> TTNNSplitter:
        return TTNNSplitter(module)

    @staticmethod
    def create_from_module_str(module_str: str) -> TTNNSplitter:
        return TTNNSplitter(TTNNSplitter._parse_module_str(module_str))

    # ----- Private methods -----

    def __init__(self, module: Module) -> None:
        super().__init__(module)

    # @override
    @staticmethod
    def _parse_module_str(module_str: str) -> Module:
        with Context() as ctx:
            ttnn.register_dialect(ctx)
            return parse_module_str(module_str, ctx)
