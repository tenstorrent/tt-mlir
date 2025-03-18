# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ttmlir.dialects import ttir
from ttmlir.ir import Context, Module

from .mlir_module_splitter import MLIRModuleSplitter
from .utils import parse_module_str


class TTIRSplitter(MLIRModuleSplitter):
    """Splits TTIR MLIR module into constituent ops."""

    # ----- Public methods -----

    @staticmethod
    def create_from_module(module: Module) -> TTIRSplitter:
        return TTIRSplitter(module)

    @staticmethod
    def create_from_module_str(module_str: str) -> TTIRSplitter:
        return TTIRSplitter(TTIRSplitter._parse_module_str(module_str))

    # ----- Private methods -----

    def __init__(self, module: Module) -> None:
        super().__init__(module)

    # @override
    @staticmethod
    def _parse_module_str(module_str: str) -> Module:
        with Context() as ctx:
            ttir.register_dialect(ctx)
            return parse_module_str(module_str, ctx)
