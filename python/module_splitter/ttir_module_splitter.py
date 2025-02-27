# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

from mlir.ir import *
from ttmlir.dialects import ttir

from .module_splitter import ModuleSplitter
from .utils import parse_module_str


class TTIRModuleSplitter(ModuleSplitter):
    """Splits TTIR MLIR module into constituent ops."""

    # ----- Public methods -----

    @staticmethod
    def create_from_module(module: Module) -> TTIRModuleSplitter:
        return TTIRModuleSplitter(module)

    @staticmethod
    def create_from_module_str(module_str: str) -> TTIRModuleSplitter:
        module = parse_module_str(
            module_str,
            TTIRModuleSplitter._get_required_dialects(),
        )
        return TTIRModuleSplitter(module)

    # ----- Private methods -----

    def __init__(self, module: Module) -> None:
        super().__init__(module)

    # @override
    @staticmethod
    def _get_required_dialects() -> List[Dialect]:
        return [ttir]


if __name__ == "__main__":
    # TODO find a convenient TTIR graph
    ttir_module_str = ""

    splitter: TTIRModuleSplitter = TTIRModuleSplitter.create_from_ttir_module_str(
        ttir_module_str
    )
    print(splitter.get_sub_ops())
    print(splitter.get_sub_modules())
