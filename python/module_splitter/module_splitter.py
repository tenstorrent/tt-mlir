# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from mlir.ir import *

from .utils import OpWrapper


class ModuleSplitter(ABC):
    """
    Abstract base class used to split a MLIR module into constituent ops.

    Parsing a module string and converting it to a MLIR Module requires proper dialects
    to be registered within the context. Thus deriving a class from this base and
    overriding `_get_required_dialects` is sufficient to be able to parse and split
    any graph. Heavy load of splitting remains in the base class, independent of
    dialect used in graph.

    Methods
    -------
    get_module -> Module:
        Returns the original MLIR module passed to the splitter upon creation.

    get_sub_ops -> List[OpWrapper]
        Returns list of constituent ops.

    get_sub_modules -> List[Module]
        Returns list of constituent ops each wrapped in a MLIR module.
    """

    # ----- Public methods -----

    def get_module(self) -> Module:
        """Returns the original MLIR module passed to the splitter upon creation."""
        return self._module

    def get_sub_ops(self) -> List[OpWrapper]:
        """Returns list of constituent ops."""
        return self._sub_ops

    def get_sub_modules(self) -> List[Module]:
        """Returns list of constituent ops each wrapped in a MLIR module."""

        return [op.as_module() for op in self._sub_ops]

    # ----- Protected methods -----

    def __init__(self, module: Module) -> None:
        # TODO ensure module consists only of ops from registered dialects, otherwise
        # split will fail.
        self._module: Module = module
        self._sub_ops: List[OpWrapper] = []

        self._split()

    @staticmethod
    @abstractmethod
    def _get_required_dialects() -> List[Dialect]:
        """
        Returns a list of dialects required to be able to parse a module string and
        convert it to a module.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _split(self) -> None:
        """Splits the original module into constituent operations."""
        for func_op in self._module.body.operations:
            assert (
                len(func_op.regions) == 1
            ), f"Expected func {func_op.name} to have only one region"

            for block in func_op.regions[0].blocks:
                for op in block.operations:
                    # TODO why skip this?
                    if op.name.startswith(("func.", "return")):
                        continue

                    self._sub_ops.append(OpWrapper(op))
