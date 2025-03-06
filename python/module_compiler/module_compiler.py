# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from mlir.ir import *
from ttmlir.module_splitter import ModuleSplitter
from ttrt.common.util import Binary


class CompileStep(Enum):
    STABLE_HLO = 1
    TTIR = 2
    TTNN = 3
    FLATBUFFER = 4


class ModuleCompiler(ABC):
    """
    Abstract base class used to compile a MLIR module.

    It provides the following public API for two types of compilation:

    Methods
    -------
    compile_full_model -> Binary
        Runs entire MLIR module through the compiler and returns generated flatbuffer.

    compile_op_by_op -> List[Binary]
        Splits MLIR module into constituent ops and runs the compiler on each one of
        them, returning a list of generated flatbuffers.
    """

    # ----- Public methods -----

    def compile_full_module(self) -> Binary:
        return self._compile(self._module)

    def compile_op_by_op(self) -> List[Binary]:
        return [
            self._compile(sub_module)
            for sub_module in self._module_splitter.get_sub_modules()
        ]

    # ----- Protected methods -----

    def __init__(
        self,
        module: Module,
        module_splitter: ModuleSplitter,
        starting_compile_step: CompileStep,
    ) -> None:
        self._module: Module = module
        self._module_splitter = module_splitter
        self._compile_step = starting_compile_step

    def _mark_compile_step(self, step: CompileStep) -> None:
        self._compile_step = step

    @abstractmethod
    def _compile(self, module: Module) -> Binary:
        raise NotImplementedError("Subclasses must implement this method.")
