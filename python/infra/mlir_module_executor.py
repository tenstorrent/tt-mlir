# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List

from ttmlir.ir import Module
from ttrt.common.util import Binary

from .mlir_module_splitter import MLIRModuleSplitter


class CompileStep(Enum):
    GENERATED_STABLE_HLO = 1
    GENERATED_TTIR = 2
    GENERATED_TTNN = 3
    GENERATED_FLATBUFFER = 4
    EXECUTED_FLATBUFFER = 5


@dataclass
class CompilationResult:
    compile_step: CompileStep


class MLIRModuleExecutor(ABC):
    """
    Abstract base class used to compile and run on device a given MLIR module.

    TODO
    """

    # ----- Public methods -----

    def compile_full_module(self) -> Binary:
        return self._compile(self._module)

    def compile_op_by_op(self) -> List[Binary]:
        return [
            self._compile(sub_module, f"ttnn_fb{i+1}.ttnn")
            for i, sub_module in enumerate(self._module_splitter.get_sub_modules())
        ]

    def compile_and_run_full_module(self) -> None:
        fb = self.compile_full_module(self._module)

    def compile_and_run_op_by_op(self) -> None:
        fbs = self.compile_op_by_op()

    # ----- Protected methods -----

    def __init__(
        self,
        module: Module,
        module_splitter: MLIRModuleSplitter,
        starting_compile_step: CompileStep,
    ) -> None:
        self._module: Module = module
        self._module_splitter = module_splitter
        self._compile_step = starting_compile_step

    def _mark_compile_step(self, step: CompileStep) -> None:
        self._compile_step = step

    @abstractmethod
    def _compile(self, module: Module, flatbuffer_name: str = "ttnn_fb.ttnn") -> Binary:
        raise NotImplementedError("Subclasses must implement this method.")
