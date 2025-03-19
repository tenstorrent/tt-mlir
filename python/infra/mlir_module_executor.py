# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from ttmlir.ir import Module
from ttrt.common.util import Binary

from .mlir_module_splitter import MLIRModuleSplitter
from .ttnn_executor import TTNNExecutor  # TODO should be here


class ExecutionPhase(Enum):
    GENERATED_STABLE_HLO = 1
    GENERATED_TTIR = 2
    GENERATED_TTNN = 3
    GENERATED_FLATBUFFER = 4
    EXECUTED_FLATBUFFER = 5


@dataclass
class ExecutionResult:
    execution_phase: ExecutionPhase
    last_generated_module: Module
    flatbuffer: Optional[Binary] = None
    device_run_passed: Optional[bool] = None

    @property
    def flatbuffer_generated(self) -> bool:
        return (
            self.execution_phase == ExecutionPhase.GENERATED_FLATBUFFER
            and self.flatbuffer is not None
        )


class MLIRModuleExecutor(ABC):
    """
    Abstract base class used to compile and run on device a given MLIR module.

    Instantiate one of concrete subclasses by giving it a MLIR module (or module str)
    and use provided public methods  TODO
    """

    # ----- Public methods -----

    def split_compile_and_run(self) -> List[ExecutionResult]:
        results = []

        for i, sub_module in enumerate(self._split()):
            compilation_result = self._compile(sub_module, f"ttnn_fb{i+1}.ttnn")

            if not compilation_result.flatbuffer_generated:
                results.append(compilation_result)
            else:
                run_result = self._run(compilation_result.flatbuffer)
                results.append(run_result)

    def compile_split_and_run(self) -> List[ExecutionResult]:
        results = []

        compilation_result = self._compile(self._module)

        if not compilation_result.flatbuffer_generated:
            results.append(compilation_result)
            return results

        ttnn_module = compilation_result.last_generated_module
        ttnn_executor = TTNNExecutor.create_from_module(ttnn_module)

        for i, sub_module in enumerate(ttnn_executor._split()):
            compilation_result = ttnn_executor._compile(
                sub_module, f"ttnn_fb{i+1}.ttnn"
            )

            if not compilation_result.flatbuffer_generated:
                results.append(compilation_result)

            run_result = ttnn_executor._run(compilation_result.flatbuffer)
            results.append(run_result)

    # ----- Protected methods -----

    def __init__(
        self,
        module: Module,
        module_splitter: MLIRModuleSplitter,
        starting_execution_phase: ExecutionPhase,
    ) -> None:
        self._module: Module = module
        self._module_splitter = module_splitter

        # Upon creation, each particular splitter starts from some module which is in
        # a particular phase. Store that starting state and it will get updated during
        # execution steps.
        self._execution_result = ExecutionResult(starting_execution_phase, module)

    def _split(self) -> List[Module]:
        self._module_splitter.get_sub_modules()

    def _mark_execution_step(
        self,
        new_phase: ExecutionPhase,
        generated_module: Optional[Module] = None,
        generated_flatbuffer: Optional[Binary] = None,
        run_passed: Optional[bool] = None,
    ) -> None:
        self._execution_result.execution_phase = new_phase
        if generated_module is not None:
            self._execution_result.last_generated_module = generated_module
        if generated_flatbuffer is not None:
            self._execution_result.flatbuffer = generated_flatbuffer
        if run_passed is not None:
            self._execution_result.device_run_passed = run_passed

    @abstractmethod
    def _compile(
        self, module: Module, flatbuffer_name: str = "ttnn_fb.ttnn"
    ) -> ExecutionResult:
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def _run(self, flatbuffer: Binary) -> ExecutionResult:
        # TODO move implementation here, run is not subclass specific.
        raise NotImplementedError("Subclasses must implement this method.")
