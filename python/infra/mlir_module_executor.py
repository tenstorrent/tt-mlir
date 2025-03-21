# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ttmlir.ir import Module
from ttrt.common.util import Binary

from .compile_and_run import run_flatbuffer, ttnn_to_flatbuffer
from .utils import convert_str_to_module


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
    device_run_passed: bool = False

    @property
    def compilation_finished(self) -> bool:
        return self.execution_phase == ExecutionPhase.GENERATED_TTNN

    @property
    def flatbuffer_generated(self) -> bool:
        return (
            self.execution_phase == ExecutionPhase.GENERATED_FLATBUFFER
            and self.flatbuffer is not None
        )

    @property
    def run_finished(self) -> bool:
        return self.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER

    def __repr__(self) -> str:
        return f"ExecutionResult({self.execution_phase.name})"


class MLIRModuleExecutor(ABC):
    """
    Abstract base class used to compile and run on device a given MLIR module.

    Instantiate one of concrete subclasses by giving it a MLIR module (or module str)
    and use provided public methods  TODO
    """

    # ----- Public methods -----

    @convert_str_to_module
    def compile(self, module: Module) -> Module:
        """
        Compiles MLIR `module`, returning a generated TTNN module.

        Asserts if compilation doesn't reach TTNN.
        """
        print("Running compile on module")
        # Each time `compile` is called, prepare for new run by forgetting results of
        # previous run and storing new module to work on.
        self._reset(module)
        # Run compilation steps on stored module.
        compiled = self._compile()

        # If we failed to generate TTNN module, there is no point in proceeding.
        assert self._execution_result.compilation_finished, (
            f"WARNING: Couldn't generate TTNN module. "
            f"Managed to get to compilation phase: "
            f"{self._execution_result.execution_phase.name}"
        )

        return compiled

    @convert_str_to_module
    def execute(self, module: Module) -> ExecutionResult:
        print("Running execute on module")
        # Each time `compile` is called, prepare for new run by forgetting results of
        # previous run and storing new module to work on.
        self._reset(module)
        # Run execution steps on stored module.
        return self._execute()

    # ----- Protected methods -----

    def __init__(
        self,
        starting_execution_phase: ExecutionPhase,
    ) -> None:
        # Upon creation, each particular splitter starts from some module which is in
        # a particular phase. Store that starting state and it will get updated during
        # execution steps.
        self._starting_execution_phase = starting_execution_phase

        self._module: Module = None
        self._execution_result: ExecutionResult = None

    def _reset(self, module: Module) -> None:
        """Resets internal state, gets ready for a new run."""
        self._module = module
        self._execution_result = ExecutionResult(self._starting_execution_phase, module)

    def _mark_execution_step(
        self,
        new_phase: ExecutionPhase,
        generated_module: Optional[Module] = None,
        generated_flatbuffer: Optional[Binary] = None,
        run_passed: Optional[bool] = None,
    ) -> None:
        """Stores execution progress."""
        self._execution_result.execution_phase = new_phase
        if generated_module is not None:
            self._execution_result.last_generated_module = generated_module
        if generated_flatbuffer is not None:
            self._execution_result.flatbuffer = generated_flatbuffer
        if run_passed is not None:
            self._execution_result.device_run_passed = run_passed

    @abstractmethod
    def _compile(self) -> Module:
        """
        Attempts compiling stored module down to TTNN.

        Returns last successfully generated module produced by compilation steps.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _generate_flatbuffer(
        self, flatbuffer_name: str = "ttnn_fb.ttnn"
    ) -> Optional[Binary]:
        """
        Attempts generating flatbuffer from TTNN module.

        Returns flatbuffer if successfully generated, None otherwise.
        """
        assert self._execution_result.compilation_finished

        try:
            flatbuffer = ttnn_to_flatbuffer(
                self._execution_result.last_generated_module, flatbuffer_name
            )
            self._mark_execution_step(
                ExecutionPhase.GENERATED_FLATBUFFER, generated_flatbuffer=flatbuffer
            )
        finally:
            return self._execution_result.flatbuffer

    def _run(self) -> bool:
        """
        Attempts running generated flatbuffer on device.

        Returns True if run didn't return any error codes, False if it did and None
        if it failed completely.
        """
        assert self._execution_result.flatbuffer_generated

        try:
            return_code = run_flatbuffer(self._execution_result.flatbuffer)
            run_passed = return_code == 0
            self._mark_execution_step(
                ExecutionPhase.EXECUTED_FLATBUFFER, run_passed=run_passed
            )
        finally:
            return self._execution_result.device_run_passed

    def _execute(self) -> ExecutionResult:
        """
        Executes stored module by compiling it to TTNN, generating flatbuffer from it
        and running it on device.

        It is possible that execution fails at any of the execution steps. Returns
        result which unambiguously represents what the execution managed to generate.
        """
        self._compile()

        if not self._execution_result.compilation_finished:
            return self._execution_result

        self._generate_flatbuffer()

        if not self._execution_result.flatbuffer_generated:
            return self._execution_result

        self._run()

        return self._execution_result
