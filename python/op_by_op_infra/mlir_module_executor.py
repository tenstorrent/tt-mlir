# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from ttmlir.compile_and_run import run_flatbuffer, ttnn_to_flatbuffer
from ttrt.common.util import Binary

from .execution_result import ExecutionPhase, ExecutionResult
from .utils import ModuleDialect, ModuleWrapper, convert_to_module_wrapper


class MLIRModuleExecutor(ABC):
    """
    Abstract base class used to compile or execute on device a given MLIR module.

    Execution consists of following phases:
        - Compilation phase. Implementation details vary and are left for subclasses to
          implement.
        - Flatbuffer generation phase. Generates flatbuffer from TTNN module. Depends on
          the previous phase.
        - Run phase. Runs generated flabuffer on device. Depends on the previous phase.
    """

    # ----- Public methods -----

    @convert_to_module_wrapper
    def compile(self, module: ModuleWrapper) -> ModuleWrapper:
        """
        Compiles MLIR `module`, returning a generated TTNN module.

        Asserts if compilation doesn't reach TTNN.
        """
        # Each time `compile` is called, prepare for new run by forgetting results of
        # previous run and storing new module to work on.
        self._reset(module)
        # Run compilation steps on stored module.
        compiled = self._compile()

        # If we failed to generate TTNN module, there is no point in proceeding.
        assert self._execution_result.compilation_finished, (
            f"ERROR: Couldn't generate TTNN module. "
            f"Managed to get to compilation phase: "
            f"{self._execution_result.execution_phase.name}"
        )

        return compiled

    @convert_to_module_wrapper
    def execute(self, module: ModuleWrapper) -> ExecutionResult:
        """
        Executes MLIR `module`, returning execution result.

        Initial `module` is passed through compilation phase, flatbuffer generation
        phase and lastly through run on device phase. If any of those steps (or steps
        within them) fail, execution stops and result is returned holding everything
        generated up to the last successful step.
        """
        # Each time `compile` is called, prepare for new run by forgetting results of
        # previous run and storing new module to work on.
        self._reset(module)

        # TODO special case where module consists solely of dealloc op. See what should
        # be done with it.
        if (
            module.has_origin_op
            and module.dialect == ModuleDialect.TTNN
            and "ttnn.dealloc" in module.origin_op.name
        ):
            return self._execution_result

        # Run execution steps on stored module.
        return self._execute()

    # ----- Protected methods -----

    def __init__(
        self,
        starting_execution_phase: ExecutionPhase,
    ) -> None:
        """Constructor."""
        # Upon creation, each particular splitter starts from some module which is in
        # a particular phase. Store that starting state and it will get updated during
        # execution steps.
        self._starting_execution_phase = starting_execution_phase

        self._module: ModuleWrapper = None
        self._execution_result: ExecutionResult = None

    def _reset(self, module: ModuleWrapper) -> None:
        """Resets internal state, gets ready for a new run."""
        self._module = module
        self._execution_result = ExecutionResult(self._starting_execution_phase, module)

    def _mark_execution_step(
        self,
        new_phase: ExecutionPhase,
        generated_module: Optional[ModuleWrapper] = None,
        generated_flatbuffer: Optional[Binary] = None,
        run_passed: Optional[bool] = None,
    ) -> None:
        """Marks execution progress."""
        self._execution_result.execution_phase = new_phase
        if generated_module is not None:
            self._execution_result.last_generated_module = generated_module
        if generated_flatbuffer is not None:
            self._execution_result.flatbuffer = generated_flatbuffer
        if run_passed is not None:
            self._execution_result.device_run_passed = run_passed

        self._execution_result.last_update = datetime.now()

    @abstractmethod
    def _compile(self) -> ModuleWrapper:
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
                self._execution_result.last_generated_module.module, flatbuffer_name
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
            if return_code == 0:
                self._mark_execution_step(
                    ExecutionPhase.EXECUTED_FLATBUFFER, run_passed=True
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
