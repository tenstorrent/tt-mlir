# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ttmlir.compile_and_run import (
    run_flatbuffer,
    stablehlo_to_ttir,
    ttir_to_ttnn,
    ttnn_to_flatbuffer,
)
from ttrt.common.util import Binary

from .execution_result import ExecutionPhase, ExecutionResult
from .utils import (
    ModuleDialect,
    ModuleWrapper,
    TTNNModuleWrapper,
    convert_to_module_wrapper,
)


class MLIRModuleExecutor:
    """
    Class used to compile or execute on device a given MLIR module.

    Execution consists of following phases:
        - Compilation phase. Passes given module through a subset of following steps:
          shlo -> ttir -> ttnn, depending on the dialect of the module.
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
        # Prepare for new run.
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
        # Prepare for new run.
        self._reset(module)

        # TODO special case where module consists solely of one of following TTNN ops
        # that cannot be executed on their own. They either fail fb generation or run.
        # See what should be done with them.
        if (
            module.has_origin_op
            and module.dialect == ModuleDialect.TTNN
            and module.origin_op_name
            in [
                "ttnn.get_device",
                "ttnn.to_device",
                "ttnn.full",
                "ttnn.empty",
                "ttnn.deallocate",
            ]
        ):
            return self._execution_result

        # Run execution steps on stored module.
        return self._execute()

    # ----- Private methods -----

    def __init__(self, compile_only: bool = False) -> None:
        """Constructor."""
        self._compile_only = compile_only
        self._module: ModuleWrapper = None
        self._execution_result: ExecutionResult = None

    def _reset(self, module: ModuleWrapper) -> None:
        """Resets internal state, gets ready for a new run."""
        self._module = module
        self._original_module_dialect = ModuleDialect.detect(module.module)

        # Detect and mark the starting phase of execution which will evolve during
        # execution steps.
        if self._original_module_dialect == ModuleDialect.STABLE_HLO:
            starting_execution_phase = ExecutionPhase.GENERATED_STABLE_HLO
        elif self._original_module_dialect == ModuleDialect.TTIR:
            starting_execution_phase = ExecutionPhase.GENERATED_TTIR
        elif self._original_module_dialect == ModuleDialect.TTNN:
            starting_execution_phase = ExecutionPhase.GENERATED_TTNN
        else:
            raise ValueError(
                f"Unsupported dialect: {self._original_module_dialect.name}"
            )

        self._execution_result = ExecutionResult(starting_execution_phase, module)

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

    def _compile(self) -> ModuleWrapper:
        """
        Attempts compiling stored module down to TTNN.

        Returns last successfully generated module produced by compilation steps.
        """
        if self._original_module_dialect == ModuleDialect.STABLE_HLO:
            return self._compile_shlo_to_ttnn()
        elif self._original_module_dialect == ModuleDialect.TTIR:
            return self._compile_ttir_to_ttnn()
        elif self._original_module_dialect == ModuleDialect.TTNN:
            # Trivial, original module was already a TTNN module.
            return self._module
        else:
            raise ValueError(
                f"Unsupported dialect: {self._original_module_dialect.name}"
            )

    def _compile_shlo_to_ttnn(self) -> ModuleWrapper:
        """
        Tries to compile SHLO module down to TTNN module.

        If any of the compilation steps fail, it returns last successfully generated
        module.
        """
        # During compilation steps, keep in mind that compilation API uses MLIR `Module`
        # which it modifies in-place. Also, don't lose track of the origin op.
        try:
            shlo = self._module.module

            ttir = stablehlo_to_ttir(shlo)
            self._mark_execution_step(
                ExecutionPhase.GENERATED_TTIR,
                ModuleWrapper(
                    ttir,
                    origin_op_name=self._module.origin_op_name,
                    origin_op_operands=self._module.origin_op_operands,
                    origin_op_results=self._module.origin_op_results,
                ),
            )

            ttnn = ttir_to_ttnn(ttir)
            self._mark_execution_step(
                ExecutionPhase.GENERATED_TTNN,
                TTNNModuleWrapper(
                    ttnn,
                    origin_op_name=self._module.origin_op_name,
                    origin_op_operands=self._module.origin_op_operands,
                    origin_op_results=self._module.origin_op_results,
                ),
            )
        finally:
            return self._execution_result.last_generated_module

    def _compile_ttir_to_ttnn(self) -> ModuleWrapper:
        """
        Tries to compile TTIR module down to TTNN module.

        If any of the compilation steps fail, it returns last successfully generated
        module.
        """
        try:
            ttir = self._module.module

            ttnn = ttir_to_ttnn(ttir)
            self._mark_execution_step(
                ExecutionPhase.GENERATED_TTNN,
                TTNNModuleWrapper(
                    ttnn,
                    origin_op_name=self._module.origin_op_name,
                    origin_op_operands=self._module.origin_op_operands,
                    origin_op_results=self._module.origin_op_results,
                ),
            )
        finally:
            return self._execution_result.last_generated_module

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

        if self._compile_only:
            self._execution_result.device_run_passed = True
            return self._execution_result

        self._generate_flatbuffer()

        if not self._execution_result.flatbuffer_generated:
            return self._execution_result

        self._run()

        return self._execution_result
