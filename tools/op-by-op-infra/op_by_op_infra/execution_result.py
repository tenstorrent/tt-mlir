# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from ttrt.common.util import Binary

from .pydantic_models import OpTest, TensorDesc
from .utils import ModuleWrapper


class ExecutionPhase(Enum):
    """Progress marks in execution pipeline."""

    GENERATED_STABLE_HLO = 1
    GENERATED_TTIR = 2
    GENERATED_TTNN = 3
    GENERATED_FLATBUFFER = 4
    EXECUTED_FLATBUFFER = 5


@dataclass
class ExecutionResult:
    """
    Final result of execution.

    Holds all info necessary to determine how far down the compilation and run pipeline
    we managed to get (i.e. which ExecutionPhase we reached).
    """

    # Execution phase reached during execution pipeline.
    execution_phase: ExecutionPhase
    # Last module generated during compilation.
    last_generated_module: ModuleWrapper
    # Flatbuffer generated from TTNN module.
    # None if execution_phase < GENERATED_FLATBUFFER.
    flatbuffer: Optional[Binary] = None
    # Flag indicating successful run on device.
    # False if execution_phase < EXECUTED_FLATBUFFER or ttrt run returned code != 0.
    device_run_passed: bool = False
    # Error message from the step that failed. None if all steps succeeded.
    error_message: Optional[str] = None

    # Timestamp taken when execution was started.
    execution_started: datetime = datetime.now()
    # Timestamp which updated with each successful execution step.
    last_update: datetime = datetime.now()

    @property
    def execution_ended(self) -> datetime:
        """Timestamp taken during last successful execution step."""
        return self.last_update

    @property
    def compilation_finished(self) -> bool:
        """Returns True if compilation phase passed and generated TTNN module."""
        return self.execution_phase == ExecutionPhase.GENERATED_TTNN

    @property
    def flatbuffer_generated(self) -> bool:
        """Returns True if flatbuffer was successfully produced from TTNN module."""
        return (
            self.execution_phase == ExecutionPhase.GENERATED_FLATBUFFER
            and self.flatbuffer is not None
        )

    @property
    def run_finished(self) -> bool:
        """Returns True if flatbuffer was successfully executed on device."""
        return (
            self.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER
            and self.device_run_passed == True
        )

    def __repr__(self) -> str:
        return f"ExecutionResult({self.execution_phase.name})"


def convert_to_pydantic_model(result: ExecutionResult) -> OpTest:
    """
    Converts ExecutionResult to OpTest pydantic model.

    NOTE some pydantic model fields cannot be filled at this point. Additional info is
    required from frontend in order to fill it completely.
    """
    assert (
        result.last_generated_module.has_origin_op
    ), f"Generated module has no origin op. Nothing to track."

    inputs = [
        TensorDesc(
            shape=input.shape,
            data_type=input.data_type,
            buffer_type=input.buffer_type,
            layout=input.layout,
            grid_shape=input.grid_shape,
        )
        for input in result.last_generated_module.inputs
    ]

    outputs = [
        TensorDesc(
            shape=output.shape,
            data_type=output.data_type,
            buffer_type=output.buffer_type,
            layout=output.layout,
            grid_shape=output.grid_shape,
        )
        for output in result.last_generated_module.outputs
    ]

    if result.device_run_passed:
        error_msg = None
    elif result.error_message:
        error_msg = result.error_message
    else:
        error_msg = f"Last step successfully finished: {result.execution_phase.name}."

    pydantic_model = OpTest(
        test_start_ts=result.execution_started,
        test_end_ts=result.execution_ended,
        success=result.device_run_passed,
        error_message=error_msg,
        op_name=result.last_generated_module.origin_op_name,
        model_name=result.last_generated_module.origin_model,
        inputs=inputs,
        outputs=outputs,
    )

    return pydantic_model
