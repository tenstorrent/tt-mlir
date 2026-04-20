# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ChiselContext / BinaryState / ProgramState — three-level state hierarchy.

ChiselContext is a singleton accessed from all four DebugHooks callbacks.
BinaryState is created once per unique binary.id and owns the IRModule.
ProgramState is created once per (binary_id, program_index) and owns
the golden_tensor_pool and op_iter.
"""
import json
import logging
from typing import Dict, Iterator, Optional

from .ops import IRModule
from .tensors import TensorPool

logger = logging.getLogger("chisel")


class ChiselContext:
    """Singleton context shared by all four DebugHooks callbacks."""

    _instance: Optional["ChiselContext"] = None

    def __init__(self):
        ChiselContext._instance = self
        self.binaries: Dict[int, "BinaryState"] = {}
        self.current_binary: Optional["BinaryState"] = None
        self.current_program: Optional["ProgramState"] = None

    @classmethod
    def get_instance(cls) -> "ChiselContext":
        if cls._instance is None:
            raise RuntimeError("ChiselContext not initialized")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def preprogram(self, binary, program_context) -> None:
        import _ttmlir_runtime as tt_runtime

        program_index = tt_runtime.runtime.get_program_index(program_context)

        if binary.id not in self.binaries:
            self.binaries[binary.id] = BinaryState(binary)
        binary_state = self.binaries[binary.id]
        self.current_binary = binary_state

        program_name = binary.get_program_name(program_index)
        program = binary_state.get_or_create_program(program_index, program_name)
        program.reset_for_new_execution()
        self.current_program = program

        logger.debug(
            "preprogram: binary_id=%d program=%s index=%d",
            binary.id,
            program_name,
            program_index,
        )

    def postprogram(self, binary, program_context) -> None:
        if self.current_program is not None:
            logger.info("postprogram: %s complete", self.current_program.program_name)
        self.current_binary = None
        self.current_program = None

    def preop(self, binary, program_context, op_context) -> None:
        import _ttmlir_runtime as tt_runtime

        from .ops import get_op_inputs
        from .utils import retrieve_torch_tensor

        program = self.current_program
        binary_state = self.current_binary
        program.current_op = next(program.op_iter)
        op = program.current_op

        asm_state = binary_state.ir_module.get_asm_state(program.program_name)
        op_inputs = get_op_inputs(op)
        input_refs = tt_runtime.runtime.get_op_input_refs(op_context, program_context)

        # Seed pool with device tensors for any input not yet in the pool.
        # An SSA name absent from the pool is a program input (never produced
        # by a prior op in this session).
        for mlir_inp, tensor_ref in zip(op_inputs, input_refs):
            name = mlir_inp.get_name(asm_state)
            if name not in program.golden_tensor_pool:
                program.golden_tensor_pool[name] = retrieve_torch_tensor(
                    program_context, tensor_ref
                )

    def postop(self, binary, program_context, op_context) -> None:
        import _ttmlir_runtime as tt_runtime

        from .executor import execute_golden_from_pool
        from .metrics import compute_metrics
        from .ops import get_op_outputs
        from .utils import retrieve_torch_tensor

        program = self.current_program
        binary_state = self.current_binary
        op = program.current_op

        op_outputs = get_op_outputs(op)
        if not op_outputs:
            return

        golden_result = execute_golden_from_pool(
            op,
            binary_state.ir_module,
            program.program_name,
            program.golden_tensor_pool,
        )

        output_ref = tt_runtime.runtime.get_op_output_ref(op_context, program_context)
        if output_ref is None:
            return
        device_tensor = retrieve_torch_tensor(program_context, output_ref)

        asm_state = binary_state.ir_module.get_asm_state(program.program_name)
        out_name = op_outputs[0].get_name(asm_state)
        compute_metrics(op.name, out_name, golden_result, device_tensor)


class BinaryState:
    """Per-binary state: owns the IRModule and per-program states."""

    def __init__(self, binary):
        mlir_json = json.loads(binary.get_mlir_as_json())
        mlir_source = mlir_json["source"]
        functions = [
            binary.get_program_name(i) for i in range(binary.get_num_programs())
        ]
        self.ir_module = IRModule(mlir_source=mlir_source, functions=functions)
        self.programs: Dict[int, "ProgramState"] = {}

    def get_or_create_program(
        self, program_index: int, program_name: str
    ) -> "ProgramState":
        if program_index not in self.programs:
            self.programs[program_index] = ProgramState(
                program_index, program_name, self.ir_module
            )
        return self.programs[program_index]


class ProgramState:
    """Per-program state: owns golden_tensor_pool and op_iter."""

    def __init__(self, program_index: int, program_name: str, ir_module: IRModule):
        self.program_index = program_index
        self.program_name = program_name
        self.golden_tensor_pool = TensorPool()
        self.ops = ir_module.get_function_ops(program_name)
        self.current_op = None
        self.op_iter: Iterator = iter(self.ops)

    def reset_for_new_execution(self) -> None:
        self.op_iter = iter(self.ops)
        self.current_op = None
        # golden_tensor_pool is intentionally NOT cleared — cross-run chaining
