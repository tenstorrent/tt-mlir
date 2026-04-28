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
from datetime import datetime
from typing import Callable, Dict, Iterator, Optional, TextIO

from ttmlir.ir import Operation
from _ttmlir_runtime import runtime as tt_runtime

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
        # Output / behavior flags
        self.strict: bool = False
        self.isolation_check: bool = True
        self.accum_check: bool = True
        # Skip-mode: caller-supplied predicate (Operation) -> bool.
        # None disables skip mode entirely.
        self.skip_criterion: Optional[Callable[[Operation], bool]] = None
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_path: Optional[str] = f"chisel_results/{_ts}.jsonl"
        self._results_file: Optional[TextIO] = None

    def write_record(self, record: dict) -> None:
        """Append one JSON record to results_path (no-op if path is None).

        The file is opened lazily on first write and reused across records so
        we do not pay open/close per op.
        """
        if self.results_path is None:
            return
        if self._results_file is None:
            self._results_file = open(self.results_path, "a")
        self._results_file.write(json.dumps(record) + "\n")
        self._results_file.flush()

    def close_results(self) -> None:
        if self._results_file is not None:
            self._results_file.close()
            self._results_file = None

    @classmethod
    def get_instance(cls) -> "ChiselContext":
        if cls._instance is None:
            raise RuntimeError("ChiselContext not initialized — call ChiselContext() first")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def should_skip(self, op: Operation) -> bool:
        """True if the caller's criterion says this op's device output should be replaced."""
        if self.skip_criterion is None:
            return False
        return self.skip_criterion(op)

    def preprogram(self, binary, program_context) -> None:
        program_index = tt_runtime.get_program_index(program_context)

        if binary.id not in self.binaries:
            self.binaries[binary.id] = BinaryState(binary)
        binary_state = self.binaries[binary.id]
        self.current_binary = binary_state

        program_name = binary.get_program_name(program_index)
        program = ProgramState(program_index, program_name, binary_state.ir_module)
        program.golden_tensor_pool.update(binary_state.global_tensor_pool)
        binary_state.programs[program_index] = program
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
            if self.current_binary is not None:
                self.current_binary.global_tensor_pool.update(
                    self.current_program.golden_tensor_pool
                )
        self.current_binary = None
        self.current_program = None


class BinaryState:
    """Per-binary state: owns the IRModule, per-program states, and cross-program golden pool."""

    def __init__(self, binary):
        mlir_json = json.loads(binary.get_mlir_as_json())
        mlir_source = mlir_json["source"]
        functions = [
            binary.get_program_name(i) for i in range(binary.get_num_programs())
        ]
        self.ir_module = IRModule(mlir_source=mlir_source, functions=functions)
        self.programs: Dict[int, "ProgramState"] = {}
        self.global_tensor_pool: TensorPool = TensorPool()


class ProgramState:
    """Per-program state: owns golden_tensor_pool, op_iter, and per-op stashed inputs."""

    def __init__(self, program_index: int, program_name: str, ir_module: IRModule):
        self.program_index = program_index
        self.program_name = program_name
        self.golden_tensor_pool = TensorPool()
        self.ops = ir_module.get_function_ops(program_name)
        self.current_op: Optional[Operation] = None
        self.op_iter: Iterator = iter(self.ops)
        # Per-op transient state (set in preop, consumed in postop). Lives on
        # ProgramState because op execution is scoped to the active program.
        self.stashed_inputs: Optional[dict] = None
