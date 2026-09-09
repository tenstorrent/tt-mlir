# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import logging
from typing import Dict, Iterator, List, Optional, Tuple, Type

from _ttmlir_runtime import runtime as tt_runtime
from _ttmlir_runtime.binary import Binary
from _ttmlir_runtime.runtime import CallbackContext, OpContext, TensorRef
from golden import GoldenMapTensor

from ttmlir.ir import AsmState

from .exceptions import UnexpectedStateError
from .op_configs import ChiselOpConfig, default_configs
from .ops import IRModule, SSAName
from .recorder import ChiselRecorder, _UNSET
from .report import ChiselRecord, ChiselReport
from .validators import ChiselChecksConfig

logger = logging.getLogger("chisel")

# Module-level handle to the active context. Set by chisel.bind() and
# cleared by chisel.unbind(); get_instance() raises if unset.
_current: Optional["ChiselContext"] = None


def get_instance() -> "ChiselContext":
    if _current is None:
        raise RuntimeError("ChiselContext is not bound. Call chisel.bind() first.")
    return _current


def is_initialized() -> bool:
    return _current is not None


def set_current(ctx: Optional["ChiselContext"]) -> None:
    global _current
    _current = ctx


class ChiselContext:
    def __init__(self) -> None:
        self.binaries: Dict[int, "BinaryState"] = {}
        # In-flight op-callback pointers; None outside an op callback.
        self._current_callback_binary: Optional["BinaryState"] = None
        self._current_callback_program: Optional["ProgramState"] = None
        self.recorder: ChiselRecorder = ChiselRecorder()
        self.op_configs: Dict[Type, ChiselOpConfig] = default_configs()
        self.checks_config: ChiselChecksConfig = ChiselChecksConfig()

    def register_op_config(self, op_type: Type, config: ChiselOpConfig) -> None:
        self.op_configs[op_type] = config

    def get_op_config(self, op: object) -> ChiselOpConfig:
        return self.op_configs.get(type(op), ChiselOpConfig())

    @property
    def report(self) -> ChiselReport:
        return self.recorder.report

    @property
    def results_path(self) -> Optional[str]:
        return self.recorder.results_path

    @property
    def debug_chisel_dir(self) -> Optional[str]:
        return self.recorder.debug_chisel_dir

    @property
    def asm_state(self) -> AsmState:
        binary_state = self._current_callback_binary
        if binary_state is None:
            raise UnexpectedStateError("asm_state")
        return binary_state.ir_module.get_asm_state()

    @property
    def mesh_shape(self) -> Tuple[int, ...]:
        """Mesh shape of the binary in the current callback scope.

        Sourced from `ttcore.meshes` on the binary's MLIR module so it tracks
        whatever mesh the program was compiled for (no need to plumb it from
        the user / Device handle).
        """
        binary_state = self._current_callback_binary
        if binary_state is None:
            raise UnexpectedStateError("mesh_shape")
        return binary_state.mesh_shape

    @property
    def is_in_op_scope(self) -> bool:
        # Guards the op-scope properties below, which raise UnexpectedStateError
        # when accessed outside begin_op/end_op.
        program = self._current_callback_program
        return program is not None and program._current_op is not None

    @property
    def op(self):
        program = self._current_callback_program
        value = program._current_op if program is not None else None
        if value is None:
            raise UnexpectedStateError("op")
        return value

    @property
    def input_refs(self) -> List[TensorRef]:
        program = self._current_callback_program
        value = program._current_input_refs if program is not None else None
        if value is None:
            raise UnexpectedStateError("input_refs")
        return value

    @property
    def output_refs(self) -> List[TensorRef]:
        program = self._current_callback_program
        value = program._current_output_refs if program is not None else None
        if value is None:
            raise UnexpectedStateError("output_refs")
        return value

    @property
    def stashed_inputs(self) -> Dict[SSAName, GoldenMapTensor]:
        program = self._current_callback_program
        value = program._stashed_inputs if program is not None else None
        if value is None:
            raise UnexpectedStateError("stashed_inputs")
        return value

    @property
    def golden_tensor_pool(self) -> Dict[SSAName, GoldenMapTensor]:
        """Program-scoped SSA -> golden tensor map; persists across ops."""
        program = self._current_callback_program
        if program is None:
            raise UnexpectedStateError("golden_tensor_pool")
        return program._golden_tensor_pool

    @property
    def device_tensor_pool(self) -> Dict[SSAName, GoldenMapTensor]:
        """Program-scoped SSA -> device tensor cache; persists across ops."""
        program = self._current_callback_program
        if program is None:
            raise UnexpectedStateError("device_tensor_pool")
        return program._device_tensor_pool

    @property
    def rt_program_context(self) -> Optional[CallbackContext]:
        program = self._current_callback_program
        return program._rt_program_context if program is not None else None

    @property
    def rt_op_context(self) -> Optional[OpContext]:
        program = self._current_callback_program
        return program._rt_op_context if program is not None else None

    @property
    def pre_failed(self) -> bool:
        # Whether the PRE handler for the current op recorded a failure.
        program = self._current_callback_program
        if program is None:
            raise UnexpectedStateError("pre_failed")
        return program._pre_failed

    @pre_failed.setter
    def pre_failed(self, value: bool) -> None:
        program = self._current_callback_program
        if program is None:
            raise UnexpectedStateError("pre_failed")
        program._pre_failed = value

    def begin_callback(
        self,
        rt_binary: Binary,
        rt_program_context: CallbackContext,
        rt_op_context: OpContext,
    ) -> "ProgramState":
        binary_state = self.binaries.get(rt_binary.id)
        if binary_state is None:
            raise UnexpectedStateError("begin_callback")

        program_index = tt_runtime.get_program_index(rt_program_context)
        program = binary_state.programs.get(program_index)
        if program is None:
            raise UnexpectedStateError("begin_callback")

        self._current_callback_binary = binary_state
        self._current_callback_program = program
        program._rt_binary = rt_binary
        program._rt_program_context = rt_program_context
        program._rt_op_context = rt_op_context
        return program

    def end_callback(self) -> None:
        self._current_callback_binary = None

        program = self._current_callback_program
        if program is None:
            raise UnexpectedStateError("end_callback")

        program._rt_binary = None
        program._rt_program_context = None
        program._rt_op_context = None
        self._current_callback_program = None

    def configure(
        self,
        *,
        results_path=_UNSET,
        report_capacity=_UNSET,
        debug_chisel_dir=_UNSET,
        checks_config=_UNSET,
    ) -> None:
        self.recorder.configure(
            results_path=results_path,
            report_capacity=report_capacity,
            debug_chisel_dir=debug_chisel_dir,
        )
        if checks_config is not _UNSET:
            self.checks_config = checks_config

    def write_record(self, record: ChiselRecord) -> None:
        self.recorder.write(
            record,
            program=self._current_callback_program,
            binary_state=self._current_callback_binary,
        )

    def close_results(self) -> None:
        self.recorder.close()

    def clear_binaries(self) -> None:
        """Evict all cached BinaryState entries (parsed MLIR modules)."""
        self.binaries.clear()

    def preprogram(
        self, rt_binary: Binary, rt_program_context: CallbackContext
    ) -> None:
        program_index = tt_runtime.get_program_index(rt_program_context)

        if rt_binary.id not in self.binaries:
            self.binaries[rt_binary.id] = BinaryState(rt_binary)
        binary_state = self.binaries[rt_binary.id]

        program_name = rt_binary.get_program_name(program_index)
        program = ProgramState(program_index, program_name, binary_state.ir_module)
        binary_state.programs[program_index] = program

        logger.debug(
            "preprogram: binary_id=%d program=%s index=%d",
            rt_binary.id,
            program_name,
            program_index,
        )

    def postprogram(
        self, rt_binary: Binary, rt_program_context: CallbackContext
    ) -> None:
        program_index = tt_runtime.get_program_index(rt_program_context)
        binary_state = self.binaries.get(rt_binary.id)
        if binary_state is not None:
            binary_state.programs.pop(program_index, None)


class BinaryState:
    # Owns the IRModule and per-program states. The rt_binary handle is
    # not stored - it may not be valid across callbacks; use the one
    # passed by the current callback.

    def __init__(self, rt_binary: Binary) -> None:
        self.binary_id: int = rt_binary.id
        mlir_json = json.loads(rt_binary.get_mlir_as_json())
        self.mlir_source: str = mlir_json["source"]
        functions = [
            rt_binary.get_program_name(i) for i in range(rt_binary.get_num_programs())
        ]
        self.ir_module = IRModule(mlir_source=self.mlir_source, functions=functions)
        self.mesh_shape: Tuple[int, ...] = self.ir_module.get_mesh_shape()
        self.programs: Dict[int, "ProgramState"] = {}


class ProgramState:
    # Two nested lifecycles:
    #   - Callback scope: rt_* handles, valid only inside one callback.
    #   - Op scope (begin_op / end_op): current_op + refs + stashed_inputs,
    #     spans the PRE/POST callback pair. begin_op reads _rt_op_context,
    #     so callback-scope handles must be set first.

    def __init__(
        self, program_index: int, program_name: str, ir_module: IRModule
    ) -> None:
        self.program_index = program_index
        self.program_name = program_name
        self._op_iter: Iterator = iter(ir_module.get_function_ops(program_name))
        self._rt_binary: Optional[Binary] = None
        self._rt_program_context: Optional[CallbackContext] = None
        self._rt_op_context: Optional[OpContext] = None
        self._current_op = None
        self._current_input_refs: Optional[List[TensorRef]] = None
        self._current_output_refs: Optional[List[TensorRef]] = None
        self._stashed_inputs: Optional[Dict[SSAName, GoldenMapTensor]] = None
        self._pre_failed: bool = False
        # Both pools persist across ops within a program; not reset per op.
        self._golden_tensor_pool: Dict[SSAName, GoldenMapTensor] = {}
        self._device_tensor_pool: Dict[SSAName, GoldenMapTensor] = {}

    def begin_op(self) -> None:
        self._current_op = next(self._op_iter).opview
        self._current_input_refs = tt_runtime.get_op_input_refs(self._rt_op_context)
        self._current_output_refs = tt_runtime.get_op_output_refs(self._rt_op_context)
        self._stashed_inputs = {}
        self._pre_failed = False

    def end_op(self) -> None:
        self._current_op = None
        self._current_input_refs = None
        self._current_output_refs = None
        self._stashed_inputs = None
        self._pre_failed = False
