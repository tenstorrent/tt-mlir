# tools/chisel/chisel/context.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Slim ChiselContext singleton for per-op isolation testing.

Holds only what's needed for PR 1: ir_module, op_iter, stashed inputs.
No BinaryState/ProgramState hierarchy (added in PR 2).
"""
import json
from typing import Iterator, Optional

from ttmlir.ir import Operation

from .ops import IRModule


class ChiselContext:
    """Singleton context for chisel callbacks during TTRT execution."""

    _instance: Optional["ChiselContext"] = None

    def __init__(self):
        ChiselContext._instance = self
        self.ir_module: IRModule | None = None
        self.op_iter: Iterator | None = None
        self._current_op: Operation | None = None
        self._stashed_inputs: dict | None = None

    def ensure_ir_module(self, binary) -> None:
        """Lazily create IRModule from the binary's MLIR source on first preOp."""
        if self.ir_module is not None:
            return
        mlir_json = json.loads(binary.get_mlir_as_json())
        mlir_source = mlir_json["source"]
        functions = [
            binary.get_program_name(i)
            for i in range(binary.get_num_programs())
        ]
        self.ir_module = IRModule(mlir_source=mlir_source, functions=functions)
        self.op_iter = iter(self.ir_module.get_function_ops())

    @classmethod
    def get_instance(cls) -> "ChiselContext":
        if cls._instance is None:
            raise RuntimeError("ChiselContext not initialized")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None
