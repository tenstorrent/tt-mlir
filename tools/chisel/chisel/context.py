# tools/chisel/chisel/context.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Slim ChiselContext singleton for per-op isolation testing.

Holds only what's needed for PR 1: ir_module, op_iter, stashed inputs.
No BinaryState/ProgramState hierarchy (added in PR 2).
"""
from typing import Iterator, Optional

from ttmlir.ir import Operation

from chisel.ops import IRModule


class ChiselContext:
    """Singleton context for chisel callbacks during TTRT execution."""

    _instance: Optional["ChiselContext"] = None

    def __init__(self, ir_module: IRModule):
        ChiselContext._instance = self
        self.ir_module = ir_module
        self.op_iter: Iterator = iter(ir_module.get_function_ops())
        self._current_op: Operation | None = None
        self._stashed_inputs: dict | None = None

    @classmethod
    def get_instance(cls) -> "ChiselContext":
        if cls._instance is None:
            raise RuntimeError("ChiselContext not initialized")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None
