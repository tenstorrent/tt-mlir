# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from typing import List

from mlir.ir import *


@dataclass(frozen=True)
class Operand:
    """Simple dataclass representing an operand of a MLIR operation."""

    name: str
    type: Type


@dataclass(frozen=True)
class Result:
    """Simple dataclass representing result of a MLIR operation."""

    name: str
    type: Type


class OpWrapper:
    """Convenience wrapper around MLIR op."""

    def __init__(self, op: OpView) -> None:
        self._op = op
        self._operands = [
            Operand(operand.get_name(), operand.type) for operand in op.operands
        ]
        self._result = Result(op.result.get_name(), op.result.type)

    def __str__(self) -> str:
        return str(self._op)

    def __repr__(self) -> str:
        return str(self)

    def as_module_str(self) -> str:
        """Returns self wrapped in a MLIR module str."""
        return wrap_in_module_str(self._op, self._operands, self._result)


def parse_module_str(module_str: str, ctx: Context) -> Module:
    """
    Parses `module_str` and returns MLIR module.

    Context `ctx` must be provided which contains all registered dialects needed to
    parse `module_str`.
    """

    def preprocess_module_str(module_str: str) -> str:
        """Preprocesses module string by removing `loc(...)` from it."""
        loc_pattern = re.compile(r"\s*loc\([^)]*\)")
        return re.sub(loc_pattern, "", module_str)

    return Module.parse(preprocess_module_str(module_str), ctx)


def wrap_in_module_str(op: OpWrapper, operands: List[Operand], result: Result) -> str:
    """
    Wraps `op` in a MLIR `func` and then in a MLIR `module` and returns string
    representation of that module.
    """
    unpacked_operands = ", ".join(
        f"{operand.name}: {operand.type}" for operand in operands
    )

    return (
        f"module {{ \n"
        f"\tfunc.func @main({unpacked_operands}) -> {result.type} {{ \n"
        f"\t\t{op} \n"
        f"\t\treturn {result.name} : {result.type} \n"
        f"\t}} \n"
        f"}}"
    )
