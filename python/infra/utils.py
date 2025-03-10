# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass
from typing import List, Optional

from ttmlir.ir import Context, Module, OpAttributeMap, OpView, Type


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

    def __init__(self, op: OpView, attrs: Optional[OpAttributeMap] = None) -> None:
        self._op = op
        self._operands = [
            Operand(operand.get_name(), operand.type) for operand in op.operands
        ]
        self._result = (
            Result(op.result.get_name(), op.result.type)
            if len(op.results) > 0
            else None
        )
        self._attributes = attrs

    def __str__(self) -> str:
        return str(self._op)

    def __repr__(self) -> str:
        return str(self)

    def as_module_str(self) -> str:
        """Returns self wrapped in a MLIR module str."""
        return wrap_in_module_str(
            self._op, self._operands, self._result, self._attributes
        )


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


def wrap_in_module_str(
    op: OpWrapper,
    operands: List[Operand],
    result: Optional[Result] = None,
    attributes: OpAttributeMap = None,
) -> str:
    """
    Wraps `op` in a MLIR `func` and then in a MLIR `module` and returns string
    representation of that module.
    """
    unpacked_operands = ", ".join(
        f"{operand.name}: {operand.type}" for operand in operands
    )

    # Handle special case of ops that don't return anything.
    if result is not None:
        fn_return_type = result.type
        return_stmt = f"return {result.name} : {result.type}"
    else:
        fn_return_type = "()"
        return_stmt = "return"

    # Handle special case of modules that carry attributes.
    if attributes is not None:
        attrs = "{" + ",\n".join(f"{a.name} = {a.attr}" for a in attributes) + "}"
    else:
        attrs = "{}"

    return (
        f"module attributes {attrs} {{ \n"
        f"\tfunc.func @main({unpacked_operands}) -> {fn_return_type} {{ \n"
        f"\t\t{op} \n"
        f"\t\t{return_stmt} \n"
        f"\t}} \n"
        f"}}"
    )
