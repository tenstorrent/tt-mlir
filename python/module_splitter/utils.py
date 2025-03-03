# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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

    def as_module(self) -> Module:
        """Returns self wrapped in a MLIR module."""
        module_str = wrap_in_module_str(self._op, self._operands, self._result)
        return parse_module_str(module_str)


def parse_module_str(
    module_str: str, required_dialects: List[Dialect] = None
) -> Module:
    """
    Parses `module_str` and returns MLIR module.

    If any specific dialects are needed in order to be able to parse the `module_str`,
    they can be provided using `required_dialects` parameter.

    TODO not all dialects have `register_dialect` method, but stablehlo, ttir and ttnn
    do. Check why and what to do with the ones that don't support it.
    """
    from mlir.dialects import stablehlo

    with Context() as ctx:
        stablehlo.register_dialect(ctx)  # TODO not like this
        if required_dialects is not None:
            for dialect in required_dialects:
                dialect.register_dialect(ctx)

        return Module.parse(module_str)


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
