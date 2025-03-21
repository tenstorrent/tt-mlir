# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

from ttmlir.dialects import stablehlo, tt, ttir, ttnn
from ttmlir.ir import Context, Module, OpAttributeMap, OperationList, OpView, Type
from ttrt.common.util import Binary


class MLIRDialect(Enum):
    """Named like this to avoid collision with builtin `Dialect`."""

    STABLE_HLO = "stablehlo"
    TTIR = "ttir"
    TTNN = "ttnn"
    TT = "tt"

    @staticmethod
    def detect(module_or_op: str | OpView | Module) -> MLIRDialect:
        """
        Factory method. Detects dialect used in the mlir module or op string
        representation.
        """
        str_repr = str(module_or_op)

        if "stablehlo." in str_repr:
            return MLIRDialect.STABLE_HLO
        elif "ttir." in str_repr:
            return MLIRDialect.TTIR
        elif "ttnn." in str_repr:
            return MLIRDialect.TTNN
        else:
            # Fallback to returning `tt` dialet if nothing else succeeds. It bundles
            # together all builtin dialects.
            return MLIRDialect.TT


class OpWrapper:
    """Convenience wrapper around MLIR op."""

    # ----- Public methods and properties -----

    def __init__(self, op: OpView, attrs: Optional[OpAttributeMap] = None) -> None:
        self._op = op
        self._operands = [
            OpWrapper.Operand(operand.get_name(), operand.type)
            for operand in op.operands
        ]
        self._result = (
            OpWrapper.Result(op.result.get_name(), op.result.type)
            if len(op.results) > 0
            else None
        )
        self._attributes = attrs

    def __str__(self) -> str:
        return str(self._op)

    def __repr__(self) -> str:
        return str(self)

    @property
    def name(self) -> str:
        return self._op.name

    def as_module_str(self) -> str:
        """Returns self wrapped in a MLIR module str."""
        return OpWrapper._wrap_in_module_str(
            self._op, self._operands, self._result, self._attributes
        )

    def as_module(self) -> ModuleWrapper:
        """
        Returns self wrapped in `ModuleWrapper`.

        Wrapper will contain original MLIR module, dialect used and op from which
        it was created (`self`).
        """
        module_wrapper = parse_module_str(self.as_module_str())
        # Store a reference to the original op.
        module_wrapper.generated_from_op = self
        return module_wrapper

    # ----- Private methods and classes -----

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

    @staticmethod
    def _wrap_in_module_str(
        op: OpWrapper,
        operands: List[OpWrapper.Operand],
        result: Optional[OpWrapper.Result] = None,
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


class ModuleWrapper:
    """
    Convenience wrapper around MLIR module.

    Provides posibility to keep track of the op from which module was generated, useful
    in op by op processing pipeline.
    """

    def __init__(
        self,
        module: Module,
        dialect: Optional[MLIRDialect] = None,
        generated_from_op: Optional[OpWrapper] = None,
    ) -> None:
        self.module: Module = module
        self.dialect: MLIRDialect = dialect or MLIRDialect.detect(module)
        self.generated_from_op: Optional[OpWrapper] = generated_from_op

    @property
    def attributes(self) -> Optional[OpAttributeMap]:
        """Returns module attributes if any, otherwise None."""
        return (
            self.module.operation.attributes
            if len(self.module.operation.attributes) > 0
            else None
        )

    @property
    def operations(self) -> OperationList:
        return self.module.body.operations

    @property
    def is_generated_from_op(self) -> bool:
        return self.generated_from_op is not None

    def __repr__(self) -> str:
        if self.generated_from_op:
            return f"ModuleWrapper(\n{self.module}), Generated from: {self.generated_from_op.name})"
        else:
            return f"ModuleWrapper(\n{self.module}\n)"


def parse_module_str(module_str: str) -> ModuleWrapper:
    """
    Within a temporary context registers necessary dialects and parses `module_str`
    returning ModuleWrapper instance.
    """

    def preprocess_module_str(module_str: str) -> str:
        """Preprocesses module string by removing `loc(...)` from it."""
        loc_pattern = re.compile(r"\s*loc\([^)]*\)")
        return re.sub(loc_pattern, "", module_str)

    def register_dialect(dialect: MLIRDialect, ctx: Context) -> None:
        """
        Detects dialect used in `module_str` and registers it with context `ctx`.
        """
        if dialect == MLIRDialect.STABLE_HLO:
            stablehlo.register_dialect(ctx)
            # TODO there must be a better way to do this. We need to register `ttir`
            # (or any other of our dialects) otherwise we'll encounter problems with
            # `func` dialect which isn't included through `stablehlo` and doesn't
            # provide `func.register_dialect(ctx)` on its own.
            ttir.register_dialect(ctx)
        elif dialect == MLIRDialect.TTIR:
            ttir.register_dialect(ctx)
        elif dialect == MLIRDialect.TTNN:
            ttnn.register_dialect(ctx)
        elif dialect == MLIRDialect.TT:
            tt.register_dialect(ctx)
        else:
            raise ValueError(f"Unknown dialect: {dialect.name}")

    with Context() as ctx:
        cleaned_module_str = preprocess_module_str(module_str)
        dialect = MLIRDialect.detect(cleaned_module_str)
        # Must register dialect in order for parsing to work.
        register_dialect(dialect, ctx)
        mlir_module = Module.parse(cleaned_module_str)
        return ModuleWrapper(mlir_module, dialect=dialect)


def convert_to_module_wrapper(func: Callable) -> Callable:
    """
    Decorator to ensure that the `module` argument is always of type `ModuleWrapper`.
    If it's a string, it will be parsed using `parse_module_str`. If it is a `Module`
    it will be wrapped.
    """

    def wrapper(
        self,
        module: str | Module | ModuleWrapper,
        *args,
        **kwargs,
    ) -> List[Module]:
        if isinstance(module, str):
            m = parse_module_str(module)
        elif isinstance(module, Module):
            m = ModuleWrapper(module)
        else:
            m = module

        # Call the original function with the converted module.
        return func(self, m, *args, **kwargs)

    return wrapper


class ExecutionPhase(Enum):
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

    execution_phase: ExecutionPhase
    last_generated_module: ModuleWrapper
    flatbuffer: Optional[Binary] = None
    device_run_passed: bool = False

    @property
    def compilation_finished(self) -> bool:
        return self.execution_phase == ExecutionPhase.GENERATED_TTNN

    @property
    def flatbuffer_generated(self) -> bool:
        return (
            self.execution_phase == ExecutionPhase.GENERATED_FLATBUFFER
            and self.flatbuffer is not None
        )

    @property
    def run_finished(self) -> bool:
        return self.execution_phase == ExecutionPhase.EXECUTED_FLATBUFFER

    @property
    def run_succeeded(self) -> bool:
        return self.run_finished and self.device_run_passed == True

    def __repr__(self) -> str:
        return f"ExecutionResult({self.execution_phase.name})"
