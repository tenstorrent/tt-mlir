# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional

from ttmlir.dialects import stablehlo, tt, ttir, ttnn
from ttmlir.ir import (
    Context,
    Module,
    OpAttributeMap,
    OperationList,
    OpView,
    RankedTensorType,
)


class ModuleDialect(Enum):
    """
    Enum for available dialects used in modules.

    Named like this to avoid collision with builtin `Dialect`.
    """

    STABLE_HLO = "stablehlo"
    TTIR = "ttir"
    TTNN = "ttnn"
    TT = "tt"

    @staticmethod
    def detect(module_or_op: str | OpView | Module) -> ModuleDialect:
        """
        Factory method. Detects dialect used in the mlir module or op string
        representation.
        """
        str_repr = str(module_or_op)

        if "stablehlo." in str_repr:
            return ModuleDialect.STABLE_HLO
        elif "ttir." in str_repr:
            return ModuleDialect.TTIR
        elif "ttnn." in str_repr:
            return ModuleDialect.TTNN
        else:
            # Fallback to returning `tt` dialet if nothing else succeeds. It bundles
            # together all builtin dialects.
            return ModuleDialect.TT


@dataclass(frozen=True)
class OperandAndResultBase(ABC):
    """
    Abstract base dataclass representing an operand or result of a MLIR operation.

    MLIR ops usually have one or multiple operands and only one result (though in
    general multi result is supported).

    Abstracting them this way makes them have the same implementation. Derived classes
    exist for each of them just to be strict and clear.
    """

    name: str
    type: RankedTensorType

    @property
    def shape(self) -> List[int]:
        return self.type.shape

    @property
    def data_type(self) -> str:
        return str(self.type.element_type)

    @property
    def buffer_type(self) -> Optional[str]:
        """String representing buffer type. Only TTNN tensors have it."""
        if not self._has_ttnn_encoding:
            return None

        buffer_type_attr = ttnn.ir.BufferTypeAttr.maybe_downcast(
            self._layout_attr.memory_space
        )
        return str(ttnn.BufferType(buffer_type_attr.value))

    @property
    def layout(self) -> Optional[str]:
        """String representing tensor memory layout. Only TTNN tensors have it."""
        if not self._has_ttnn_encoding:
            return None

        return str(
            ttnn.TensorMemoryLayout(self._layout_attr.tensor_memory_layout_as_int)
        )

    @property
    def grid_shape(self) -> Optional[List[int]]:
        """
        Shape of grid of cores which are used to store tensor in memory.
        Only TTNN tensors have it.
        """
        if not self._has_ttnn_encoding:
            return None

        return self._layout_attr.grid_shape

    @property
    def _layout_attr(self) -> ttnn.ir.TTNNLayoutAttr:
        """Returns TTNNLayoutAttr from tensor type. Only TTNN tensors have it."""
        assert self._has_ttnn_encoding
        return ttnn.ir.TTNNLayoutAttr.maybe_downcast(self.type.encoding)

    @property
    def _has_ttnn_encoding(self) -> bool:
        return self.type.encoding is not None and "ttnn" in str(self.type.encoding)


@dataclass(frozen=True)
class Operand(OperandAndResultBase):
    """Simple dataclass representing an operand of a MLIR operation."""


@dataclass(frozen=True)
class Result(OperandAndResultBase):
    """Simple dataclass representing result of a MLIR operation."""


class OpWrapper:
    """Convenience wrapper around MLIR op."""

    # ----- Public methods and properties -----

    def __init__(self, op: OpView, attrs: Optional[OpAttributeMap] = None) -> None:
        """Constructor."""
        self.op = op
        self.operands = [
            Operand(operand.get_name(), operand.type) for operand in op.operands
        ]
        self.result = (
            Result(op.result.get_name(), op.result.type)
            if len(op.results) > 0
            else None
        )
        self.attributes = attrs

    def __str__(self) -> str:
        return str(self.op)

    def __repr__(self) -> str:
        return str(self)

    @property
    def name(self) -> str:
        return self.op.name

    def as_module_str(self) -> str:
        """Returns self wrapped in a MLIR module str."""
        return OpWrapper._wrap_in_module_str(
            self.op, self.operands, self.result, self.attributes
        )

    def as_module(self) -> ModuleWrapper:
        """
        Returns self wrapped in `ModuleWrapper`.

        Wrapper will contain original MLIR module, dialect used and op from which
        it was created (`self`).
        """
        module_wrapper = parse_module_str(self.as_module_str())
        # Store a reference to the original op.
        module_wrapper.origin_op = self
        return module_wrapper

    # ----- Private methods and classes -----

    @staticmethod
    def _wrap_in_module_str(
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


class ModuleWrapper:
    """
    Convenience wrapper around MLIR module.

    Provides posibility to keep track of the op from which module was generated, useful
    in op by op processing pipeline.
    """

    def __init__(
        self,
        module: Module,
        dialect: Optional[ModuleDialect] = None,
        origin_op: Optional[OpWrapper] = None,
    ) -> None:
        self.module: Module = module
        self.dialect: ModuleDialect = dialect or ModuleDialect.detect(module)
        self.origin_op: Optional[OpWrapper] = origin_op

    def __repr__(self) -> str:
        s = f"ModuleWrapper(\n{self.module})"

        if self.has_origin_op:
            s += f", Generated from: {self.origin_op.name}"

        return s

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
        """Returns list of operations in module's body."""
        return self.module.body.operations

    @property
    def inputs(self) -> List[Operand]:
        """
        Shorthand accessor for operands of origin op.

        It asserts that module wrapper was generated by wrapping an op. In case of
        module with multiple ops in func body it can be made to reflect inputs of the
        func, but there was no use case for that in current state of things.
        """
        assert self.has_origin_op
        return self.origin_op.operands

    @property
    def outputs(self) -> List[Result]:
        """
        Shorthand accessor for results of origin op.

        NOTE hardcoded to return list with only one result if it exists. If needed can
        be easily expanded to return multiple results.
        """
        assert self.has_origin_op
        return [self.origin_op.result] if self.origin_op.result is not None else []

    @property
    def has_origin_op(self) -> bool:
        """Returns True if module was generated by wrapping an op."""
        return self.origin_op is not None

    def copy(self) -> ModuleWrapper:
        """
        Creates a copy of self.

        Take note that copy is shallow: reference to the origin op stays the same.

        This is useful to not mess up the original module in compilation steps which are
        done in-place.
        """
        copy = parse_module_str(str(self.module))
        copy.origin_op = self.origin_op
        return copy


def parse_module_str(module_str: str) -> ModuleWrapper:
    """
    Within a temporary context registers necessary dialects and parses `module_str`
    returning ModuleWrapper instance.
    """

    def preprocess_module_str(module_str: str) -> str:
        """Preprocesses module string by removing `loc(...)` from it."""
        loc_pattern = re.compile(r"\s*loc\([^)]*\)")
        return re.sub(loc_pattern, "", module_str)

    def register_dialect(dialect: ModuleDialect, ctx: Context) -> None:
        """
        Detects dialect used in `module_str` and registers it with context `ctx`.
        """
        if dialect == ModuleDialect.STABLE_HLO:
            stablehlo.register_dialect(ctx)
            # TODO there must be a better way to do this. We need to register `ttir`
            # (or any other of our dialects) otherwise we'll encounter problems with
            # `func` dialect which isn't included through `stablehlo` and doesn't
            # provide `func.register_dialect(ctx)` on its own.
            ttir.register_dialect(ctx)
        elif dialect == ModuleDialect.TTIR:
            ttir.register_dialect(ctx)
        elif dialect == ModuleDialect.TTNN:
            ttnn.register_dialect(ctx)
        elif dialect == ModuleDialect.TT:
            tt.register_dialect(ctx)
        else:
            raise ValueError(f"Unknown dialect: {dialect.name}")

    with Context() as ctx:
        cleaned_module_str = preprocess_module_str(module_str)
        dialect = ModuleDialect.detect(cleaned_module_str)
        # Must register dialect in order for parsing to work.
        register_dialect(dialect, ctx)
        mlir_module = Module.parse(cleaned_module_str)
        return ModuleWrapper(mlir_module, dialect=dialect)


def convert_to_module_wrapper(func: Callable) -> Callable:
    """
    Decorator to ensure that the `module` argument is always of type `ModuleWrapper`.
    If it's a string, it will be parsed using `parse_module_str`. If it is a `Module`
    it will be wrapped.

    This allows decorated function to accept str or Module or ModuleWrapper.
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
