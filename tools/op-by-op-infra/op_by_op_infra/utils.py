# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from ttmlir.compile_and_run_utils import ModuleDialect, create_mlir_module_from_string
from ttmlir.dialects import func, ttcore, ttnn
from ttmlir.ir import (
    Module,
    OpAttributeMap,
    OperationList,
    OpView,
    RankedTensorType,
    Type,
)


@dataclass(frozen=True)
class OperandAndResultBase(ABC):
    """
    Abstract base dataclass representing an operand or result of a MLIR operation.

    Abstracting them this way makes them have the same implementation. Derived classes
    exist for each of them just to be strict and clear.
    """

    name: str
    type: Type

    @property
    def shape(self) -> List[int]:
        return self.type.shape if isinstance(self.type, RankedTensorType) else []

    @property
    def data_type(self) -> str:
        return (
            str(self.type.element_type)
            if isinstance(self.type, RankedTensorType)
            else ""
        )

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

        mem_layout = self._layout_attr.tensor_memory_layout_as_int

        return (
            str(ttnn.TensorMemoryLayout(mem_layout)) if mem_layout is not None else None
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
        return (
            isinstance(self.type, RankedTensorType)
            and self.type.encoding is not None
            and "ttnn" in str(self.type.encoding)
        )


@dataclass(frozen=True)
class Operand(OperandAndResultBase):
    """Simple dataclass representing an operand of a MLIR operation."""


@dataclass(frozen=True)
class Result(OperandAndResultBase):
    """Simple dataclass representing result of a MLIR operation."""


class OpWrapper:
    """Convenience wrapper around MLIR op."""

    # ----- Public methods and properties -----

    def __init__(
        self,
        op: OpView,
        attrs: Optional[OpAttributeMap] = None,
        func_op: Optional[func.FuncOp] = None,
    ) -> None:
        """Constructor."""
        self.op = op
        self.operands = [
            Operand(operand.get_name(), operand.type) for operand in op.operands
        ]
        self.results = [Result(result.get_name(), result.type) for result in op.results]
        self.attributes = attrs
        self.func_op = func_op

    def __str__(self) -> str:
        return str(self.op)

    def __repr__(self) -> str:
        return str(self)

    @property
    def name(self) -> str:
        return self.op.name

    def as_module_str(self) -> str:
        """
        Wraps `self.op` in a MLIR `func` and then in a MLIR `module` and returns string
        representation of that module.

        Example
        ------
        ```
        module attributes {...} {
            func.func main(...) -> ... {
                %0 = self.op ...
                return %0
            }
        }
        ```
        """
        # Make operands unique to handle cases where the same operand is used multiple times
        unique_operands = {operand.name: operand for operand in self.operands}.values()
        unpacked_operands = ", ".join(
            f"{operand.name}: {operand.type}" for operand in unique_operands
        )

        if len(self.results) > 1:
            results = f"({', '.join(result.name for result in self.results)})"
            return_type = f"({', '.join(str(result.type) for result in self.results)})"
            return_stmt = f"return {results} : {return_type}"
        elif len(self.results) == 1:
            return_type = self.results[0].type
            return_stmt = f"return {self.results[0].name} : {self.results[0].type}"
        else:
            return_type = "()"
            return_stmt = "return"

        # Handle special case of modules that carry attributes.
        if self.attributes is not None:
            attrs = (
                "{" + ",\n".join(f"{a.name} = {a.attr}" for a in self.attributes) + "}"
            )
        else:
            attrs = "{}"

        # Add func_op if present
        func_op_str = f"  {self.func_op} \n" if self.func_op is not None else ""

        return (
            f"module attributes {attrs} {{ \n"
            f"  func.func @main({unpacked_operands}) -> {return_type} {{ \n"
            f"    {self.op} \n"
            f"    {return_stmt} \n"
            f"  }} \n"
            f"{func_op_str}"
            f"}}"
        )

    def as_module(self) -> ModuleWrapper:
        """
        Returns self represented as a single-op module, wrapped in `ModuleWrapper`.
        """
        module_wrapper = parse_module_str(self.as_module_str())
        # Store important references to the original op.
        module_wrapper.origin_op_name = str(self.name)
        module_wrapper.origin_op_operands = self.operands
        module_wrapper.origin_op_results = self.results
        return module_wrapper


class TTNNOpWrapper(OpWrapper):
    """
    Aux op wrapper for TTNN ops carrying additional ttcore.device op which will be embeded
    in inner-most module above the wrapped op itself.

    See docstring for `as_module_str` and `TTNNModuleWrapper`.
    """

    def __init__(
        self,
        op: OpView,
        tt_device_op: ttcore.DeviceOp,
        attrs: Optional[OpAttributeMap] = None,
        func_op: Optional[func.FuncOp] = None,
    ) -> None:
        super().__init__(op, attrs, func_op)
        self.tt_device_op = tt_device_op

    # @override
    def as_module_str(self) -> str:
        """
        Implements wrapping of `self.op` in a TTNN module which is a bit more complex
        than what base class does.

        Example
        -------
        ```
        module {
            ttcore.device_module {
                builtin.module attributes {...} {
                    self.tt_device_op ...
                    func.func main(...) -> ... {
                        %0 = self.op ...
                        return %0
                    }
                }
            }
        }
        ```
        """
        # Make operands unique to handle cases where the same operand is used multiple times
        unique_operands = {operand.name: operand for operand in self.operands}.values()
        unpacked_operands = ", ".join(
            f"{operand.name}: {operand.type}" for operand in unique_operands
        )

        if len(self.results) > 1:
            results = f"({', '.join(result.name for result in self.results)})"
            return_type = f"({', '.join(str(result.type) for result in self.results)})"
            return_stmt = f"return {results} : {return_type}"
        elif len(self.results) == 1:
            return_type = self.results[0].type
            return_stmt = f"return {self.results[0].name} : {self.results[0].type}"
        else:
            return_type = "()"
            return_stmt = "return"

        # Handle special case of modules that carry attributes.
        if self.attributes is not None:
            attrs = (
                "{" + ",\n".join(f"{a.name} = {a.attr}" for a in self.attributes) + "}"
            )
        else:
            attrs = "{}"

        # Add func_op if present
        func_op_str = f"  {self.func_op} \n" if self.func_op is not None else ""

        return (
            f"module {{ \n"
            f"ttcore.device_module {{ \n"
            f"builtin.module attributes {attrs} {{ \n"
            f"  {self.tt_device_op} \n"
            f"{func_op_str}"
            f"  func.func @main({unpacked_operands}) -> {return_type} {{ \n"
            f"    {self.op} \n"
            f"    {return_stmt} \n"
            f"  }} \n"
            f"}} \n"
            f"}} \n"
            f"}}"
        )


class ModuleWrapper:
    """
    Convenience wrapper around MLIR module.

    Provides posibility to keep track of the op from which module was generated, useful
    in op by op processing pipeline.
    """

    # ----- Public methods and properties -----

    def __init__(
        self,
        module: Module,
        dialect: Optional[ModuleDialect] = None,
        *,
        origin_op_name: Optional[str] = None,
        origin_op_operands: Optional[List[Operand]] = None,
        origin_op_results: Optional[List[Result]] = None,
    ) -> None:
        self.module: Module = module
        self.dialect: ModuleDialect = dialect or ModuleDialect.detect(module)

        self.origin_op_name = origin_op_name
        self.origin_op_operands = origin_op_operands
        self.origin_op_results = origin_op_results

    def __repr__(self) -> str:
        s = f"ModuleWrapper(\ndialect: {self.dialect.value}\n{self.module}"

        if self.has_origin_op:
            s += f"generated from: {self.origin_op_name}"

        s += ")"

        return s

    @property
    def operations(self) -> OperationList:
        """Returns list of operations in module's body."""
        return self.module.body.operations

    @property
    def inputs(self) -> List[Operand]:
        """Shorthand accessor for operands of origin op."""
        assert self.has_origin_op
        return self.origin_op_operands if self.origin_op_operands is not None else []

    @property
    def outputs(self) -> List[Result]:
        """Shorthand accessor for results of origin op."""
        assert self.has_origin_op
        return self.origin_op_results if self.origin_op_results is not None else []

    @property
    def has_origin_op(self) -> bool:
        """Returns True if module originated as a single op."""
        return self.origin_op_name is not None

    def wrap_op(self, op: OpView, func_op: Optional[func.FuncOp] = None) -> OpWrapper:
        return OpWrapper(op, self._attributes, func_op)

    # ----- Private methods and properties -----

    @property
    def _attributes(self) -> Optional[OpAttributeMap]:
        """Returns module attributes if any, otherwise None."""
        return (
            self.module.operation.attributes
            if len(self.module.operation.attributes) > 0
            else None
        )


class TTNNModuleWrapper(ModuleWrapper):
    """
    In general, modules look like
    ```
    module attributes {...} {
        func func1() {...}
        func func2() {...}
        ...
    }
    ```
    TTNN modules are a bit more complicated. They look like
    ```
    module {
        ttcore.device_module {
            module attributes {...} {
                ttcore.device ...
                func func1() {...}
                func func2() {...}
            }
        }
    }
    ```

    This class is implemented to abstract away those complexities so MLIRModuleSplitter
    doesn't have to handle them.
    """

    # ----- Public methods and properties -----

    def __init__(
        self,
        module: Module,
        dialect: Optional[ModuleDialect] = None,
        *,
        origin_op_name: Optional[str] = None,
        origin_op_operands: Optional[List[Operand]] = None,
        origin_op_results: Optional[List[Result]] = None,
    ) -> None:
        super().__init__(
            module,
            dialect,
            origin_op_name=origin_op_name,
            origin_op_operands=origin_op_operands,
            origin_op_results=origin_op_results,
        )

        self._tt_device_module_op: ttcore.DeviceModuleOp = self.module.body.operations[
            0
        ]
        assert isinstance(self._tt_device_module_op, ttcore.DeviceModuleOp)

        self._nested_module_op: OpView = self._tt_device_module_op.bodyRegion.blocks[
            0
        ].operations[0]
        self._nested_module = parse_module_str(str(self._nested_module_op))

        self._tt_device_op: ttcore.DeviceOp = self._operations[0]
        assert isinstance(self._tt_device_op, ttcore.DeviceOp)

    def __repr__(self) -> str:
        s = f"TTNNModuleWrapper(\ndialect: {self.dialect.value}\n{self.module}"

        if self.has_origin_op:
            s += f"generated from: {self.origin_op_name}"

        s += ")"

        return s

    # @override
    @property
    def operations(self) -> List[OpView]:
        # Skip the first ttcore.device op.
        return list(self._operations)[1:]

    # @override
    def wrap_op(
        self, op: OpView, func_op: Optional[func.FuncOp] = None
    ) -> TTNNOpWrapper:
        return TTNNOpWrapper(op, self._tt_device_op, self._attributes, func_op)

    # ----- Private methods and properties -----

    # @override
    @property
    def _attributes(self) -> Optional[OpAttributeMap]:
        return (
            self._nested_module_op.attributes
            if len(self._nested_module_op.attributes) > 0
            else None
        )

    @property
    def _operations(self) -> OperationList:
        return self._nested_module.module.body.operations


def is_top_level_ttnn_module(
    module: Module, dialect: Optional[ModuleDialect] = None
) -> bool:
    """
    Returns True only if module is in TTNN dialect and contains one nested
    ttcore.device_module.
    """
    dialect = dialect or ModuleDialect.detect(module)

    # Check if the module has the expected TTNN structure
    if dialect == ModuleDialect.TTNN and len(module.body.operations) == 1:
        # Check the name of the operation rather than its type
        op = module.body.operations[0]
        op_name = str(op).split("\n")[0].strip()
        return "ttcore.device_module" in op_name
    return False


def parse_module_str(module_str: str) -> ModuleWrapper:
    """
    Within a temporary context registers necessary dialects and parses `module_str`
    returning ModuleWrapper instance.
    """
    mlir_module = create_mlir_module_from_string(module_str)
    return (
        TTNNModuleWrapper(mlir_module)
        if is_top_level_ttnn_module(mlir_module)
        else ModuleWrapper(mlir_module)
    )


def preprocess_module_str(module_str: str) -> str:
    """
    Preprocesses module string by removing:
    - lines that start with `#loc`
    - `loc(...(...)...)` from other lines
    - `.sdy.mesh...` lines
    """
    loc_pattern = re.compile(r"^#loc.*$", re.MULTILINE)
    module_str = re.sub(loc_pattern, "", module_str)
    loc_pattern = re.compile(r"\s*loc\((?:[^()]*(?:\([^()]*\))*)*\)")
    module_str = re.sub(loc_pattern, "", module_str)
    loc_pattern = re.compile(r"\s*sdy.mesh.*$", re.MULTILINE)
    return re.sub(loc_pattern, "", module_str)


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
            module = preprocess_module_str(module)
            m = parse_module_str(module)
        elif isinstance(module, Module):
            m = (
                TTNNModuleWrapper(module)
                if is_top_level_ttnn_module(module)
                else ModuleWrapper(module)
            )
        elif isinstance(module, ModuleWrapper):
            m = module
        else:
            raise TypeError(f"Unexpected module type {type(module)}")

        # Call the original function with the converted module.
        return func(self, m, *args, **kwargs)

    return wrapper
