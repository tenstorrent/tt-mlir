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

# Patterns for preprocessing MLIR module strings
_LOC_LINE_PATTERN = re.compile(r"^#loc.*$", re.MULTILINE)  # Match #loc lines
_LOC_INLINE_PATTERN = re.compile(
    r"\s*loc\((?:[^()]*(?:\([^()]*\))*)*\)"
)  # Match inline loc(...) annotations
_SDY_MESH_PATTERN = re.compile(r"\s*sdy.mesh.*$", re.MULTILINE)  # Match sdy.mesh lines

# Operations that are preserved in the function body rather than parameterized as inputs.
# - ttnn.get_device: Returns device handle, cannot be replaced with test input
# - stablehlo.constant: Required for reduce/reduce_window pattern matching, otherwise improves test accuracy
# - stablehlo.iota: Required for reduce/sort pattern matching, otherwise improves test accuracy
PRESERVED_OP_NAMES = [
    "ttnn.get_device",
    "stablehlo.constant",
    "stablehlo.iota",
]

# Operations that preserve constant semantics when their input is a constant.
# Mirrors the logic in StableHLOToTTIRPatterns.cpp::getStableHLOConstantDefiningOp.
CONSTANT_PRESERVING_OPS = [
    "stablehlo.reshape",
    "stablehlo.broadcast_in_dim",
    "stablehlo.convert",
]


def _get_constant_chain(operand) -> List:
    """Traverses backward through constant-preserving ops to find a constant chain."""
    defining_op = operand.owner
    # owner is None for function args, or a Block for block args (not an Operation)
    if not defining_op or not hasattr(defining_op, "name"):
        return []

    # Direct constant case
    if defining_op.name in PRESERVED_OP_NAMES:
        return [defining_op]

    # Traverse backward through constant-preserving ops
    chain = []
    current_op = defining_op
    while current_op and hasattr(current_op, "name"):
        if current_op.name in CONSTANT_PRESERVING_OPS:
            chain.append(current_op)
            current_op = current_op.operands[0].owner
        elif current_op.name in PRESERVED_OP_NAMES:
            chain.append(current_op)
            # Reverse to get [constant, ..., final_preserving_op]
            chain.reverse()
            return chain
        else:
            break

    return []


def _get_captured_values_from_regions(op) -> List:
    """Collects values used inside op's regions that are defined outside (captured)."""
    captured = []
    for region in op.regions:
        for block in region.blocks:
            # Collect all values defined within this block:
            # 1. Block arguments (e.g., %arg204, %arg205 in reducer)
            # 2. Results of operations within the block
            defined_in_block = set()
            for arg in block.arguments:
                defined_in_block.add(arg)
            for inner_op in block.operations:
                for result in inner_op.results:
                    defined_in_block.add(result)

            # Find operands that are used but not defined in the block = captured
            for inner_op in block.operations:
                for operand in inner_op.operands:
                    if operand not in defined_in_block and operand not in captured:
                        captured.append(operand)

    return captured


def _replace_mlir_identifier(text: str, old_name: str, new_name: str) -> str:
    """Replace SSA value name (e.g., %arg0), avoiding partial matches like %arg01."""
    pattern = re.escape(old_name) + r"(?=[^a-zA-Z0-9_]|$)"
    return re.sub(pattern, new_name, text)


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
    """
    Convenience wrapper around MLIR op.

    Extracts and caches all necessary information from the live OpView at construction
    time, so the wrapper can outlive the MLIR context.
    If the op has preserved operands, they are stored, to later be added to the module.
    """

    # ----- Public methods and properties -----

    def __init__(
        self,
        op: OpView,
        attrs: Optional[OpAttributeMap] = None,
        func_op: Optional[func.FuncOp] = None,
        origin_model: str = "",
    ) -> None:
        """Constructor."""
        self.op_string = str(op)
        self.op_name = op.name
        self.func_op_string = str(func_op) if func_op is not None else ""

        # First pass: collect unique operands by original_name (keep first occurrence)
        unique_operands = {}  # original_name -> operand
        for operand in op.operands:
            original_name = operand.get_name()
            if original_name not in unique_operands:
                unique_operands[original_name] = operand

        # Also collect captured values from regions (e.g., constants used inside reducer blocks)
        for captured in _get_captured_values_from_regions(op):
            original_name = captured.get_name()
            if original_name not in unique_operands:
                unique_operands[original_name] = captured

        # Second pass: process unique operands, detect constant chains
        operand_mapping = {}  # original_name -> standardized_name
        parameterized_operands = []
        preserved_chains = (
            []
        )  # List of (chain_ops, chain_original_names, final_standardized_name)
        pres_counter = 0
        arg_counter = 0

        for original_name, operand in unique_operands.items():
            chain = _get_constant_chain(operand)
            if chain:
                # This operand comes from a constant or constant-preserving chain
                final_standardized_name = f"%pres{pres_counter}"
                operand_mapping[original_name] = final_standardized_name

                # Collect original names for all values in the chain
                chain_original_names = [
                    chain_op.results[0].get_name() for chain_op in chain
                ]

                preserved_chains.append((chain, chain_original_names, pres_counter))
                pres_counter += 1
            else:
                # Parameterized operand
                standardized_name = f"%operand{arg_counter}"
                operand_mapping[original_name] = standardized_name
                parameterized_operands.append(Operand(standardized_name, operand.type))
                arg_counter += 1

        # Build result mapping - track which results have users for selective return
        result_mapping = {}
        num_results = len(op.results)
        for i, result in enumerate(op.results):
            original_name = result.get_name()
            standardized_name = f"%res{i}"
            result_mapping[original_name] = standardized_name

        # For multi-result ops, only return results that have users in the original graph.
        # This is important for pattern matching (e.g., argmax checks that first result is unused).
        if num_results > 1:
            used_result_indices = [
                i
                for i, result in enumerate(op.results)
                if any(True for _ in result.uses)
            ]
            # If no results have users (all dead), return all of them to avoid empty return
            if not used_result_indices:
                used_result_indices = list(range(num_results))
        else:
            used_result_indices = list(range(num_results))

        # For multi-result ops, replace the defining syntax %base:N with %res0, %res1, ...
        # MLIR defines multi-result ops as %name:N = op(...) but they're referenced as %name#0, %name#1
        if len(op.results) > 1:
            first_result_name = op.results[0].get_name()
            if "#" in first_result_name:
                base_name = first_result_name.split("#")[0]
                old_def_pattern = f"{base_name}:{len(op.results)}"
                new_def_pattern = ", ".join(f"%res{i}" for i in range(len(op.results)))
                self.op_string = self.op_string.replace(
                    old_def_pattern, new_def_pattern
                )

        # Replace all identifiers in main op string
        for original, standardized in {**result_mapping, **operand_mapping}.items():
            self.op_string = _replace_mlir_identifier(
                self.op_string, original, standardized
            )

        # Build preserved_ops_strings with proper SSA naming for chains
        self.preserved_ops_strings = []
        for chain, chain_original_names, pres_idx in preserved_chains:
            # Build SSA name mapping for this chain
            # Intermediate values: %pres{pres_idx}_{i}, final value: %pres{pres_idx}
            chain_name_mapping = {}
            for i, orig_name in enumerate(chain_original_names):
                if orig_name is None:
                    continue
                if i == len(chain) - 1:
                    # Final op in chain gets the main preserved name
                    chain_name_mapping[orig_name] = f"%pres{pres_idx}"
                else:
                    # Intermediate ops get indexed names
                    chain_name_mapping[orig_name] = f"%pres{pres_idx}_{i}"

            # Convert each op in chain to string and apply renaming
            for i, chain_op in enumerate(chain):
                op_str = str(chain_op)
                for orig, standardized in chain_name_mapping.items():
                    op_str = _replace_mlir_identifier(op_str, orig, standardized)
                self.preserved_ops_strings.append(op_str)

        self.operands = parameterized_operands
        self.results = [
            Result(f"%res{i}", result.type) for i, result in enumerate(op.results)
        ]
        # Track which results to return (only those with users in original graph)
        self.used_result_indices = used_result_indices
        # Convert attributes to string to avoid MLIR context issues
        if attrs is not None:
            self.attributes_str = (
                "{" + ",\n".join(f"{a.name} = {a.attr}" for a in attrs) + "}"
            )
        else:
            self.attributes_str = "{}"
        self.origin_model = [origin_model]

    def __str__(self) -> str:
        return self.op_string

    def __repr__(self) -> str:
        return str(self)

    @property
    def name(self) -> str:
        return self.op_name

    def add_origin_model(self, model: str) -> None:
        """Adds a new origin model to the list if it's not already present."""
        if model and model not in self.origin_model:
            self.origin_model.append(model)

    def as_module_str(self) -> str:
        """
        Wraps the cached op string in a MLIR `func` and then in a MLIR `module` and
        returns string representation of that module.

        If the op has preserved operands, they are emitted at the beginning of the
        function body.

        Example
        ------
        ```
        module attributes {...} {
            func.func main(%arg0: tensor<...>) -> ... {
                %pres1 = stablehlo.constant dense<...> : tensor<...>
                %res0 = <op>(%arg0, %pres1) ...
                return %res0
            }
        }
        ```
        """
        unpacked_operands = ", ".join(
            f"{operand.name}: {operand.type}" for operand in self.operands
        )

        # Prepare preserved ops to emit in function body
        preserved_body = ""
        if self.preserved_ops_strings:
            preserved_body = "    " + "\n    ".join(self.preserved_ops_strings) + "\n"

        # Only return results that had users in the original graph
        returned_results = [self.results[i] for i in self.used_result_indices]

        if len(returned_results) > 1:
            results = f"{', '.join(result.name for result in returned_results)}"
            return_type = (
                f"{', '.join(str(result.type) for result in returned_results)}"
            )
            return_stmt = f"return {results} : {return_type}"
        elif len(returned_results) == 1:
            return_type = returned_results[0].type
            return_stmt = (
                f"return {returned_results[0].name} : {returned_results[0].type}"
            )
        else:
            return_type = "()"
            return_stmt = "return"

        return (
            f"module attributes {self.attributes_str} {{ \n"
            f'  func.func @main({unpacked_operands}) -> ({return_type}) attributes {{tt.function_type = "forward_device"}} {{ \n'
            f"{preserved_body}"
            f"    {self.op_string} \n"
            f"    {return_stmt} \n"
            f"  }} \n"
            f"  {self.func_op_string}"
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
        # Convert list of origin models to a single string (comma-separated if multiple)
        module_wrapper.origin_model = ", ".join(self.origin_model)
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
        origin_model: str = "",
    ) -> None:
        super().__init__(op, attrs, func_op, origin_model)
        self.tt_device_op_string = str(tt_device_op)

    # @override
    def as_module_str(self) -> str:
        """
        Implements wrapping of the cached op string in a TTNN module which is a bit more
        complex than what base class does.

        If the op has preserved operands, they are emitted at the beginning of the
        function body.

        Example
        -------
        ```
        module {
            ttcore.device_module {
                builtin.module attributes {...} {
                    <tt_device_op> ...
                    func.func main(%arg0: tensor<...>) -> ... {
                        %pres1 = stablehlo.constant dense<...> : tensor<...>
                        %res0 = <op>(%arg0, %pres1) ...
                        return %res0
                    }
                }
            }
        }
        ```
        """
        unpacked_operands = ", ".join(
            f"{operand.name}: {operand.type}" for operand in self.operands
        )

        # Prepare preserved ops to emit in function body
        preserved_body = ""
        if self.preserved_ops_strings:
            preserved_body = "    " + "\n    ".join(self.preserved_ops_strings) + "\n"

        # Only return results that had users in the original graph
        returned_results = [self.results[i] for i in self.used_result_indices]

        if len(returned_results) > 1:
            results = f"{', '.join(result.name for result in returned_results)}"
            return_type = (
                f"{', '.join(str(result.type) for result in returned_results)}"
            )
            return_stmt = f"return {results} : {return_type}"
        elif len(returned_results) == 1:
            return_type = returned_results[0].type
            return_stmt = (
                f"return {returned_results[0].name} : {returned_results[0].type}"
            )
        else:
            return_type = "()"
            return_stmt = "return"

        return (
            f"module {{ \n"
            f"ttcore.device_module {{ \n"
            f"builtin.module attributes {self.attributes_str} {{ \n"
            f"  {self.tt_device_op_string} \n"
            f"  {self.func_op_string}"
            f'  func.func @main({unpacked_operands}) -> ({return_type}) attributes {{tt.function_type = "forward_device"}} {{ \n'
            f"{preserved_body}"
            f"    {self.op_string} \n"
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
        origin_model: str = "",
    ) -> None:
        self.module: Module = module
        self.dialect: ModuleDialect = dialect or ModuleDialect.detect(module)

        self.origin_op_name = origin_op_name
        self.origin_op_operands = origin_op_operands
        self.origin_op_results = origin_op_results
        self.origin_model = origin_model

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

    def wrap_op(
        self,
        op: OpView,
        func_op: Optional[func.FuncOp] = None,
        origin_model: str = "",
    ) -> OpWrapper:
        return OpWrapper(op, self._attributes, func_op, origin_model)

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
        origin_model: str = "",
    ) -> None:
        super().__init__(
            module,
            dialect,
            origin_op_name=origin_op_name,
            origin_op_operands=origin_op_operands,
            origin_op_results=origin_op_results,
            origin_model=origin_model,
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
        self,
        op: OpView,
        func_op: Optional[func.FuncOp] = None,
        origin_model: str = "",
    ) -> TTNNOpWrapper:
        return TTNNOpWrapper(
            op, self._tt_device_op, self._attributes, func_op, origin_model
        )

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
    module_str = re.sub(_LOC_LINE_PATTERN, "", module_str)
    module_str = re.sub(_LOC_INLINE_PATTERN, "", module_str)
    return re.sub(_SDY_MESH_PATTERN, "", module_str)


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
