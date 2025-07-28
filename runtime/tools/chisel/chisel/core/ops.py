# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MLIR Operation Utilities

This module provides essential utilities for working with MLIR operations, including:
- Operation input/output analysis
- MLIR module parsing and manipulation
- Function and operation traversal
- Location tracking and management

Key Components:
- get_op_outputs: Extracts output tensors from operations
- get_op_inputs: Extracts input tensors from operations
- IRModule: A wrapper around MLIR Module with enhanced functionality
"""
from functools import cache
from typing import Dict, List, Tuple

from ttmlir.dialects import func
from ttmlir.ir import (
    AsmState,
    Context,
    Module,
    Operation,
    WalkOrder,
    WalkResult,
    BlockArgument,
)

from ..utils.location import hash_location
from .enums import ExecutionType


@cache
def get_op_outputs(op: Operation) -> list:
    """
    Extract output tensors from an MLIR operation.

    This function filters and returns only the operation results that have tensor-like
    properties (shape and element_type). The results are cached for performance.

    Args:
        op (Operation): The MLIR operation to extract outputs from

    Returns:
        list: List of output values that are tensors
    """
    outputs = []
    for result in op.results:
        # Only include results that have tensor-like properties
        if hasattr(result.type, "shape") and hasattr(result.type, "element_type"):
            outputs.append(result)
    return outputs


@cache
def get_op_inputs(op: Operation) -> list:
    """
    Extract input tensors from an MLIR operation.

    This function filters and returns only the operation operands that have tensor-like
    properties (shape and element_type). The results are cached for performance.

    Args:
        op (Operation): The MLIR operation to extract inputs from

    Returns:
        list: List of input values that are tensors
    """
    inputs = []
    for operand in op.operands:
        # Only include operands that have tensor-like properties
        if hasattr(operand.type, "shape") and hasattr(operand.type, "element_type"):
            inputs.append(operand)
    return inputs


class IRModule:
    """
    Enhanced MLIR module wrapper with helper methods for operation analysis and traversal.

    This class provides a higher-level interface for working with MLIR modules, including:
    - Function management and lookup
    - Operation traversal and analysis
    - Location tracking for debugging and optimization
    - Caching for improved performance

    Note: Currently designed with a single main function in mind, but can be extended
    to handle multiple functions more flexibly.

    Attributes:
        mlir_module (Module): The underlying MLIR module
        context (Context): MLIR context associated with the module
        execution_type (ExecutionType): Type of execution (GOLDEN/DEVICE)
        main_function_name (str): Name of the main function
        _functions (Dict[str, Operation]): Cache of function `func.FuncOp` by name
        _function_ops (Dict[str, List[Operation]]): Cache of function main body operations by name
        _asm_state (Dict[str, AsmState]): Assembly state cache for faster lookups
        _last_loc_line (Dict[Tuple[int, int], int]): Mapping of locations to line numbers
        ignored_ops (List[str]): List of operation names to ignore during traversal
    """

    def __init__(
        self,
        mlir_module: Module,
        context: Context,
        execution_type: ExecutionType,
        functions: List[str],
        current_function_name: str | None = None,
        ignored_ops: List[str] = [],
    ):
        """
        Create an IRModule from MLIR text

        Args:
            mlir_text (str): MLIR text
            context (Context): Context
            execution_type (ExecutionType): Execution type
            functions (List[str]): List of function names
            current_function_name (str, optional): Current function name. If not provided, the first function is used.
            ignored_ops (List[str], optional): Ops in MLIR to ignore.
        """
        self.mlir_module: Module = mlir_module
        self.context: Context = context
        self.execution_type: ExecutionType = execution_type
        # Ops that can be ignored for example: `ttir.empty`, `ttnn.deallocate`
        self.ignored_ops: List[str] = ignored_ops

        # Function name -> Operation
        self._functions: Dict[str, Operation] = {
            name: self._add_function(name) for name in functions
        }
        # Function name -> List of main body operations
        self._function_ops: Dict[str, List[Operation]] = {
            name: self._extract_function_ops(name) for name in functions
        }
        # AsmState reduces time of .get_asm and .get_name methods
        self._asm_state: Dict[str, AsmState] = {
            name: AsmState(self._functions[name]) for name in functions
        }
        # Last line number for each location
        self._last_loc_line: Dict[str, Dict[Tuple[int, int], int]] = {
            name: self._generate_last_loc_line(name) for name in functions
        }

        # Set current function name if provided, otherwise use the first function
        if current_function_name is not None:
            self.current_function_name = current_function_name
        else:
            self.current_function_name = functions[0]

    def get_asm_state(self) -> AsmState:
        """
        Returns the `AsmState` for the `func.FuncOp` with the given name.
        Used for faster execution of .get_asm and .get_name methods on ir.Values and ir.BlockArguments
        """
        return self._asm_state[self.current_function_name]

    def get_function(self) -> Operation:
        """
        Returns the `func.FuncOp` with the given name
        """
        return self._functions[self.current_function_name]

    def get_function_inputs(self) -> List[BlockArgument]:
        """
        Returns the input arguments of the `func.FuncOp` with the given name
        """
        return self._functions[self.current_function_name].arguments

    def get_function_ops(self) -> List[Operation]:
        """
        Returns the operations of the `func.FuncOp` with the given name
        """
        return self._function_ops[self.current_function_name]

    @property
    def last_loc_line(self) -> Dict[Tuple[int, int], int]:
        """
        Get the mapping of operation locations to their indices in the operation list.

        This property provides access to a dictionary that maps (line, column) location
        tuples to their corresponding operation indices in the function's operation list.
        This is primarily used for operation grouping and execution order tracking.

        Returns:
            Dict[Tuple[int, int], int]: A dictionary mapping location hashes to operation indices

        Note:
            The mapping must be populated using populate_last_loc_line() before use.
        """
        return self._last_loc_line[self.current_function_name]

    def _extract_function_ops(self, name: str) -> List[Operation]:
        assert name in self._functions

        ops = []
        for region in self._functions[name].regions:
            for block in region.blocks:
                for op in block.operations:
                    if op.name in self.ignored_ops:
                        continue
                    ops.append(op)

        return ops

    def _add_function(self, name: str) -> Operation:
        # Search through all operations to find the function
        for op in self._dfs(self.mlir_module.operation):
            if isinstance(op, func.FuncOp) and op.name.value == name:
                return op
        raise ValueError(f"Function {name} not found in module")

    def _generate_last_loc_line(self, name: str) -> Dict[Tuple[int, int], int]:
        last_loc_line = {}
        for i, op in enumerate(self._function_ops[name]):
            last_loc_line[hash_location(op.location)] = i
        return last_loc_line

    def _dfs(self, op: Operation, walk_order: WalkOrder = WalkOrder.POST_ORDER) -> List[Operation]:
        assert op is not None
        ops = []

        def _walk_ops(op):
            """Callback function for MLIR's walk operation"""
            nonlocal ops
            ops.append(op.opview)
            return WalkResult.ADVANCE

        # Perform the actual walk using MLIR's built-in walker
        op.operation.walk(_walk_ops, walk_order=walk_order)
        return ops
