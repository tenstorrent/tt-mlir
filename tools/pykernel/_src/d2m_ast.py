# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect
import logging
import os
from typing import Dict, Any, Optional, List
from ttmlir.ir import *
from ttmlir.dialects import (
    ttcore,
    d2m,
    func,
    arith,
)
from ttmlir.dialects._ods_common import get_default_loc_context

logger = logging.getLogger(__name__)


def _get_ast_location(node: ast.AST, source_file: str = None) -> str:
    """Get source location information from an AST node."""
    if hasattr(node, "lineno") and hasattr(node, "col_offset"):
        location = f"line {node.lineno}, col {node.col_offset}"
        if source_file:
            # Get just the filename without the full path
            filename = os.path.basename(source_file)
            location = f"{filename}:{location}"
        return location
    return "unknown location"


def _trace_ast_node(node: ast.AST, operation: str, source_file: str = None):
    """Log trace information for AST node processing."""
    location = _get_ast_location(node, source_file)
    node_type = type(node).__name__
    logger.trace(f"[AST TRACE] {operation} {node_type} at {location}")


from .kernel_types import *
from .utils import _discover_dialect_ops, _cast
from .kernel_ast import TTCompilerBase

from .stream import Stream


class D2MGenericCompiler(TTCompilerBase):
    _syntax = {}

    def __init__(self, name, kernel_type=None, captures={}, *args, **kwargs):
        super().__init__(name, kernel_type, *args, **kwargs)
        self.loc = Location.name(self.name)
        self.captures = captures
        self.streams = set()
        self.supported_nodes.append(ast.AsyncFunctionDef)

        self.grid: List[int] = list(kwargs.get("grid", [1, 1]))
        self.memory_space: str = kwargs.get("memory_space", "L1")
        self.tiled: bool = kwargs.get("tiled", True)

        # Track source file for AST tracing
        self.source_file = kwargs.get("source_file", None)

        self._fn_map = {}
        self._fn_map["iter_index"] = (
            d2m.iter_index,
            [True],
        )  # True for arg as attribute
        self._fn_map["core_index"] = (
            d2m.core_index,
            [True],
        )  # True for arg as attribute
        for name, val in D2MGenericCompiler._syntax.items():
            self._fn_map[name] = val

    def _emit_entry(self, node):
        _trace_ast_node(node, "Processing function entry", self.source_file)
        # TODO: add alloca args name into symbol table
        assert not self.func_entry, "Cannot declare function within a function"

        func_operand_types = []
        for i in range(len(node.args.args)):
            arg = node.args.args[i]
            _trace_ast_node(arg, f"Processing argument {i}", self.source_file)

            if not arg.annotation:
                raise TypeError("All kernel arguments must have a type annotation")
            elif arg.annotation.id == "TensorBlock":
                shape = self.args[i].shape
                logger.debug(
                    f"TensorBlock arg {i}: shape = {shape}, type = {type(shape)}"
                )
                dtype = F32Type.get(self.ctx)
                from ..d2m_api import create_metal_layout

                layout = create_metal_layout(
                    self.ctx, shape, self.grid, self.tiled, self.memory_space
                )
                tile_shape = [32, 32] if self.tiled else [1, 1]

                # Create grid shape that matches logical rank (like builder_utils.py)
                logical_rank = len(shape)
                if len(self.grid) == 2 and logical_rank == 2:
                    # For 2D tensors with 2D grid, use the grid as-is
                    grid_shape = list(self.grid)
                else:
                    # For other cases, pad grid with 1s to match logical rank
                    grid_shape = list(self.grid) + [1] * (logical_rank - len(self.grid))

                # Downcast to properly typed MetalLayoutAttr for getDeviceShape
                typed_layout = ttcore.ir.MetalLayoutAttr.maybe_downcast(layout)
                if typed_layout is None:
                    raise RuntimeError("Failed to downcast MetalLayoutAttr")
                device_shape = typed_layout.getDeviceShape(grid_shape, tile_shape)

                logger.debug(
                    f"TensorBlock device_shape from getDeviceShape: {device_shape}"
                )

                element_type = (
                    ttcore.ir.TileType.get(self.ctx, 32, 32, ttcore.DataType.Float32)
                    if self.tiled
                    else dtype
                )
                tensor_type = RankedTensorType.get(device_shape, element_type, layout)
                func_operand_types.append(tensor_type)
            elif arg.annotation.id == "CircularBuffer":
                shape = self.args[i].shape
                logger.debug(
                    f"CircularBuffer arg {i}: shape = {shape}, type = {type(shape)}"
                )
                dtype = F32Type.get(self.ctx)
                from ..d2m_api import create_metal_layout

                # Create layout to compute device shape (for shard calculation)
                layout = create_metal_layout(
                    self.ctx, shape, self.grid, self.tiled, self.memory_space
                )
                tile_shape = [32, 32] if self.tiled else [1, 1]

                # Create grid shape that matches logical rank
                logical_rank = len(shape)
                if len(self.grid) == 2 and logical_rank == 2:
                    grid_shape = list(self.grid)
                else:
                    grid_shape = list(self.grid) + [1] * (logical_rank - len(self.grid))

                # Get full device shape to extract shard portion
                typed_layout = ttcore.ir.MetalLayoutAttr.maybe_downcast(layout)
                if typed_layout is None:
                    raise RuntimeError("Failed to downcast MetalLayoutAttr")
                device_shape = typed_layout.getDeviceShape(grid_shape, tile_shape)

                # CircularBuffer is LOCAL per-core - use only shard shape (last N dimensions)
                # For 2D tensor with 2D grid: device_shape = [grid_y, grid_x, shard_y, shard_x]
                # We want the shard portion: [shard_y, shard_x]
                logical_rank = len(shape)
                shard_shape = device_shape[-logical_rank:]

                logger.debug(
                    f"CircularBuffer full device_shape: {device_shape}, shard_shape: {shard_shape}"
                )

                element_type = (
                    ttcore.ir.TileType.get(self.ctx, 32, 32, ttcore.DataType.Float32)
                    if self.tiled
                    else dtype
                )

                # Create tensor WITHOUT MetalLayoutAttr - CircularBuffer is local!
                tensor = RankedTensorType.get(shard_shape, element_type, None)
                func_operand_types.append(d2m.ir.CBType.get(self.ctx, tensor))
            elif arg.annotation.id == "Semaphore":
                func_operand_types.append(d2m.ir.SemaphoreType.get(self.ctx))
            else:
                raise TypeError(
                    f"Unknown kernel arguments type annotation {arg.annotation.id}"
                )

        self.func_entry = func.FuncOp(name=node.name, type=(func_operand_types, []))

        self.func_entry.attributes[d2m.ir.ThreadAttr.name] = d2m.ir.ThreadAttr.get(
            self.ctx, self.kernel_type
        )

        self.symbol_tables.append({})

        # prepopulate bb arguments into symbol table
        func_bb = self.func_entry.add_entry_block()
        for i, bb_arg in enumerate(func_bb.arguments):
            self.symbol_tables[-1][node.args.args[i].arg] = bb_arg

        # Add d2m module to symbol table
        self.symbol_tables[-1]["d2m"] = d2m

        self.module_symbol_table = SymbolTable(self.module.operation)

        # update basic block
        with InsertionPoint(func_bb):
            # prepopulate captures at the top of the scope
            for name, val in self.captures.items():
                assert isinstance(name, str)
                if isinstance(val, int):
                    self.symbol_tables[-1][name] = arith.ConstantOp(
                        IndexType.get(self.ctx), val
                    )
                elif isinstance(val, Stream):
                    logger.debug(
                        f"Stream '{name}': val.shape = {val.shape}, type = {type(val.shape)}"
                    )
                    with InsertionPoint.at_block_begin(self.module.body):
                        from ..d2m_api import create_metal_layout

                        layout = create_metal_layout(
                            self.ctx,
                            val.shape,
                            self.grid,
                            self.tiled,
                            self.memory_space,
                        )
                        tile_shape = [32, 32] if self.tiled else [1, 1]

                        # Create grid shape that matches logical rank (like builder_utils.py)
                        logical_rank = len(val.shape)
                        if len(self.grid) == 2 and logical_rank == 2:
                            # For 2D tensors with 2D grid, use the grid as-is
                            grid_shape = list(self.grid)
                        else:
                            # For other cases, pad grid with 1s to match logical rank
                            grid_shape = list(self.grid) + [1] * (
                                logical_rank - len(self.grid)
                            )

                        # Downcast to properly typed MetalLayoutAttr for getDeviceShape
                        typed_layout = ttcore.ir.MetalLayoutAttr.maybe_downcast(layout)
                        if typed_layout is None:
                            raise RuntimeError("Failed to downcast MetalLayoutAttr")
                        device_shape = typed_layout.getDeviceShape(
                            grid_shape, tile_shape
                        )

                        logger.debug(
                            f"Stream '{name}' device_shape from getDeviceShape: {device_shape}"
                        )
                        element_type = (
                            ttcore.ir.TileType.get(
                                self.ctx, 32, 32, ttcore.DataType.Float32
                            )
                            if self.tiled
                            else F32Type.get(self.ctx)
                        )
                        tensor = RankedTensorType.get(
                            device_shape, element_type, layout
                        )
                        globalTensor = ttcore.GlobalOp(val.name, tensor)
                        self.module_symbol_table.insert(globalTensor.operation)
                    self.symbol_tables[-1][name] = ttcore.get_global(tensor, val.name)
                    self.streams.add(val.name)
                else:
                    raise TypeError(f"Invalid capture type for var {name}: {type(val)}")

            for target in node.body:
                _trace_ast_node(
                    target, "Processing function body statement", self.source_file
                )
                self.visit(target)

            func.ReturnOp([])

        self.symbol_tables.pop()

    def visit_FunctionDef(self, node):
        _trace_ast_node(node, "Visiting FunctionDef", self.source_file)
        with self.loc:
            return self._emit_entry(node)

    def visit_AsyncFunctionDef(self, node):
        _trace_ast_node(node, "Visiting AsyncFunctionDef", self.source_file)
        with self.loc:
            return self._emit_entry(node)

    def visit(self, node, **kwargs):
        """Override visit method to add trace logging for all AST nodes."""
        if node is not None:
            _trace_ast_node(node, "Visiting", self.source_file)
        return super().visit(node, **kwargs)


def syntax(syntax_name):
    if syntax_name.startswith("!"):

        def _class_wrapper(cls):
            assert isinstance(cls, type)

            for name, method in cls.__dict__.items():
                if callable(method):
                    sig = inspect.signature(method)
                    first_arg_name = next(iter(sig.parameters.keys()))
                    if first_arg_name == "ast_self":
                        setattr(cls, name, staticmethod(method))
                        qualified = f"{syntax_name}.{name}"
                        D2MGenericCompiler._syntax[qualified] = method

            return cls

        return _class_wrapper
    else:

        def _fn_wrapper(fn):
            assert callable(fn)
            D2MGenericCompiler._syntax[fn.__name__] = fn
            return fn

        return _fn_wrapper
