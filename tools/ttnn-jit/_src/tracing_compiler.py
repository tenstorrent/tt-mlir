# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect

from ttmlir.dialects import func, ttir
from ttmlir.ir import (
    Context,
    Location,
    Module,
    InsertionPoint,
    RankedTensorType,
    FunctionType,
    TypeAttr,
)

from ttnn_jit._src.utils import (
    cleanup_source_code,
    get_maximal_block_sharding_grid,
    get_core_grid_from_tensor_arg,
)
from ttnn_jit._src.tensor_translator import create_tensor_layout
from ttnn_jit._src.conversions import (
    mlir_dtype_from_ttnn_dtype,
    ttnn_dtype_from_mlir_dtype,
)
from ttnn_jit._src.jit_functions import TTNNJitNamespaceUpdater, ResultWrapper
from ttnn_jit._src.tensor_translator import (
    _create_dram_tensor_layout,
    _create_sharded_tensor_layout,
    _calculate_tile_shape,
)
import ttnn


class JitContext:
    """Context for tracking MLIR values during tracing."""

    def __init__(self, func_bb, ctx):
        self.func_bb = func_bb
        self.ctx = ctx
        self.value_map = {}  # Maps id(python_obj) -> MLIR value
        self.func_arg_ids = set()  # Track IDs of original function arguments


class TracingCompiler:
    """Compiler for generating TTIR from tracing execution."""

    def __init__(self, func, *args, memory_config=None, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.tensor_args = kwargs.get("_tensor_args", {})
        self.memory_config = memory_config

    def compile(self):
        """Compile function to TTIR module."""
        ctx = Context()
        cursor = Location.unknown(ctx)
        module = Module.create(cursor)
        insert_point = module.body

        # Create function signature with input types (with layouts)
        input_types = self._create_input_types(ctx)

        # Create function with dummy output type (will be updated later)
        # Use first input type as placeholder
        dummy_output_type = input_types[0] if input_types else None

        with Location.unknown(ctx):
            with InsertionPoint(insert_point):
                func_op = func.FuncOp(
                    name=self.func.__name__,
                    type=(
                        input_types,
                        [dummy_output_type] if dummy_output_type else [],
                    ),
                )
                func_bb = func_op.add_entry_block()

        # Create JIT context
        jit_ctx = JitContext(func_bb, ctx)

        # Map original function arguments to MLIR block arguments
        for i, arg in enumerate(self.args):
            arg_id = id(arg)
            jit_ctx.value_map[arg_id] = func_bb.arguments[i]
            jit_ctx.func_arg_ids.add(arg_id)

        # Modify function to use ttnn_jit namespace
        modified_func = self._modify_function(jit_ctx)

        # Execute modified function
        try:
            # Filter out metadata kwargs that aren't actual function arguments
            call_kwargs = {k: v for k, v in self.kwargs.items() if k != "_tensor_args"}
            result = modified_func(*self.args, **call_kwargs)

            # Extract return value
            return_value = self._extract_return_value(result, jit_ctx, func_bb)
            return_type = return_value.type

            # Update function signature with correct return type
            # We need to recreate the function with the correct signature
            self._update_function_signature(
                module, func_op, input_types, [return_type], func_bb, return_value, ctx
            )
        except Exception as e:
            import traceback

            print(f"Error executing the code: {e}")
            traceback.print_exc()
            raise e

        # Insert output layout conversion if memory_config provided
        try:

            if self.memory_config is None:
                # If no memory_config is provided, set output layout to block-sharded
                output_tensor_shape = [int(dim) for dim in return_type.shape]
                # Get the device core grid from the first tensor arg
                core_grid = get_core_grid_from_tensor_arg(
                    next(iter(self.tensor_args.values()))
                )
                block_sharded_grid = get_maximal_block_sharding_grid(
                    output_tensor_shape, core_grid
                )

                block_sharded_memory_config = ttnn.create_sharded_memory_config(
                    shape=output_tensor_shape,
                    core_grid=ttnn.CoreGrid(
                        x=block_sharded_grid[0] + 1, y=block_sharded_grid[1] + 1
                    ),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    use_height_and_width_as_shard_shape=False,
                )
                self.memory_config = block_sharded_memory_config

            module = self._insert_output_layout_conversion(module, self.memory_config)
        except Exception as e:
            print(f"Output layout conversion insertion failed: {e}")
            raise e

        return module, return_type

    def _create_input_types(self, ctx):
        """
        Create input tensor types.

        For TTIR dialect with ttnn-mode=false, we use plain RankedTensorType
        without TTNN layout encoding. The layout information is preserved in the
        tensor's memory config but not encoded in the MLIR type.
        """
        input_types = []

        # Get function parameter names
        sig = inspect.signature(self.func)
        param_names = list(sig.parameters.keys())

        # Wrap in Location context for MLIR operations
        with Location.unknown(ctx):
            for i, param_name in enumerate(param_names):
                if i < len(self.args) and param_name in self.tensor_args:
                    tensor_arg = self.tensor_args[param_name]
                    shape = list(tensor_arg.shape)
                    dtype = mlir_dtype_from_ttnn_dtype(tensor_arg.dtype, ctx)
                    encoding = create_tensor_layout(ctx, tensor_arg)
                    tensor_type = RankedTensorType.get(shape, dtype, encoding)
                    input_types.append(tensor_type)

        return input_types

    def _modify_function(self, jit_ctx):
        """
        Create a modified version of function that replaces ttnn with ttnn_jit.
        """
        # Get the source code of the function (without decorators, already dedented)
        source = cleanup_source_code(self.func)

        # Replace ttnn with ttnn_jit in the source code
        modified_source = source.replace("ttnn.", "ttnn_jit.")

        # Parse the modified source code into an AST
        tree = ast.parse(modified_source)

        # Fix missing locations in the AST
        ast.fix_missing_locations(tree)

        # Compile the modified AST back to code
        code = compile(tree, filename="<ast>", mode="exec")

        # Execute the code in a namespace that includes the original function's globals
        namespace = self.func.__globals__.copy()

        # Create ttnn_jit module object with jit functions as attributes
        ttnn_jit_namespace = TTNNJitNamespaceUpdater(jit_ctx)
        namespace["ttnn_jit"] = ttnn_jit_namespace

        # Add closure variables to the namespace if the function has any
        # Replace any ttnn functions in closure variables with their ttnn_jit equivalents
        if self.func.__closure__:
            closure_vars = {}
            if self.func.__code__.co_freevars:
                for var_name, cell in zip(
                    self.func.__code__.co_freevars, self.func.__closure__
                ):
                    cell_value = cell.cell_contents

                    # Check if this closure variable is a ttnn function
                    if callable(cell_value) and hasattr(cell_value, "__module__"):
                        module = getattr(cell_value, "__module__", "")
                        func_name = getattr(cell_value, "__name__", "")

                        # Check if it's a ttnn function (module contains 'ttnn' and name starts with 'ttnn.')
                        if "ttnn" in module and func_name.startswith("ttnn."):
                            # Extract the simple function name (e.g., 'max' from 'ttnn.max')
                            simple_name = func_name.split(".")[-1]
                            assert hasattr(
                                ttnn_jit_namespace, simple_name
                            ), f"ttnn function '{simple_name}' not found in ttnn_jit namespace"

                            closure_vars[var_name] = getattr(
                                ttnn_jit_namespace, simple_name
                            )
                            continue

                    # Not a ttnn function, keep original value
                    closure_vars[var_name] = cell_value

            namespace.update(closure_vars)

        exec(code, namespace)

        # Get the function from the namespace (it will have the same name as the original)
        modified_func = namespace[self.func.__name__]

        return modified_func

    def _extract_return_value(self, result, jit_ctx, func_bb):
        """Extract MLIR value from return result."""
        # Check if result is a ResultWrapper
        if isinstance(result, ResultWrapper):
            return result.mlir_value

        # Check if result is in value_map
        result_id = id(result)
        if result_id in jit_ctx.value_map:
            return jit_ctx.value_map[result_id]

        # Fallback: use first argument (shouldn't happen in normal cases)
        if len(func_bb.arguments) > 0:
            return func_bb.arguments[0]

        raise ValueError("Could not extract return value from function result")

    def _update_function_signature(
        self, module, func_op, input_types, output_types, func_bb, return_value, ctx
    ):
        """
        Update function signature with correct return type.

        MLIR doesn't allow direct modification of function signatures,
        so we need to recreate the function with the correct signature.
        """
        # Get the old block before creating new function
        old_block = func_op.regions[0].blocks[0]

        # Add return statement if not already present
        # Check if the last operation is already a return
        if not old_block.operations or not isinstance(
            old_block.operations[-1], func.ReturnOp
        ):
            with InsertionPoint(old_block), Location.unknown(ctx):
                func.ReturnOp([return_value])

        # Create new function with correct return type
        # The new function's region is empty (no blocks), so directly append the old block to it
        with InsertionPoint(module.body), Location.unknown(ctx):
            new_func_op = func.FuncOp(
                name=func_op.name.value, type=(input_types, output_types)
            )

        # Move the old block to the new function's region
        new_region = new_func_op.regions[0]
        old_block.append_to(new_region)

        # Erase the old function
        func_op.erase()

    class _MockTensor:
        """Mock tensor_arg used to call _create_*_tensor_layout functions."""

        def __init__(self, shape, dtype, memory_config):
            self.shape = shape
            self.dtype = dtype
            self._memory_config = memory_config

        def memory_config(self):
            return self._memory_config

    def _create_output_layout_from_memory_config(
        self, ctx, memory_config, tensor_shape, element_type
    ):
        """Passes memory_config wrapped in a MockTensor to create output layout."""
        dtype = ttnn_dtype_from_mlir_dtype(element_type)
        mock_tensor_arg = TracingCompiler._MockTensor(
            tensor_shape, dtype, memory_config
        )
        if memory_config.buffer_type == ttnn.BufferType.DRAM:
            return _create_dram_tensor_layout(ctx, mock_tensor_arg)
        else:
            return _create_sharded_tensor_layout(ctx, mock_tensor_arg)

    def _convert_return_operands(self, return_op, module_ctx, memory_config):
        """Convert all operands of a return op to their target layouts.

        Creates ttir.empty + ttir.to_layout ops for tensor types, returns list of
        new operands (converted or original on error/non-tensor).
        """
        return_values = list(return_op.operands)
        new_return_values = []

        with InsertionPoint(return_op), Location.unknown():
            for idx, ret_val in enumerate(return_values):
                ret_type = ret_val.type

                # only convert tensor types
                if not isinstance(ret_type, RankedTensorType):
                    new_return_values.append(ret_val)
                    continue

                # get shape and element type
                try:
                    tensor_shape = [int(dim) for dim in ret_type.shape]
                except (TypeError, ValueError) as e:
                    print(f"  Failed to extract shape from return type: {e}")
                    new_return_values.append(ret_val)
                    continue

                element_type = ret_type.element_type

                try:
                    # create output layout from memory_config
                    output_layout = self._create_output_layout_from_memory_config(
                        module_ctx, memory_config, tensor_shape, element_type
                    )
                    # create output type with new layout
                    output_type = RankedTensorType.get(
                        tensor_shape, element_type, output_layout
                    )
                    # insert ttir.empty and ttir.to_layout
                    empty_op = ttir.EmptyOp(output_type)
                    to_layout_op = ttir.ToLayoutOp(
                        [output_type], ret_val, empty_op.result
                    )

                    new_return_values.append(to_layout_op.result)

                except Exception as e:
                    print(f"Failed to insert layout conversion: {e}")
                    new_return_values.append(ret_val)

        return new_return_values

    def _replace_return_op(self, return_op, new_operands):
        """Replace return op with a new one using the provided operands."""
        with InsertionPoint(return_op), Location.unknown():
            func.ReturnOp(new_operands)
        return_op.operation.erase()

    def _update_function_type(self, func_op, module_ctx):
        """Update function signature to match actual return operand types."""
        input_types = [arg.type for arg in func_op.arguments]

        # get output types from return ops
        output_types = []
        for block in func_op.regions[0].blocks:
            for op in block.operations:
                if isinstance(op, func.ReturnOp):
                    output_types = [v.type for v in op.operands]
                    break
            if output_types:
                break

        # update function type attribute
        new_func_type = FunctionType.get(input_types, output_types, module_ctx)
        func_op.attributes["function_type"] = TypeAttr.get(new_func_type)

    def _insert_output_layout_conversion(self, module, memory_config):
        """Insert output layout conversion (ttir.to_layout) for return values."""
        if memory_config is None:
            print("No memory_config provided, skipping output layout conversion")
            return module

        print(f"Inserting output layout conversion for memory_config: {memory_config}")

        with module.context:
            # iterate over all func ops and their return ops
            for op in module.body.operations:
                if not isinstance(op, func.FuncOp):
                    continue

                # collect all return ops from all blocks
                return_ops = []
                for block in op.regions[0].blocks:
                    for block_op in block.operations:
                        if isinstance(block_op, func.ReturnOp):
                            return_ops.append(block_op)

                # process each return op
                for return_op in return_ops:
                    new_operands = self._convert_return_operands(
                        return_op, module.context, memory_config
                    )
                    self._replace_return_op(return_op, new_operands)

                # update function signature once per func
                self._update_function_type(op, module.context)

        return module
