# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ast
import inspect

from ttmlir.dialects import func
from ttmlir.ir import Context, Location, Module, InsertionPoint, RankedTensorType

from ttnn_jit._src.utils import cleanup_source_code
from ttnn_jit._src.tensor_translator import create_tensor_layout
from ttnn_jit._src.conversions import mlir_dtype_from_ttnn_dtype
from ttnn_jit._src.jit_functions import TTNNJitNamespaceUpdater, ResultWrapper


class JitContext:
    """Context for tracking MLIR values during tracing."""

    def __init__(self, func_bb, ctx):
        self.func_bb = func_bb
        self.ctx = ctx
        self.value_map = {}  # Maps id(python_obj) -> MLIR value
        self.func_arg_ids = set()  # Track IDs of original function arguments


class TracingCompiler:
    """Compiler for generating TTIR from tracing execution."""

    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.tensor_args = kwargs.get("_tensor_args", {})

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

        return module

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

        # Add closure variables to the namespace if the function has any
        if self.func.__closure__:
            closure_vars = {}
            if self.func.__code__.co_freevars:
                for var_name, cell in zip(
                    self.func.__code__.co_freevars, self.func.__closure__
                ):
                    closure_vars[var_name] = cell.cell_contents
            namespace.update(closure_vars)

        # Create ttnn_jit module object with jit functions as attributes
        namespace["ttnn_jit"] = TTNNJitNamespaceUpdater(jit_ctx)

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
