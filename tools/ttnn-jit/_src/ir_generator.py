# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttnn_jit._src.tracing_compiler import TracingCompiler
from ttnn_jit._src.conversions import ttnn_dtype_from_mlir_dtype
from ttnn_jit._src.tensor_translator import (
    _create_dram_tensor_layout,
    _create_sharded_tensor_layout,
    _calculate_tile_shape,
)
from ttmlir.ir import InsertionPoint, RankedTensorType, Location, FunctionType, TypeAttr
from ttmlir.dialects import ttir, func
import ttnn


class MockTensor:
    """Mock tensor_arg used to call _create_*_tensor_layout functions."""

    def __init__(self, shape, dtype, memory_config):
        self.shape = shape
        self.dtype = dtype
        self._memory_config = memory_config

    def memory_config(self):
        return self._memory_config


def create_output_layout_from_memory_config(
    ctx, memory_config, tensor_shape, element_type
):
    """Passes memory_config wrapped in a MockTensor to create output layout."""
    dtype = ttnn_dtype_from_mlir_dtype(element_type)
    mock_tensor_arg = MockTensor(tensor_shape, dtype, memory_config)
    if memory_config.buffer_type == ttnn.BufferType.DRAM:
        return _create_dram_tensor_layout(ctx, mock_tensor_arg)
    else:
        return _create_sharded_tensor_layout(ctx, mock_tensor_arg)


def _convert_return_operands(return_op, module_ctx, memory_config, debug):
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
                if debug:
                    print(f"  Failed to extract shape from return type: {e}")
                new_return_values.append(ret_val)
                continue

            element_type = ret_type.element_type

            if debug:
                print(f"  Converting return value {idx}: {ret_type}")

            try:
                # create output layout from memory_config
                output_layout = create_output_layout_from_memory_config(
                    module_ctx, memory_config, tensor_shape, element_type
                )
                # create output type with new layout
                output_type = RankedTensorType.get(
                    tensor_shape, element_type, output_layout
                )
                # insert ttir.empty and ttir.to_layout
                empty_op = ttir.EmptyOp(output_type)
                to_layout_op = ttir.ToLayoutOp([output_type], ret_val, empty_op.result)

                new_return_values.append(to_layout_op.result)

            except Exception as e:
                if debug:
                    print(f"Failed to insert layout conversion: {e}")
                new_return_values.append(ret_val)

    return new_return_values


def _replace_return_op(return_op, new_operands):
    """Replace return op with a new one using the provided operands."""
    with InsertionPoint(return_op), Location.unknown():
        func.ReturnOp(new_operands)
    return_op.operation.erase()


def _update_function_type(func_op, module_ctx):
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


def insert_output_layout_conversion(module, debug, memory_config):
    """Insert output layout conversion (ttir.to_layout) for return values."""
    if memory_config is None:
        if debug:
            print("No memory_config provided, skipping output layout conversion")
        return module

    if debug:
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
                new_operands = _convert_return_operands(
                    return_op, module.context, memory_config, debug
                )
                _replace_return_op(return_op, new_operands)

            # update function signature once per func
            _update_function_type(op, module.context)

    return module


def print_and_verify_ir(ir, method_name, debug):
    if debug:
        print("---- IR Dump after " + method_name + " ----")
        print(ir)
    ir.operation.verify()


def generate_ir(f, debug, memory_config, *args, **kwargs):
    """Generate IR from tracing compilation."""
    compiler = TracingCompiler(f, *args, **kwargs)
    ir = compiler.compile()
    # insert output layout conversion if memory_config is provided
    ir = insert_output_layout_conversion(ir, debug, memory_config)
    print_and_verify_ir(ir, "TracingCompiler (Tracing-based)", debug)
    return ir
