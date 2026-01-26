# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttnn_jit._src.tracing_compiler import TracingCompiler
from ttnn_jit._src.tensor_translator import (
    _create_dram_tensor_layout,
    _create_sharded_tensor_layout,
    _calculate_tile_shape,
)
from ttmlir.ir import InsertionPoint, RankedTensorType, Location
from ttmlir.dialects import ttir, func
import ttnn


def print_and_verify_ir(ir, method_name, debug):
    if debug:
        print("---- IR Dump after " + method_name + " ----")
        print(ir)
    ir.operation.verify()


def create_output_layout_from_memory_config(
    ctx, memory_config, tensor_shape, element_type, debug
):

    # Validate tensor_shape is iterable
    if not isinstance(tensor_shape, (list, tuple)):
        raise TypeError(
            f"tensor_shape must be a list or tuple, got {type(tensor_shape)}: {tensor_shape}"
        )

    # Ensure all elements in tensor_shape are Python integers
    try:
        tensor_shape = [int(dim) for dim in tensor_shape]
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"tensor_shape must contain only integers, got {tensor_shape}: {e}"
        )

    # Infer dtype from element type if it's a tile type
    if hasattr(element_type, "element_type"):
        # It's a TileType, extract the scalar element type
        scalar_type = element_type.element_type
    else:
        scalar_type = element_type
    if debug:
        print(f"scalar type: {scalar_type}")

    # Convert scalar type back to ttnn dtype for layout creation
    dtype_str = str(scalar_type)
    if "bf16" in dtype_str or "bfloat16" in dtype_str:
        dtype = ttnn.bfloat16
    elif "f32" in dtype_str or "float32" in dtype_str:
        dtype = ttnn.float32
    elif "f16" in dtype_str or "float16" in dtype_str:
        dtype = ttnn.float16
    else:
        dtype = ttnn.bfloat16  # Default fallback
    if debug:
        print(f"Converted dtype: {dtype}")

    if memory_config.buffer_type == ttnn.BufferType.DRAM:
        # Create a mock tensor argument for DRAM layout
        class MockTensor:
            def __init__(self, shape, dtype, memory_config):
                self.shape = shape
                self.dtype = dtype
                self._memory_config = memory_config

            def memory_config(self):
                return self._memory_config

        mock_tensor = MockTensor(tensor_shape, dtype, memory_config)
        return _create_dram_tensor_layout(ctx, mock_tensor)
    else:
        # L1 with sharding
        if memory_config.shard_spec is None:
            raise ValueError("L1 memory config must have shard_spec for output layout")

        class MockTensor:
            def __init__(self, shape, dtype, memory_config):
                self.shape = shape
                self.dtype = dtype
                self._memory_config = memory_config

            def memory_config(self):
                return self._memory_config

        mock_tensor = MockTensor(tensor_shape, dtype, memory_config)
        return _create_sharded_tensor_layout(ctx, mock_tensor)


def insert_output_layout_conversion(module, memory_config, debug):

    if memory_config is None:
        if debug:
            print("No memory_config provided, skipping output layout conversion")
        return module

    if debug:
        print(f"Inserting output layout conversion for memory_config: {memory_config}")

    with module.context:
        # Find all function operations
        for op in module.body.operations:
            if not isinstance(op, func.FuncOp):
                continue

            if debug:
                print(f"Processing function: {op.name.value}")

            # Find the function body blocks
            for block in op.regions[0].blocks:
                return_ops = []
                for block_op in block.operations:
                    if isinstance(block_op, func.ReturnOp):
                        return_ops.append(block_op)

                if debug:
                    print(f"Found {len(return_ops)} return operation(s)")

                # modify each return
                for return_op in return_ops:
                    return_values = list(return_op.operands)
                    new_return_values = []

                    # Set insertion point BEFORE the return op and establish location context
                    with InsertionPoint(return_op), Location.unknown():
                        for idx, ret_val in enumerate(return_values):
                            ret_type = ret_val.type

                            # Only convert tensor types
                            if not isinstance(ret_type, RankedTensorType):
                                if debug:
                                    print(
                                        f"  Return value {idx} is not a RankedTensorType, skipping"
                                    )
                                new_return_values.append(ret_val)
                                continue

                            # Get tensor shape and element type
                            # Ensure shape is a proper Python list of integers
                            try:
                                tensor_shape = [int(dim) for dim in ret_type.shape]
                            except (TypeError, ValueError) as e:
                                if debug:
                                    print(
                                        f"  ✗ Failed to extract shape from return type: {e}"
                                    )
                                new_return_values.append(ret_val)
                                continue

                            element_type = ret_type.element_type

                            if debug:
                                print(
                                    f"  Converting return value {idx}: shape={tensor_shape}, element_type={element_type}"
                                )
                                print(f"    element_type type: {type(element_type)}")
                                print(f"    ret_type: {ret_type}")

                            try:
                                # Create output layout from memory_config
                                output_layout = create_output_layout_from_memory_config(
                                    module.context,
                                    memory_config,
                                    tensor_shape,
                                    element_type,
                                    debug,
                                )

                                # Create new tensor type with output layout
                                output_type = RankedTensorType.get(
                                    tensor_shape, element_type, output_layout
                                )

                                # Create ttir.empty
                                empty_op = ttir.EmptyOp(output_type)

                                # Create ttir.to_layout
                                to_layout_op = ttir.ToLayoutOp(
                                    [output_type],
                                    ret_val,
                                    empty_op.result,
                                )

                                new_return_values.append(to_layout_op.result)

                                if debug:
                                    print(
                                        f"Inserted ttir.empty + ttir.to_layout for return value {idx}"
                                    )
                                    print(f"Output type: {output_type}")
                            except TypeError as e:
                                if debug:
                                    print(
                                        f"Failed to insert layout conversion (TypeError): {e}"
                                    )
                                new_return_values.append(ret_val)
                            except Exception as e:
                                if debug:
                                    print(f"Failed to insert layout conversion: {e}")
                                new_return_values.append(ret_val)

                    # Replace the return operation with a new one with updated operands
                    with InsertionPoint(return_op), Location.unknown():
                        func.ReturnOp(new_return_values)

                    return_op.operation.erase()

                    # Update function signature to match new return types
                    input_types = [arg.type for arg in op.arguments]
                    output_types = [v.type for v in new_return_values]

                    from ttmlir.ir import FunctionType, TypeAttr

                    new_func_type = FunctionType.get(
                        input_types, output_types, module.context
                    )
                    op.attributes["function_type"] = TypeAttr.get(new_func_type)

                    if debug:
                        print(
                            f"Updated function signature and replaced return operation"
                        )

    if debug:
        print("Output layout conversion complete")

    return module


def generate_ir(f, memory_config, debug, *args, **kwargs):
    """Generate IR from tracing compilation."""
    compiler = TracingCompiler(f, *args, **kwargs)
    ir = compiler.compile()
    # Insert output layout conversion if memory_config is provided
    ir = insert_output_layout_conversion(ir, debug, memory_config)
    print_and_verify_ir(ir, "TracingCompiler (Tracing-based)", debug)
    return ir
