# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.ir import *
from ttmlir.dialects import ttnn, func, ttcore


class GraphTraceCompiler:
    """
    Compiler that generates MLIR IR from TTNN graph traces.
    This is an alternative to TTIRCompiler that uses runtime graph capture
    instead of AST parsing.
    """

    def __init__(self, captured_graph, function_name, *args, **kwargs):
        self.ctx = Context()
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.captured_graph = captured_graph
        self.function_name = function_name
        self.tensor_args = kwargs.get("_tensor_args", {})
        self.backend = kwargs.get("_backend")
        self.max_grid = kwargs.get("_max_grid")
        self.args = args

    def _mlir_dtype_from_ttnn_dtype(self, dtype):
        match int(dtype):
            case 0:
                return BF16Type.get(self.ctx)
            case 1:
                return F32Type.get(self.ctx)
            case 2:
                return U32Type.get(self.ctx)
            case 5:
                return U8Type.get(self.ctx)
            case 6:
                return U16Type.get(self.ctx)
            case 7:
                return I32Type.get(self.ctx)
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

    def _ttcore_dtype_from_ttnn_dtype(self, dtype):
        match int(dtype):
            case 0:
                return ttcore.DataType.BFloat16
            case 1:
                return ttcore.DataType.Float32
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

    def _ttcore_dtype_from_mlir_dtype(self, dtype):
        match str(dtype):
            case "f32":
                return ttcore.DataType.Float32
            case "bf16":
                return ttcore.DataType.BFloat16
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

    def _create_tensor_layout(self, tensor_arg):
        """Create TTNN layout attribute from tensor."""
        # Only rank 2 tensors supported
        assert len(tensor_arg.shape) == 2

        with Location.unknown(self.ctx):
            shard_spec = tensor_arg.memory_config().shard_spec
            shard_shape = shard_spec.shape

            # Create identity affine map
            identity_map = AffineMap.get_identity(2, self.ctx)

            # Create ttcore grid attr based off max_grid
            grid_size_x = self.max_grid[0] + 1
            grid_size_y = self.max_grid[1] + 1
            grid = ttcore.ir.GridAttr.get(self.ctx, [grid_size_x, grid_size_y])

            # Create memref with tile type
            data_type = self._ttcore_dtype_from_ttnn_dtype(tensor_arg.dtype)
            shard_shape_tile_x = shard_shape[0] // 32
            shard_shape_tile_y = shard_shape[1] // 32
            tile_type = ttcore.ir.TileType.get(self.ctx, 32, 32, data_type)
            buffer_type = ttnn.ir.BufferTypeAttr.get(self.ctx, ttnn.BufferType.L1)
            memref = MemRefType.get(
                [shard_shape_tile_x, shard_shape_tile_y], tile_type, None, buffer_type
            )

            # Create TTNN layout
            ttnn_layout = ttnn.ir.TTNNLayoutAttr.get(
                self.ctx,
                identity_map,
                grid,
                memref,
                ttnn.TensorMemoryLayout.BlockSharded,
                None,
            )
            return ttnn_layout

    def _extract_operation_name(self, node):
        """Extract the operation name from a function_start node."""
        if node['node_type'] == 'function_start':
            name = node['params'].get('name', '')
            # Extract the actual operation name (e.g., 'abs' from 'ttnn::abs')
            if '::' in name:
                parts = name.split('::')
                return parts[-1]  # Get the last part (e.g., 'abs')
            return name
        return None

    def _find_main_operation(self):
        """Find the main TTNN operation from the graph trace."""
        # Look for function_start nodes with ttnn:: prefix (not prim or internal ops)
        for node in self.captured_graph:
            if node['node_type'] == 'function_start':
                name = node['params'].get('name', '')
                # We want top-level ttnn operations like ttnn::abs, not ttnn::prim::*
                if name.startswith('ttnn::') and '::prim::' not in name:
                    op_name = self._extract_operation_name(node)
                    return op_name, node
        return None, None

    def compile(self):
        """Compile the graph trace into an MLIR module."""
        # Extract the main operation from graph trace
        op_name, op_node = self._find_main_operation()
        
        if not op_name:
            raise ValueError("Could not find main operation in graph trace")

        with Location.unknown(self.ctx):
            # Build input types from tensor args
            input_types = []
            for arg in self.args:
                shape = list(arg.shape)
                layout = self._create_tensor_layout(arg)
                dtype = self._mlir_dtype_from_ttnn_dtype(arg.dtype)
                tensor_type = RankedTensorType.get(shape, dtype, layout)
                input_types.append(tensor_type)

            # Output types same as input for unary ops
            output_types = [input_types[0]]

            # Create function
            with InsertionPoint(self.insert_point):
                func_op = func.FuncOp(
                    name=self.function_name, 
                    type=(input_types, output_types)
                )
                func_bb = func_op.add_entry_block()

                # Build the function body
                with InsertionPoint(func_bb):
                    # Create get_device op
                    mesh_shape_attr = ttnn.ir.MeshShapeAttr.get(self.ctx, 1, 1)
                    mesh_offset_attr = ttnn.ir.MeshOffsetAttr.get(self.ctx, 0, 0)
                    device = ttnn.get_device(
                        mesh_shape=mesh_shape_attr, 
                        mesh_offset=mesh_offset_attr
                    )

                    # Get input argument
                    input_tensor = func_bb.arguments[0]
                    result_type = input_tensor.type

                    # Create the operation based on the traced operation
                    op_result = self._create_ttnn_op(
                        op_name, result_type, input_tensor, device
                    )

                    # Return the result
                    func.ReturnOp([op_result])

    def _create_ttnn_op(self, op_name, result_type, input_tensor, device):
        """Create a TTNN operation based on the operation name."""
        # Map operation names to TTNN dialect operations
        op_map = {
            'abs': ttnn.abs,
            'exp': ttnn.exp,
            'neg': ttnn.neg,
            'sqrt': ttnn.sqrt,
            'rsqrt': ttnn.rsqrt,
            'log': ttnn.log,
            'cos': ttnn.cos,
            'sin': ttnn.sin,
            'ceil': ttnn.ceil,
            'floor': ttnn.floor,
            # Add more operations as needed
        }

        if op_name not in op_map:
            raise NotImplementedError(f"Operation {op_name} not yet supported in GraphTraceCompiler")

        # Create the operation
        op_func = op_map[op_name]
        op = op_func(result_type, input_tensor)
        
        # Add required attributes
        op.owner.attributes["ttnn.hoist_generic_via_d2m"] = UnitAttr.get(self.ctx)

        return op

