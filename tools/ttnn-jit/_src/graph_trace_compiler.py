# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Graph Trace Compiler - IR generation from captured execution traces.

This module provides a clean, scalable implementation for generating MLIR IR
from levelized graphs extracted from TTNN operation traces.

Key features:
- Incremental depth exploration to handle composite operations
- Clear tensor-to-argument mapping based on tensor IDs
- Support for both simple and composite operations
- Extensible design using operation registry pattern
- All operation-specific logic decoupled into separate modules
"""

from ttmlir.ir import *
from ttmlir.dialects import ttnn, func, ttcore
from typing import Dict, List, Any, Optional, Tuple
import ttnn_jit._src.supported_ops as supported_ops
from ttnn_jit._src.tensor_translator import (
    _get_collapsed_linear_affine_map,
    create_tensor_layout,
    _calculate_tile_shape,
    TILE_WIDTH,
    TILE_HEIGHT,
)
from ttnn_jit._src.levelized_graph import LevelizedGraph, LevelizedGraphVertex
from ttnn_jit._src.op_registry import get_registry
from ttnn_jit._src.conversions import (
    mlir_dtype_from_ttnn_dtype,
    ttcore_dtype_from_mlir_dtype,
    buffer_type_from_string,
    memory_layout_from_string,
)
from ttnn._ttnn.graph import extract_levelized_graph


class GraphToIRTranslator:
    """
    Generates MLIR IR from captured execution traces.

    This is the main compiler class that takes a captured graph trace and produces
    MLIR IR. It handles:
    - Finding optimal depth (k value)
    - Creating function signature from tensor arguments
    - Processing vertices in topological order
    - Generating MLIR operations
    - Handling both supported and composite operations
    """

    def __init__(
        self,
        captured_graph: List[Dict[str, Any]],
        function_name: str,
        tensor_args: Dict[str, Any],
    ):
        """
        Initialize the translator.

        Args:
            captured_graph: Raw captured graph from graph capture
            function_name: Name for the generated function
            tensor_args: Dictionary mapping argument names to tensor objects
        """
        self.ctx = Context()
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.captured_graph = captured_graph
        self.function_name = function_name
        self.tensor_args = tensor_args
        self.levelized_graph_ir: Optional[LevelizedGraph] = None

    def _find_optimal_depth(self, max_depth: int = 10) -> Tuple[int, LevelizedGraph]:
        """
        Find the minimum depth k where all level 1 operations are visitable.
        Level 1 operations are essentially the top level operations that can be executed on the device.
        Eg. ttnn.add is a level 1 operation, but its internal node (ttnn.prim_binary_ng) is a level 2 operation.
        Note that we don't need to include level 2 ops in the IR (unless their parent is a composite op, such as ttnn.digamma).

        Args:
            max_depth: Maximum depth to try before giving up

        Returns:
            Tuple of (optimal_k, levelized_graph_ir)

        Raises:
            ValueError: If no valid depth found within max_depth
        """
        for k in range(1, max_depth + 1):
            levelized_graph_data = extract_levelized_graph(self.captured_graph, k)
            graph_ir = LevelizedGraph(levelized_graph_data)

            if graph_ir.can_visit_all_level_1_ops():
                return k, graph_ir

        raise ValueError(
            f"Cannot visit all level 1 operations even at max_depth={max_depth}"
        )

    def _parse_memory_config_from_output_info(
        self, output_info_str: str
    ) -> Dict[str, Any]:
        """
        Parse memory configuration from output_info string.

        Extracts buffer_type, memory_layout, and shard_spec from the captured
        tensor specification string.

        Args:
            output_info_str: String like "Tensor(storage=DeviceStorage(),tensor_spec=...)"

        Returns:
            Dict with keys: buffer_type, memory_layout, shard_shape, grid
        """
        import re

        config = {
            "buffer_type": "DRAM",  # Default
            "memory_layout": "INTERLEAVED",  # Default
            "shard_shape": None,
            "grid": [1, 1],  # Default
        }

        # Extract buffer_type (BufferType::L1 or BufferType::DRAM)
        buffer_match = re.search(r"buffer_type=BufferType::(\w+)", output_info_str)
        if buffer_match:
            config["buffer_type"] = buffer_match.group(1)

        # Extract memory_layout (TensorMemoryLayout::BLOCK_SHARDED, INTERLEAVED, etc.)
        layout_match = re.search(
            r"memory_layout=TensorMemoryLayout::(\w+)", output_info_str
        )
        if layout_match:
            config["memory_layout"] = layout_match.group(1)

        # Extract shard_shape if sharded
        # Format: shard_spec=ShardSpec(...,shape={64, 128},...)
        # Note: Use .*? for non-greedy match since ShardSpec contains nested parens
        shard_shape_match = re.search(
            r"shard_spec=ShardSpec\(.*?shape=\{([^}]+)\}", output_info_str
        )
        if shard_shape_match:
            shape_str = shard_shape_match.group(1)
            config["shard_shape"] = [int(x.strip()) for x in shape_str.split(",")]

        # Extract grid if sharded
        # Format: grid={[(x=0,y=0) - (x=0,y=0)]}
        # This represents a grid from (x_min, y_min) to (x_max, y_max)
        # Note: Multiple CoreRanges are not supported in JIT/D2M
        # Only parse grid from ShardSpec, not from NdShardSpec (which may also contain grid)
        grid_pattern = r"shard_spec=ShardSpec\(.*?grid=\{\[\(x=(\d+),y=(\d+)\)\s*-\s*\(x=(\d+),y=(\d+)\)\]\}"
        grid_match = re.search(grid_pattern, output_info_str)
        grid_matches = [grid_match.groups()] if grid_match else []

        if len(grid_matches) > 1:
            print(f"Multiple CoreRanges detected in output_info:")
            print(f"  output_info: {output_info_str}")
            print(f"  Found {len(grid_matches)} CoreRange(s):")
            for i, match in enumerate(grid_matches):
                x_min, y_min, x_max, y_max = map(int, match)
                print(
                    f"    CoreRange {i+1}: (x={x_min},y={y_min}) - (x={x_max},y={y_max})"
                )
            raise BaseException(
                f"Multiple CoreRanges in grid attribute are not supported in JIT/D2M. "
                f"Found {len(grid_matches)} CoreRange(s) in output_info. "
                f"Only single CoreRange grids are currently supported."
            )

        if len(grid_matches) == 1:
            x_min, y_min, x_max, y_max = map(int, grid_matches[0])
            # Grid size is (max - min + 1) for each dimension
            # But TTNN uses (width, height) while compiler uses (height, width)
            grid_width = x_max - x_min + 1
            grid_height = y_max - y_min + 1
            config["grid"] = [grid_height, grid_width]

        return config

    def _parse_shape_from_vertex(
        self, vertex: "LevelizedGraphVertex"
    ) -> Optional[List[int]]:
        """
        Parse the output shape from a vertex's output_shape field.

        Args:
            vertex: The vertex containing output_shape information

        Returns:
            List of dimensions, or None if parsing fails

        Raises:
            NotImplementedError: If shape is scalar (empty list)
        """
        if not vertex.output_shape or len(vertex.output_shape) == 0:
            return None

        # Extract shape from string like "Shape([64, 1])" or "Shape([])"
        shape_str = vertex.output_shape[0]
        import ast

        # Find the list part between brackets
        start = shape_str.find("[")
        end = shape_str.rfind("]")  # Use rfind to get the last bracket
        if start == -1 or end == -1:
            return None

        shape_list_str = shape_str[start : end + 1]
        shape = ast.literal_eval(shape_list_str)

        # Scalar outputs (empty shape) are not currently supported
        if len(shape) == 0:
            raise NotImplementedError(
                f"Scalar output (empty shape) for vertex {vertex.name} is not currently supported. "
                f"Operations that reduce to scalars should be avoided or handled differently."
            )

        return shape

    def _calculate_memref_and_affine_map(
        self, shape: List[int], memory_config: Dict[str, Any]
    ) -> Tuple[List[int], Any]:
        """
        Calculate memref shape and affine map based on tensor shape and memory configuration.

        Args:
            shape: Tensor shape (list of dimensions)
            memory_config: Memory configuration dict with keys:
                - shard_shape: Optional[Tuple[int, int]]
                - buffer_type, memory_layout, grid

        Returns:
            Tuple of (memref_shape, affine_map)
        """
        import math

        if memory_config["shard_shape"]:
            # Use shard shape for sharded tensors
            shard_h, shard_w = memory_config["shard_shape"]
            tiles_h = math.ceil(shard_h / TILE_HEIGHT)
            tiles_w = math.ceil(shard_w / TILE_WIDTH)
            memref_shape = [tiles_h, tiles_w]
            affine_map = _get_collapsed_linear_affine_map(
                self.ctx, shape, memory_config["grid"]
            )
        else:
            memref_shape = _calculate_tile_shape(shape)
            affine_shape = shape if len(shape) > 0 else [1, 1]
            affine_map = _get_collapsed_linear_affine_map(
                self.ctx, affine_shape, (0, 0)
            )

        return memref_shape, affine_map

    def _get_output_type_from_vertex(
        self,
        vertex: "LevelizedGraphVertex",
        default_type,
        parent_output_info: Optional[List[str]] = None,
    ) -> Any:
        """
        Extract MLIR output type from vertex output_shape and output_info.

        For operations with shape changes (like reductions), we need to use the
        output_shape from the vertex instead of assuming the same shape as input.

        This function now properly parses the output_info to extract the actual
        memory configuration (buffer type, memory layout, shard spec) captured
        during graph execution.

        Args:
            vertex: The vertex containing output information
            default_type: Fallback type to inherit dtype and layout properties from
            parent_output_info: Optional parent composite op's output_info to inherit
                              when vertex.output_info is empty (for internal ops)

        Returns:
            MLIR tensor type with correct shape and memory configuration
        """
        try:
            # Parse shape from vertex
            shape = self._parse_shape_from_vertex(vertex)
            if shape is None:
                return default_type

            # Get dtype from default_type
            dtype = default_type.element_type

            # Parse memory configuration from output_info
            memory_config = {
                "buffer_type": "DRAM",
                "memory_layout": "INTERLEAVED",
                "shard_shape": None,
                "grid": [1, 1],
            }
            # Use vertex's output_info if available, otherwise fall back to parent's output_info
            output_info_to_use = None
            if vertex.output_info and len(vertex.output_info) > 0:
                output_info_to_use = vertex.output_info[0]
            elif parent_output_info and len(parent_output_info) > 0:
                output_info_to_use = parent_output_info[0]

            if output_info_to_use:
                memory_config = self._parse_memory_config_from_output_info(
                    output_info_to_use
                )

            # Create tile type
            data_type_ttcore = ttcore_dtype_from_mlir_dtype(dtype)
            tile_type = ttcore.ir.TileType.get(
                self.ctx, TILE_WIDTH, TILE_HEIGHT, data_type_ttcore
            )

            # Map buffer type and memory layout to enums
            buffer_type_enum = buffer_type_from_string(memory_config["buffer_type"])
            memory_layout_enum = memory_layout_from_string(
                memory_config["memory_layout"]
            )

            # Create buffer type and grid attributes
            buffer_type = ttnn.ir.BufferTypeAttr.get(self.ctx, buffer_type_enum)
            grid = ttcore.ir.GridAttr.get(self.ctx, memory_config["grid"])

            # Calculate memref shape and affine map
            memref_shape, affine_map = self._calculate_memref_and_affine_map(
                shape, memory_config
            )

            # Create memref and layout
            memref = MemRefType.get(memref_shape, tile_type, None, buffer_type)

            exact_grid = True
            tensor_mesh = None

            layout = ttnn.ir.TTNNLayoutAttr.get_with_linear(
                self.ctx,
                affine_map,
                grid,
                memref,
                memory_layout_enum,
                tensor_mesh,
                exact_grid,
            )

            return RankedTensorType.get(shape, dtype, layout)

        except Exception as e:
            # If parsing fails, return default type
            import traceback

            print(f"Warning: Failed to parse output type from vertex: {e}")
            traceback.print_exc()
            return default_type

    def _build_function_signature(
        self, tensor_vertices: List[Tuple[LevelizedGraphVertex, int]]
    ) -> Tuple[List, List]:
        """
        Build function input and output types from tensor arguments.

        Args:
            tensor_vertices: List of (vertex, tensor_id) tuples sorted by tensor_id

        Returns:
            Tuple of (input_types, output_types)
        """
        input_types = []

        # Get tensor arguments in the order determined by tensor_id sorting
        tensor_args_list = list(self.tensor_args.values())

        for i, (vertex, tensor_id) in enumerate(tensor_vertices):
            # Use the i-th tensor arg (assumes tensor_args order matches)
            if i < len(tensor_args_list):
                tensor_arg = tensor_args_list[i]
                shape = list(tensor_arg.shape)
                layout = create_tensor_layout(self.ctx, tensor_arg)
                dtype = mlir_dtype_from_ttnn_dtype(tensor_arg.dtype, self.ctx)
                tensor_type = RankedTensorType.get(shape, dtype, layout)
                input_types.append(tensor_type)

        # Infer output type from final operation's output_shape
        assert len(input_types) >= 1, "At least one input tensor is needed"

        # Find the final output vertex (operation with no consumers)
        final_vertex = self.levelized_graph_ir.find_output_vertex()
        if final_vertex is not None and final_vertex.output_shape:
            # Use the output shape from the final operation
            output_types = [
                self._get_output_type_from_vertex(final_vertex, input_types[0])
            ]
        else:
            # Fallback to using first input type
            output_types = [input_types[0]]

        return input_types, output_types

    def _process_vertex(
        self,
        vertex: LevelizedGraphVertex,
        tensor_arg_map: Dict[int, Any],
        operation_results: Dict[int, Any],
        device: Any,
        result_type: Any,
    ) -> Optional[Any]:
        """
        Process a single vertex and generate MLIR operation.

        Args:
            vertex: The vertex to process
            tensor_arg_map: Map from vertex counter to MLIR argument values
            operation_results: Map from vertex counter to MLIR operation results
            device: MLIR device value
            result_type: Expected result type

        Returns:
            MLIR value representing the result, or None for tensors
        """
        # Tensors map directly to arguments - already handled
        if vertex.is_tensor():
            return None

        # Assert output_shape is always populated for non-placeholder operations
        # Placeholder ops are just markers and may have empty output_info
        if not vertex.is_placeholder():
            assert (
                vertex.output_shape and len(vertex.output_shape) > 0
            ), f"output_shape must be populated for operation {vertex.name} (vertex {vertex.counter})"

        # Get operation name
        op_name = vertex.get_op_name()
        if op_name is None:
            raise ValueError(
                f"Could not extract operation name from vertex: {vertex.name}"
            )

        # Check if this is a composite operation that needs expansion (eg, digamma)
        if supported_ops.is_composite(op_name):
            # This operation should be expanded into its internals
            # Process each internal operation and return the final result
            return self._process_composite_vertex(
                vertex, tensor_arg_map, operation_results, device, result_type
            )

        # Collect operands from in_edges
        # Note: in_edges may contain duplicates (e.g., [6, 6] for multiply(x, x))
        operands = []
        for in_edge_id in vertex.in_edges:
            if in_edge_id in tensor_arg_map:
                # This is a tensor argument
                operands.append(tensor_arg_map[in_edge_id])
            elif in_edge_id in operation_results:
                # This is an intermediate result
                operands.append(operation_results[in_edge_id])
            else:
                raise ValueError(f"Could not find input for edge {in_edge_id}")

        # Generate MLIR operation (handler will parse arguments)
        return self._create_ttnn_op(op_name, result_type, operands, device, vertex)

    def _process_composite_vertex(
        self,
        vertex: LevelizedGraphVertex,
        tensor_arg_map: Dict[int, Any],
        operation_results: Dict[int, Any],
        device: Any,
        result_type: Any,
    ) -> Optional[Any]:
        """
        Process a composite operation by expanding it into its internals.

        Composite operations (eg, digamma) are expanded into their
        constituent operations. Each internal operation is processed recursively,
        and can reference both tensor arguments and previous operation results.

        Args:
            vertex: The composite operation vertex
            tensor_arg_map: Map from vertex counter to MLIR argument values
            operation_results: Map from vertex counter to MLIR operation results
            device: MLIR device value
            result_type: Expected result type for the final output

        Returns:
            MLIR value representing the final result of the composite operation
        """
        if not vertex.has_internals():
            raise ValueError(
                f"Composite operation {vertex.name} has no internals to expand"
            )

        # Parse parent composite vertex's output_info once to inherit for internal ops
        parent_output_info = vertex.output_info if vertex.output_info else None

        # Process each internal operation in order
        for internal_id in vertex.internals:
            internal_vertex = self.levelized_graph_ir.get_vertex(internal_id)
            if internal_vertex is None:
                raise ValueError(f"Internal vertex {internal_id} not found")

            # Skip placeholder operations in internals
            if internal_vertex.is_placeholder():
                continue

            # For internal operations, infer the result type from the vertex's output shape
            # Use the parent's result_type as a fallback, and inherit parent's output_info
            # if the internal op doesn't have its own output_info
            internal_result_type = self._get_output_type_from_vertex(
                internal_vertex, result_type, parent_output_info=parent_output_info
            )

            # Recursively process internal vertex
            internal_result = self._process_vertex(
                internal_vertex,
                tensor_arg_map,
                operation_results,
                device,
                internal_result_type,
            )
            if internal_result is not None:
                operation_results[internal_id] = internal_result

        # Return the last internal operation's result (the final output)
        # Find the last non-placeholder internal operation
        for internal_id in reversed(vertex.internals):
            internal_vertex = self.levelized_graph_ir.get_vertex(internal_id)
            if internal_vertex and not internal_vertex.is_placeholder():
                result = operation_results.get(internal_id)
                if result is not None:
                    return result

        # Fallback: return the last internal's result
        final_internal_id = vertex.internals[-1]
        return operation_results.get(final_internal_id)

    def _create_ttnn_op(
        self,
        op_name: str,
        result_type,
        operands: List,
        device,
        vertex: LevelizedGraphVertex,
    ) -> OpResult:
        """
        Create a TTNN MLIR operation using the operation registry.

        The registry system allows operations to be added without modifying this method.
        Each operation category (unary, binary, reduction) or individual operation can
        register its own handler that defines how it should be processed.

        Args:
            op_name: Name of the operation
            result_type: MLIR result type
            operands: List of MLIR operand values
            device: MLIR device value
            vertex: The operation vertex (used for operations that need shape/type information)

        Returns:
            MLIR operation result

        Raises:
            NotImplementedError: If operation is not registered
        """
        registry = get_registry()
        handler = registry.get_handler(op_name)

        if handler is None:
            raise NotImplementedError(
                f"Operation {op_name} not yet supported. "
                f"Register a handler in op_registry.py to add support."
            )

        # Parse arguments using the handler
        parsed_args = handler.parse_arguments(op_name, vertex.arguments)

        # Prepare result type (some ops like reduction may modify it)
        result_type = handler.prepare_result_type(op_name, result_type, vertex, self)

        # Prepare operands (some ops may need to add constant tensors or transform operands)
        # This is where ops like binary ops can add constant tensors to the operand list
        operands = handler.prepare_operands(
            op_name, operands, parsed_args, result_type, device, self
        )

        # Validate operands
        handler.validate_operands(op_name, operands)

        # Create the operation
        result = handler.create_operation(
            op_name, result_type, operands, device, parsed_args, self.ctx
        )

        return result

    def _create_constant_tensor(self, value: float, result_type, device) -> OpResult:
        """Create a constant tensor with the given value."""
        with Location.unknown(self.ctx):
            shape = list(result_type.shape)
            dtype = result_type.element_type

            # Create the full tensor using ttnn.full
            shape_attr = ttnn.ir.ShapeAttr.get(self.ctx, shape)
            dtype_attr = ttcore.ir.DataTypeAttr.get(
                self.ctx, ttcore_dtype_from_mlir_dtype(dtype)
            )
            fill_value_attr = FloatAttr.get(F32Type.get(self.ctx), value)

            # Extract layout from result_type if present
            layout_attr = None
            if result_type.encoding:
                layout = ttnn.ir.TTNNLayoutAttr.maybe_downcast(result_type.encoding)
                if layout:
                    layout_attr = ttnn.ir.LayoutAttr.get(
                        self.ctx, layout.memory_layout_as_int
                    )

            full_tensor = ttnn.FullOp(
                result_type,
                shape_attr,
                fill_value_attr,
                device=device,
                dtype=dtype_attr,
                layout=layout_attr,
            )
            full_tensor.attributes["ttnn.hoist_generic_via_d2m"] = UnitAttr.get(
                self.ctx
            )

            return full_tensor.result

    def compile(self) -> Module:
        """
        Generate MLIR module from captured graph.

        This is the main entry point for compilation. It:
        1. Finds optimal depth k (as the minimum depth where all level 1 operations are visitable)
        2. Extracts tensor arguments
        3. Builds function signature
        4. Processes vertices in topological order
        5. Generates MLIR operations

        Returns:
            MLIR Module containing the generated function
        """
        # Step 1: Find optimal depth and extract levelized graph
        k, self.levelized_graph_ir = self._find_optimal_depth()

        # Step 2: Extract tensor arguments
        tensor_vertices = self.levelized_graph_ir.extract_tensor_arguments()

        # Step 3: Build function signature
        with Location.unknown(self.ctx):
            input_types, output_types = self._build_function_signature(tensor_vertices)

            # Step 4: Create function
            with InsertionPoint(self.insert_point):
                func_op = func.FuncOp(
                    name=self.function_name, type=(input_types, output_types)
                )
                func_bb = func_op.add_entry_block()

                # Step 5: Build function body
                self._build_function_body(
                    func_bb, tensor_vertices, input_types, output_types
                )

        return self.module

    def _create_device(self) -> OpResult:
        """
        Create a device handle for TTNN operations.

        Returns:
            MLIR device value representing the target device.
        """
        mesh_shape_attr = ttnn.ir.MeshShapeAttr.get(self.ctx, 1, 1)
        mesh_offset_attr = ttnn.ir.MeshOffsetAttr.get(self.ctx, 0, 0)
        device = ttnn.get_device(
            mesh_shape=mesh_shape_attr, mesh_offset=mesh_offset_attr
        )
        return device

    def _build_function_body(
        self,
        func_bb,
        tensor_vertices: List[Tuple[LevelizedGraphVertex, int]],
        input_types: List,
        output_types: List,
    ) -> None:
        """Build the function body from the levelized graph."""
        with InsertionPoint(func_bb):
            # Create device handle
            device = self._create_device()

            # Map tensor vertices to function arguments
            tensor_arg_map = {}
            for i, (vertex, tensor_id) in enumerate(tensor_vertices):
                tensor_arg_map[vertex.counter] = func_bb.arguments[i]

            # Map operation vertices to results
            operation_results = {}

            # Process all level 1 operations in topological order
            level_1_ops = self.levelized_graph_ir.get_level_1_operations()

            for vertex in level_1_ops:
                result = self._process_vertex(
                    vertex, tensor_arg_map, operation_results, device, output_types[0]
                )
                if result is not None:
                    operation_results[vertex.counter] = result

            # Return the last result
            final_vertex = self.levelized_graph_ir.find_output_vertex()
            if final_vertex is not None:
                # Final vertex could be either a tensor argument or an operation result
                if final_vertex.counter in operation_results:
                    final_result = operation_results[final_vertex.counter]
                # Edge case: no ops are run and we are returning input tensor.
                elif final_vertex.counter in tensor_arg_map:
                    final_result = tensor_arg_map[final_vertex.counter]
                else:
                    raise ValueError(
                        f"Final vertex {final_vertex.counter} not found in "
                        f"operation_results or tensor_arg_map"
                    )
                func.ReturnOp([final_result])
