# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.ir import *
from ttmlir.dialects import ttnn, func, ttcore
from typing import Dict, List, Any, Optional, Set, Union
from dataclasses import dataclass
import ttnn_jit._src.supported_ops as supported_ops
import json
import re


@dataclass
class ParsedArgument:
    """
    Represents a parsed argument with its type and value.
    """

    arg_type: str  # "tensor_ref", "input_tensor", "constant", "nullopt", "unsupported"
    raw_value: str  # Original string value
    parsed_value: Optional[
        Union[int, float]
    ] = None  # Parsed numeric value if applicable


def parse_argument(arg: str) -> ParsedArgument:
    """
    Parse an argument string and return a structured representation.
    This function is called once during graph creation to avoid repeated regex matching.
    """
    if arg == "nullopt":
        return ParsedArgument(arg_type="nullopt", raw_value=arg)

    if arg.startswith("[ unsupported type"):
        return ParsedArgument(arg_type="unsupported", raw_value=arg)

    # Check if it's a tensor reference (e.g., "tensor: 0" or "tensor:0")
    match = re.match(r"tensor:\s*(\d+)", arg)
    if match:
        tensor_ref = int(match.group(1))
        return ParsedArgument(
            arg_type="tensor_ref", raw_value=arg, parsed_value=tensor_ref
        )

    # Check if it's an input tensor (starts with "Tensor(...")
    if arg.startswith("Tensor("):
        return ParsedArgument(arg_type="input_tensor", raw_value=arg)

    # Try to parse as a numeric constant (float)
    try:
        float_val = float(arg)
        return ParsedArgument(
            arg_type="constant", raw_value=arg, parsed_value=float_val
        )
    except (ValueError, TypeError):
        pass

    # Default: treat as a generic string argument
    return ParsedArgument(arg_type="other", raw_value=arg)


class Vertex:
    """
    Represents a vertex in the graph with properties from the JSON data.
    """

    def __init__(
        self,
        counter: int,
        node_type: str,
        connections: List[int],
        arguments: List[Any] = None,
        params: Dict[str, Any] = None,
        parsed_arguments: Optional[List[ParsedArgument]] = None,
    ):
        self.id = counter
        self.node_type = node_type
        self.connections = sorted(list(set(connections)))
        self.arguments = arguments or []
        self.params = params or {}
        self.parsed_arguments = parsed_arguments  # Pre-parsed arguments for efficiency

        # For traversal
        self.stacking_level = 0

        if len(connections) != len(set(connections)):
            print(
                f"Warning: Duplicate connections found for vertex {counter}: {connections}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert the vertex back to a dictionary representation."""
        return {
            "index": self.id,
            "name": self.params.get("name"),
            "children": self.connections,
            "arguments": self.arguments,
        }

    def __str__(self) -> str:
        """String representation of the vertex."""
        # print id, type, name and stacking level
        return f"Vertex(id={self.id}, type={self.node_type}, name={self.params.get('name')}, level={self.stacking_level})"

    def __repr__(self) -> str:
        return self.__str__()


class ClusterVertex:
    """
    Represents a cluster of vertices.
    """

    def __init__(self, cluster_idx: int):
        self.cluster_idx = cluster_idx
        self.vertices = []
        self.connections = []
        self.cluster_connections = []

    def add_vertex(self, vertex: Vertex) -> None:
        self.vertices.append(vertex)
        self.connections = sorted(list(set(self.connections + vertex.connections)))

    def add_cluster_connection(self, cluster_idx: int) -> None:
        if cluster_idx == self.cluster_idx:
            return
        self.cluster_connections = sorted(
            list(set(self.cluster_connections + [cluster_idx]))
        )

    def __str__(self) -> str:
        return (
            f"ClusterVertex(vertices={self.vertices}, connections={self.connections})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class JitGraph:
    """
    Represents a graph structure built from the JSON data.
    """

    def __init__(self):
        self.vertices: Dict[int, Vertex] = {}
        self.cluster_vertices: List[ClusterVertex] = []

    def add_vertex(self, vertex_data: Dict[str, Any]) -> Vertex:
        """
        Create and add a vertex to the graph from vertex data.
        """
        counter = vertex_data.get("counter")
        node_type = vertex_data.get("node_type")
        connections = vertex_data.get("connections", [])
        arguments = vertex_data.get("arguments", [])
        params = vertex_data.get("params", {})

        vertex = Vertex(counter, node_type, connections, arguments, params)
        self.vertices[counter] = vertex

        return vertex

    def add_cluster_vertex(self, cluster_vertex: ClusterVertex) -> None:
        self.cluster_vertices.append(cluster_vertex)

    def process_stacking_levels(self) -> None:
        vertices = sorted(self.vertices.values(), key=lambda v: v.id)

        curr_stacking_level = 1
        for vertex in vertices:
            if vertex.node_type == "function_end":
                curr_stacking_level -= 1

            vertex.stacking_level = curr_stacking_level

            if vertex.node_type == "function_start":
                curr_stacking_level += 1

        assert curr_stacking_level == 1

    def get_vertex(self, vertex_id: int) -> Optional[Vertex]:
        """Get a vertex by its ID."""
        return self.vertices.get(vertex_id)

    def get_children_vertices(self, vertex_id: int) -> List[Vertex]:
        """Get all vertices that this vertex connects to."""
        vertex = self.get_vertex(vertex_id)
        return [self.vertices[conn_id] for conn_id in vertex.connections]

    def get_parent_vertices(self, vertex_id: int) -> List[Vertex]:
        """Get all vertices that connect to this vertex."""
        return [v for v in self.vertices.values() if vertex_id in v.connections]

    def clusterize(self) -> None:
        # iterate graph vertices with index
        cluster_idx = -1
        curr_cluster_vertex = None
        for idx, vertex in enumerate(self.vertices.values()):
            assert idx == vertex.id

            if vertex.node_type == "function_start" and vertex.stacking_level == 1:
                cluster_idx += 1
                curr_cluster_vertex = ClusterVertex(cluster_idx)
                curr_cluster_vertex.add_vertex(vertex)
            elif vertex.node_type == "function_end" and vertex.stacking_level == 1:
                curr_cluster_vertex.add_vertex(vertex)
                self.add_cluster_vertex(curr_cluster_vertex)
                curr_cluster_vertex = None
            elif vertex.stacking_level > 1:
                curr_cluster_vertex.add_vertex(vertex)
            else:
                # skip these
                assert (
                    (vertex.stacking_level == 1 and vertex.node_type == "tensor")
                    or (vertex.stacking_level == 1 and vertex.node_type == "buffer")
                    or (
                        vertex.stacking_level == 1
                        and vertex.node_type == "capture_start"
                    )
                    or (
                        vertex.stacking_level == 1
                        and vertex.node_type == "buffer_deallocate"
                    )
                    or (
                        vertex.stacking_level == 1 and vertex.node_type == "capture_end"
                    )
                )

    def unify_clusters(self) -> None:
        vertex_to_cluster_map: Dict[int, ClusterVertex] = {}
        for cluster in self.cluster_vertices:
            for vertex in cluster.vertices:
                vertex_to_cluster_map[vertex.id] = cluster

        # Iterate clusters and update connections to point to clusters instead of vertices
        for cluster in self.cluster_vertices:
            for conn_id in cluster.connections:
                if conn_id not in vertex_to_cluster_map:
                    # This is expected for non-clustered vertices (tensors, function_end, etc.)
                    # Check if it's a vertex that exists
                    if conn_id in self.vertices:
                        vertex = self.vertices[conn_id]
                        # Skip non-clustered vertices - they're not operations
                        if vertex.node_type in [
                            "tensor",
                            "function_end",
                            "capture_end",
                            "buffer_deallocate",
                        ]:
                            continue
                        else:
                            print(
                                f"Warning: Connection {conn_id} (type={vertex.node_type}, name={vertex.params.get('name', 'N/A')}) not clustered"
                            )
                    else:
                        print(
                            f"Error: Connection {conn_id} not found in graph vertices"
                        )
                    continue
                cluster.add_cluster_connection(
                    vertex_to_cluster_map[conn_id].cluster_idx
                )

    def get_clusters(self) -> List[ClusterVertex]:
        return self.cluster_vertices

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert the graph back to a list of dictionaries."""
        return [vertex.to_dict() for vertex in self.vertices.values()]

    def fix_backward_links(self, names: List[str]) -> None:
        # Remove all connections from nodes to previous nodes
        for vertex in self.vertices.values():
            if vertex.params["name"] not in names:
                continue
            if len(vertex.connections) == 0:
                continue

            conns = [conn for conn in vertex.connections if conn >= vertex.id]
            vertex.connections = conns

    def fix_links_to_ones(self) -> None:
        for vertex in self.vertices.values():
            for child in self.get_children_vertices(vertex.id):
                if child.params["name"] == "ttnn::ones":
                    vertex.connections.remove(child.id)

    def dump_to_json(self, output_file: str) -> None:
        with open(output_file, "w") as f:
            json.dump(self.to_dict_list(), f, indent=2)


def load_graph_from_captured_graph(captured_graph: List[Dict[str, Any]]) -> JitGraph:
    graph = JitGraph()
    for vertex_data in captured_graph:
        graph.add_vertex(vertex_data)
    graph.process_stacking_levels()

    for vertex in graph.vertices.values():
        assert len(vertex.connections) == len(set(vertex.connections))

    return graph


def create_simplified_graph_from_clusterized(graph: JitGraph) -> JitGraph:
    """
    Create a simplified graph from clusters, preserving correct tensor references using CONNECTIONS.

    Key insight: The RUNTIME captured graph uses "Tensor(...)" for ALL tensor arguments, not "tensor: N".
    We need to use the CONNECTIONS in the graph to determine dataflow:
    - If a cluster has incoming connections from other clusters, those are its input tensors
    - We need to map those to either input arguments or results of previous simplified operations
    """
    new_graph = JitGraph()

    # Build mapping: which clusters connect to which clusters
    # cluster_connections already gives us this information

    # Build mapping: cluster index -> which input tensors (from capture) it consumes
    # This is done by looking at which TENSOR nodes connect into this cluster
    cluster_to_input_tensors = {}
    for cluster in graph.get_clusters():
        input_tensor_nodes = []
        # Find all tensor nodes that connect to any vertex in this cluster
        for vertex in cluster.vertices:
            # Find parent vertices (those that connect TO this vertex)
            parent_vertices = graph.get_parent_vertices(vertex.id)
            for parent in parent_vertices:
                if parent.node_type == "tensor" and parent.stacking_level == 1:
                    # This is an input tensor (stacking_level 1 means top-level)
                    if "tensor_id" in parent.params:
                        tensor_id = parent.params["tensor_id"]
                        if tensor_id not in [t["id"] for t in input_tensor_nodes]:
                            input_tensor_nodes.append(
                                {"id": tensor_id, "counter": parent.id}
                            )
        cluster_to_input_tensors[cluster.cluster_idx] = input_tensor_nodes

    # Build mapping: cluster index -> list of parent clusters that produce its inputs
    # We do this by looking at TENSOR nodes that connect to this cluster, and finding
    # which clusters produce those tensors
    tensor_to_producer_cluster = {}  # Map tensor_id -> cluster that produces it

    # First, identify which cluster produces which tensor
    # Only look at function_end nodes to find output tensors
    for cluster in graph.get_clusters():
        # Find function_end vertices in this cluster
        for vertex in cluster.vertices:
            if vertex.node_type == "function_end":
                # This function_end produces output tensors
                for conn_id in vertex.connections:
                    if conn_id in graph.vertices:
                        conn_vertex = graph.vertices[conn_id]
                        if (
                            conn_vertex.node_type == "tensor"
                            and "tensor_id" in conn_vertex.params
                        ):
                            tensor_id = conn_vertex.params["tensor_id"]
                            tensor_to_producer_cluster[tensor_id] = cluster.cluster_idx

    # Now, for each cluster, find which clusters produce the tensors it consumes
    cluster_to_parent_clusters = {}
    for cluster in graph.get_clusters():
        parent_cluster_idxs = []
        # Find tensor nodes that connect INTO this cluster
        for vertex in cluster.vertices:
            parent_vertices = graph.get_parent_vertices(vertex.id)
            for parent in parent_vertices:
                if parent.node_type == "tensor" and "tensor_id" in parent.params:
                    # This is a tensor node that this cluster consumes
                    tensor_id = parent.params["tensor_id"]
                    # Check if this tensor is produced by another cluster
                    if tensor_id in tensor_to_producer_cluster:
                        producer_cluster = tensor_to_producer_cluster[tensor_id]
                        if producer_cluster not in parent_cluster_idxs:
                            parent_cluster_idxs.append(producer_cluster)
        cluster_to_parent_clusters[cluster.cluster_idx] = sorted(parent_cluster_idxs)

    # Create simplified graph with corrected arguments
    # For each "Tensor(...)" argument, we need to determine if it's:
    # 1. An input tensor (counter < first operation's counter)
    # 2. Output from a previous cluster (use parent cluster mapping)
    for cluster_vertex in graph.get_clusters():
        first_vertex = cluster_vertex.vertices[0]

        # Process arguments to replace "Tensor(...)" with correct references
        corrected_arguments = []
        # Index for tracking which Tensor(...) argument we're on
        tensor_arg_idx = 0

        parent_clusters = cluster_to_parent_clusters.get(cluster_vertex.cluster_idx, [])

        for arg in first_vertex.arguments:
            if arg.startswith("Tensor("):
                # This is a tensor argument - determine what it references
                # It could be:
                # 1. An input tensor (if this cluster has input tensor connections)
                # 2. Output from a parent cluster (if this cluster has parent clusters)

                if tensor_arg_idx < len(parent_clusters):
                    # This tensor comes from a parent cluster
                    parent_cluster_idx = parent_clusters[tensor_arg_idx]
                    corrected_arguments.append(f"tensor: {parent_cluster_idx}")
                else:
                    # This might be an input tensor - keep it as Tensor(...)
                    # The IR compiler will handle it
                    corrected_arguments.append(arg)

                tensor_arg_idx += 1
            else:
                # Not a tensor argument, keep as-is
                corrected_arguments.append(arg)

        # Parse arguments once during graph creation for efficiency
        parsed_args = [parse_argument(arg) for arg in corrected_arguments]

        added_vertex = new_graph.add_vertex(
            {
                "counter": cluster_vertex.cluster_idx,
                "node_type": first_vertex.node_type,
                "connections": cluster_vertex.cluster_connections,
                "arguments": corrected_arguments,
                "params": first_vertex.params,
            }
        )

        # Store the parsed arguments for efficient access during IR generation
        added_vertex.parsed_arguments = parsed_args

        # For some reason, ttnn::ones calls somehow always point to next node in the graph + the correct node where its resulting tensor is consumed - fixing this by taking only the consumption node
        if added_vertex.params["name"] == "ttnn::ones":
            if len(added_vertex.connections) == 2:
                added_vertex.connections = [added_vertex.connections[-1]]

    return new_graph


class GraphToIRTranslator:
    """
    Compiler that generates MLIR IR from a captured graph.
    Takes a captured graph, simplifies it, and generates MLIR IR.
    """

    def __init__(self, captured_graph, function_name, tensor_args, max_grid):
        self.ctx = Context()
        self.cursor = Location.unknown(self.ctx)
        self.module = Module.create(self.cursor)
        self.insert_point = self.module.body
        self.captured_graph = captured_graph
        self.function_name = function_name
        self.tensor_args = tensor_args
        self.max_grid = max_grid
        self.simplified_graph = None  # Will be set during compilation

    def _mlir_dtype_from_ttnn_dtype(self, dtype):
        match int(dtype):
            case 0:
                return BF16Type.get(self.ctx)
            case 1:
                return F32Type.get(self.ctx)
            case 2:
                return IntegerType.get_unsigned(32, self.ctx)
            case 5:
                return IntegerType.get_unsigned(8, self.ctx)
            case 6:
                return IntegerType.get_unsigned(16, self.ctx)
            case 7:
                return IntegerType.get_signless(32, self.ctx)
            case _:
                raise ValueError(f"Unsupported dtype: {dtype}")

    def _ttcore_dtype_from_ttnn_dtype(self, dtype):
        match str(dtype):
            case "DataType.BFLOAT16":
                return ttcore.DataType.BFloat16
            case "DataType.FLOAT32":
                return ttcore.DataType.Float32
            case "DataType.INT32":
                return ttcore.DataType.Int32
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

    def _extract_dtype_from_tensor_arg(self, arg_str):
        """Extract DataType from argument string."""
        if "DataType::BFLOAT16" in arg_str:
            return 0  # BF16
        elif "DataType::FLOAT32" in arg_str:
            return 1  # F32
        # Add more as needed
        return None

    def _create_tensor_layout(self, tensor_arg):
        """Create TTNN layout attribute from tensor."""
        assert len(tensor_arg.shape) == 2
        data_type = self._ttcore_dtype_from_ttnn_dtype(tensor_arg.dtype)
        tile_type = ttcore.ir.TileType.get(self.ctx, 32, 32, data_type)
        identity_map = AffineMap.get_identity(2, self.ctx)

        if tensor_arg.memory_config().is_sharded():
            shard_spec = tensor_arg.memory_config().shard_spec
            shard_shape = shard_spec.shape
            grid_size_x = self.max_grid[0] + 1
            grid_size_y = self.max_grid[1] + 1

            # TTNN writes grids as (width, height) but compiler expects (height, width)
            grid = ttcore.ir.GridAttr.get(self.ctx, [grid_size_y, grid_size_x])
            shard_shape_tile_x = shard_shape[0] // 32
            shard_shape_tile_y = shard_shape[1] // 32
            buffer_type = ttnn.ir.BufferTypeAttr.get(self.ctx, ttnn.BufferType.L1)
            memref = MemRefType.get(
                [shard_shape_tile_x, shard_shape_tile_y], tile_type, None, buffer_type
            )
            ttnn_layout = ttnn.ir.TTNNLayoutAttr.get_with_linear(
                self.ctx,
                identity_map,
                grid,
                memref,
                ttnn.TensorMemoryLayout.BlockSharded,
                None,
            )
            return ttnn_layout
        else:
            assert (
                self.max_grid[0] == 0 and self.max_grid[1] == 0
            ), "The grid for DRAM interleaved tensors is always 1x1"
            buffer_type = ttnn.ir.BufferTypeAttr.get(self.ctx, ttnn.BufferType.DRAM)
            grid = ttcore.ir.GridAttr.get(self.ctx, [1, 1])
            shape = [tensor_arg.shape[0] // 32, tensor_arg.shape[1] // 32]
            memref = MemRefType.get(shape, tile_type, None, buffer_type)
            return ttnn.ir.TTNNLayoutAttr.get_with_linear(
                self.ctx,
                identity_map,
                grid,
                memref,
                ttnn.TensorMemoryLayout.Interleaved,
                None,
            )

    def _adjust_op_name(self, op_name):
        """Adjust operation name to match TTNN dialect names."""
        if op_name == "sub":
            op_name = "subtract"
        elif op_name == "div":
            op_name = "divide"
        elif op_name == "pow":
            op_name = "pow_tensor"
        return op_name

    def _extract_op_name(self, node):
        """Extract operation name from node."""
        name = node["name"]
        op_name = name
        if "::" in name:
            op_name = name.split("::")[-1]
        op_name = self._adjust_op_name(op_name)
        return op_name

    def _extract_op_name_from_vertex(self, vertex: Vertex):
        """Extract operation name from vertex."""
        name = vertex.params.get("name", "")
        op_name = name
        if "::" in name:
            op_name = name.split("::")[-1]
        op_name = self._adjust_op_name(op_name)
        return op_name

    def compile(self):
        """Generate MLIR module from captured graph."""
        # Step 1: Simplify the captured graph
        # Load graph from captured JSON
        graph = load_graph_from_captured_graph(self.captured_graph)

        # Clusterize and unify
        graph.clusterize()
        graph.unify_clusters()

        # Create simplified graph from clusters
        self.simplified_graph = create_simplified_graph_from_clusterized(graph)

        # Apply graph fixes
        self.simplified_graph.fix_backward_links(
            ["ttnn::conv2d", "ttnn::to_memory_config"]
        )
        self.simplified_graph.fix_links_to_ones()

        # Step 2: Generate MLIR IR from simplified graph
        with Location.unknown(self.ctx):
            # Build input types from tensor args
            input_types = []
            for arg_name, tensor_arg in self.tensor_args.items():
                shape = list(tensor_arg.shape)
                layout = self._create_tensor_layout(tensor_arg)
                dtype = self._mlir_dtype_from_ttnn_dtype(tensor_arg.dtype)
                tensor_type = RankedTensorType.get(shape, dtype, layout)
                input_types.append(tensor_type)

            # Output type is same as first input for now
            # How to determine output type dynamically?
            assert len(input_types) >= 1, "At least one input tensor is needed for now"
            output_types = [input_types[0]]

            # Create function
            with InsertionPoint(self.insert_point):
                func_op = func.FuncOp(
                    name=self.function_name, type=(input_types, output_types)
                )
                func_bb = func_op.add_entry_block()

                # Build the function body
                self._build_function_body(func_bb, input_types, output_types)

        # Return the generated module
        return self.module

    def _build_function_body(self, func_bb, input_types, output_types):
        """Build the function body from the simplified graph."""
        with InsertionPoint(func_bb):
            # Create get_device op
            mesh_shape_attr = ttnn.ir.MeshShapeAttr.get(self.ctx, 1, 1)
            mesh_offset_attr = ttnn.ir.MeshOffsetAttr.get(self.ctx, 0, 0)
            device = ttnn.get_device(
                mesh_shape=mesh_shape_attr, mesh_offset=mesh_offset_attr
            )

            # Map tensor_id to MLIR values
            # tensor_id is a string from the captured graph that identifies tensors
            tensor_id_to_value = {}

            # Map input tensors - assuming they start from tensor_id 0
            for i, arg in enumerate(func_bb.arguments):
                tensor_id_to_value[i] = arg

            # Map operation index to MLIR values (for tracking intermediate results)
            operation_results = {}

            # Process each vertex in the simplified graph
            # Use vertices directly to access pre-parsed arguments
            vertices = sorted(
                self.simplified_graph.vertices.values(), key=lambda v: v.id
            )
            for vertex in vertices:
                op_name = self._extract_op_name_from_vertex(vertex)
                result = self._process_node(
                    vertex,
                    op_name,
                    tensor_id_to_value,
                    operation_results,
                    device,
                    output_types[0],
                )

                # Store result for this operation index
                if result is not None:
                    operation_results[vertex.id] = result

            # Return the last result
            if len(vertices) > 0:
                last_vertex = vertices[-1]
                final_result = operation_results[last_vertex.id]
                func.ReturnOp([final_result])

    def _process_node(
        self,
        vertex,
        op_name,
        tensor_id_to_value,
        operation_results,
        device,
        result_type,
    ):
        """Process a single node and generate the corresponding MLIR operation.

        Uses pre-parsed arguments from the vertex for efficiency, avoiding repeated regex matching.
        """
        # Use pre-parsed arguments if available, otherwise fall back to parsing raw arguments
        if vertex.parsed_arguments is not None:
            parsed_args = vertex.parsed_arguments
        else:
            # Fallback for vertices that weren't parsed (shouldn't happen in normal flow)
            parsed_args = [parse_argument(arg) for arg in vertex.arguments]

        # Collect operands for the operation
        operands = []
        constant_value = None

        # Track the count of input tensors seen so far
        input_tensor_idx = 0

        for parsed_arg in parsed_args:
            if parsed_arg.arg_type == "nullopt" or parsed_arg.arg_type == "unsupported":
                # Skip nullopt and unsupported arguments
                continue

            if parsed_arg.arg_type == "tensor_ref":
                # This is a tensor reference (e.g., "tensor: 0")
                tensor_ref = parsed_arg.parsed_value

                # Check if it's an intermediate result (operation index)
                if tensor_ref in operation_results:
                    operands.append(operation_results[tensor_ref])
                # Otherwise check if it's an input tensor (tensor_id matches input index)
                elif tensor_ref in tensor_id_to_value:
                    operands.append(tensor_id_to_value[tensor_ref])
                else:
                    raise ValueError(
                        f"Tensor reference {tensor_ref} not found. "
                        f"Available operations: {list(operation_results.keys())}, "
                        f"Available inputs: {list(tensor_id_to_value.keys())}"
                    )
                continue

            if parsed_arg.arg_type == "input_tensor":
                # This is an input tensor (starts with "Tensor(...")
                # This happens for operations that directly take the input
                # All "Tensor(...)" arguments refer to the original input(s)
                # For now, we assume they reference the same input tensor(s) in order
                # If we have multiple inputs, use input_tensor_idx to track which one
                if len(tensor_id_to_value) > input_tensor_idx:
                    operands.append(tensor_id_to_value[input_tensor_idx])
                elif len(tensor_id_to_value) == 1:
                    # Only one input - all Tensor(...) references map to it
                    operands.append(tensor_id_to_value[0])
                else:
                    print(
                        f"Warning: Cannot map Tensor argument at index {input_tensor_idx}, only {len(tensor_id_to_value)} inputs available"
                    )
                input_tensor_idx += 1
                continue

            if parsed_arg.arg_type == "constant":
                # This is a numeric constant
                constant_value = parsed_arg.parsed_value

        # Generate the MLIR operation
        return self._create_ttnn_op(
            op_name, result_type, operands, device, constant_value
        )

    def _create_ttnn_op(
        self, op_name, result_type, operands, device, constant_value=None
    ):
        """Create a TTNN operation based on the operation name."""
        if op_name in supported_ops.unary_ops:
            # Unary operation
            if len(operands) != 1:
                raise ValueError(
                    f"Unary operation {op_name} expects 1 operand, got {len(operands)}"
                )

            op_func = getattr(ttnn, op_name)
            result = op_func(result_type, operands[0])
            result.owner.attributes["ttnn.hoist_generic_via_d2m"] = UnitAttr.get(
                self.ctx
            )
            return result

        elif op_name in supported_ops.binary_ops:
            # Binary operation
            if constant_value is not None:
                # Need to create a constant tensor
                constant_tensor = self._create_constant_tensor(
                    constant_value, result_type, device
                )
                operands.append(constant_tensor)

            if len(operands) != 2:
                raise ValueError(
                    f"Binary operation {op_name} expects 2 operands, got {len(operands)}"
                )

            op_func = getattr(ttnn, op_name)
            # Call the operation without dtype parameter
            result = op_func(result_type, operands[0], operands[1])
            result.owner.attributes["ttnn.hoist_generic_via_d2m"] = UnitAttr.get(
                self.ctx
            )

            # Add dtype as an attribute (not a parameter) for binary ops
            dtype_attr = ttcore.ir.DataTypeAttr.get(
                self.ctx, self._ttcore_dtype_from_mlir_dtype(result_type.element_type)
            )
            result.owner.attributes["dtype"] = dtype_attr
            return result

        else:
            raise NotImplementedError(
                f"Operation {op_name} not yet supported in GraphToIRTranslator"
            )

    def _create_constant_tensor(self, value, result_type, device):
        """Create a constant tensor with the given value."""
        with Location.unknown(self.ctx):
            shape = list(result_type.shape)
            dtype = result_type.element_type

            # Create the full tensor using ttnn.full
            shape_attr = ttnn.ir.ShapeAttr.get(self.ctx, shape)
            dtype_attr = ttcore.ir.DataTypeAttr.get(
                self.ctx, self._ttcore_dtype_from_mlir_dtype(dtype)
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

            return full_tensor
