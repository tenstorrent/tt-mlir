# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Levelized Graph Data Structures.

This module provides data structures for working with levelized graphs
extracted from TTNN operation traces.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re
from ttnn_jit._src.return_modifier import get_placeholder_op_name
import ttnn_jit._src.supported_ops as supported_ops


@dataclass
class LevelizedGraphVertex:
    """
    Wrapper for a vertex in the levelized graph with helper methods.

    A vertex can represent either:
    - An input tensor (name="tensor[<id>]", in_edges=[], stacking_level=1)
    - An operation (name="ttnn::<op_name>", stacking_level>=1)

    Attributes:
        counter: Unique vertex ID
        stacking_level: Nesting depth (1=top-level, 2=internal, etc.)
        name: Vertex name (tensor ID or operation name)
        arguments: List of serialized arguments
        in_edges: IDs of vertices that produce inputs
        out_edges: IDs of vertices that consume outputs
        internals: IDs of internal operations at stacking_level+1
        output_info: Layout and memory config info (may be empty for final outputs)
        output_shape: Shape information (always present)
    """

    counter: int
    stacking_level: int
    name: str
    arguments: List[str]
    in_edges: List[int]
    out_edges: List[int]
    internals: List[int]
    output_info: List[str]
    output_shape: List[str]

    @staticmethod
    def from_dict(vertex_dict: Dict[str, Any]) -> "LevelizedGraphVertex":
        """Create a LevelizedGraphVertex from a dictionary."""
        return LevelizedGraphVertex(
            counter=vertex_dict["counter"],
            stacking_level=vertex_dict["stacking_level"],
            name=vertex_dict["name"],
            arguments=vertex_dict.get("arguments", []),
            in_edges=vertex_dict.get("in_edges", []),
            out_edges=vertex_dict.get("out_edges", []),
            internals=vertex_dict.get("internals", []),
            output_info=vertex_dict.get("output_info", []),
            output_shape=vertex_dict.get("output_shape", []),
        )

    def is_tensor(self) -> bool:
        """Check if this vertex represents an input tensor."""
        return (
            "tensor[" in self.name
            and len(self.in_edges) == 0
            and self.stacking_level == 1
        )

    def is_operation(self) -> bool:
        """Check if this vertex represents an operation."""
        return not self.is_tensor()

    def get_tensor_id(self) -> Optional[int]:
        """
        Extract tensor ID from tensor vertex name.

        Returns:
            Tensor ID if this is a tensor vertex, None otherwise.
            Example: "tensor[5]" -> 5
        """
        if not self.is_tensor():
            return None
        match = re.search(r"tensor\[(\d+)\]", self.name)
        return int(match.group(1)) if match else None

    def get_op_name(self) -> Optional[str]:
        """
        Extract operation name from operation vertex.

        Returns:
            Operation name with namespace prefix removed and adjusted.
            Example: "ttnn::multiply" -> "multiply"
                     "ttnn::sub" -> "subtract"
        """
        if not self.is_operation():
            return None

        # Extract base name
        op_name = self.name
        if "::" in op_name:
            op_name = op_name.split("::")[-1]

        # Apply name adjustments for MLIR dialect compatibility
        name_map = {
            "sub": "subtract",
            "div": "divide",
            "pow": "pow_tensor",
        }
        return name_map.get(op_name, op_name)

    def is_supported(self) -> bool:
        """
        Check if this operation is supported by the compiler.

        Returns:
            True if this is a tensor or a supported operation.
        """
        if self.is_tensor():
            return True

        op_name = self.get_op_name()
        return op_name is not None and supported_ops.is_supported(op_name)

    def has_internals(self) -> bool:
        """Check if this operation has internal operations."""
        return len(self.internals) > 0

    def is_final_output(self) -> bool:
        """Check if this vertex is a final output (no consumers)."""
        return len(self.out_edges) == 0

    def is_placeholder(self) -> bool:
        """
        Check if this vertex is a placeholder operation.

        A placeholder op is used to mark outputs without performing computation.
        One example is the identity operation (ttnn::identity).
        """
        return self.is_operation() and get_placeholder_op_name() in self.name.lower()

    def __repr__(self) -> str:
        return f"LevelizedGraphVertex(counter={self.counter}, level={self.stacking_level}, name={self.name})"


class LevelizedGraph:
    """
    Manages the levelized graph structure and provides graph operations.

    This class provides high-level operations on the levelized graph such as:
    - Extracting tensor arguments
    - Topological sorting
    - Finding output vertices
    - Checking visitability
    """

    def __init__(self, levelized_graph_data: List[Dict[str, Any]]):
        """
        Initialize from levelized graph data.

        Args:
            levelized_graph_data: List of vertex dictionaries from extract_levelized_graph
        """
        # Convert to LevelizedGraphVertex objects
        self.vertices: Dict[int, LevelizedGraphVertex] = {}
        for vertex_dict in levelized_graph_data:
            vertex = LevelizedGraphVertex.from_dict(vertex_dict)
            self.vertices[vertex.counter] = vertex

    def get_vertex(self, counter: int) -> Optional[LevelizedGraphVertex]:
        """Get a vertex by its counter/ID."""
        return self.vertices.get(counter)

    def get_vertices_at_level(self, level: int) -> List[LevelizedGraphVertex]:
        """Get all vertices at a specific stacking level."""
        return [v for v in self.vertices.values() if v.stacking_level == level]

    def extract_tensor_arguments(self) -> List[Tuple[LevelizedGraphVertex, int]]:
        """
        Extract input tensor vertices and order them by tensor ID.

        This determines the mapping from tensor vertices to function arguments:
        - tensor[1] -> %arg0
        - tensor[3] -> %arg1
        - tensor[5] -> %arg2

        Returns:
            List of (vertex, tensor_id) tuples sorted by tensor_id.
        """
        tensor_vertices = []
        for vertex in self.vertices.values():
            if vertex.is_tensor():
                tensor_id = vertex.get_tensor_id()
                if tensor_id is not None:
                    tensor_vertices.append((vertex, tensor_id))

        # Sort by tensor_id to determine argument order
        tensor_vertices.sort(key=lambda x: x[1])
        return tensor_vertices

    def get_level_1_operations(self) -> List[LevelizedGraphVertex]:
        """
        Get all operation vertices at stacking level 1 in topological order.
        Excludes placeholder operations which are just used to mark outputs.

        Returns:
            List of operation vertices at level 1, sorted by counter.
        """
        ops = [
            v
            for v in self.vertices.values()
            if v.stacking_level == 1 and v.is_operation() and not v.is_placeholder()
        ]
        # Counter order should already be topologically sorted
        ops.sort(key=lambda v: v.counter)
        return ops

    def find_output_vertex(self) -> Optional[LevelizedGraphVertex]:
        """
        Find the final output vertex at stacking level 1.
        Skips placeholder operations since they're just used to mark outputs.
        Returns the actual operation that produces the output.

        Returns:
            The vertex with no out_edges at stacking_level=1, or None if not found.
        """
        level_1_vertices = self.get_vertices_at_level(1)

        # First, try to find placeholder operation (it marks the output)
        placeholder_vertices = [v for v in level_1_vertices if v.is_placeholder()]

        if placeholder_vertices:
            # Placeholder found - return the operation that feeds into it
            placeholder_vertex = placeholder_vertices[-1]
            if placeholder_vertex.in_edges:
                # Get the vertex that feeds into the placeholder
                input_vertex_id = placeholder_vertex.in_edges[0]
                return self.get_vertex(input_vertex_id)

        # Fallback: find any operation with no out_edges (excluding placeholder)
        output_vertices = [
            v
            for v in level_1_vertices
            if v.is_final_output() and v.is_operation() and not v.is_placeholder()
        ]

        if len(output_vertices) == 0:
            return None

        # Should be exactly one, but return the last one if multiple
        return output_vertices[-1]

    def can_visit_all_level_1_ops(self) -> bool:
        """
        Check if all level 1 operations can be visited.

        An operation is visitable if:
        - It's a tensor (always visitable)
        - It's a placeholder operation (skip, used only for marking outputs)
        - It's a supported operation
        - It's a composite operation with internals (can expand)

        Returns:
            True if all level 1 operations are visitable.
        """
        for vertex in self.get_vertices_at_level(1):
            if vertex.is_tensor():
                continue  # Tensors are always visitable

            if vertex.is_placeholder():
                continue  # Placeholder ops are just markers, skip them

            if vertex.is_supported():
                continue  # Supported ops are visitable

            if (
                supported_ops.is_composite(vertex.get_op_name())
                and vertex.has_internals()
            ):
                continue  # Composite ops with internals should be expanded

            # This op is unsupported and has no internals - not visitable
            return False

        return True
