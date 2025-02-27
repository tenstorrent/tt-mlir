# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Utility library for parsing MLIR
import re
from collections import defaultdict
from model_explorer import graph_builder, node_data_builder

from ttmlir.dialects import tt, ttnn, ttir
from ttmlir import ir, util
from . import utils


def parse_loc_string(loc_str):
    """
    This can be replaced by ttmlir.ir.Module.parse, but requires some further wodo to extract the actual location object from the module.
    """
    match = re.match(r'^loc\("([^"]+)"', loc_str)
    if match:
        return match.group(1)
    return None


class AttrHandler:
    """
    A class that handles parsing and registering handlers for MLIR attribute types.
    """

    ATTR_HANDLERS = {}

    @staticmethod
    def default_parser(attr):
        return [graph_builder.KeyValue(key=attr.name, value=str(attr.attr))]

    @staticmethod
    def parse_attr(attr):
        if attr.name in AttrHandler.ATTR_HANDLERS:
            return AttrHandler.ATTR_HANDLERS[attr.name](attr.attr)
        else:
            # Unknown Attr Type, return default parser
            return AttrHandler.default_parser(attr)

    @staticmethod
    def register_handler(attr_name):
        """
        Decorator function to register a handler for a specific attribute name.

        Usage:

        @AttrHandler.register_handler("attr_name")
        def parse_attr_name(attr: ir.Attribute) -> List[graph_builder.KeyValue]:
            pass

        registers a handler for any NamedAttribute present in the MLIR module with the name "attr_name".

        The handler itself is the function that is decorated with this decorator. It must follow the function signature of
        `parse_attr_name` as shown above.
        """

        def decorator(handler):
            AttrHandler.ATTR_HANDLERS[attr_name] = handler
            return handler

        return decorator


@AttrHandler.register_handler("tt.device")
def parse_tt_device(attr):
    device = tt.ir.DeviceAttr.maybe_downcast(attr)
    result = []
    result.append(
        graph_builder.KeyValue(
            key="device_chip_ids", value=", ".join(map(str, device.chip_ids))
        )
    )
    result.append(
        graph_builder.KeyValue(
            key="device_grid_shape", value=str(device.grid_attr.shape)
        )
    )
    if device.mesh_shape:
        result.append(
            graph_builder.KeyValue(
                key="device_mesh_shape", value=str(device.mesh_shape)
            )
        )
    result.append(graph_builder.KeyValue(key="device_l1_map", value=str(device.l1_map)))
    result.append(
        graph_builder.KeyValue(key="device_dram_map", value=str(device.dram_map))
    )
    return result


@AttrHandler.register_handler("tt.system_desc")
def parse_tt_system_desc(attr):
    system_desc = tt.ir.SystemDescAttr.maybe_downcast(attr)
    result = []
    for i, chip_desc, chip_coord, chip_capability in zip(
        system_desc.chip_desc_indices,
        system_desc.chip_descs,
        system_desc.chip_coords,
        system_desc.chip_capabilities,
    ):
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-arch", value=str(tt.Arch(chip_desc.arch.arch_as_int))
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-capability",
                value=str(tt.ChipCapability(chip_capability.capability_as_int)),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-coord",
                value="x".join(
                    map(
                        str,
                        (chip_coord.rack, chip_coord.shelf, chip_coord.y, chip_coord.x),
                    )
                ),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-dram-channel-size",
                value=str(chip_desc.dram_channel_size),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-dram-unreserved-base",
                value=str(chip_desc.dram_unreserved_base),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-dram-unreserved-end",
                value=str(chip_desc.dram_unreserved_end),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-erisc-l1-unreserved-size",
                value=str(chip_desc.erisc_l1_unreserved_base),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-grid", value="x".join(map(str, chip_desc.grid))
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-l1-size", value=str(chip_desc.l1_size)
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-l1-unreserved-base",
                value=str(chip_desc.l1_unreserved_base),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-noc-dram-address-align-bytes",
                value=str(chip_desc.noc_dram_address_align_bytes),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-noc-l1-address-align-bytes",
                value=str(chip_desc.noc_l1_address_align_bytes),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-num-cbs", value=str(chip_desc.num_cbs)
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-num-dram-channels",
                value=str(chip_desc.num_dram_channels),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-pcie-address-align-bytes",
                value=str(chip_desc.pcie_address_align_bytes),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-usable-dram-channel-size",
                value=str(chip_desc.usable_dram_channel_size),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-usable-l1-size", value=str(chip_desc.usable_l1_size)
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-supported-data-types",
                value=", ".join(
                    [
                        str(tt.DataType(dt.data_type_as_int))
                        for dt in chip_desc.supported_data_types
                    ]
                ),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-supported-tile-sizes",
                value=", ".join(
                    [
                        "x".join(map(str, (tsize.y, tsize.x)))
                        for tsize in chip_desc.supported_tile_sizes
                    ]
                ),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-dram-core-coords",
                value=", ".join(
                    [
                        "x".join(map(str, (coord.y, coord.x)))
                        for coord in chip_desc.chip_physical_cores.dram
                    ]
                ),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-eth-core-coords",
                value=", ".join(
                    [
                        "x".join(map(str, (coord.y, coord.x)))
                        for coord in chip_desc.chip_physical_cores.eth
                    ]
                ),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-eth-inactive-core-coords",
                value=", ".join(
                    [
                        "x".join(map(str, (coord.y, coord.x)))
                        for coord in chip_desc.chip_physical_cores.eth_inactive
                    ]
                ),
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-worker-core-coords",
                value=", ".join(
                    [
                        "x".join(map(str, (coord.y, coord.x)))
                        for coord in chip_desc.chip_physical_cores.worker
                    ]
                ),
            )
        )
    return result


@AttrHandler.register_handler("mesh_shape")
def parse_mesh_shape(attr):
    mesh_shape = ttnn.ir.MeshShapeAttr.maybe_downcast(attr)
    return [
        graph_builder.KeyValue(
            key="mesh_shape", value="x".join(map(str, (mesh_shape.y, mesh_shape.x)))
        )
    ]


@AttrHandler.register_handler("layout")
def parse_layout(attr):
    # This is for parsing TTNN Layouts (Enum)
    layout = ttnn.ir.LayoutAttr.maybe_downcast(attr)
    return [graph_builder.KeyValue(key="layout", value=str(ttnn.Layout(layout.value)))]


@AttrHandler.register_handler("memory_config")
def parse_memory_config(attr):
    memory_config = ttnn.ir.MemoryConfigAttr.maybe_downcast(attr)
    result = []
    result.append(
        graph_builder.KeyValue(
            key="buffer-type",
            value=str(ttnn.BufferType(memory_config.buffer_type.value)),
        )
    )
    result.append(
        graph_builder.KeyValue(
            key="shard-shape",
            value="x".join(map(str, memory_config.shard_spec.shard_shape.shape)),
        )
    )
    result.append(
        graph_builder.KeyValue(
            key="tensor-memory-layout",
            value=str(
                ttnn.TensorMemoryLayout(memory_config.tensor_memory_layout.value)
            ),
        )
    )
    return result


@AttrHandler.register_handler("force")
def parse_force(attr):
    return [graph_builder.KeyValue(key="force", value=str(attr.value))]


@AttrHandler.register_handler("dtype")
def parse_dtype(attr):
    dtype = tt.ir.DataTypeAttr.maybe_downcast(attr)
    if dtype is None:
        # Potential for dtype to be StringAttr instead of tt.DataTypeAttr
        return [graph_builder.KeyValue(key="dtype", value=str(attr))]
    return [
        graph_builder.KeyValue(
            key="dtype", value=str(tt.DataType(dtype.data_type_as_int))
        )
    ]


@AttrHandler.register_handler("shape")
def parse_shape(attr):
    shape = ttnn.ir.ShapeAttr.maybe_downcast(attr)
    if not shape:
        return [graph_builder.KeyValue(key="shape", value=str(attr))]
    return [graph_builder.KeyValue(key="shape", value="x".join(map(str, shape.shape)))]


@AttrHandler.register_handler("operandSegmentSizes")
def parse_operandSegmentSizes(attr):
    return [graph_builder.KeyValue(key="operandSegmentSizes", value=str(list(attr)))]


@AttrHandler.register_handler("dimension")
def parse_dimension(attr):
    return [graph_builder.KeyValue(key="dimension", value=str(attr.value))]


@AttrHandler.register_handler("tt.layout")
def parse_tt_layout(attr):
    layout = tt.ir.MetalLayoutAttr.maybe_downcast(attr)
    result = []
    result.append(graph_builder.KeyValue(key="linear", value=str(layout.linear)))
    result.append(
        graph_builder.KeyValue(
            key="memory_space", value=str(tt.MemorySpace(layout.memory_space_as_int))
        )
    )
    result.append(
        graph_builder.KeyValue(
            key="grid_shape", value="x".join(map(str, layout.grid_attr.shape))
        )
    )
    result.append(
        graph_builder.KeyValue(key="memref_shape", value=str(layout.memref.shape))
    )
    result.append(
        graph_builder.KeyValue(key="memref_rank", value=str(layout.memref.rank))
    )
    tile_type = tt.ir.TileType.maybe_downcast(layout.memref.element_type)
    if tile_type is not None:
        result.append(
            graph_builder.KeyValue(
                key="tile_datatype", value=str(tt.DataType(tile_type.data_type_as_int))
            )
        )
        result.append(
            graph_builder.KeyValue(
                key="tile_shape", value="x".join(map(str, tile_type.shape))
            )
        )
    return result


@AttrHandler.register_handler("ttnn_layout")
def parse_ttnn_ttnn_layout(attr):
    layout = ttnn.ir.TTNNLayoutAttr.maybe_downcast(attr)
    result = []
    result.append(graph_builder.KeyValue(key="linear", value=str(layout.linear)))
    memory_layout = layout.tensor_memory_layout_as_int
    if memory_layout is not None:
        result.append(
            utils.make_editable_kv(
                graph_builder.KeyValue(
                    key="tensor_memory_layout",
                    value=str(ttnn.TensorMemoryLayout(memory_layout)),
                ),
                editable={
                    "input_type": "value_list",
                    "options": [str(o) for o in ttnn.TensorMemoryLayout],
                },
            )
        )
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="grid_shape", value="x".join(map(str, layout.grid_attr.shape))
            ),
            editable={
                "input_type": "grid",
                "separator": "x",
                "min_value": 1,
                "max_value": 100,
                "step": 1,
            },
        )
    )
    result.append(
        graph_builder.KeyValue(key="memref_shape", value=str(layout.memref.shape))
    )
    buffer_attr = ttnn.ir.BufferTypeAttr.maybe_downcast(layout.memref.memory_space)
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="buffer_type", value=str(ttnn.BufferType(buffer_attr.value))
            ),
            editable={
                "input_type": "value_list",
                "options": [str(o) for o in ttnn.BufferType],
            },
        )
    )

    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="memory_layout",
                value=str(ttnn.Layout(layout.memory_layout_as_int)),
            ),
            editable={
                "input_type": "value_list",
                "options": [str(o) for o in ttnn.Layout],
            },
        )
    )

    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="data_type",
                value=str(tt.DataType(layout.data_type_as_int)),
            ),
            editable={
                "input_type": "value_list",
                "options": [str(o) for o in tt.DataType],
            },
        )
    )
    return result


class OpHandler:
    # Help create unique ids for ops with the same location name.
    name_dict = defaultdict(int)
    schedule = 0

    def __init__(self, op):
        self.op = op
        self.named_location = util.get_loc_name(self.op.location)
        self.full_location = util.get_loc_full(self.op.location)
        self.id = self._create_unique_id()

    def _create_unique_id(self):
        name = self.full_location if self.full_location else "unknown"
        name_num = self.name_dict[name]
        id = name + "__" + str(name_num)
        self.name_dict[name] += 1
        return id

    def get_namespace(self, parent_op=None):
        op = self.op if not parent_op else parent_op
        name = util.get_loc_name(op.location)
        if op.parent and op.parent.name != "builtin.module":
            parent_name = self.get_namespace(op.parent)
            if parent_name:
                return parent_name + "/" + name
        return name or ""

    def get_attributes(self):
        # Parse Op Attributes themselves
        result = []
        for attr in self.op.attributes:
            result.extend(AttrHandler.parse_attr(attr))

        # Add location as an attribute
        if self.named_location:
            result.append(
                graph_builder.KeyValue(key="named_location", value=self.named_location)
            )
        if self.full_location:
            result.append(
                graph_builder.KeyValue(key="full_location", value=self.full_location)
            )

        # Add output tensor attriributes to the op itself
        if self.op.results:
            # Examples like the Pooling Op Contain more than 1 Result Tensor
            # Since the output of a pool op is currently the same shape we don't have to add any extra logic
            # In the future we may have to obfuscate with output_shape_1, etc...
            # For now let's just set the output_tensor to the first result
            output_tensor = list(self.op.results)[0]
            output_attrs = []
            if isinstance(output_tensor.type, ir.RankedTensorType):
                output_attrs = [
                    graph_builder.KeyValue(
                        key="shape", value=str(output_tensor.type.shape)
                    ),
                    graph_builder.KeyValue(
                        key="dtype", value=str(output_tensor.type.element_type)
                    ),
                    graph_builder.KeyValue(
                        key="rank", value=str(output_tensor.type.rank)
                    ),
                ]
            if hasattr(output_tensor.type, "encoding") and output_tensor.type.encoding:
                if "ttnn_layout" in str(output_tensor.type.encoding):
                    output_attrs.extend(
                        AttrHandler.parse_attr(
                            output_tensor.type.encoding.get_named("ttnn_layout")
                        )
                    )
                else:
                    # Parse as a standard layout
                    output_attrs.extend(
                        AttrHandler.parse_attr(
                            output_tensor.type.encoding.get_named("tt.layout")
                        )
                    )
            result.extend(output_attrs)

        # Add schedule as an attribute
        result.append(
            graph_builder.KeyValue(key="schedule", value=str(OpHandler.schedule))
        )
        OpHandler.schedule += 1

        return result

    def make_graph_node(self):
        return graph_builder.GraphNode(
            id=self.id,
            label=str(self.op.name),
            namespace=self.get_namespace(),
            attrs=self.get_attributes(),
        )

    def make_constant_node(self, constant_name):
        return graph_builder.GraphNode(
            id=self._create_unique_id(),
            label=str(constant_name),
            namespace=self.get_namespace(),
        )


EMPTY_OPS = [
    "ttnn.empty",
    "tensor.empty",
]

FILTERED_OPS = [
    "ttnn.deallocate",
    "ttnn.get_device",
    *EMPTY_OPS,
]


def build_graph(module, perf_trace=None):
    output_connections = defaultdict(int)
    graph = graph_builder.Graph(id="tt-graph")

    op_to_graph_node = {}
    # Track operands already added to graph to avoid duplicates
    operands_in_graph = set()

    # Prepare perf data for color overlay
    perf_node_data = {}
    loc_to_perf = {}
    if perf_trace is not None:
        for _, row in perf_trace.iterrows():
            loc = parse_loc_string(row["LOC"])
            assert loc not in loc_to_perf
            if loc:
                loc_to_perf[loc] = row["DEVICE FW DURATION [ns]"]

    # Process the module hierarchy recursively
    process_module(
        module,
        graph,
        op_to_graph_node,
        operands_in_graph,
        output_connections,
        loc_to_perf,
        perf_node_data,
    )

    # Add performance data to the graph color overlay, if it exists
    overlay_data = None
    if perf_node_data:
        gradient = [
            node_data_builder.GradientItem(stop=0, bgColor="yellow"),
            node_data_builder.GradientItem(stop=1, bgColor="red"),
        ]
        graph_node_data = node_data_builder.GraphNodeData(
            results=perf_node_data, gradient=gradient
        )
        overlay_data = node_data_builder.ModelNodeData(
            graphsData={"tt-graph": graph_node_data}
        )

    OpHandler.schedule = 0
    return graph, overlay_data


def process_module(
    module,
    graph,
    op_to_graph_node,
    operands_in_graph,
    output_connections,
    loc_to_perf,
    perf_node_data,
):
    """
    Recursively process a module and its nested submodules.

    Args:
        module: The module to process
        graph: The graph being built
        op_to_graph_node: Mapping from operations to graph nodes
        operands_in_graph: Set of operands already added to graph
        output_connections: Tracking of output connections
        loc_to_perf: Mapping from locations to performance data
        perf_node_data: Performance data for nodes
    """
    module_op = OpHandler(module.operation)
    module_attrs = module_op.get_attributes()
    module_attrs = dict((attr.key, attr.value) for attr in module_attrs)

    # Add module attributes to the graph as "namespace attributes"
    if not graph.groupNodeAttributes:
        graph.groupNodeAttributes = {}

    # Add this module's namespace attributes
    namespace = module_op.get_namespace()
    if namespace not in graph.groupNodeAttributes:
        graph.groupNodeAttributes[namespace] = module_attrs
    else:
        # Merge with existing attributes if namespace already exists
        graph.groupNodeAttributes[namespace].update(module_attrs)

    # Process operations in this module
    process_operations(
        module.body.operations,
        graph,
        op_to_graph_node,
        operands_in_graph,
        output_connections,
        loc_to_perf,
        perf_node_data,
    )


def process_operations(
    operations,
    graph,
    op_to_graph_node,
    operands_in_graph,
    output_connections,
    loc_to_perf,
    perf_node_data,
):
    """
    Process a list of operations, including handling nested modules.

    Args:
        operations: List of operations to process
        graph: The graph being built
        op_to_graph_node: Mapping from operations to graph nodes
        operands_in_graph: Set of operands already added to graph
        output_connections: Tracking of output connections
        loc_to_perf: Mapping from locations to performance data
        perf_node_data: Performance data for nodes
    """
    append_later = []

    # First pass: create all nodes and constants
    for op in operations:
        # Check if this operation is a nested module
        if is_module_op(op):
            # Process the nested module's ops recursively
            process_operations(
                op.regions[0].blocks[0],
                graph,
                op_to_graph_node,
                operands_in_graph,
                output_connections,
                loc_to_perf,
                perf_node_data,
            )
            continue

        # Process regions in the operation
        for region in op.regions:
            for block in region.blocks:
                # Recursively process operations in this block
                process_operations(
                    block.operations,
                    graph,
                    op_to_graph_node,
                    operands_in_graph,
                    output_connections,
                    loc_to_perf,
                    perf_node_data,
                )

        # Create graph node for this operation
        operation = OpHandler(op)
        graph_node = operation.make_graph_node()

        if (
            operation.named_location in loc_to_perf
            and operation.op.name not in EMPTY_OPS
        ):
            perf_node_data[operation.id] = node_data_builder.NodeDataResult(
                loc_to_perf[operation.named_location]
            )

        if op.name not in FILTERED_OPS and op.name in EMPTY_OPS:
            append_later.append(graph_node)
        elif op.name not in FILTERED_OPS:
            graph.nodes.append(graph_node)

        op_to_graph_node[op] = graph_node

        # Process operands
        for operand in op.operands:
            if isinstance(operand, ir.Value) and not isinstance(
                operand.owner, ir.Operation
            ):
                # If the owner is not an op, then it is a constant provided from the toplevel FuncOp.
                if operand not in operands_in_graph:
                    # This is a constant and we need to create a node for it.
                    operand_node = operation.make_constant_node(operand.get_name())
                    graph.nodes.append(operand_node)
                    op_to_graph_node[operand] = operand_node
                    operands_in_graph.add(operand)

    # Add the nodes that should be appended later
    for node in append_later:
        graph.nodes.append(node)

    # Second pass: create all edges
    for op in operations:
        # Skip module operations as they've been processed recursively
        if is_module_op(op):
            continue

        # Process regions in the operation
        for region in op.regions:
            for block in region.blocks:
                create_edges_for_block(block, op_to_graph_node, output_connections)


def create_edges_for_block(block, op_to_graph_node, output_connections):
    """
    Create edges between nodes for operations in a block.

    Args:
        block: The block containing operations
        op_to_graph_node: Mapping from operations to graph nodes
        output_connections: Tracking of output connections
    """
    for op in block.operations:
        # Skip module operations as they've been processed recursively
        if is_module_op(op):
            continue

        # Create edges for this operation
        for operand_index, operand in enumerate(op.operands):
            if operand.owner == block:
                source_node = op_to_graph_node[operand]
            else:
                source_node = op_to_graph_node[operand.owner]

            target_node = op_to_graph_node[op]

            target_node.incomingEdges.append(
                graph_builder.IncomingEdge(
                    sourceNodeId=source_node.id,
                    sourceNodeOutputId=str(output_connections[source_node.id]),
                    targetNodeInputId=str(operand_index),
                )
            )

            output_attrs = []
            if isinstance(operand.type, ir.RankedTensorType):
                output_attrs = [
                    graph_builder.KeyValue(key="shape", value=str(operand.type.shape)),
                    graph_builder.KeyValue(
                        key="dtype", value=str(operand.type.element_type)
                    ),
                    graph_builder.KeyValue(key="rank", value=str(operand.type.rank)),
                ]
            if hasattr(operand.type, "encoding") and operand.type.encoding:
                if "ttnn_layout" in str(operand.type.encoding):
                    output_attrs.extend(
                        AttrHandler.parse_attr(
                            operand.type.encoding.get_named("ttnn_layout")
                        )
                    )
                else:
                    # Parse as a standard layout
                    output_attrs.extend(
                        AttrHandler.parse_attr(
                            operand.type.encoding.get_named("tt.layout")
                        )
                    )
            source_node.outputsMetadata.append(
                graph_builder.MetadataItem(
                    id=str(output_connections[source_node.id]),
                    attrs=[
                        graph_builder.KeyValue(
                            key="__tensor_tag", value=str(target_node.label)
                        ),
                    ]
                    + output_attrs,
                )
            )
            output_connections[source_node.id] += 1


def is_module_op(op):
    """
    Check if an operation represents a module.

    Args:
        op: The operation to check

    Returns:
        bool: True if the operation is a module, False otherwise
    """
    # Check for tt.device_module or builtin.module operations
    return op.name == "tt.device_module" or op.name == "builtin.module"
