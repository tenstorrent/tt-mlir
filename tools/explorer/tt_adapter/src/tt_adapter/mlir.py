# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Utility library for parsing MLIR

from collections import defaultdict
from model_explorer import graph_builder

from ttmlir.dialects import tt, ttnn, ttir
from ttmlir import ir


def get_loc_str(loc):
    try:
        res = str(loc).split('"')[1]
    except:
        res = "unknown"
    return res


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
            key="memory_layout",
            value=str(tt.TensorMemoryLayout(layout.memory_layout_as_int)),
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
    result.append(
        graph_builder.KeyValue(
            key="memory_layout",
            value=str(ttnn.TensorMemoryLayout(layout.memory_layout_as_int)),
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
    buffer_attr = ttnn.ir.BufferTypeAttr.maybe_downcast(layout.memref.memory_space)
    result.append(
        graph_builder.KeyValue(
            key="memref_memory_space", value=str(ttnn.BufferType(buffer_attr.value))
        )
    )
    return result


class OpHandler:
    def __init__(self, op):
        self.op = op

    def get_id(self, names: defaultdict):
        name = get_loc_str(self.op.location)
        name_num = names[name]
        id = name + "__" + str(name_num)
        names[name] += 1
        return id

    def get_namespace(self, parent_op=None):
        op = self.op if not parent_op else parent_op
        name = get_loc_str(op.location)
        if op.parent and op.parent.name != "builtin.module":
            return self.get_namespace(op.parent) + "/" + name
        return name

    def get_attributes(self):
        # Parse Op Attributes themselves
        result = []
        for attr in self.op.attributes:
            result.extend(AttrHandler.parse_attr(attr))
        return result

    def make_graph_node(self, name_dict):
        return graph_builder.GraphNode(
            id=self.get_id(name_dict),
            label=self.op.name,
            namespace=self.get_namespace(),
            attrs=self.get_attributes(),
        )

    def make_constant_node(self, name_dict, constant_name):
        return graph_builder.GraphNode(
            id=self.get_id(name_dict),
            label=constant_name,
            namespace=self.get_namespace(),
        )


EMPTY_OPS = [
    "ttnn.empty",
    "tensor.empty",
]

FILTERED_OPS = [
    "ttnn.deallocate",
    "ttnn.get_device",
]


def build_graph(module):
    name_dict = defaultdict(int)
    output_connections = defaultdict(int)
    graph = graph_builder.Graph(id="tt-graph")

    op_to_graph_node = {}

    module_op = OpHandler(module.operation)
    graph.nodes.append(module_op.make_graph_node(name_dict))

    for op in module.body.operations:
        append_later = []
        for region in op.regions:
            for block in region.blocks:
                for op in block.operations:
                    # Create all the nodes and constants in the first pass.
                    operation = OpHandler(op)
                    graph_node = operation.make_graph_node(name_dict)

                    if op.name in EMPTY_OPS:
                        append_later.append(graph_node)
                    elif op.name not in FILTERED_OPS:
                        graph.nodes.append(graph_node)

                    op_to_graph_node[op] = graph_node

                    for operand in op.operands:
                        if isinstance(operand, ir.Value):
                            # This is a constant and we need to create a node for it.
                            operand_node = operation.make_constant_node(
                                name_dict, operand.get_name()
                            )
                            graph.nodes.append(operand_node)
                            op_to_graph_node[operand] = operand_node

                # This puts the node at the far right when viewing which is a bit more consistant with it being the last operand.
                for node in append_later:
                    graph.nodes.append(node)

                for op in block.operations:
                    # Create all the edges in the second pass.
                    for operand_index, operand in enumerate(op.operands):
                        if operand.owner == block:
                            source_node = op_to_graph_node[operand]
                        else:
                            source_node = op_to_graph_node[operand.owner]

                        target_node = op_to_graph_node[op]

                        target_node.incomingEdges.append(
                            graph_builder.IncomingEdge(
                                sourceNodeId=source_node.id,
                                sourceNodeOutputId=output_connections[source_node.id],
                                targetNodeInputId=operand_index,
                            )
                        )

                        output_attrs = []
                        if isinstance(operand.type, ir.RankedTensorType):
                            output_attrs = [
                                graph_builder.KeyValue(
                                    key="shape", value=str(operand.type.shape)
                                ),
                                graph_builder.KeyValue(
                                    key="dtype", value=str(operand.type.element_type)
                                ),
                                graph_builder.KeyValue(
                                    key="rank", value=str(operand.type.rank)
                                ),
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
                                        key="__tensor_tag", value=target_node.label
                                    ),
                                ]
                                + output_attrs,
                            )
                        )
                        output_connections[source_node.id] += 1

    return graph
