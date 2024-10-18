# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Library to manipulate TTIR Modules

from model_explorer import graph_builder
from ttmlir.dialects import tt, ttir, ttkernel
from . import utils
from collections import defaultdict


def create_id(op, name_dict):
    name = utils.get_loc_str(op.location)
    name_num = name_dict[name]
    id = name + "__" + str(name_num)
    name_dict[name] += 1
    return id


def get_attrs(op):
    result = []
    for attr in op.attributes:
        result.append(graph_builder.KeyValue(key=attr.name, value=str(attr.attr)))
    return result


def create_namespace(op):
    name = utils.get_loc_str(op.location)
    if op.parent and op.parent.name != "builtin.module":
        return create_namespace(op.parent) + "/" + name
    return name


def get_layout_attrs(tensor):
    if hasattr(tensor.type, "encoding") and tensor.type.encoding is None:
        source_node_attrs = [
            graph_builder.KeyValue(key="shape", value=str(tensor.type.shape)),
            graph_builder.KeyValue(
                key="element_type",
                value=str(tensor.type.element_type),
            ),
            graph_builder.KeyValue(key="rank", value=str(tensor.type.rank)),
        ]
    else:
        layout = tt.ir.LayoutAttr.getLayout(tensor.type)

        source_node_attrs = [
            graph_builder.KeyValue(key="shape", value=str(tensor.type.shape)),
            graph_builder.KeyValue(
                key="element_type",
                value=str(tensor.type.element_type),
            ),
            graph_builder.KeyValue(key="rank", value=str(tensor.type.rank)),
            # graph_builder.KeyValue(
            #    key="strides",
            #    value=array_ref_repr(layout.stride),
            # ),
            # graph_builder.KeyValue(
            #    key="Out of Bounds Value",
            #    value=layout.oobval.name,
            # ),
            utils.make_editable_kv(
                graph_builder.KeyValue(
                    key="Memory Space",
                    value=str(tt.MemorySpace(layout.memory_space_as_int)),
                ),
                editable={
                    "input_type": "value_list",
                    "options": utils.get_enum_options(tt.MemorySpace),
                },
            ),
            utils.make_editable_kv(
                graph_builder.KeyValue(
                    key="Memory Layout",
                    value=str(tt.TensorMemoryLayout(layout.memory_layout_as_int)),
                ),
                editable={
                    "input_type": "value_list",
                    "options": utils.get_enum_options(tt.TensorMemoryLayout),
                },
            ),
            utils.make_editable_kv(
                graph_builder.KeyValue(
                    key="Grid Shape",
                    value=array_ref_repr(layout.grid_attr.shape),
                ),
                editable={
                    "input_type": "value_list",
                    "options": ["1x1", "4x4", "8x8"],
                },
            ),
        ]

    # source_node.outputsMetadata.append(
    #     graph_builder.MetadataItem(
    #         id=str(connections[source_node.id]),
    #         attrs=[
    #             graph_builder.KeyValue(key="__tensor_tag", value=id)
    #         ]
    #         + source_node_attrs,
    #     )
    # )
    # return []
    return source_node_attrs

    source_node.attrs.extend(source_node_attrs)


def ttir_to_graph(module, ctx):
    # Can assume that to-layout pass has already been run on the module.
    name_dict = defaultdict(int)
    output_connections = dict()
    graph = graph_builder.Graph(id="ttir-graph")

    op_to_graph_node = dict()

    for op in module.body.operations:
        append_later = []
        for region in op.regions:
            for block in region.blocks:
                for op in block.operations:
                    # Create all the nodes and constants in the first pass.
                    graph_node = graph_builder.GraphNode(
                        id=create_id(op, name_dict),
                        label=op.name,
                        namespace=create_namespace(op),
                        attrs=get_attrs(op),
                    )

                    if op.name == "tensor.empty":
                        append_later.append(graph_node)
                    else:
                        graph.nodes.append(graph_node)

                    op_to_graph_node[op] = graph_node
                    output_connections[graph_node.id] = 0

                    for operand in op.operands:
                        if operand.owner == block and operand not in op_to_graph_node:
                            # This is a constant and we need to create a node for it.
                            operand_node = graph_builder.GraphNode(
                                id=create_id(op, name_dict),
                                label=operand.get_name(),
                                namespace=create_namespace(op),
                            )
                            graph.nodes.append(operand_node)
                            op_to_graph_node[operand] = operand_node
                            output_connections[operand_node.id] = 0

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

                        output_attrs = get_layout_attrs(operand)
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
