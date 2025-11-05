# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Utility library for parsing MLIR
import logging
import re
from pathlib import Path
from collections import defaultdict
from model_explorer import graph_builder, node_data_builder
from datetime import datetime, timezone

from ttmlir.dialects import ttcore, ttnn, ttir, func
from ttmlir import ir, util
from . import utils

OVERRIDE_PARAMETER_DISABLED_STR = "None"


def parse_loc_string(loc_str):
    """
    This can be replaced by ttmlir.ir.Module.parse, but requires some further work to extract the actual location object from the module.
    """
    if not isinstance(loc_str, str):
        logging.error(
            "Invalid LOC type in perf_trace: %s, expected string", type(loc_str)
        )
        # raise IndexError("Invalid LOC type in perf_trace: %s, expected string", type(loc_str))
        return None
    match = re.match(r'^loc\("([^"]+)"', loc_str)
    if not match:
        logging.error("Failed to match location string: %s", loc_str)
        return None
    return match.group(1)


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


@AttrHandler.register_handler("ttcore.device")
def parse_tt_device(attr):
    device = ttcore.ir.DeviceAttr.maybe_downcast(attr)
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


@AttrHandler.register_handler("ttcore.system_desc")
def parse_tt_system_desc(attr):
    system_desc = ttcore.ir.SystemDescAttr.maybe_downcast(attr)
    result = []
    for i, chip_desc, chip_coord, chip_capability in zip(
        system_desc.chip_desc_indices,
        system_desc.chip_descs,
        system_desc.chip_coords,
        system_desc.chip_capabilities,
    ):
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-arch", value=str(ttcore.Arch(chip_desc.arch.arch_as_int))
            )
        )
        result.append(
            graph_builder.KeyValue(
                key=f"chip#{i}-capability",
                value=str(ttcore.ChipCapability(chip_capability.capability_as_int)),
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
                        str(ttcore.DataType(dt.data_type_as_int))
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

    if memory_config.shard_spec:
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
                ttnn.TensorMemoryLayout(int(memory_config.tensor_memory_layout.value))
            ),
        )
    )

    return result


@AttrHandler.register_handler("force")
def parse_force(attr):
    return [graph_builder.KeyValue(key="force", value=str(attr.value))]


@AttrHandler.register_handler("dtype")
def parse_dtype(attr):
    dtype = ttcore.ir.DataTypeAttr.maybe_downcast(attr)
    if dtype is None:
        # Potential for dtype to be StringAttr instead of ttcore.DataTypeAttr
        return [graph_builder.KeyValue(key="dtype", value=str(attr))]
    return [
        graph_builder.KeyValue(
            key="dtype", value=str(ttcore.DataType(dtype.data_type_as_int))
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


@AttrHandler.register_handler("ttcore.layout")
def parse_tt_layout(attr):
    layout = ttcore.ir.MetalLayoutAttr.maybe_downcast(attr)
    result = []
    result.append(graph_builder.KeyValue(key="linear", value=str(layout.linear)))
    result.append(
        graph_builder.KeyValue(
            key="memory_space",
            value=str(ttcore.MemorySpace(layout.memory_space_as_int)),
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
    tile_type = ttcore.ir.TileType.maybe_downcast(layout.memref.element_type)
    if tile_type is not None:
        result.append(
            graph_builder.KeyValue(
                key="tile_datatype",
                value=str(ttcore.DataType(tile_type.data_type_as_int)),
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
                value=str(ttcore.DataType(layout.data_type_as_int)),
            ),
            editable={
                "input_type": "value_list",
                "options": [str(o) for o in ttcore.DataType],
            },
        )
    )

    return result


@AttrHandler.register_handler("conv2d_config")
def parse_conv2d_config(attr):
    conv2d_config = ttnn.ir.Conv2dConfigAttr.maybe_downcast(attr)
    result = []
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="weights_dtype",
                value=str(ttcore.DataType(conv2d_config.weights_dtype_as_int)),
            ),
            editable={
                "input_type": "value_list",
                "options": [str(o) for o in ttcore.DataType],
            },
        )
    )
    activation = (
        str(conv2d_config.activation.op_type)
        if conv2d_config.activation is not None
        else OVERRIDE_PARAMETER_DISABLED_STR
    )
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="activation",
                value=activation,
            ),
            editable={
                "input_type": "value_list",
                "options": ["relu", OVERRIDE_PARAMETER_DISABLED_STR],
            },
        )
    )
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="deallocate_activation",
                value=str(conv2d_config.deallocate_activation),
            ),
            editable={
                "input_type": "value_list",
                "options": ["True", "False"],
            },
        )
    )
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="reallocate_halo_output",
                value=str(conv2d_config.reallocate_halo_output),
            ),
            editable={
                "input_type": "value_list",
                "options": ["True", "False"],
            },
        )
    )
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="act_block_h_override",
                value=str(conv2d_config.act_block_h_override),
            ),
            editable={
                "input_type": "int_list",
                "min_value": 0,
                "step": 32,
            },
        )
    )
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="act_block_w_div",
                value=str(conv2d_config.act_block_w_div),
            ),
            editable={"input_type": "int_list", "min_value": 1, "step": 1},
        )
    )
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="reshard_if_not_optimal",
                value=str(conv2d_config.reshard_if_not_optimal),
            ),
            editable={
                "input_type": "value_list",
                "options": ["True", "False"],
            },
        )
    )
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="override_sharding_config",
                value=str(conv2d_config.override_sharding_config),
            ),
            editable={
                "input_type": "value_list",
                "options": ["True", "False"],
            },
        )
    )
    shard_layout = OVERRIDE_PARAMETER_DISABLED_STR
    if conv2d_config.shard_layout_as_int:
        shard_layout = str(ttnn.TensorMemoryLayout(conv2d_config.shard_layout_as_int))
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="shard_layout",
                value=shard_layout,
            ),
            editable={
                "input_type": "value_list",
                "options": [str(o) for o in ttnn.TensorMemoryLayout]
                + [OVERRIDE_PARAMETER_DISABLED_STR],
            },
        )
    )
    result.append(
        graph_builder.KeyValue(
            key="core_grid",
            value=str(conv2d_config.core_grid),
        )
    )
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="transpose_shards",
                value=str(conv2d_config.transpose_shards),
            ),
            editable={
                "input_type": "value_list",
                "options": ["True", "False"],
            },
        )
    )
    output_layout = OVERRIDE_PARAMETER_DISABLED_STR
    if conv2d_config.output_layout_as_int is not None:
        output_layout = str(ttnn.Layout(conv2d_config.output_layout_as_int))
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="output_layout",
                value=output_layout,
            ),
            editable={
                "input_type": "value_list",
                "options": [str(o) for o in ttnn.Layout]
                + [OVERRIDE_PARAMETER_DISABLED_STR],
            },
        )
    )
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="enable_act_double_buffer",
                value=str(conv2d_config.enable_act_double_buffer),
            ),
            editable={
                "input_type": "value_list",
                "options": ["True", "False"],
            },
        )
    )
    result.append(
        utils.make_editable_kv(
            graph_builder.KeyValue(
                key="enable_weights_double_buffer",
                value=str(conv2d_config.enable_weights_double_buffer),
            ),
            editable={
                "input_type": "value_list",
                "options": ["True", "False"],
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

        if util.is_fused_loc(self.op.location):
            self.locations = util.get_fused_locations(self.op.location)
            for loc in self.locations:
                # Use the first locations to fit the bill for "legacy" location support.
                if util.is_name_loc(loc):
                    self.named_location = util.get_loc_name(loc)
                elif util.is_file_line_col_loc(loc):
                    self.full_location = util.get_loc_full(loc)
        else:
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

        # Fused Loc Logic
        if util.is_fused_loc(op.location):
            locs = util.get_fused_locations(op.location)
            for loc in locs:
                if util.is_name_loc(loc):
                    name = util.get_loc_name(loc)

        name = name or ""

        # Don't process returns since they should be on the bottom of the graph
        if op.name == "func.return":
            return ""

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
                            output_tensor.type.encoding.get_named("ttcore.layout")
                        )
                    )
            result.extend(output_attrs)

        # Add schedule as an attribute
        result.append(
            graph_builder.KeyValue(key="schedule", value=str(OpHandler.schedule))
        )
        OpHandler.schedule += 1

        return result

    def make_graph_node(self, node_properties):
        extra_attrs = node_properties.get("extra_attrs", None)
        pin_to_group_top = node_properties.get("pinToGroupTop", False)
        parent_namespace = node_properties.get("namespace", "")

        # Handle attributes
        attrs = self.get_attributes()
        if extra_attrs is not None:
            attrs.extend(extra_attrs)

        # Handle Graph node configs
        config = graph_builder.GraphNodeConfig(pinToGroupTop=pin_to_group_top)

        # Handle namespace
        self.namespace = self.get_namespace()
        if parent_namespace:
            self.namespace = (
                parent_namespace + "/" + self.namespace
                if self.namespace
                else parent_namespace
            )

        return graph_builder.GraphNode(
            id=self.id,
            label=str(self.op.name),
            namespace=self.namespace,
            attrs=attrs,
            config=config,
        )


EMPTY_OPS = [
    "ttnn.empty",
    "ttir.empty",
]

FILTERED_OPS = [
    "ttnn.deallocate",
    "ttnn.get_device",
    *EMPTY_OPS,
]


class GraphHandler:
    def __init__(self):
        self.output_connections = defaultdict(int)
        self.op_to_graph_node = {}
        self.operands_in_graph = set()
        self.perf_node_data = {}
        self.loc_to_perf = {}
        self.memory_data = {}
        self.accuracy_node_data = {}
        self.loc_to_accuracy = {}
        self.processed_locs = set()
        self.global_counter = 0
        self.namespace_counter = {}

    def build_graph(
        self,
        model_path: str,
        module,
        model_runner,
        perf_trace=None,
        memory_trace=None,
        golden_results=None,
        cpp_code=None,
    ):

        self.model_path = model_path
        self.module = module
        self.model_runner = model_runner

        self.graph_id = Path(model_path).name

        if model_runner.get_last_run(model_path):
            self.graph_id = f'{self.graph_id} - Execution {datetime.now(timezone.utc).strftime("%H:%M:%S")}'

        self.graph = graph_builder.Graph(id=self.graph_id)

        # Prepare perf data for color overlay
        print(f"DEBUG [GraphHandler.build_graph]: perf_trace is {'None' if perf_trace is None else f'provided with {len(perf_trace)} rows'}")
        if perf_trace is not None:
            for _, row in perf_trace.iterrows():
                loc = parse_loc_string(row["LOC"])
                if not loc:
                    continue
                # Force the full location here=,
                loc = row["LOC"]
                if loc not in self.loc_to_perf:
                    self.loc_to_perf[loc] = 0
                self.loc_to_perf[loc] += row["DEVICE FW DURATION [ns]"]
            print(f"DEBUG [GraphHandler.build_graph]: Processed perf_trace, loc_to_perf has {len(self.loc_to_perf)} entries")

        if memory_trace is not None:
            for node in memory_trace:
                loc = memory_trace[node]["loc"]
                self.memory_data[loc] = {}
                self.memory_data[loc]["dram"] = round(
                    memory_trace[node]["dram"]["total_bytes_allocated_per_bank"]
                    / memory_trace[node]["dram"]["total_bytes_per_bank"],
                    4,
                )
                self.memory_data[loc]["l1"] = round(
                    memory_trace[node]["l1"]["total_bytes_allocated_per_bank"]
                    / memory_trace[node]["l1"]["total_bytes_per_bank"],
                    4,
                )
                self.memory_data[loc]["l1_small"] = round(
                    memory_trace[node]["l1_small"]["total_bytes_allocated_per_bank"]
                    / memory_trace[node]["l1_small"]["total_bytes_per_bank"],
                    4,
                )

        if golden_results is not None:
            for loc, res in golden_results.items():
                _loc = parse_loc_string(loc)
                if not _loc:
                    continue
                if loc in self.loc_to_accuracy:
                    logging.error("Double locations presented in golden_results")
                    raise IndexError("Double locations present in golden_results")
                # Store the full result here, just need to parse the loc accordingly
                self.loc_to_accuracy[loc] = res

        module_op = OpHandler(self.module.operation)
        module_attrs = module_op.get_attributes()
        module_attrs = dict((attr.key, attr.value) for attr in module_attrs)

        # Add module attributes to the graph as "namespace attributes"
        if not self.graph.groupNodeAttributes:
            self.graph.groupNodeAttributes = {}

        # Add this module's namespace attributes
        namespace = module_op.get_namespace()
        if namespace not in self.graph.groupNodeAttributes:
            self.graph.groupNodeAttributes[namespace] = module_attrs
        else:
            # Merge with existing attributes if namespace already exists
            self.graph.groupNodeAttributes[namespace].update(module_attrs)

        self.module_to_json(self.module)

        # Check if all perf locations match some graph node
        for loc in self.loc_to_perf.keys():
            if loc not in self.processed_locs:
                logging.error(f"Perf location {loc} not found in graph nodes")

        # Add Overlay Data if it exists
        overlays = {}

        print(f"DEBUG [GraphHandler.build_graph]: self.perf_node_data length = {len(self.perf_node_data) if self.perf_node_data else 0}")
        print(f"DEBUG [GraphHandler.build_graph]: self.accuracy_node_data length = {len(self.accuracy_node_data) if self.accuracy_node_data else 0}")
        print(f"DEBUG [GraphHandler.build_graph]: self.loc_to_perf length = {len(self.loc_to_perf) if self.loc_to_perf else 0}")

        # Add performance data to the graph color overlay, if it exists
        if self.perf_node_data:
            print("DEBUG [GraphHandler.build_graph]: Adding perf_data to overlays")
            gradient = [
                node_data_builder.GradientItem(stop=0, bgColor="yellow"),
                node_data_builder.GradientItem(stop=1, bgColor="red"),
            ]
            graph_node_data = node_data_builder.GraphNodeData(
                results=self.perf_node_data, gradient=gradient
            )
            overlays["perf_data"] = node_data_builder.ModelNodeData(
                graphsData={self.graph_id: graph_node_data}
            ).graphsData
        else:
            print("DEBUG [GraphHandler.build_graph]: NOT adding perf_data - perf_node_data is empty!")

        if self.accuracy_node_data:
            thres = [
                # Show Red if ActualPCC - ExpectedPCC is 0 and below (ActualPCC < ExpectedPCC)
                node_data_builder.ThresholdItem(value=0, bgColor="red"),
                # Show Green if ActualPCC - ExpectedPCC is 1 and below (Actual PCC >= ExpectedPCC)
                node_data_builder.ThresholdItem(value=1, bgColor="green"),
            ]
            graph_node_data = node_data_builder.GraphNodeData(
                results=self.accuracy_node_data, thresholds=thres
            )
            overlays["accuracy_data"] = node_data_builder.ModelNodeData(
                graphsData={self.graph_id: graph_node_data}
            ).graphsData

        OpHandler.schedule = 0
        return self.graph, overlays

    def module_to_json(self, module):
        if isinstance(module, ir.Module):
            operations = module.body.operations
        else:
            operations = module.regions[0].blocks[0].operations

        for operation in operations:
            if utils.is_nested_module(operation):
                self.module_to_json(operation)
                continue
            if isinstance(operation, func.FuncOp):
                self.process_function(operation)

    def process_function(self, func):
        for region in func.regions:
            self.append_after = []
            self.process_region(region)
            for node in self.append_after:
                self.graph.nodes.append(node)

    def process_region(self, region, namespace=""):
        # For each operation in the region/block, recursively process
        for block in region.blocks:
            block_args = []
            for op in block.operations:
                node_properties = {"namespace": namespace, "pinToGroupTop": False}
                # Handle Nested ops
                if op.regions and op.name:
                    new_namespace = self.generate_new_namespace(op.name)
                    node_properties["namespace"] = (
                        namespace + "/" + new_namespace
                        if namespace != ""
                        else new_namespace
                    )
                    node_properties["pinToGroupTop"] = True
                    self.process_operation(op, node_properties, block_args)
                    for subregion in op.regions:
                        self.process_region(subregion, node_properties["namespace"])
                # Non-nested op
                else:
                    self.process_operation(op, node_properties, block_args)
            self.process_args(block_args, namespace)
            self.add_edges(block)

    def process_operation(self, op, node_properties, block_args):

        # Create graph node for this operation
        operation = OpHandler(op)
        self.processed_locs.add(operation.full_location)

        if (
            operation.full_location in self.loc_to_perf
            and operation.op.name not in EMPTY_OPS
        ):
            self.perf_node_data[operation.id] = node_data_builder.NodeDataResult(
                self.loc_to_perf[operation.full_location]
            )

        if (
            operation.full_location in self.loc_to_accuracy
            and operation.op.name not in EMPTY_OPS
        ):
            self.accuracy_node_data[operation.id] = node_data_builder.NodeDataResult(
                self.loc_to_accuracy[operation.full_location]["actual_pcc"]
                - self.loc_to_accuracy[operation.full_location]["expected_pcc"]
            )

        extra_attrs = []
        if self.memory_data and operation.full_location in self.memory_data:
            extra_attrs.append(
                utils.add_to_dataclass(
                    graph_builder.KeyValue(
                        key="dram_memory",
                        value=str(self.memory_data[operation.full_location]["dram"]),
                    ),
                    "display_type",
                    "memory",
                )
            )
            extra_attrs.append(
                utils.add_to_dataclass(
                    graph_builder.KeyValue(
                        key="l1_memory",
                        value=str(self.memory_data[operation.full_location]["l1"]),
                    ),
                    "display_type",
                    "memory",
                )
            )
            extra_attrs.append(
                utils.add_to_dataclass(
                    graph_builder.KeyValue(
                        key="l1_small_memory",
                        value=str(
                            self.memory_data[operation.full_location]["l1_small"]
                        ),
                    ),
                    "display_type",
                    "memory",
                )
            )

        node_properties["extra_attrs"] = extra_attrs
        graph_node = operation.make_graph_node(node_properties)

        if op.name not in FILTERED_OPS and op.name in EMPTY_OPS:
            self.append_later.append(graph_node)
        elif op.name not in FILTERED_OPS:
            self.graph.nodes.append(graph_node)

        self.op_to_graph_node[op] = graph_node

        for operand in op.operands:
            if isinstance(operand, ir.Value) and not isinstance(
                operand.owner, ir.Operation
            ):
                if operand not in self.operands_in_graph:
                    block_args.append(operand)

    def process_args(self, args, namespace):
        name = "input"
        namespace = namespace + "/Inputs" if namespace else "Inputs"
        for arg in args:
            if arg not in self.operands_in_graph:
                new_id = name + "_" + str(self.global_counter)
                arg_node = graph_builder.GraphNode(
                    id=new_id,
                    label=str(arg.get_name()),
                    namespace=namespace,
                )
                self.graph.nodes.append(arg_node)
                self.op_to_graph_node[arg] = arg_node
                self.operands_in_graph.add(arg)
                self.global_counter += 1

    def add_edges(self, block):
        for op in block.operations:
            # Create edges for this operation
            for operand_index, operand in enumerate(op.operands):
                if operand.owner == block:
                    source_node = self.op_to_graph_node[operand]
                else:
                    source_node = self.op_to_graph_node[operand.owner]

                target_node = self.op_to_graph_node[op]

                target_node.incomingEdges.append(
                    graph_builder.IncomingEdge(
                        sourceNodeId=source_node.id,
                        sourceNodeOutputId=str(self.output_connections[source_node.id]),
                        targetNodeInputId=str(operand_index),
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
                                operand.type.encoding.get_named("ttcore.layout")
                            )
                        )
                source_node.outputsMetadata.append(
                    graph_builder.MetadataItem(
                        id=str(self.output_connections[source_node.id]),
                        attrs=[
                            graph_builder.KeyValue(
                                key="__tensor_tag", value=str(target_node.label)
                            ),
                        ]
                        + output_attrs,
                    )
                )
                self.output_connections[source_node.id] += 1

    def generate_new_namespace(self, base_name):
        counter = self.namespace_counter.get(base_name, 0)
        self.namespace_counter[base_name] = counter + 1
        return f"{base_name}_{counter}"
