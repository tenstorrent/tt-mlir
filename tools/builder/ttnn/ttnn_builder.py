# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Callable, Any
import torch

from ttmlir.ir import *
from ttmlir import util
from ttmlir.dialects import ttnn, ttcore, func

from builder.base.builder import *
from builder.base.builder_utils import *

from golden import *


class TTNNBuilder(Builder):

    # ----- Methods -----

    def __init__(
        self,
        ctx: Context,
        location: Location,
        mesh_name: Union[List[str], str] = "mesh",
        mesh_dict: Union[
            List[OrderedDict[str, int]], OrderedDict[str, int]
        ] = OrderedDict([("x", 1), ("y", 1)]),
        disable_golden_check: bool = False,
    ):
        super().__init__(ctx, location, mesh_name, mesh_dict, disable_golden_check)

    # ----- Private Methods ----

    def _organize_eltwise_ttnn(
        self,
        inputs: List[Operand],
        ttnn_tensor: RankedTensorType,
    ):
        return (ttnn_tensor, *inputs)

    def _op_proxy(
        self,
        op_ttnn_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
        organize_ttnn_args: Optional[Callable] = None,
        organize_golden_args: Optional[Callable] = None,
        output_shape: Optional[Shape] = None,
        output_type: Optional[Type] = None,
        output_create_fn: Optional[Callable] = None,
        golden_kwargs: dict = {},
        ttnn_kwargs: dict = {},
        loc: Optional[Union[str, Location]] = None,
        skip_golden: bool = False,
    ) -> Any:
        if organize_ttnn_args is None:
            organize_ttnn_args = self._organize_eltwise_ttnn

        if organize_golden_args is None:
            organize_golden_args = self._organize_eltwise_golden

        with self._ctx, self._loc:
            # If output shape or type is not provided, calculate it using golden function.
            # This is needed because ttnn ops do not have shape or type MLIR inference trait.

            output_shape_and_type = self._get_output_shape_and_type(
                organize_golden_args, inputs, op_ttnn_function, golden_kwargs
            )
            if not output_shape_and_type:
                assert (
                    output_shape is not None
                ), "Output shape must be provided if there is no golden function for this op"
                assert (
                    output_type is not None
                ), "Output type must be provided if there is no golden function for this op"
            else:
                (
                    calculated_output_shape,
                    calculated_output_type,
                ) = output_shape_and_type

            # Use provided output shape/type if available, otherwise use calculated ones.
            output_shape = calculated_output_shape if not output_shape else output_shape
            output_type = (
                self._get_type_from_torch_dtype(calculated_output_type)
                if not output_type
                else output_type
            )

            # Create output tensor using provided function or create empty tensor.
            result_tensor = self.create_ttnn_tensor(output_shape, output_type)

            # Prepare location for the op.
            id = self._get_next_global_id()
            loc = (
                self._get_loc_from_str(loc)
                if loc is not None
                else self._get_loc_of_extra_file_callee(id=id)
            )

            # Organize arguments and create the ttnn op.
            if organize_ttnn_args(inputs, result_tensor) == 0:
                op = op_ttnn_function(
                    loc=loc,
                    **ttnn_kwargs,
                )
            else:
                op = op_ttnn_function(
                    *organize_ttnn_args(inputs, result_tensor),
                    loc=loc,
                    **ttnn_kwargs,
                )

            # Set unit attributes if provided.
            if unit_attrs is not None:
                for attr_name in unit_attrs:
                    op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

            if not skip_golden and not self._disable_golden_check:
                op_golden_function = get_golden_function(
                    op_ttnn_function, **golden_kwargs
                )
                if op_golden_function is not None:
                    if len(inputs) == 0:
                        golden_output = op_golden_function(**golden_kwargs)
                    else:
                        golden_output = op_golden_function(
                            *(organize_golden_args(inputs)), **golden_kwargs
                        )
                    self._set_golden_tensor(op.result, golden_output)

            return op.result

    def _get_data_type_attribute(self, operand: Operand) -> ttcore.ir.DataTypeAttr:
        with self._ctx, self._loc:
            dtype = ttnn.ir.TTNNLayoutAttr.maybe_downcast(
                self._get_type(operand).encoding
            ).data_type_as_int
            return ttcore.ir.DataTypeAttr.get(self._ctx, dtype)

    # ----- Public Helper Methods ----

    def create_tensor_encoding(
        self, shape: Shape, element_type: Union[torch.dtype, TypeInfo]
    ) -> ttnn.ir.TTNNLayoutAttr:
        """
        TTNN tensors require that encoding information is present.
        This method creates a TTNN tensor with encoding information.
        For simplicity we will always create DRAM/Interlaved tiled tensor.
        """
        if isinstance(element_type, torch.dtype):
            element_type = self._get_type_from_torch_dtype(element_type)
        with self._ctx, self._loc:
            data_type = util.element_type_to_data_type(element_type)
            tile_element_type = ttcore.ir.TileType.get(self._ctx, 32, 32, data_type)
            buffer_type = ttnn.BufferType.DRAM
            grid_attr = ttcore.ir.GridAttr.get(self._ctx, [1, 1])
            ttnn_layout_attr = ttnn.ir.TTNNLayoutAttr.get(
                self._ctx,
                shape,
                tile_element_type,
                buffer_type,
                grid_attr,
                ttnn.TensorMemoryLayout.Interleaved,
            )
            return ttnn_layout_attr

    def create_ttnn_tensor(self, shape: Shape, element_type: Type) -> RankedTensorType:
        """
        TTNN tensors require that encoding information is present.
        This method creates a TTNN tensor with encoding information.
        For simplicity we will always create DRAM/Interlaved tiled tensor.
        """
        with self._ctx, self._loc:
            ttnn_layout_attr = self.create_tensor_encoding(shape, element_type)
            return RankedTensorType.get(shape, element_type, ttnn_layout_attr)

    def create_l1_width_sharded_tiled_encoding(
        self, shape: Shape, element_type: Type
    ) -> ttnn.ir.TTNNLayoutAttr:
        with self._ctx, self._loc:
            data_type = util.element_type_to_data_type(element_type)
            tile_element_type = ttcore.ir.TileType.get(self._ctx, 32, 32, data_type)
            buffer_type = ttnn.BufferType.L1
            grid_attr = ttcore.ir.GridAttr.get(self._ctx, [1, 1])
            return ttnn.ir.TTNNLayoutAttr.get(
                self._ctx,
                shape,
                tile_element_type,
                buffer_type,
                grid_attr,
                ttnn.TensorMemoryLayout.WidthSharded,
            )

    def create_l1_width_sharded_tiled_ttnn_tensor(
        self, shape: Shape, element_type: Type
    ) -> RankedTensorType:
        with self._ctx, self._loc:
            ttnn_layout_attr = self.create_l1_width_sharded_tiled_encoding(
                shape, element_type
            )
            return RankedTensorType.get(shape, element_type, ttnn_layout_attr)

    def _get_location(self) -> Location:
        stack = inspect.stack()
        caller_frame = stack[2]
        filename = caller_frame.filename
        lineno = caller_frame.lineno
        return Location.name(f"{filename}:{lineno}")

    # ----- TTCore Op Parsers ----

    @parse(ttcore.LoadCachedOp)
    def load_cached_parser(
        self,
        old_op: ttcore.LoadCachedOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """
        Parser for ttcore.LoadCachedOp.

        This op calls a precomputed function with given arguments and returns
        its results. It is typically used to load constant or hoisted computation
        results.
        """
        callee_name = old_op.callee.value
        nested_func_op = self._func_name_to_op[callee_name]

        # Get golden inputs for the operands
        new_golden_inputs = []
        if not self._disable_golden_check:
            for operand in old_op.inputs:
                owner = operand.owner
                if isinstance(owner, Block):
                    queried_operand = operand
                else:
                    queried_operand = owner.result
                new_golden_inputs.append(
                    self._get_golden_tensor(global_dict[queried_operand])
                )

        # Parse the nested function
        with InsertionPoint(self._current_module_insertion_point):
            new_func_op = self.parse_nested_func(nested_func_op, new_golden_inputs)

        # Map operands to new operands
        new_operands = [global_dict[operand] for operand in old_op.inputs]

        # Get result types from old op
        result_types = [result.type for result in old_op.results_]

        # Create new LoadCachedOp
        new_op = ttcore.LoadCachedOp(
            results_=result_types,
            callee=new_func_op.name.value,
            inputs=new_operands,
            loc=old_op.location,
        )

        # Set goldens for new results
        if not self._disable_golden_check:
            ordered_inputs, ordered_outputs = self._func_ops_generated[new_func_op]
            for index, output in enumerate(ordered_outputs):
                self._set_golden_tensor(
                    new_op.results_[index], self._get_golden_tensor(output)
                )

        # Map old results to new results
        op_map_dictionary = {}
        for old_result, new_result in zip(old_op.results_, new_op.results_):
            op_map_dictionary[old_result] = new_result

        return new_op, op_map_dictionary

    # ----- Device Op Generators and Parsers ----

    @tag(ttnn.GetDeviceOp)
    def get_device(
        self,
        mesh_shape: Optional[List[int]] = None,
        mesh_offset: Optional[List[int]] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.GetDeviceOp.

        This op returns a device handle with optional mesh_shape and mesh_offset
        attributes that define the submesh.
        """
        with self._ctx, self._loc:
            # Create result type for device
            result_type = ttnn.ir.DeviceType.get(self._ctx)

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            # Build optional attributes
            mesh_shape_attr = None
            if mesh_shape is not None:
                mesh_shape_attr = ttnn.ir.MeshShapeAttr.get(self._ctx, mesh_shape)

            mesh_offset_attr = None
            if mesh_offset is not None:
                mesh_offset_attr = ttnn.ir.MeshShapeAttr.get(self._ctx, mesh_offset)

            op = ttnn.GetDeviceOp(
                results=[result_type],
                mesh_shape=mesh_shape_attr,
                mesh_offset=mesh_offset_attr,
                loc=loc,
            )

            return op.result

    @parse(ttnn.GetDeviceOp)
    def get_device_parser(
        self,
        old_op: ttnn.GetDeviceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """
        Parser for ttnn.GetDeviceOp.

        This op takes no operands and returns a device handle. It has optional
        mesh_shape and mesh_offset attributes that define the submesh.
        """
        result_type = old_op.result.type

        # Create the new GetDeviceOp with the same attributes
        # Note: GetDeviceOp uses keyword-only arguments
        new_op = ttnn.GetDeviceOp(
            results=[result_type],
            mesh_shape=old_op.mesh_shape,
            mesh_offset=old_op.mesh_offset,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.GetDeviceOp)
    def get_device_split(
        self,
        old_op: ttnn.GetDeviceOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.GetDeviceOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(name="get_device_module")
                def decorated_func():
                    result_type = old_op.result.type
                    new_op = ttnn.GetDeviceOp(
                        results=[result_type],
                        mesh_shape=old_op.mesh_shape,
                        mesh_offset=old_op.mesh_offset,
                        loc=old_op.location,
                    )
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.ConstantOp)
    def constant(
        self,
        value: Attribute,
        shape: List[int],
        element_type: Type,
        device: Optional[Operand] = None,
        dtype: Optional[Attribute] = None,
        layout: Optional[Attribute] = None,
        memory_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.ConstantOp.

        This op creates a tensor filled with a constant value.
        """
        with self._ctx, self._loc:
            result_type = self.create_ttnn_tensor(shape, element_type)

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.ConstantOp(
                result_type,
                value,
                device=device,
                dtype=dtype,
                layout=layout,
                memory_config=memory_config,
                loc=loc,
            )

            return op.result

    @parse(ttnn.ConstantOp)
    def constant_parser(
        self,
        old_op: ttnn.ConstantOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """
        Parser for ttnn.ConstantOp.

        This op creates a tensor filled with a constant value.
        """
        result_type = old_op.result.type

        # Get the device operand if present
        device = None
        if old_op.device is not None:
            device = global_dict[old_op.device]

        new_op = ttnn.ConstantOp(
            result_type,
            old_op.value,
            device=device,
            dtype=old_op.dtype,
            layout=old_op.layout,
            memory_config=old_op.memory_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.ConstantOp)
    def constant_split(
        self,
        old_op: ttnn.ConstantOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.ConstantOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = []
            if old_op.device is not None:
                op_input_types.append(old_op.device.type)

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="constant_module")
                def decorated_func(*inputs):
                    device = inputs[0] if len(inputs) > 0 else None
                    result_type = old_op.result.type

                    new_op = ttnn.ConstantOp(
                        result_type,
                        old_op.value,
                        device=device,
                        dtype=old_op.dtype,
                        layout=old_op.layout,
                        memory_config=old_op.memory_config,
                        loc=old_op.location,
                    )

                    if device is not None:
                        ordered_inputs.append(device)
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.FullOp)
    def full(
        self,
        shape: List[int],
        fill_value: float,
        element_type: Type,
        device: Optional[Operand] = None,
        dtype: Optional[Attribute] = None,
        layout: Optional[Attribute] = None,
        memory_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.FullOp.

        This op creates a tensor filled with a specified value.
        """
        with self._ctx, self._loc:
            result_type = self.create_ttnn_tensor(shape, element_type)
            shape_attr = ttnn.ir.ShapeAttr.get(self._ctx, shape)
            fill_value_attr = FloatAttr.get_f32(fill_value)

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.FullOp(
                result_type,
                shape_attr,
                fill_value_attr,
                device=device,
                dtype=dtype,
                layout=layout,
                memory_config=memory_config,
                loc=loc,
            )

            return op.result

    @parse(ttnn.FullOp)
    def full_parser(
        self,
        old_op: ttnn.FullOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """
        Parser for ttnn.FullOp.

        This op creates a tensor filled with a specified value.
        """
        result_type = old_op.result.type

        # Get the device operand if present
        device = None
        if old_op.device is not None:
            device = global_dict[old_op.device]

        new_op = ttnn.FullOp(
            result_type,
            old_op.shape,
            old_op.fill_value,
            device=device,
            dtype=old_op.dtype,
            layout=old_op.layout,
            memory_config=old_op.memory_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.FullOp)
    def full_split(
        self,
        old_op: ttnn.FullOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.FullOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = []
            if old_op.device is not None:
                op_input_types.append(old_op.device.type)

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="full_module")
                def decorated_func(*inputs):
                    device = inputs[0] if len(inputs) > 0 else None
                    result_type = old_op.result.type

                    new_op = ttnn.FullOp(
                        result_type,
                        old_op.shape,
                        old_op.fill_value,
                        device=device,
                        dtype=old_op.dtype,
                        layout=old_op.layout,
                        memory_config=old_op.memory_config,
                        loc=old_op.location,
                    )

                    if device is not None:
                        ordered_inputs.append(device)
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.ReshapeOp)
    def reshape(
        self,
        in0: Operand,
        shape: List[int],
        memory_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.ReshapeOp.

        This op reshapes an input tensor to a new shape.
        """
        with self._ctx, self._loc:
            element_type = self.get_type(in0)
            result_type = self.create_ttnn_tensor(shape, element_type)
            shape_attr = ttnn.ir.ShapeAttr.get(self._ctx, shape)

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.ReshapeOp(
                result_type,
                in0,
                shape_attr,
                memory_config=memory_config,
                loc=loc,
            )

            # Propagate golden if available
            if not self._disable_golden_check:
                input_golden = self._get_golden_tensor(in0)
                if input_golden is not None:
                    golden_output = input_golden.apply_shardwise(
                        lambda shard: shard.reshape(shape)
                    )
                    self._set_golden_tensor(op.result, golden_output)

            return op.result

    @parse(ttnn.ReshapeOp)
    def reshape_parser(
        self,
        old_op: ttnn.ReshapeOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """
        Parser for ttnn.ReshapeOp.

        This op reshapes an input tensor to a new shape.
        """
        result_type = old_op.result.type
        input_tensor = global_dict[old_op.input]

        new_op = ttnn.ReshapeOp(
            result_type,
            input_tensor,
            old_op.shape,
            memory_config=old_op.memory_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.ReshapeOp)
    def reshape_split(
        self,
        old_op: ttnn.ReshapeOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.ReshapeOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="reshape_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    result_type = old_op.result.type

                    new_op = ttnn.ReshapeOp(
                        result_type,
                        input_tensor,
                        old_op.shape,
                        memory_config=old_op.memory_config,
                        loc=old_op.location,
                    )

                    ordered_inputs.append(input_tensor)
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.DeallocateOp)
    def deallocate(
        self,
        in0: Operand,
        force: bool = False,
        loc: Optional[str] = None,
    ) -> None:
        """
        Creates ttnn.DeallocateOp.

        This op deallocates the memory for the input tensor.
        """
        with self._ctx, self._loc:
            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            force_attr = BoolAttr.get(force, self._ctx)

            ttnn.DeallocateOp(
                in0,
                force=force_attr,
                loc=loc,
            )

    @parse(ttnn.DeallocateOp)
    def deallocate_parser(
        self,
        old_op: ttnn.DeallocateOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.DeallocateOp."""
        input_tensor = global_dict[old_op.input]

        new_op = ttnn.DeallocateOp(
            input_tensor,
            force=old_op.force,
            loc=old_op.location,
        )

        return new_op, {}

    @split(ttnn.DeallocateOp)
    def deallocate_split(
        self,
        old_op: ttnn.DeallocateOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.DeallocateOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="deallocate_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]

                    new_op = ttnn.DeallocateOp(
                        input_tensor,
                        force=old_op.force,
                        loc=old_op.location,
                    )

                    ordered_inputs.append(input_tensor)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.FromDeviceOp)
    def from_device(
        self,
        in0: Operand,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.FromDeviceOp.

        This op transfers a tensor from device to host.
        """
        with self._ctx, self._loc:
            result_type = in0.type

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.FromDeviceOp(
                result_type,
                in0,
                loc=loc,
            )

            # Propagate golden if available
            if not self._disable_golden_check:
                input_golden = self._get_golden_tensor(in0)
                if input_golden is not None:
                    self._set_golden_tensor(op.result, input_golden)

            return op.result

    @parse(ttnn.FromDeviceOp)
    def from_device_parser(
        self,
        old_op: ttnn.FromDeviceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.FromDeviceOp."""
        result_type = old_op.result.type
        input_tensor = global_dict[old_op.input]

        new_op = ttnn.FromDeviceOp(
            result_type,
            input_tensor,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.FromDeviceOp)
    def from_device_split(
        self,
        old_op: ttnn.FromDeviceOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.FromDeviceOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="from_device_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    result_type = old_op.result.type

                    new_op = ttnn.FromDeviceOp(
                        result_type,
                        input_tensor,
                        loc=old_op.location,
                    )

                    ordered_inputs.append(input_tensor)
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.ToDeviceOp)
    def to_device(
        self,
        in0: Operand,
        device: Operand,
        memory_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.ToDeviceOp.

        This op transfers a tensor from host to device.
        """
        with self._ctx, self._loc:
            result_type = in0.type

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.ToDeviceOp(
                result_type,
                in0,
                device,
                memory_config=memory_config,
                loc=loc,
            )

            # Propagate golden if available
            if not self._disable_golden_check:
                input_golden = self._get_golden_tensor(in0)
                if input_golden is not None:
                    self._set_golden_tensor(op.result, input_golden)

            return op.result

    @parse(ttnn.ToDeviceOp)
    def to_device_parser(
        self,
        old_op: ttnn.ToDeviceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.ToDeviceOp."""
        result_type = old_op.result.type
        input_tensor = global_dict[old_op.input]
        device = global_dict[old_op.device]

        new_op = ttnn.ToDeviceOp(
            result_type,
            input_tensor,
            device,
            memory_config=old_op.memory_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.ToDeviceOp)
    def to_device_split(
        self,
        old_op: ttnn.ToDeviceOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.ToDeviceOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type, old_op.device.type]

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="to_device_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    device = inputs[1]
                    result_type = old_op.result.type

                    new_op = ttnn.ToDeviceOp(
                        result_type,
                        input_tensor,
                        device,
                        memory_config=old_op.memory_config,
                        loc=old_op.location,
                    )

                    ordered_inputs.extend([input_tensor, device])
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.ToLayoutOp)
    def to_layout(
        self,
        in0: Operand,
        layout: Attribute,
        dtype: Optional[Attribute] = None,
        memory_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.ToLayoutOp.

        This op converts a tensor to a specified layout.
        """
        with self._ctx, self._loc:
            result_type = in0.type

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.ToLayoutOp(
                result_type,
                in0,
                layout,
                dtype=dtype,
                memory_config=memory_config,
                loc=loc,
            )

            # Propagate golden if available
            if not self._disable_golden_check:
                input_golden = self._get_golden_tensor(in0)
                if input_golden is not None:
                    self._set_golden_tensor(op.result, input_golden)

            return op.result

    @parse(ttnn.ToLayoutOp)
    def to_layout_parser(
        self,
        old_op: ttnn.ToLayoutOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.ToLayoutOp."""
        result_type = old_op.result.type
        input_tensor = global_dict[old_op.input]

        new_op = ttnn.ToLayoutOp(
            result_type,
            input_tensor,
            old_op.layout,
            dtype=old_op.dtype,
            memory_config=old_op.memory_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.ToLayoutOp)
    def to_layout_split(
        self,
        old_op: ttnn.ToLayoutOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.ToLayoutOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="to_layout_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    result_type = old_op.result.type

                    new_op = ttnn.ToLayoutOp(
                        result_type,
                        input_tensor,
                        old_op.layout,
                        dtype=old_op.dtype,
                        memory_config=old_op.memory_config,
                        loc=old_op.location,
                    )

                    ordered_inputs.append(input_tensor)
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.ToMemoryConfigOp)
    def to_memory_config(
        self,
        in0: Operand,
        memory_config: Attribute,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.ToMemoryConfigOp.

        This op converts a tensor to a specified memory configuration.
        """
        with self._ctx, self._loc:
            result_type = in0.type

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.ToMemoryConfigOp(
                result_type,
                in0,
                memory_config,
                loc=loc,
            )

            # Propagate golden if available
            if not self._disable_golden_check:
                input_golden = self._get_golden_tensor(in0)
                if input_golden is not None:
                    self._set_golden_tensor(op.result, input_golden)

            return op.result

    @parse(ttnn.ToMemoryConfigOp)
    def to_memory_config_parser(
        self,
        old_op: ttnn.ToMemoryConfigOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.ToMemoryConfigOp."""
        result_type = old_op.result.type
        input_tensor = global_dict[old_op.input]

        new_op = ttnn.ToMemoryConfigOp(
            result_type,
            input_tensor,
            old_op.memory_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.ToMemoryConfigOp)
    def to_memory_config_split(
        self,
        old_op: ttnn.ToMemoryConfigOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.ToMemoryConfigOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="to_memory_config_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    result_type = old_op.result.type

                    new_op = ttnn.ToMemoryConfigOp(
                        result_type,
                        input_tensor,
                        old_op.memory_config,
                        loc=old_op.location,
                    )

                    ordered_inputs.append(input_tensor)
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.PermuteOp)
    def permute(
        self,
        in0: Operand,
        permutation: List[int],
        memory_config: Optional[Attribute] = None,
        pad_value: Optional[float] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.PermuteOp.

        This op permutes the dimensions of the input tensor.
        """
        with self._ctx, self._loc:
            # Calculate output shape based on permutation
            input_shape = list(self._get_type(in0).shape)
            output_shape = [input_shape[i] for i in permutation]
            element_type = self.get_type(in0)
            result_type = self.create_ttnn_tensor(output_shape, element_type)

            permutation_attr = DenseI64ArrayAttr.get(permutation, self._ctx)
            pad_value_attr = None
            if pad_value is not None:
                pad_value_attr = FloatAttr.get_f32(pad_value)

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.PermuteOp(
                result_type,
                in0,
                permutation_attr,
                memory_config=memory_config,
                pad_value=pad_value_attr,
                loc=loc,
            )

            # Propagate golden if available
            if not self._disable_golden_check:
                input_golden = self._get_golden_tensor(in0)
                if input_golden is not None:
                    golden_output = input_golden.apply_shardwise(
                        lambda shard: shard.permute(permutation)
                    )
                    self._set_golden_tensor(op.result, golden_output)

            return op.result

    @parse(ttnn.PermuteOp)
    def permute_parser(
        self,
        old_op: ttnn.PermuteOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.PermuteOp."""
        result_type = old_op.result.type
        input_tensor = global_dict[old_op.input]

        new_op = ttnn.PermuteOp(
            result_type,
            input_tensor,
            old_op.permutation,
            memory_config=old_op.memory_config,
            pad_value=old_op.pad_value,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.PermuteOp)
    def permute_split(
        self,
        old_op: ttnn.PermuteOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.PermuteOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="permute_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    result_type = old_op.result.type

                    new_op = ttnn.PermuteOp(
                        result_type,
                        input_tensor,
                        old_op.permutation,
                        memory_config=old_op.memory_config,
                        pad_value=old_op.pad_value,
                        loc=old_op.location,
                    )

                    ordered_inputs.append(input_tensor)
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.ConcatenateHeadsOp)
    def concatenate_heads(
        self,
        in0: Operand,
        output_shape: List[int],
        memory_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.ConcatenateHeadsOp.

        This op concatenates attention heads back together.
        """
        with self._ctx, self._loc:
            element_type = self.get_type(in0)
            result_type = self.create_ttnn_tensor(output_shape, element_type)

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.ConcatenateHeadsOp(
                result_type,
                in0,
                memory_config=memory_config,
                loc=loc,
            )

            return op.result

    @parse(ttnn.ConcatenateHeadsOp)
    def concatenate_heads_parser(
        self,
        old_op: ttnn.ConcatenateHeadsOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.ConcatenateHeadsOp."""
        result_type = old_op.result.type
        input_tensor = global_dict[old_op.input]

        new_op = ttnn.ConcatenateHeadsOp(
            result_type,
            input_tensor,
            memory_config=old_op.memory_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.ConcatenateHeadsOp)
    def concatenate_heads_split(
        self,
        old_op: ttnn.ConcatenateHeadsOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.ConcatenateHeadsOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="concatenate_heads_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    result_type = old_op.result.type

                    new_op = ttnn.ConcatenateHeadsOp(
                        result_type,
                        input_tensor,
                        memory_config=old_op.memory_config,
                        loc=old_op.location,
                    )

                    ordered_inputs.append(input_tensor)
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.Conv2dOp)
    def conv2d(
        self,
        input_tensor: Operand,
        weight: Operand,
        device: Operand,
        in_channels: int,
        out_channels: int,
        batch_size: int,
        input_height: int,
        input_width: int,
        kernel_size: List[int],
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        groups: int,
        output_shape: List[int],
        bias: Optional[Operand] = None,
        dtype: Optional[Attribute] = None,
        conv2d_config: Optional[Attribute] = None,
        compute_config: Optional[Attribute] = None,
        conv2d_slice_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.Conv2dOp.

        This op performs 2D convolution.
        """
        with self._ctx, self._loc:
            element_type = self.get_type(input_tensor)
            result_type = self.create_ttnn_tensor(output_shape, element_type)

            in_channels_attr = IntegerAttr.get(
                IntegerType.get_unsigned(32), in_channels
            )
            out_channels_attr = IntegerAttr.get(
                IntegerType.get_unsigned(32), out_channels
            )
            batch_size_attr = IntegerAttr.get(IntegerType.get_unsigned(32), batch_size)
            input_height_attr = IntegerAttr.get(
                IntegerType.get_unsigned(32), input_height
            )
            input_width_attr = IntegerAttr.get(
                IntegerType.get_unsigned(32), input_width
            )
            kernel_size_attr = DenseI32ArrayAttr.get(kernel_size, self._ctx)
            stride_attr = DenseI32ArrayAttr.get(stride, self._ctx)
            padding_attr = DenseI32ArrayAttr.get(padding, self._ctx)
            dilation_attr = DenseI32ArrayAttr.get(dilation, self._ctx)
            groups_attr = IntegerAttr.get(IntegerType.get_unsigned(32), groups)

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.Conv2dOp(
                result_type,
                input_tensor,
                weight,
                device,
                in_channels_attr,
                out_channels_attr,
                batch_size_attr,
                input_height_attr,
                input_width_attr,
                kernel_size_attr,
                stride_attr,
                padding_attr,
                dilation_attr,
                groups_attr,
                bias=bias,
                dtype=dtype,
                conv2d_config=conv2d_config,
                compute_config=compute_config,
                conv2d_slice_config=conv2d_slice_config,
                loc=loc,
            )

            return op.result

    @parse(ttnn.Conv2dOp)
    def conv2d_parser(
        self,
        old_op: ttnn.Conv2dOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.Conv2dOp."""
        result_type = old_op.result.type
        input_tensor = global_dict[old_op.input]
        weight = global_dict[old_op.weight]
        device = global_dict[old_op.device]

        # Get optional bias
        bias = None
        if old_op.bias is not None:
            bias = global_dict[old_op.bias]

        new_op = ttnn.Conv2dOp(
            result_type,
            input_tensor,
            weight,
            device,
            old_op.in_channels,
            old_op.out_channels,
            old_op.batch_size,
            old_op.input_height,
            old_op.input_width,
            old_op.kernel_size,
            old_op.stride,
            old_op.padding,
            old_op.dilation,
            old_op.groups,
            bias=bias,
            dtype=old_op.dtype,
            conv2d_config=old_op.conv2d_config,
            compute_config=old_op.compute_config,
            conv2d_slice_config=old_op.conv2d_slice_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.Conv2dOp)
    def conv2d_split(
        self,
        old_op: ttnn.Conv2dOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.Conv2dOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type, old_op.weight.type, old_op.device.type]
            if old_op.bias is not None:
                op_input_types.append(old_op.bias.type)

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="conv2d_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    weight = inputs[1]
                    device = inputs[2]
                    bias = inputs[3] if len(inputs) > 3 else None
                    result_type = old_op.result.type

                    new_op = ttnn.Conv2dOp(
                        result_type,
                        input_tensor,
                        weight,
                        device,
                        old_op.in_channels,
                        old_op.out_channels,
                        old_op.batch_size,
                        old_op.input_height,
                        old_op.input_width,
                        old_op.kernel_size,
                        old_op.stride,
                        old_op.padding,
                        old_op.dilation,
                        old_op.groups,
                        bias=bias,
                        dtype=old_op.dtype,
                        conv2d_config=old_op.conv2d_config,
                        compute_config=old_op.compute_config,
                        conv2d_slice_config=old_op.conv2d_slice_config,
                        loc=old_op.location,
                    )

                    ordered_inputs.extend([input_tensor, weight, device])
                    if bias is not None:
                        ordered_inputs.append(bias)
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.MeanOp)
    def mean(
        self,
        in0: Operand,
        keep_dim: bool,
        output_shape: List[int],
        dim_arg: Optional[List[int]] = None,
        compute_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.MeanOp.

        This op computes the mean of the input tensor along specified dimensions.
        """
        with self._ctx, self._loc:
            element_type = self.get_type(in0)
            result_type = self.create_ttnn_tensor(output_shape, element_type)

            keep_dim_attr = BoolAttr.get(keep_dim, self._ctx)
            dim_arg_attr = None
            if dim_arg is not None:
                dim_arg_attr = DenseI32ArrayAttr.get(dim_arg, self._ctx)

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.MeanOp(
                result_type,
                in0,
                keep_dim_attr,
                dim_arg=dim_arg_attr,
                compute_config=compute_config,
                loc=loc,
            )

            # Compute golden if available
            if not self._disable_golden_check:
                input_golden = self._get_golden_tensor(in0)
                if input_golden is not None:
                    if dim_arg is not None:
                        golden_output = input_golden.apply_shardwise(
                            lambda shard: torch.mean(
                                shard, dim=dim_arg, keepdim=keep_dim
                            )
                        )
                    else:
                        golden_output = input_golden.apply_shardwise(
                            lambda shard: torch.mean(shard, keepdim=keep_dim)
                        )
                    self._set_golden_tensor(op.result, golden_output)

            return op.result

    @parse(ttnn.MeanOp)
    def mean_parser(
        self,
        old_op: ttnn.MeanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.MeanOp."""
        result_type = old_op.result.type
        input_tensor = global_dict[old_op.input]

        new_op = ttnn.MeanOp(
            result_type,
            input_tensor,
            old_op.keep_dim,
            dim_arg=old_op.dim_arg,
            compute_config=old_op.compute_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.MeanOp)
    def mean_split(
        self,
        old_op: ttnn.MeanOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.MeanOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="mean_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    result_type = old_op.result.type

                    new_op = ttnn.MeanOp(
                        result_type,
                        input_tensor,
                        old_op.keep_dim,
                        dim_arg=old_op.dim_arg,
                        compute_config=old_op.compute_config,
                        loc=old_op.location,
                    )

                    ordered_inputs.append(input_tensor)
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.PrepareConv2dBiasOp)
    def prepare_conv2d_bias(
        self,
        bias_tensor: Operand,
        device: Operand,
        input_memory_config: Attribute,
        input_tensor_layout: Attribute,
        in_channels: int,
        out_channels: int,
        batch_size: int,
        input_height: int,
        input_width: int,
        kernel_size: List[int],
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        groups: int,
        input_dtype: Attribute,
        output_shape: List[int],
        output_dtype: Optional[Attribute] = None,
        conv2d_config: Optional[Attribute] = None,
        compute_config: Optional[Attribute] = None,
        conv2d_slice_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.PrepareConv2dBiasOp.

        This op prepares the bias tensor for Conv2d operation.
        """
        with self._ctx, self._loc:
            element_type = self.get_type(bias_tensor)
            result_type = self.create_ttnn_tensor(output_shape, element_type)

            in_channels_attr = IntegerAttr.get(
                IntegerType.get_unsigned(32), in_channels
            )
            out_channels_attr = IntegerAttr.get(
                IntegerType.get_unsigned(32), out_channels
            )
            batch_size_attr = IntegerAttr.get(IntegerType.get_unsigned(32), batch_size)
            input_height_attr = IntegerAttr.get(
                IntegerType.get_unsigned(32), input_height
            )
            input_width_attr = IntegerAttr.get(
                IntegerType.get_unsigned(32), input_width
            )
            kernel_size_attr = DenseI32ArrayAttr.get(kernel_size, self._ctx)
            stride_attr = DenseI32ArrayAttr.get(stride, self._ctx)
            padding_attr = DenseI32ArrayAttr.get(padding, self._ctx)
            dilation_attr = DenseI32ArrayAttr.get(dilation, self._ctx)
            groups_attr = IntegerAttr.get(IntegerType.get_unsigned(32), groups)

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.PrepareConv2dBiasOp(
                result_type,
                bias_tensor,
                input_memory_config,
                input_tensor_layout,
                in_channels_attr,
                out_channels_attr,
                batch_size_attr,
                input_height_attr,
                input_width_attr,
                kernel_size_attr,
                stride_attr,
                padding_attr,
                dilation_attr,
                groups_attr,
                device,
                input_dtype,
                output_dtype=output_dtype,
                conv2d_config=conv2d_config,
                compute_config=compute_config,
                conv2d_slice_config=conv2d_slice_config,
                loc=loc,
            )

            return op.result

    @parse(ttnn.PrepareConv2dBiasOp)
    def prepare_conv2d_bias_parser(
        self,
        old_op: ttnn.PrepareConv2dBiasOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.PrepareConv2dBiasOp."""
        result_type = old_op.result.type
        bias_tensor = global_dict[old_op.bias_tensor]
        device = global_dict[old_op.device]

        new_op = ttnn.PrepareConv2dBiasOp(
            result_type,
            bias_tensor,
            old_op.input_memory_config,
            old_op.input_tensor_layout,
            old_op.in_channels,
            old_op.out_channels,
            old_op.batch_size,
            old_op.input_height,
            old_op.input_width,
            old_op.kernel_size,
            old_op.stride,
            old_op.padding,
            old_op.dilation,
            old_op.groups,
            device,
            old_op.input_dtype,
            output_dtype=old_op.output_dtype,
            conv2d_config=old_op.conv2d_config,
            compute_config=old_op.compute_config,
            conv2d_slice_config=old_op.conv2d_slice_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.PrepareConv2dBiasOp)
    def prepare_conv2d_bias_split(
        self,
        old_op: ttnn.PrepareConv2dBiasOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.PrepareConv2dBiasOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.bias_tensor.type, old_op.device.type]

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="prepare_conv2d_bias_module")
                def decorated_func(*inputs):
                    bias_tensor = inputs[0]
                    device = inputs[1]
                    result_type = old_op.result.type

                    new_op = ttnn.PrepareConv2dBiasOp(
                        result_type,
                        bias_tensor,
                        old_op.input_memory_config,
                        old_op.input_tensor_layout,
                        old_op.in_channels,
                        old_op.out_channels,
                        old_op.batch_size,
                        old_op.input_height,
                        old_op.input_width,
                        old_op.kernel_size,
                        old_op.stride,
                        old_op.padding,
                        old_op.dilation,
                        old_op.groups,
                        device,
                        old_op.input_dtype,
                        output_dtype=old_op.output_dtype,
                        conv2d_config=old_op.conv2d_config,
                        compute_config=old_op.compute_config,
                        conv2d_slice_config=old_op.conv2d_slice_config,
                        loc=old_op.location,
                    )

                    ordered_inputs.extend([bias_tensor, device])
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.PrepareConv2dWeightsOp)
    def prepare_conv2d_weights(
        self,
        weight_tensor: Operand,
        device: Operand,
        input_memory_config: Attribute,
        input_tensor_layout: Attribute,
        weights_format: Attribute,
        in_channels: int,
        out_channels: int,
        batch_size: int,
        input_height: int,
        input_width: int,
        kernel_size: List[int],
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        has_bias: bool,
        groups: int,
        input_dtype: Attribute,
        output_shape: List[int],
        output_dtype: Optional[Attribute] = None,
        conv2d_config: Optional[Attribute] = None,
        compute_config: Optional[Attribute] = None,
        conv2d_slice_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.PrepareConv2dWeightsOp.

        This op prepares the weight tensor for Conv2d operation.
        """
        with self._ctx, self._loc:
            element_type = self.get_type(weight_tensor)
            result_type = self.create_ttnn_tensor(output_shape, element_type)

            in_channels_attr = IntegerAttr.get(
                IntegerType.get_unsigned(32), in_channels
            )
            out_channels_attr = IntegerAttr.get(
                IntegerType.get_unsigned(32), out_channels
            )
            batch_size_attr = IntegerAttr.get(IntegerType.get_unsigned(32), batch_size)
            input_height_attr = IntegerAttr.get(
                IntegerType.get_unsigned(32), input_height
            )
            input_width_attr = IntegerAttr.get(
                IntegerType.get_unsigned(32), input_width
            )
            kernel_size_attr = DenseI32ArrayAttr.get(kernel_size, self._ctx)
            stride_attr = DenseI32ArrayAttr.get(stride, self._ctx)
            padding_attr = DenseI32ArrayAttr.get(padding, self._ctx)
            dilation_attr = DenseI32ArrayAttr.get(dilation, self._ctx)
            has_bias_attr = BoolAttr.get(has_bias, self._ctx)
            groups_attr = IntegerAttr.get(IntegerType.get_unsigned(32), groups)

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.PrepareConv2dWeightsOp(
                result_type,
                weight_tensor,
                input_memory_config,
                input_tensor_layout,
                weights_format,
                in_channels_attr,
                out_channels_attr,
                batch_size_attr,
                input_height_attr,
                input_width_attr,
                kernel_size_attr,
                stride_attr,
                padding_attr,
                dilation_attr,
                has_bias_attr,
                groups_attr,
                device,
                input_dtype,
                output_dtype=output_dtype,
                conv2d_config=conv2d_config,
                compute_config=compute_config,
                conv2d_slice_config=conv2d_slice_config,
                loc=loc,
            )

            return op.result

    @parse(ttnn.PrepareConv2dWeightsOp)
    def prepare_conv2d_weights_parser(
        self,
        old_op: ttnn.PrepareConv2dWeightsOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.PrepareConv2dWeightsOp."""
        result_type = old_op.result.type
        weight_tensor = global_dict[old_op.weight_tensor]
        device = global_dict[old_op.device]

        new_op = ttnn.PrepareConv2dWeightsOp(
            result_type,
            weight_tensor,
            old_op.input_memory_config,
            old_op.input_tensor_layout,
            old_op.weights_format,
            old_op.in_channels,
            old_op.out_channels,
            old_op.batch_size,
            old_op.input_height,
            old_op.input_width,
            old_op.kernel_size,
            old_op.stride,
            old_op.padding,
            old_op.dilation,
            old_op.has_bias,
            old_op.groups,
            device,
            old_op.input_dtype,
            output_dtype=old_op.output_dtype,
            conv2d_config=old_op.conv2d_config,
            compute_config=old_op.compute_config,
            conv2d_slice_config=old_op.conv2d_slice_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.PrepareConv2dWeightsOp)
    def prepare_conv2d_weights_split(
        self,
        old_op: ttnn.PrepareConv2dWeightsOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.PrepareConv2dWeightsOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.weight_tensor.type, old_op.device.type]

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="prepare_conv2d_weights_module")
                def decorated_func(*inputs):
                    weight_tensor = inputs[0]
                    device = inputs[1]
                    result_type = old_op.result.type

                    new_op = ttnn.PrepareConv2dWeightsOp(
                        result_type,
                        weight_tensor,
                        old_op.input_memory_config,
                        old_op.input_tensor_layout,
                        old_op.weights_format,
                        old_op.in_channels,
                        old_op.out_channels,
                        old_op.batch_size,
                        old_op.input_height,
                        old_op.input_width,
                        old_op.kernel_size,
                        old_op.stride,
                        old_op.padding,
                        old_op.dilation,
                        old_op.has_bias,
                        old_op.groups,
                        device,
                        old_op.input_dtype,
                        output_dtype=old_op.output_dtype,
                        conv2d_config=old_op.conv2d_config,
                        compute_config=old_op.compute_config,
                        conv2d_slice_config=old_op.conv2d_slice_config,
                        loc=old_op.location,
                    )

                    ordered_inputs.extend([weight_tensor, device])
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.RMSNormOp)
    def rms_norm_tag(
        self,
        input_tensor: Operand,
        weight: Optional[Operand] = None,
        bias: Optional[Operand] = None,
        epsilon: float = 1.0e-5,
        memory_config: Optional[Attribute] = None,
        compute_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.RMSNormOp.

        Applies Root Mean Square (RMS) normalization to the input tensor.
        """
        with self._ctx, self._loc:
            input_shape = list(self._get_type(input_tensor).shape)
            element_type = self.get_type(input_tensor)
            result_type = self.create_ttnn_tensor(input_shape, element_type)

            epsilon_attr = FloatAttr.get_f32(epsilon)

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.RMSNormOp(
                result_type,
                input_tensor,
                weight=weight,
                bias=bias,
                epsilon=epsilon_attr,
                memory_config=memory_config,
                compute_config=compute_config,
                loc=loc,
            )

            return op.result

    @parse(ttnn.RMSNormOp)
    def rms_norm_parser(
        self,
        old_op: ttnn.RMSNormOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.RMSNormOp."""
        result_type = old_op.result.type
        input_tensor = global_dict[old_op.input]

        # Get optional weight and bias
        weight = None
        if old_op.weight is not None:
            weight = global_dict[old_op.weight]
        bias = None
        if old_op.bias is not None:
            bias = global_dict[old_op.bias]

        new_op = ttnn.RMSNormOp(
            result_type,
            input_tensor,
            weight=weight,
            bias=bias,
            epsilon=old_op.epsilon,
            memory_config=old_op.memory_config,
            compute_config=old_op.compute_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.RMSNormOp)
    def rms_norm_split(
        self,
        old_op: ttnn.RMSNormOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.RMSNormOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]
            if old_op.weight is not None:
                op_input_types.append(old_op.weight.type)
            if old_op.bias is not None:
                op_input_types.append(old_op.bias.type)

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="rms_norm_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    idx = 1
                    weight = None
                    bias = None
                    if old_op.weight is not None:
                        weight = inputs[idx]
                        idx += 1
                    if old_op.bias is not None:
                        bias = inputs[idx]
                    result_type = old_op.result.type

                    new_op = ttnn.RMSNormOp(
                        result_type,
                        input_tensor,
                        weight=weight,
                        bias=bias,
                        epsilon=old_op.epsilon,
                        memory_config=old_op.memory_config,
                        compute_config=old_op.compute_config,
                        loc=old_op.location,
                    )

                    ordered_inputs.append(input_tensor)
                    if weight is not None:
                        ordered_inputs.append(weight)
                    if bias is not None:
                        ordered_inputs.append(bias)
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.SoftmaxOp)
    def softmax(
        self,
        in0: Operand,
        dimension: int,
        numericStable: bool = True,
        compute_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.SoftmaxOp.

        This op computes softmax along the specified dimension.
        """
        with self._ctx, self._loc:
            input_shape = list(self._get_type(in0).shape)
            element_type = self.get_type(in0)
            result_type = self.create_ttnn_tensor(input_shape, element_type)

            dimension_attr = IntegerAttr.get(IntegerType.get_signed(32), dimension)
            numeric_stable_attr = BoolAttr.get(numericStable, self._ctx)

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.SoftmaxOp(
                result_type,
                in0,
                dimension_attr,
                numericStable=numeric_stable_attr,
                compute_config=compute_config,
                loc=loc,
            )

            # Compute golden if available
            if not self._disable_golden_check:
                input_golden = self._get_golden_tensor(in0)
                if input_golden is not None:
                    golden_output = input_golden.apply_shardwise(
                        lambda shard: torch.nn.functional.softmax(shard, dim=dimension)
                    )
                    self._set_golden_tensor(op.result, golden_output)

            return op.result

    @parse(ttnn.SoftmaxOp)
    def softmax_parser(
        self,
        old_op: ttnn.SoftmaxOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.SoftmaxOp."""
        result_type = old_op.result.type
        input_tensor = global_dict[old_op.input]

        new_op = ttnn.SoftmaxOp(
            result_type,
            input_tensor,
            old_op.dimension,
            numericStable=old_op.numericStable,
            compute_config=old_op.compute_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.SoftmaxOp)
    def softmax_split(
        self,
        old_op: ttnn.SoftmaxOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.SoftmaxOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="softmax_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    result_type = old_op.result.type

                    new_op = ttnn.SoftmaxOp(
                        result_type,
                        input_tensor,
                        old_op.dimension,
                        numericStable=old_op.numericStable,
                        compute_config=old_op.compute_config,
                        loc=old_op.location,
                    )

                    ordered_inputs.append(input_tensor)
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.SplitQueryKeyValueAndSplitHeadsOp)
    def split_query_key_value_and_split_heads(
        self,
        input_tensor: Operand,
        num_heads: int,
        transpose_key: bool,
        query_shape: List[int],
        key_shape: List[int],
        value_shape: List[int],
        kv_input_tensor: Optional[Operand] = None,
        num_kv_heads: Optional[int] = None,
        memory_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> Tuple[OpResult, OpResult, OpResult]:
        """
        Creates ttnn.SplitQueryKeyValueAndSplitHeadsOp.

        This op splits the QKV tensor and splits heads for attention computation.
        """
        with self._ctx, self._loc:
            element_type = self.get_type(input_tensor)
            query_type = self.create_ttnn_tensor(query_shape, element_type)
            key_type = self.create_ttnn_tensor(key_shape, element_type)
            value_type = self.create_ttnn_tensor(value_shape, element_type)

            num_heads_attr = IntegerAttr.get(IntegerType.get_unsigned(32), num_heads)
            transpose_key_attr = BoolAttr.get(transpose_key, self._ctx)
            num_kv_heads_attr = None
            if num_kv_heads is not None:
                num_kv_heads_attr = IntegerAttr.get(
                    IntegerType.get_unsigned(32), num_kv_heads
                )

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.SplitQueryKeyValueAndSplitHeadsOp(
                query_type,
                key_type,
                value_type,
                input_tensor,
                num_heads_attr,
                transpose_key_attr,
                kv_input_tensor=kv_input_tensor,
                num_kv_heads=num_kv_heads_attr,
                memory_config=memory_config,
                loc=loc,
            )

            return op.query, op.key, op.value

    @parse(ttnn.SplitQueryKeyValueAndSplitHeadsOp)
    def split_query_key_value_and_split_heads_parser(
        self,
        old_op: ttnn.SplitQueryKeyValueAndSplitHeadsOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.SplitQueryKeyValueAndSplitHeadsOp."""
        query_type = old_op.query.type
        key_type = old_op.key.type
        value_type = old_op.value.type
        input_tensor = global_dict[old_op.input_tensor]

        # Get optional kv_input_tensor
        kv_input_tensor = None
        if old_op.kv_input_tensor is not None:
            kv_input_tensor = global_dict[old_op.kv_input_tensor]

        new_op = ttnn.SplitQueryKeyValueAndSplitHeadsOp(
            query_type,
            key_type,
            value_type,
            input_tensor,
            old_op.num_heads,
            old_op.transpose_key,
            kv_input_tensor=kv_input_tensor,
            num_kv_heads=old_op.num_kv_heads,
            memory_config=old_op.memory_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.query] = new_op.query
        op_map_dictionary[old_op.key] = new_op.key
        op_map_dictionary[old_op.value] = new_op.value
        return new_op, op_map_dictionary

    @split(ttnn.SplitQueryKeyValueAndSplitHeadsOp)
    def split_query_key_value_and_split_heads_split(
        self,
        old_op: ttnn.SplitQueryKeyValueAndSplitHeadsOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.SplitQueryKeyValueAndSplitHeadsOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input_tensor.type]
            if old_op.kv_input_tensor is not None:
                op_input_types.append(old_op.kv_input_tensor.type)

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="split_qkv_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    kv_input_tensor = inputs[1] if len(inputs) > 1 else None
                    query_type = old_op.query.type
                    key_type = old_op.key.type
                    value_type = old_op.value.type

                    new_op = ttnn.SplitQueryKeyValueAndSplitHeadsOp(
                        query_type,
                        key_type,
                        value_type,
                        input_tensor,
                        old_op.num_heads,
                        old_op.transpose_key,
                        kv_input_tensor=kv_input_tensor,
                        num_kv_heads=old_op.num_kv_heads,
                        memory_config=old_op.memory_config,
                        loc=old_op.location,
                    )

                    ordered_inputs.append(input_tensor)
                    if kv_input_tensor is not None:
                        ordered_inputs.append(kv_input_tensor)
                    ordered_outputs.extend([new_op.query, new_op.key, new_op.value])
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    @tag(ttnn.SumOp)
    def sum(
        self,
        in0: Operand,
        keep_dim: bool,
        output_shape: List[int],
        dim_arg: Optional[List[int]] = None,
        compute_config: Optional[Attribute] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """
        Creates ttnn.SumOp.

        This op computes the sum of the input tensor along specified dimensions.
        """
        with self._ctx, self._loc:
            element_type = self.get_type(in0)
            result_type = self.create_ttnn_tensor(output_shape, element_type)

            keep_dim_attr = BoolAttr.get(keep_dim, self._ctx)
            dim_arg_attr = None
            if dim_arg is not None:
                dim_arg_attr = DenseI32ArrayAttr.get(dim_arg, self._ctx)

            if loc is None:
                loc = self._get_location()
            else:
                loc = Location.name(loc)

            op = ttnn.SumOp(
                result_type,
                in0,
                keep_dim_attr,
                dim_arg=dim_arg_attr,
                compute_config=compute_config,
                loc=loc,
            )

            # Compute golden if available
            if not self._disable_golden_check:
                input_golden = self._get_golden_tensor(in0)
                if input_golden is not None:
                    if dim_arg is not None:
                        golden_output = input_golden.apply_shardwise(
                            lambda shard: torch.sum(
                                shard, dim=dim_arg, keepdim=keep_dim
                            )
                        )
                    else:
                        golden_output = input_golden.apply_shardwise(
                            lambda shard: torch.sum(shard, keepdim=keep_dim)
                        )
                    self._set_golden_tensor(op.result, golden_output)

            return op.result

    @parse(ttnn.SumOp)
    def sum_parser(
        self,
        old_op: ttnn.SumOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parser for ttnn.SumOp."""
        result_type = old_op.result.type
        input_tensor = global_dict[old_op.input]

        new_op = ttnn.SumOp(
            result_type,
            input_tensor,
            old_op.keep_dim,
            dim_arg=old_op.dim_arg,
            compute_config=old_op.compute_config,
            loc=old_op.location,
        )

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(ttnn.SumOp)
    def sum_split(
        self,
        old_op: ttnn.SumOp,
    ) -> Tuple[Module, TTNNBuilder]:
        """Split function for ttnn.SumOp."""
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_module = Module.create()
            split_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(split_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sum_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    result_type = old_op.result.type

                    new_op = ttnn.SumOp(
                        result_type,
                        input_tensor,
                        old_op.keep_dim,
                        dim_arg=old_op.dim_arg,
                        compute_config=old_op.compute_config,
                        loc=old_op.location,
                    )

                    ordered_inputs.append(input_tensor)
                    ordered_outputs.append(new_op.result)
                    return new_op

                new_func_op = decorated_func.func_op
                split_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_module, split_builder

    # ----- Public TTNN Op Generators ----

    ############### ttnn.AddOp ###############

    @tag(ttnn.AddOp)
    def add(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.add)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.AddOp)
    def add_parser(
        self,
        old_op: ttnn.AddOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.add_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.AbsOp ###############

    @tag(ttnn.AbsOp)
    def abs(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.abs)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.AbsOp)
    def abs_parser(
        self,
        old_op: ttnn.AbsOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.abs_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.CbrtOp ###############

    @tag(ttnn.CbrtOp)
    def cbrt(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.cbrt)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.CbrtOp)
    def cbrt_parser(
        self,
        old_op: ttnn.CbrtOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.cbrt_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.CeilOp ###############

    @tag(ttnn.CeilOp)
    def ceil(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.ceil)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.CeilOp)
    def ceil_parser(
        self,
        old_op: ttnn.CeilOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.ceil_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.CosOp ###############

    @tag(ttnn.CosOp)
    def cos(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.cos)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.CosOp)
    def cos_parser(
        self,
        old_op: ttnn.CosOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.cos_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.ErfOp ###############

    @tag(ttnn.ErfOp)
    def erf(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.erf)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.ErfOp)
    def erf_parser(
        self,
        old_op: ttnn.ErfOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.erf_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.ErfcOp ###############

    @tag(ttnn.ErfcOp)
    def erfc(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.erfc)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.ErfcOp)
    def erfc_parser(
        self,
        old_op: ttnn.ErfcOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.erfc_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.ExpOp ###############

    @tag(ttnn.ExpOp)
    def exp(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.exp)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.ExpOp)
    def exp_parser(
        self,
        old_op: ttnn.ExpOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.exp_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.FloorOp ###############

    @tag(ttnn.FloorOp)
    def floor(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.floor)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.FloorOp)
    def floor_parser(
        self,
        old_op: ttnn.FloorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.floor_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.GeluOp ###############

    @tag(ttnn.GeluOp)
    def gelu(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.gelu)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.GeluOp)
    def gelu_parser(
        self,
        old_op: ttnn.GeluOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.gelu_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.IsFiniteOp ###############

    @tag(ttnn.IsFiniteOp)
    def isfinite(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.isfinite)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.IsFiniteOp)
    def isfinite_parser(
        self,
        old_op: ttnn.IsFiniteOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.isfinite_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.LogicalNotOp ###############

    @tag(ttnn.LogicalNotOp)
    def logical_not(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.logical_not)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LogicalNotOp)
    def logical_not_parser(
        self,
        old_op: ttnn.LogicalNotOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.logical_not_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.BitwiseNotOp ###############

    @tag(ttnn.BitwiseNotOp)
    def bitwise_not(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.bitwise_not)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.BitwiseNotOp)
    def bitwise_not_parser(
        self,
        old_op: ttnn.BitwiseNotOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.bitwise_not_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.NegOp ###############

    @tag(ttnn.NegOp)
    def neg(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.neg)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.NegOp)
    def neg_parser(
        self,
        old_op: ttnn.NegOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.neg_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.TanOp ###############

    @tag(ttnn.TanOp)
    def tan(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.tan)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.TanOp)
    def tan_parser(
        self,
        old_op: ttnn.TanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.tan_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.AtanOp ###############

    @tag(ttnn.AtanOp)
    def atan(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.atan)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.AtanOp)
    def atan_parser(
        self,
        old_op: ttnn.AtanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.atan_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.TanhOp ###############

    @tag(ttnn.TanhOp)
    def tanh(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.tanh)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.TanhOp)
    def tanh_parser(
        self,
        old_op: ttnn.TanhOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.tanh_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.ReciprocalOp ###############

    @tag(ttnn.ReciprocalOp)
    def reciprocal(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.reciprocal)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.ReciprocalOp)
    def reciprocal_parser(
        self,
        old_op: ttnn.ReciprocalOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.reciprocal_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.ReluOp ###############

    @tag(ttnn.ReluOp)
    def relu(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.relu)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.ReluOp)
    def relu_parser(
        self,
        old_op: ttnn.ReluOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.relu_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.Relu6Op ###############

    @tag(ttnn.Relu6Op)
    def relu6(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.relu6)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.Relu6Op)
    def relu6_parser(
        self,
        old_op: ttnn.Relu6Op,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.relu6_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.RsqrtOp ###############

    @tag(ttnn.RsqrtOp)
    def rsqrt(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.rsqrt)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.RsqrtOp)
    def rsqrt_parser(
        self,
        old_op: ttnn.RsqrtOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.rsqrt_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.SigmoidOp ###############

    @tag(ttnn.SigmoidOp)
    def sigmoid(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.sigmoid)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.SigmoidOp)
    def sigmoid_parser(
        self,
        old_op: ttnn.SigmoidOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.sigmoid_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.SiluOp ###############

    @tag(ttnn.SiluOp)
    def silu(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.silu)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.SiluOp)
    def silu_parser(
        self,
        old_op: ttnn.SiluOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.silu_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.SignOp ###############

    @tag(ttnn.SignOp)
    def sign(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.sign)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.SignOp)
    def sign_parser(
        self,
        old_op: ttnn.SignOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.sign_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.SinOp ###############

    @tag(ttnn.SinOp)
    def sin(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.sin)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.SinOp)
    def sin_parser(
        self,
        old_op: ttnn.SinOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.sin_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.SqrtOp ###############

    @tag(ttnn.SqrtOp)
    def sqrt(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.sqrt)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.SqrtOp)
    def sqrt_parser(
        self,
        old_op: ttnn.SqrtOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.sqrt_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.TypecastOp ###############

    @tag(ttnn.TypecastOp)
    def typecast(
        self,
        in0: Operand,
        output_type: torch.dtype,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.typecast)
        mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            mlir_output_type,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.TypecastOp)
    def typecast_parser(
        self,
        old_op: ttnn.TypecastOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.typecast_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, old_op.dtype, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.LogOp ###############

    @tag(ttnn.LogOp)
    def log(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.log)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LogOp)
    def log_parser(
        self,
        old_op: ttnn.LogOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.log_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.Log1pOp ###############

    @tag(ttnn.Log1pOp)
    def log1p(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.log1p)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.Log1pOp)
    def log1p_parser(
        self,
        old_op: ttnn.Log1pOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.log1p_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.Expm1Op ###############

    @tag(ttnn.Expm1Op)
    def expm1(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.expm1)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.Expm1Op)
    def expm1_parser(
        self,
        old_op: ttnn.Expm1Op,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.expm1_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.EqualOp ###############

    @tag(ttnn.EqualOp)
    def eq(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.eq)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.EqualOp)
    def eq_parser(
        self,
        old_op: ttnn.EqualOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.eq_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.NotEqualOp ###############

    @tag(ttnn.NotEqualOp)
    def ne(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.ne)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.NotEqualOp)
    def ne_parser(
        self,
        old_op: ttnn.NotEqualOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.ne_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.GreaterEqualOp ###############

    @tag(ttnn.GreaterEqualOp)
    def ge(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.ge)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.GreaterEqualOp)
    def ge_parser(
        self,
        old_op: ttnn.GreaterEqualOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.ge_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.GreaterThanOp ###############

    @tag(ttnn.GreaterThanOp)
    def gt(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.gt)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.GreaterThanOp)
    def gt_parser(
        self,
        old_op: ttnn.GreaterThanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.gt_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.LessEqualOp ###############

    @tag(ttnn.LessEqualOp)
    def le(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.le)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LessEqualOp)
    def le_parser(
        self,
        old_op: ttnn.LessEqualOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.le_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.LessThanOp ###############

    @tag(ttnn.LessThanOp)
    def lt(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.lt)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LessThanOp)
    def lt_parser(
        self,
        old_op: ttnn.LessThanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.lt_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.LogicalAndOp ###############

    @tag(ttnn.LogicalAndOp)
    def logical_and(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.logical_and)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LogicalAndOp)
    def logical_and_parser(
        self,
        old_op: ttnn.LogicalAndOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.logical_and_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.LogicalLeftShiftOp ###############

    @tag(ttnn.LogicalLeftShiftOp)
    def logical_left_shift(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.logical_left_shift)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LogicalLeftShiftOp)
    def logical_left_shift_parser(
        self,
        old_op: ttnn.LogicalLeftShiftOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.logical_left_shift_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.LogicalOrOp ###############

    @tag(ttnn.LogicalOrOp)
    def logical_or(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.logical_or)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LogicalOrOp)
    def logical_or_parser(
        self,
        old_op: ttnn.LogicalOrOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.logical_or_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.LogicalRightShiftOp ###############

    @tag(ttnn.LogicalRightShiftOp)
    def logical_right_shift(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.logical_right_shift)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LogicalRightShiftOp)
    def logical_right_shift_parser(
        self,
        old_op: ttnn.LogicalRightShiftOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.logical_right_shift_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.LogicalXorOp ###############

    @tag(ttnn.LogicalXorOp)
    def logical_xor(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.logical_xor)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LogicalXorOp)
    def logical_xor_parser(
        self,
        old_op: ttnn.LogicalXorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.logical_xor_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.BitwiseAndOp ###############

    @tag(ttnn.BitwiseAndOp)
    def bitwise_and(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.bitwise_and)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.BitwiseAndOp)
    def bitwise_and_parser(
        self,
        old_op: ttnn.BitwiseAndOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.bitwise_and_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.BitwiseOrOp ###############

    @tag(ttnn.BitwiseOrOp)
    def bitwise_or(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.bitwise_or)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.BitwiseOrOp)
    def bitwise_or_parser(
        self,
        old_op: ttnn.BitwiseOrOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.bitwise_or_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.BitwiseXorOp ###############

    @tag(ttnn.BitwiseXorOp)
    def bitwise_xor(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.bitwise_xor)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.BitwiseXorOp)
    def bitwise_xor_parser(
        self,
        old_op: ttnn.BitwiseXorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.bitwise_xor_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.MinimumOp ###############

    @tag(ttnn.MinimumOp)
    def minimum(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.minimum)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.MinimumOp)
    def minimum_parser(
        self,
        old_op: ttnn.MinimumOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.minimum_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.MaximumOp ###############

    @tag(ttnn.MaximumOp)
    def maximum(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.maximum)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.MaximumOp)
    def maximum_parser(
        self,
        old_op: ttnn.MaximumOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.maximum_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

        return max_module, max_builder

    ############### ttnn.SubtractOp ###############

    @tag(ttnn.SubtractOp)
    def subtract(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.subtract)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.SubtractOp)
    def subtract_parser(
        self,
        old_op: ttnn.SubtractOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.subtract_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.RemainderOp ###############

    @tag(ttnn.RemainderOp)
    def remainder(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.remainder)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.RemainderOp)
    def remainder_parser(
        self,
        old_op: ttnn.RemainderOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.remainder_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.PowTensorOp ###############

    @tag(ttnn.PowTensorOp)
    def pow_tensor(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.pow_tensor)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.PowTensorOp)
    def pow_tensor_parser(
        self,
        old_op: ttnn.PowTensorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.pow_tensor_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.Atan2Op ###############

    @tag(ttnn.Atan2Op)
    def atan2(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.atan2)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.Atan2Op)
    def atan2_parser(
        self,
        old_op: ttnn.Atan2Op,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.atan2_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.MultiplyOp ###############

    @tag(ttnn.MultiplyOp)
    def multiply(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.multiply)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.MultiplyOp)
    def multiply_parser(
        self,
        old_op: ttnn.MultiplyOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.multiply_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.DivideOp ###############

    @tag(ttnn.DivideOp)
    def divide(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.divide)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.DivideOp)
    def divide_parser(
        self,
        old_op: ttnn.DivideOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.divide_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.ClampTensorOp ###############

    @tag(ttnn.ClampTensorOp)
    def clamp_tensor(
        self,
        in0: Operand,
        min_tensor: Operand,
        max_tensor: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.clamp_tensor)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        min_tensor_golden = self._get_golden_tensor(min_tensor)
        max_tensor_golden = self._get_golden_tensor(max_tensor)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0, min_tensor_golden, max_tensor_golden, mlir_output_type
        )
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            min_tensor,
            max_tensor,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.ClampTensorOp)
    def clamp_tensor_parser(
        self,
        old_op: ttnn.ClampTensorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.clamp_tensor_parser)
        in0 = global_dict[old_op.input]
        min_tensor = global_dict[old_op.min]
        max_tensor = global_dict[old_op.max]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            in0,
            min_tensor,
            max_tensor,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            min_tensor_golden = self._get_golden_tensor(min_tensor)
            max_tensor_golden = self._get_golden_tensor(max_tensor)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(
                input0, min_tensor_golden, max_tensor_golden, result.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.ConcatOp ###############

    @tag(ttnn.ConcatOp)
    def concat(
        self,
        ins: List[Operand],
        dim: int,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.concat)
        dim_attr = IntegerAttr.get(IntegerType.get_signed(32), dim)

        if output_type is None:
            mlir_output_type = self.get_type(ins[0])
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input_tensors = tuple([self._get_golden_tensor(i) for i in ins])
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input_tensors, dim_attr, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            ins,
            dim_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.ConcatOp)
    def concat_parser(
        self,
        old_op: ttnn.ConcatOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.concat_parser)
        inputs = [global_dict[in0] for in0 in old_op.inputs]
        result = old_op.result.type
        dim_attr = old_op.dim

        new_op = ttnn_op(
            result,
            inputs,
            dim=dim_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input_tensors = tuple([self._get_golden_tensor(in0) for in0 in inputs])
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(
                input_tensors, dim_attr, result.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.RepeatOp ###############

    @tag(ttnn.RepeatOp)
    def repeat(
        self,
        in0: Operand,
        repeat_dims: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.repeat)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        repeat_dims_attr = ttnn.ir.ShapeAttr.get(self._ctx, repeat_dims)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, repeat_dims_attr, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            repeat_dims_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.RepeatOp)
    def repeat_parser(
        self,
        old_op: ttnn.RepeatOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.repeat_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type
        repeat_dims_attr = old_op.repeat_dims

        new_op = ttnn_op(
            result,
            in0,
            repeat_dims_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(
                input0, repeat_dims_attr, old_op.result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.WhereOp ###############

    @tag(ttnn.WhereOp)
    def where(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.where)

        if output_type is None:
            mlir_output_type = self.get_type(in1)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        # Handle golden condition tensor
        in0_tensor = self._get_golden_tensor(in0)
        condition = in0_tensor.apply_shardwise(
            lambda shard: torch.where(
                shard > 0,
                torch.tensor(True, device=shard.device),
                torch.tensor(False, device=shard.device),
            )
        )
        input1 = self._get_golden_tensor(in1)
        input2 = self._get_golden_tensor(in2)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(condition, input1, input2, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            in2,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.WhereOp)
    def where_parser(
        self,
        old_op: ttnn.WhereOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.where_parser)
        first = global_dict[old_op.first]
        second = global_dict[old_op.second]
        third = global_dict[old_op.third]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            first,
            second,
            third,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            first_tensor = self._get_golden_tensor(first)
            condition = first_tensor.apply_shardwise(
                lambda shard: torch.where(
                    shard > 0,
                    torch.tensor(True, device=shard.device),
                    torch.tensor(False, device=shard.device),
                )
            )
            input1 = self._get_golden_tensor(second)
            input2 = self._get_golden_tensor(third)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(
                condition, input1, input2, result.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttnn.MatmulOp ###############

    @tag(ttnn.MatmulOp)
    def matmul(
        self,
        in0: Operand,
        in1: Operand,
        transpose_a: bool = False,
        transpose_b: bool = False,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.matmul)
        transpose_a_attr = BoolAttr.get(transpose_a, self._ctx)
        transpose_b_attr = BoolAttr.get(transpose_b, self._ctx)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0,
            input1,
            transpose_a_attr,
            transpose_b_attr,
            mlir_output_type,
        )
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            transpose_a=transpose_a_attr,
            transpose_b=transpose_b_attr,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.MatmulOp)
    def matmul_parser(
        self,
        old_op: ttnn.MatmulOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.matmul_parser)
        lhs = global_dict[old_op.a]
        rhs = global_dict[old_op.b]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            transpose_a=old_op.transpose_a,
            transpose_b=old_op.transpose_b,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(ttnn_op)
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            golden_output = op_golden_function(
                input0,
                input1,
                old_op.transpose_a,
                old_op.transpose_b,
                result.element_type,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {old_op.result: new_op_result}
        return new_op, op_map_dictionary

    ############### ttnn.LinearOp ###############

    @tag(ttnn.LinearOp)
    def linear(
        self,
        in0: Operand,
        in1: Operand,
        bias: Optional[Operand] = None,
        transpose_a: bool = False,
        transpose_b: bool = False,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.linear)
        transpose_a_attr = BoolAttr.get(transpose_a, self._ctx)
        transpose_b_attr = BoolAttr.get(transpose_b, self._ctx)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        bias_golden = self._get_golden_tensor(bias) if bias is not None else None
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0,
            input1,
            bias_golden,
            transpose_a_attr,
            transpose_b_attr,
            mlir_output_type,
        )
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            bias=bias,
            transpose_a=transpose_a_attr,
            transpose_b=transpose_b_attr,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LinearOp)
    def linear_parser(
        self,
        old_op: ttnn.LinearOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.linear_parser)
        a = global_dict[old_op.a]
        b = global_dict[old_op.b]
        bias = global_dict[old_op.bias] if old_op.bias is not None else None
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            a,
            b,
            loc=old_op.location,
            bias=bias,
            transpose_a=old_op.transpose_a,
            transpose_b=old_op.transpose_b,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(ttnn_op)
            input0 = self._get_golden_tensor(a)
            input1 = self._get_golden_tensor(b)
            bias_golden = self._get_golden_tensor(bias) if bias is not None else None
            golden_output = op_golden_function(
                input0,
                input1,
                bias_golden,
                old_op.transpose_a,
                old_op.transpose_b,
                result.element_type,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {old_op.result: new_op_result}
        return new_op, op_map_dictionary

    ############### ttnn.ClampScalarOp ###############

    @tag(ttnn.ClampScalarOp)
    def clamp_scalar(
        self,
        in0: Operand,
        min_arg: float,
        max_arg: float,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.clamp_scalar)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        min_attr = FloatAttr.get_f32(min_arg)
        max_attr = FloatAttr.get_f32(max_arg)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, min_attr, max_attr, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(result, in0, min_attr, max_attr, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.ClampScalarOp)
    def clamp_scalar_parser(
        self,
        old_op: ttnn.ClampScalarOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.clamp_scalar_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, old_op.min, old_op.max, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(ttnn_op)
            input0 = self._get_golden_tensor(in0)
            golden_output = op_golden_function(
                input0, old_op.min, old_op.max, result.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {old_op.result: new_op_result}
        return new_op, op_map_dictionary

    ############### ttnn.RepeatInterleaveOp ###############

    @tag(ttnn.RepeatInterleaveOp)
    def repeat_interleave(
        self,
        in0: Operand,
        repeats: int,
        dim: int,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.repeat_interleave)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        repeats_attr = IntegerAttr.get(IntegerType.get_unsigned(32), repeats)
        dim_attr = IntegerAttr.get(IntegerType.get_signed(32), dim)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0, repeats_attr, dim_attr, mlir_output_type
        )
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(result, in0, repeats_attr, dim_attr, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.RepeatInterleaveOp)
    def repeat_interleave_parser(
        self,
        old_op: ttnn.RepeatInterleaveOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.repeat_interleave_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, old_op.repeats, old_op.dim, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(ttnn_op)
            input0 = self._get_golden_tensor(in0)
            golden_output = op_golden_function(
                input0, old_op.repeats, old_op.dim, result.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    ############### ttnn.LeakyReluOp ###############

    @tag(ttnn.LeakyReluOp)
    def leaky_relu(
        self,
        in0: Operand,
        parameter: float,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.leaky_relu)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        parameter_attr = FloatAttr.get_f32(parameter)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, parameter_attr, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(result, in0, parameter_attr, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LeakyReluOp)
    def leaky_relu_parser(
        self,
        old_op: ttnn.LeakyReluOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.leaky_relu_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, old_op.parameter, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(ttnn_op)
            input0 = self._get_golden_tensor(in0)
            golden_output = op_golden_function(
                input0, old_op.parameter, result.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    ############### ttnn.MishOp ###############

    @tag(ttnn.MishOp)
    def mish(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.mish)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(result, in0, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.MishOp)
    def mish_parser(
        self,
        old_op: ttnn.MishOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.mish_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(ttnn_op)
            input0 = self._get_golden_tensor(in0)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    ############### ttnn.DumpTensorOp ###############

    @tag(ttnn.DumpTensorOp)
    def dump_tensor(
        self,
        in0: Operand,
        file_path: str,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> None:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.dump_tensor)
        file_path_attr = StringAttr.get(file_path)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        ttnn_op(
            file_path_attr,
            in0,
            loc=loc,
        )

        return

    @parse(ttnn.DumpTensorOp)
    def dump_tensor_parser(
        self,
        old_op: ttnn.DumpTensorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.dump_tensor_parser)
        in0 = global_dict[old_op.input]

        ttnn_op(
            old_op.file_path,
            in0,
            loc=old_op.location,
        )

        return None, {}

    def _op_proxy_l1_sharded_executed_op_with_dram_final_output(
        self,
        op_function: Callable,
        inputs: List[Operand],
        ttnn_kwargs: dict,
        unit_attrs: Optional[List[str]] = None,
        golden_kwargs: Optional[dict] = None,
        skip_golden: bool = False,
    ) -> OpView:
        """
        Helper method to create L1 width-sharded operations with DRAM output conversion.
        """
        with self._ctx, self._loc:
            # Create L1 width-sharded output tensor
            sharded_output_type = self.create_l1_width_sharded_tiled_ttnn_tensor(
                shape=inputs[0].type.shape,
                element_type=inputs[0].type.element_type,
            )

            # Prepare location for the operation
            id = self._get_next_global_id()
            loc = self._get_loc_of_extra_file_callee(id=id)

            # Create the operation with L1 sharded output
            op = op_function(
                sharded_output_type,
                *inputs,
                loc=loc,
                **ttnn_kwargs,
            )

            # Set unit attributes if provided
            if unit_attrs is not None:
                for attr_name in unit_attrs:
                    op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

            # Convert L1 sharded output to DRAM, to match expected final output layout
            final_output_type = self.create_ttnn_tensor(
                shape=op.result.type.shape,
                element_type=op.result.type.element_type,
            )

            tensor_memory_layout_attr = ttnn.ir.TensorMemoryLayoutAttr.get(
                self._ctx, ttnn.TensorMemoryLayout.Interleaved
            )
            buffer_type_attr = ttnn.ir.BufferTypeAttr.get(
                self._ctx, ttnn.BufferType.DRAM
            )
            memoryConfigAttr = ttnn.ir.MemoryConfigAttr.get(
                self._ctx, tensor_memory_layout_attr, buffer_type_attr
            )
            data_type = self._get_data_type_attribute(op.result)

            # Prepare location for the helper ToLayout operation
            id = self._get_next_global_id()
            loc = self._get_loc_of_extra_file_callee(id=id)

            output_to_dram = ttnn.ToLayoutOp(
                final_output_type,
                op.result,
                layout=ttnn.ir.LayoutAttr.get(self._ctx, ttnn.Layout.Tile),
                memory_config=memoryConfigAttr,
                loc=loc,
                dtype=data_type,
            )

            if not skip_golden and not self._disable_golden_check:
                golden_func = get_golden_function(op_function)
                if golden_func:
                    golden_args = self._organize_eltwise_golden(inputs)
                    golden = golden_func(*golden_args, **golden_kwargs or {})
                    self._set_golden_tensor(output_to_dram, golden)

            return output_to_dram

    def rms_norm(
        self,
        input_tensor: Operand,
        weight: Optional[Operand] = None,
        bias: Optional[Operand] = None,
        epsilon: float = 1.0e-5,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttnn.rms_norm``.

        *RMS Normalization operation.*

        Applies Root Mean Square (RMS) normalization to the input tensor.
        RMS normalization normalizes the input tensor based on the root mean square of its elements.

        Mathematical definition: rms_norm(x) = (x / sqrt(mean(x^2) + epsilon)) * weight + bias

        Parameters
        ----------
        input_tensor : Operand
            Input tensor to be normalized
        weight : Optional[Operand]
            Optional weight tensor for scaling (default: None)
        bias : Optional[Operand]
            Optional bias tensor for shifting (default: None)
        epsilon : float
            Small constant to avoid division by zero (default is 1.0e-5)
        unit_attrs : Optional[List[str]]
            Optional list of unit attributes, here used to specify L1 width sharding

        Returns
        -------
        OpView
            A tensor containing the RMS normalized output
        """
        # Check if L1 width sharding is requested
        l1_width_sharded = unit_attrs is not None and "l1_width_sharded" in unit_attrs
        if l1_width_sharded:
            # Remove the l1_width_sharded attribute for internal processing
            unit_attrs = [attr for attr in unit_attrs if attr != "l1_width_sharded"]

        # Build TTNN kwargs - only include non-None optional parameters
        ttnn_kwargs = {"epsilon": epsilon}
        if weight is not None:
            ttnn_kwargs["weight"] = weight
        if bias is not None:
            ttnn_kwargs["bias"] = bias

        # Determine normalized_shape from weight/bias tensor if available, otherwise use last input dimension
        if weight is not None:
            normalized_shape = list(weight.type.shape)
        elif bias is not None:
            normalized_shape = list(bias.type.shape)
        else:
            normalized_shape = [input_tensor.type.shape[-1]]

        golden_kwargs = {"epsilon": epsilon, "normalized_shape": normalized_shape}
        if weight is not None:
            golden_kwargs["weight"] = self._get_golden_tensor(weight)
        if bias is not None:
            golden_kwargs["bias"] = self._get_golden_tensor(bias)

        if l1_width_sharded:
            return self._op_proxy_l1_sharded_executed_op_with_dram_final_output(
                ttnn.RMSNormOp,
                [input_tensor],
                ttnn_kwargs=ttnn_kwargs,
                unit_attrs=unit_attrs,
                golden_kwargs=golden_kwargs,
            )
        else:
            return self._op_proxy(
                ttnn.RMSNormOp,
                [input_tensor],
                output_shape=input_tensor.type.shape,
                output_type=input_tensor.type.element_type,
                ttnn_kwargs=ttnn_kwargs,
                unit_attrs=unit_attrs,
                golden_kwargs=golden_kwargs,
            )

    # ----- Parse ttnn module ----

    @staticmethod
    def from_module(
        ctx: Context,
        mlir_text: str,
        golden_inputs: Dict[str, List[torch.tensor]] = None,
    ) -> Tuple(Module, TTNNBuilder):
        if golden_inputs is None:
            golden_inputs = {}

        # Disable golden checking when loading from MLIR file without golden inputs
        disable_golden = len(golden_inputs) == 0

        root_module = Module.parse(mlir_text, ctx)
        loc = Location.unknown(ctx)
        with ctx, loc:
            ttnn_builder = TTNNBuilder(ctx, loc, disable_golden_check=disable_golden)
            new_module = ttnn_builder.parse_root_module(root_module, golden_inputs)

        return new_module, ttnn_builder

    # ----- Split ttnn module ----

    def split_op(
        self,
        parsed_op: Operation,
    ) -> Tuple[Module, TTNNBuilder]:
        split_function = self.get_split_from_opview(type(parsed_op))
        return split_function(self, parsed_op)

    @staticmethod
    def split_module(
        module: Module,
        builder: TTNNBuilder,
    ) -> List[Tuple[Module, TTNNBuilder]]:
        sub_modules_and_builders = []
        old_ctx = module.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            for entry in module.body.operations:
                if isinstance(entry, ttcore.DeviceModuleOp):
                    for device_op_module in entry.regions[0].blocks[0].operations:
                        for device_module_op in (
                            device_op_module.regions[0].blocks[0].operations
                        ):
                            if isinstance(device_module_op, func.FuncOp):
                                if device_module_op.name.value in builder._nested_funcs:
                                    continue

                                for block in device_module_op.body:
                                    for op in block.operations:
                                        if isinstance(op, func.ReturnOp):
                                            continue
                                        elif isinstance(op, func.CallOp):
                                            sub_op_module_builder = (
                                                builder.split_call_op(op)
                                            )
                                            if len(sub_op_module_builder) != 0:
                                                sub_modules_and_builders.append(
                                                    sub_op_module_builder
                                                )
                                        else:
                                            sub_op_module_builder = builder.split_op(op)
                                            if len(sub_op_module_builder) != 0:
                                                sub_modules_and_builders.append(
                                                    sub_op_module_builder
                                                )
                elif isinstance(entry, func.FuncOp):
                    if entry.name.value in builder._nested_funcs:
                        continue

                    for block in entry.body:
                        for op in block.operations:
                            if isinstance(op, func.ReturnOp):
                                continue
                            elif isinstance(op, func.CallOp):
                                sub_op_module_builder = builder.split_call_op(op)
                                if len(sub_op_module_builder) != 0:
                                    sub_modules_and_builders.append(
                                        sub_op_module_builder
                                    )
                            else:
                                sub_op_module_builder = builder.split_op(op)
                                if len(sub_op_module_builder) != 0:
                                    sub_modules_and_builders.append(
                                        sub_op_module_builder
                                    )

        return sub_modules_and_builders
