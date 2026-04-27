# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Callable, Any
import torch
import numpy as np

from ttmlir.ir import *
from ttmlir import util
from ttmlir.dialects import ttnn, ttcore, func

from builder.base.builder import *
from builder.base.builder_utils import *
from builder.base.builder_enums import *

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
    ):
        super().__init__(ctx, location, mesh_name, mesh_dict)
        self.create_tensor_encoding = self._create_tensor_encoding

    def func(
        self,
        input_shapes,
        input_types,
        host_inputs: bool = False,
    ):
        if not host_inputs:
            return super().func(input_shapes, input_types)

        def wrapper(fn):
            # Create a wrapper that matches create_tensor_encoding signature
            # but sets layout to RowMajor and buffer type to SystemMemory for host tensors.
            def host_row_major_wrapper(
                shape, element_type, layout=None, buffer_type=None
            ):
                return self._create_tensor_encoding(
                    shape,
                    element_type,
                    ttnn.Layout.RowMajor,
                    ttnn.BufferType.SystemMemory,
                )

            self.create_tensor_encoding = host_row_major_wrapper

            try:
                result = super(TTNNBuilder, self).func(input_shapes, input_types)(fn)
            finally:
                self.create_tensor_encoding = self._create_tensor_encoding
            return result

        return wrapper

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

            if not skip_golden:
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
            tensor_type = (
                operand
                if isinstance(operand, RankedTensorType)
                else self._get_type(operand)
            )
            dtype = ttnn.ir.TTNNLayoutAttr.maybe_downcast(tensor_type.encoding)
            return ttcore.ir.DataTypeAttr.get(self._ctx, dtype.data_type_as_int)

    def _get_data_type_attribute_from_torch_dtype(
        self, dtype: torch.dtype
    ) -> ttcore.ir.DataTypeAttr:
        with self._ctx, self._loc:
            data_type = self._get_datatype_from_torch_dtype(dtype).data_type_as_int
            return ttcore.ir.DataTypeAttr.get(self._ctx, data_type)

    # ----- Public Helper Methods ----

    def _create_tensor_encoding(
        self,
        shape: Shape,
        element_type: Union[torch.dtype, TypeInfo],
        layout: ttnn.ir.LayoutAttr = ttnn.Layout.Tile,
        buffer_type: ttnn.ir.BufferType = ttnn.BufferType.DRAM,
    ) -> ttnn.ir.TTNNLayoutAttr:
        """
        TTNN tensors require that encoding information is present.
        This method creates a TTNN tensor with encoding information.
        For simplicity we will always create DRAM/Interleaved tiled tensor.
        """
        if isinstance(element_type, torch.dtype):
            element_type = self._get_type_from_torch_dtype(element_type)
        with self._ctx, self._loc:
            if layout == ttnn.Layout.Tile:
                data_type = util.element_type_to_data_type(element_type)
                layout_element_type = ttcore.ir.TileType.get(
                    self._ctx, 32, 32, data_type
                )
            elif layout == ttnn.Layout.RowMajor:
                layout_element_type = element_type
            else:
                raise ValueError(f"Unsupported layout: {layout}")

            if buffer_type == ttnn.BufferType.SystemMemory:
                tensor_memory_layout = None
            elif buffer_type == ttnn.BufferType.L1:
                tensor_memory_layout = ttnn.TensorMemoryLayout.WidthSharded
            else:
                tensor_memory_layout = ttnn.TensorMemoryLayout.Interleaved

            grid_attr = ttcore.ir.GridAttr.get(self._ctx, [1, 1])
            return ttnn.ir.TTNNLayoutAttr.get(
                self._ctx,
                shape,
                layout_element_type,
                buffer_type,
                grid_attr,
                tensor_memory_layout,
            )

    def create_ttnn_tensor(
        self,
        shape: Shape,
        element_type: Union[torch.dtype, TypeInfo],
        layout: ttnn.ir.LayoutAttr = ttnn.Layout.Tile,
        buffer_type: ttnn.ir.BufferType = ttnn.BufferType.DRAM,
    ) -> RankedTensorType:
        """
        TTNN tensors require that encoding information is present.
        This method creates a TTNN tensor with encoding information.
        For simplicity we will always create DRAM/Interleaved tiled tensor.
        """
        with self._ctx, self._loc:
            ttnn_layout_attr = self._create_tensor_encoding(
                shape, element_type, layout, buffer_type
            )
            return RankedTensorType.get(shape, element_type, ttnn_layout_attr)

    def _get_location(self) -> Location:
        stack = inspect.stack()
        caller_frame = stack[2]
        filename = caller_frame.filename
        lineno = caller_frame.lineno
        return Location.name(f"{filename}:{lineno}")

    # ----- Private CCL Helpers -----

    def _create_memory_config_attr(
        self,
        buffer_type: ttnn.ir.BufferType = ttnn.BufferType.DRAM,
        tensor_memory_layout: ttnn.ir.TensorMemoryLayout = ttnn.TensorMemoryLayout.Interleaved,
    ) -> ttnn.ir.MemoryConfigAttr:
        if buffer_type == ttnn.BufferType.SystemMemory:
            return self._create_system_memory_memory_config()

        tensor_memory_layout_attr = ttnn.ir.TensorMemoryLayoutAttr.get(
            self._ctx, tensor_memory_layout
        )
        buffer_type_attr = ttnn.ir.BufferTypeAttr.get(self._ctx, buffer_type)
        return ttnn.ir.MemoryConfigAttr.get(
            self._ctx, tensor_memory_layout_attr, buffer_type_attr
        )

    def _create_system_memory_memory_config(self):
        # The pybound MemoryConfigAttr.get() requires a TensorMemoryLayoutAttr, but system_memory requires no TensorMemoryLayoutAttr
        memory_config_str = "#ttnn.memory_config<#ttnn.buffer_type<system_memory>>"
        return Attribute.parse(memory_config_str)

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.AddOp)
    def add_split(
        self,
        old_op: ttnn.AddOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.add_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            add_module = Module.create()
            add_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(add_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="add_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    add_builder._set_golden_tensor(new_op_result, old_op_result)
                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    add_builder._set_golden_tensor(in0, input0)
                    add_builder._set_golden_tensor(in1, input1)
                    add_builder._annotate_presharded_arg(in0)
                    add_builder._annotate_presharded_arg(in1)
                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                add_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return add_module, add_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.AbsOp)
    def abs_split(
        self,
        old_op: ttnn.AbsOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.abs_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            abs_module = Module.create()
            abs_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(abs_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="abs_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    abs_builder._set_golden_tensor(new_op_result, old_op_result)
                    input0 = self._get_golden_tensor(old_op.input)
                    abs_builder._set_golden_tensor(in0, input0)
                    abs_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                abs_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return abs_module, abs_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.CbrtOp)
    def cbrt_split(
        self,
        old_op: ttnn.CbrtOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.cbrt_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            cbrt_module = Module.create()
            cbrt_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(cbrt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="cbrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    cbrt_builder._set_golden_tensor(new_op_result, old_op_result)
                    input0 = self._get_golden_tensor(old_op.input)
                    cbrt_builder._set_golden_tensor(in0, input0)
                    cbrt_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                cbrt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return cbrt_module, cbrt_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.CeilOp)
    def ceil_split(
        self,
        old_op: ttnn.CeilOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.ceil_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            ceil_module = Module.create()
            ceil_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(ceil_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="ceil_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    ceil_builder._set_golden_tensor(new_op_result, old_op_result)
                    input0 = self._get_golden_tensor(old_op.input)
                    ceil_builder._set_golden_tensor(in0, input0)
                    ceil_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                ceil_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return ceil_module, ceil_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.CosOp)
    def cos_split(
        self,
        old_op: ttnn.CosOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.cos_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            cos_module = Module.create()
            cos_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(cos_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="cos_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    cos_builder._set_golden_tensor(new_op_result, old_op_result)
                    input0 = self._get_golden_tensor(old_op.input)
                    cos_builder._set_golden_tensor(in0, input0)
                    cos_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                cos_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return cos_module, cos_builder

    ############### ttnn.AcosOp ###############

    @tag(ttnn.AcosOp)
    def acos(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.acos)

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

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.AcosOp)
    def acos_parser(
        self,
        old_op: ttnn.AcosOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.acos_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.AcosOp)
    def acos_split(
        self,
        old_op: ttnn.AcosOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.acos_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            acos_module = Module.create()
            acos_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(acos_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="acos_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    acos_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    acos_builder._set_golden_tensor(in0, input0)
                    acos_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                acos_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return acos_module, acos_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.ErfOp)
    def erf_split(
        self,
        old_op: ttnn.ErfOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.erf_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            erf_module = Module.create()
            erf_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(erf_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="erf_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    erf_builder._set_golden_tensor(new_op_result, old_op_result)
                    input0 = self._get_golden_tensor(old_op.input)
                    erf_builder._set_golden_tensor(in0, input0)
                    erf_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                erf_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return erf_module, erf_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.ErfcOp)
    def erfc_split(
        self,
        old_op: ttnn.ErfcOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.erfc_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            erfc_module = Module.create()
            erfc_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(erfc_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="erfc_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    erfc_builder._set_golden_tensor(new_op_result, old_op_result)
                    input0 = self._get_golden_tensor(old_op.input)
                    erfc_builder._set_golden_tensor(in0, input0)
                    erfc_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                erfc_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return erfc_module, erfc_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.ExpOp)
    def exp_split(
        self,
        old_op: ttnn.ExpOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.exp_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            exp_module = Module.create()
            exp_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(exp_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="exp_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    exp_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    exp_builder._set_golden_tensor(in0, input0)
                    exp_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                exp_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return exp_module, exp_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.FloorOp)
    def floor_split(
        self,
        old_op: ttnn.FloorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.floor_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            floor_module = Module.create()
            floor_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(floor_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="floor_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    floor_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    floor_builder._set_golden_tensor(in0, input0)
                    floor_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                floor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return floor_module, floor_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.GeluOp)
    def gelu_split(
        self,
        old_op: ttnn.GeluOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.gelu_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            gelu_module = Module.create()
            gelu_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(gelu_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="gelu_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    gelu_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    gelu_builder._set_golden_tensor(in0, input0)
                    gelu_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                gelu_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return gelu_module, gelu_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.IsFiniteOp)
    def isfinite_split(
        self,
        old_op: ttnn.IsFiniteOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.isfinite_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            isfinite_module = Module.create()
            isfinite_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(isfinite_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="isfinite_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    isfinite_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    isfinite_builder._set_golden_tensor(in0, input0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                isfinite_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return isfinite_module, isfinite_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LogicalNotOp)
    def logical_not_split(
        self,
        old_op: ttnn.LogicalNotOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_not_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            logical_not_module = Module.create()
            logical_not_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(logical_not_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_not_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    logical_not_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    logical_not_builder._set_golden_tensor(in0, input0)
                    logical_not_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                logical_not_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return logical_not_module, logical_not_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.BitwiseNotOp)
    def bitwise_not_split(
        self,
        old_op: ttnn.BitwiseNotOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.bitwise_not_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            bitwise_not_module = Module.create()
            bitwise_not_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(bitwise_not_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="bitwise_not_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    bitwise_not_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    bitwise_not_builder._set_golden_tensor(in0, input0)
                    bitwise_not_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                bitwise_not_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return bitwise_not_module, bitwise_not_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.NegOp)
    def neg_split(
        self,
        old_op: ttnn.NegOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.neg_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            neg_module = Module.create()
            neg_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(neg_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="neg_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    neg_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    neg_builder._set_golden_tensor(in0, input0)
                    neg_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                neg_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return neg_module, neg_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.TanOp)
    def tan_split(
        self,
        old_op: ttnn.TanOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.tan_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            tan_module = Module.create()
            tan_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(tan_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="tan_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    tan_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    tan_builder._set_golden_tensor(in0, input0)
                    tan_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                tan_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return tan_module, tan_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.AtanOp)
    def atan_split(
        self,
        old_op: ttnn.AtanOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.atan_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            atan_module = Module.create()
            atan_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(atan_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="atan_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    atan_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    atan_builder._set_golden_tensor(in0, input0)
                    atan_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                atan_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return atan_module, atan_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.TanhOp)
    def tanh_split(
        self,
        old_op: ttnn.TanhOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.tanh_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            tanh_module = Module.create()
            tanh_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(tanh_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="tanh_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    tanh_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    tanh_builder._set_golden_tensor(in0, input0)
                    tanh_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                tanh_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return tanh_module, tanh_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.ReciprocalOp)
    def reciprocal_split(
        self,
        old_op: ttnn.ReciprocalOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.reciprocal_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            reciprocal_module = Module.create()
            reciprocal_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(reciprocal_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="reciprocal_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    reciprocal_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    reciprocal_builder._set_golden_tensor(in0, input0)
                    reciprocal_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                reciprocal_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return reciprocal_module, reciprocal_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.ReluOp)
    def relu_split(
        self,
        old_op: ttnn.ReluOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.relu_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            relu_module = Module.create()
            relu_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(relu_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="relu_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    relu_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    relu_builder._set_golden_tensor(in0, input0)
                    relu_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                relu_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return relu_module, relu_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.Relu6Op)
    def relu6_split(
        self,
        old_op: ttnn.Relu6Op,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.relu6_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            relu6_module = Module.create()
            relu6_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(relu6_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="relu6_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    relu6_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    relu6_builder._set_golden_tensor(in0, input0)
                    relu6_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                relu6_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return relu6_module, relu6_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.RsqrtOp)
    def rsqrt_split(
        self,
        old_op: ttnn.RsqrtOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.rsqrt_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            rsqrt_module = Module.create()
            rsqrt_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(rsqrt_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="rsqrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    rsqrt_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    rsqrt_builder._set_golden_tensor(in0, input0)
                    rsqrt_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                rsqrt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return rsqrt_module, rsqrt_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.SigmoidOp)
    def sigmoid_split(
        self,
        old_op: ttnn.SigmoidOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.sigmoid_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            sigmoid_module = Module.create()
            sigmoid_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(sigmoid_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sigmoid_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    sigmoid_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    sigmoid_builder._set_golden_tensor(in0, input0)
                    sigmoid_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                sigmoid_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sigmoid_module, sigmoid_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.SiluOp)
    def silu_split(
        self,
        old_op: ttnn.SiluOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.silu_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            silu_module = Module.create()
            silu_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(silu_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="silu_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    silu_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    silu_builder._set_golden_tensor(in0, input0)
                    silu_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                silu_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return silu_module, silu_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.SignOp)
    def sign_split(
        self,
        old_op: ttnn.SignOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.sign_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            sign_module = Module.create()
            sign_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(sign_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sign_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    sign_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    sign_builder._set_golden_tensor(in0, input0)
                    sign_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                sign_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sign_module, sign_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.SinOp)
    def sin_split(
        self,
        old_op: ttnn.SinOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.sin_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            sin_module = Module.create()
            sin_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(sin_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sin_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    sin_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    sin_builder._set_golden_tensor(in0, input0)
                    sin_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                sin_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sin_module, sin_builder

    ############### ttnn.AsinOp ###############

    @tag(ttnn.AsinOp)
    def asin(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.asin)

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

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.AsinOp)
    def asin_parser(
        self,
        old_op: ttnn.AsinOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.asin_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.AsinOp)
    def asin_split(
        self,
        old_op: ttnn.AsinOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.asin_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            asin_module = Module.create()
            asin_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(asin_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="asin_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    asin_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    asin_builder._set_golden_tensor(in0, input0)
                    asin_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                asin_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return asin_module, asin_builder

    ############### ttnn.AsinhOp ###############

    @tag(ttnn.AsinhOp)
    def asinh(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.asinh)

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

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.AsinhOp)
    def asinh_parser(
        self,
        old_op: ttnn.AsinhOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.asinh_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.AsinhOp)
    def asinh_split(
        self,
        old_op: ttnn.AsinhOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.asinh_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            asinh_module = Module.create()
            asinh_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(asinh_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="asinh_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    asinh_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    asinh_builder._set_golden_tensor(in0, input0)
                    asinh_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                asinh_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return asinh_module, asinh_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.SqrtOp)
    def sqrt_split(
        self,
        old_op: ttnn.SqrtOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.sqrt_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            sqrt_module = Module.create()
            sqrt_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(sqrt_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sqrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    sqrt_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    sqrt_builder._set_golden_tensor(in0, input0)
                    sqrt_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                sqrt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sqrt_module, sqrt_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.TypecastOp)
    def typecast_split(
        self,
        old_op: ttnn.TypecastOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.typecast_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            typecast_module = Module.create()
            typecast_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(typecast_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="typecast_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, old_op.dtype, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    typecast_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    typecast_builder._set_golden_tensor(in0, input0)
                    typecast_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                typecast_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return typecast_module, typecast_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LogOp)
    def log_split(
        self,
        old_op: ttnn.LogOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.log_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            log_module = Module.create()
            log_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(log_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="log_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    log_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    log_builder._set_golden_tensor(in0, input0)
                    log_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                log_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return log_module, log_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.Log1pOp)
    def log1p_split(
        self,
        old_op: ttnn.Log1pOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.log1p_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            log1p_module = Module.create()
            log1p_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(log1p_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="log1p_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    log1p_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    log1p_builder._set_golden_tensor(in0, input0)
                    log1p_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                log1p_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return log1p_module, log1p_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.Expm1Op)
    def expm1_split(
        self,
        old_op: ttnn.Expm1Op,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.expm1_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            expm1_module = Module.create()
            expm1_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(expm1_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="expm1_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    expm1_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    expm1_builder._set_golden_tensor(in0, input0)
                    expm1_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                expm1_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return expm1_module, expm1_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.EqualOp)
    def eq_split(
        self,
        old_op: ttnn.EqualOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.eq_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            eq_module = Module.create()
            eq_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(eq_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="eq_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    eq_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    eq_builder._set_golden_tensor(in0, input0)
                    eq_builder._set_golden_tensor(in1, input1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                eq_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return eq_module, eq_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.NotEqualOp)
    def ne_split(
        self,
        old_op: ttnn.NotEqualOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.ne_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            ne_module = Module.create()
            ne_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(ne_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="ne_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    ne_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    ne_builder._set_golden_tensor(in0, input0)
                    ne_builder._set_golden_tensor(in1, input1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                ne_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return ne_module, ne_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.GreaterEqualOp)
    def ge_split(
        self,
        old_op: ttnn.GreaterEqualOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.ge_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            ge_module = Module.create()
            ge_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(ge_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="ge_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    ge_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    ge_builder._set_golden_tensor(in0, input0)
                    ge_builder._set_golden_tensor(in1, input1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                ge_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return ge_module, ge_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.GreaterThanOp)
    def gt_split(
        self,
        old_op: ttnn.GreaterThanOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.gt_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            gt_module = Module.create()
            gt_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(gt_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="gt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    gt_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    gt_builder._set_golden_tensor(in0, input0)
                    gt_builder._set_golden_tensor(in1, input1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                gt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return gt_module, gt_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LessEqualOp)
    def le_split(
        self,
        old_op: ttnn.LessEqualOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.le_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            le_module = Module.create()
            le_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(le_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="le_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    le_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    le_builder._set_golden_tensor(in0, input0)
                    le_builder._set_golden_tensor(in1, input1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                le_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return le_module, le_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LessThanOp)
    def lt_split(
        self,
        old_op: ttnn.LessThanOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.lt_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            lt_module = Module.create()
            lt_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(lt_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="lt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    lt_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    lt_builder._set_golden_tensor(in0, input0)
                    lt_builder._set_golden_tensor(in1, input1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                lt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return lt_module, lt_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LogicalAndOp)
    def logical_and_split(
        self,
        old_op: ttnn.LogicalAndOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_and_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            logical_and_module = Module.create()
            logical_and_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(logical_and_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_and_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    logical_and_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    logical_and_builder._set_golden_tensor(in0, input0)
                    logical_and_builder._set_golden_tensor(in1, input1)
                    logical_and_builder._annotate_presharded_arg(in0)
                    logical_and_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                logical_and_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return logical_and_module, logical_and_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LogicalLeftShiftOp)
    def logical_left_shift_split(
        self,
        old_op: ttnn.LogicalLeftShiftOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_left_shift_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            logical_left_shift_module = Module.create()
            logical_left_shift_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(logical_left_shift_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_left_shift_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    logical_left_shift_builder._set_golden_tensor(
                        new_op_result, old_op_result
                    )

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    logical_left_shift_builder._set_golden_tensor(in0, input0)
                    logical_left_shift_builder._set_golden_tensor(in1, input1)
                    logical_left_shift_builder._annotate_presharded_arg(in0)
                    logical_left_shift_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                logical_left_shift_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return logical_left_shift_module, logical_left_shift_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LogicalOrOp)
    def logical_or_split(
        self,
        old_op: ttnn.LogicalOrOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_or_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            logical_or_module = Module.create()
            logical_or_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(logical_or_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_or_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    logical_or_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    logical_or_builder._set_golden_tensor(in0, input0)
                    logical_or_builder._set_golden_tensor(in1, input1)
                    logical_or_builder._annotate_presharded_arg(in0)
                    logical_or_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                logical_or_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return logical_or_module, logical_or_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LogicalRightShiftOp)
    def logical_right_shift_split(
        self,
        old_op: ttnn.LogicalRightShiftOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_right_shift_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            logical_right_shift_module = Module.create()
            logical_right_shift_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(logical_right_shift_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_right_shift_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    logical_right_shift_builder._set_golden_tensor(
                        new_op_result, old_op_result
                    )

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    logical_right_shift_builder._set_golden_tensor(in0, input0)
                    logical_right_shift_builder._set_golden_tensor(in1, input1)
                    logical_right_shift_builder._annotate_presharded_arg(in0)
                    logical_right_shift_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                logical_right_shift_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return logical_right_shift_module, logical_right_shift_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LogicalXorOp)
    def logical_xor_split(
        self,
        old_op: ttnn.LogicalXorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_xor_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            logical_xor_module = Module.create()
            logical_xor_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(logical_xor_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_xor_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    logical_xor_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    logical_xor_builder._set_golden_tensor(in0, input0)
                    logical_xor_builder._set_golden_tensor(in1, input1)
                    logical_xor_builder._annotate_presharded_arg(in0)
                    logical_xor_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                logical_xor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return logical_xor_module, logical_xor_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.BitwiseAndOp)
    def bitwise_and_split(
        self,
        old_op: ttnn.BitwiseAndOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.bitwise_and_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            bitwise_and_module = Module.create()
            bitwise_and_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(bitwise_and_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="bitwise_and_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    bitwise_and_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    bitwise_and_builder._set_golden_tensor(in0, input0)
                    bitwise_and_builder._set_golden_tensor(in1, input1)
                    bitwise_and_builder._annotate_presharded_arg(in0)
                    bitwise_and_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                bitwise_and_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return bitwise_and_module, bitwise_and_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.BitwiseOrOp)
    def bitwise_or_split(
        self,
        old_op: ttnn.BitwiseOrOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.bitwise_or_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            bitwise_or_module = Module.create()
            bitwise_or_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(bitwise_or_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="bitwise_or_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    bitwise_or_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    bitwise_or_builder._set_golden_tensor(in0, input0)
                    bitwise_or_builder._set_golden_tensor(in1, input1)
                    bitwise_or_builder._annotate_presharded_arg(in0)
                    bitwise_or_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                bitwise_or_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return bitwise_or_module, bitwise_or_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.BitwiseXorOp)
    def bitwise_xor_split(
        self,
        old_op: ttnn.BitwiseXorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.bitwise_xor_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            bitwise_xor_module = Module.create()
            bitwise_xor_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(bitwise_xor_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="bitwise_xor_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    bitwise_xor_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    bitwise_xor_builder._set_golden_tensor(in0, input0)
                    bitwise_xor_builder._set_golden_tensor(in1, input1)
                    bitwise_xor_builder._annotate_presharded_arg(in0)
                    bitwise_xor_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                bitwise_xor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return bitwise_xor_module, bitwise_xor_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.MinimumOp)
    def minimum_split(
        self,
        old_op: ttnn.MinimumOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.minimum_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            minimum_module = Module.create()
            minimum_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(minimum_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="minimum_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    minimum_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    minimum_builder._set_golden_tensor(in0, input0)
                    minimum_builder._set_golden_tensor(in1, input1)
                    minimum_builder._annotate_presharded_arg(in0)
                    minimum_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                minimum_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return minimum_module, minimum_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

        return max_module, max_builder

    @split(ttnn.MaximumOp)
    def maximum_split(
        self,
        old_op: ttnn.MaximumOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.maximum_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            maximum_module = Module.create()
            maximum_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(maximum_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="maximum_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    maximum_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    maximum_builder._set_golden_tensor(in0, input0)
                    maximum_builder._set_golden_tensor(in1, input1)
                    maximum_builder._annotate_presharded_arg(in0)
                    maximum_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                maximum_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return maximum_module, maximum_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.SubtractOp)
    def subtract_split(
        self,
        old_op: ttnn.SubtractOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.subtract_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            subtract_module = Module.create()
            subtract_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(subtract_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="subtract_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    subtract_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    subtract_builder._set_golden_tensor(in0, input0)
                    subtract_builder._set_golden_tensor(in1, input1)
                    subtract_builder._annotate_presharded_arg(in0)
                    subtract_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                subtract_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return subtract_module, subtract_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.RemainderOp)
    def remainder_split(
        self,
        old_op: ttnn.RemainderOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.remainder_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            remainder_module = Module.create()
            remainder_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(remainder_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="remainder_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    remainder_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    remainder_builder._set_golden_tensor(in0, input0)
                    remainder_builder._set_golden_tensor(in1, input1)
                    remainder_builder._annotate_presharded_arg(in0)
                    remainder_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                remainder_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return remainder_module, remainder_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.PowTensorOp)
    def pow_tensor_split(
        self,
        old_op: ttnn.PowTensorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.pow_tensor_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            pow_tensor_module = Module.create()
            pow_tensor_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(pow_tensor_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="pow_tensor_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    pow_tensor_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    pow_tensor_builder._set_golden_tensor(in0, input0)
                    pow_tensor_builder._set_golden_tensor(in1, input1)
                    pow_tensor_builder._annotate_presharded_arg(in0)
                    pow_tensor_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                pow_tensor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return pow_tensor_module, pow_tensor_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.Atan2Op)
    def atan2_split(
        self,
        old_op: ttnn.Atan2Op,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.atan2_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            atan2_module = Module.create()
            atan2_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(atan2_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="atan2_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    atan2_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    atan2_builder._set_golden_tensor(in0, input0)
                    atan2_builder._set_golden_tensor(in1, input1)
                    atan2_builder._annotate_presharded_arg(in0)
                    atan2_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                atan2_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return atan2_module, atan2_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.MultiplyOp)
    def multiply_split(
        self,
        old_op: ttnn.MultiplyOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.multiply_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            multiply_module = Module.create()
            multiply_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(multiply_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="multiply_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    multiply_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    multiply_builder._set_golden_tensor(in0, input0)
                    multiply_builder._set_golden_tensor(in1, input1)
                    multiply_builder._annotate_presharded_arg(in0)
                    multiply_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                multiply_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return multiply_module, multiply_builder

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

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.DivideOp)
    def divide_split(
        self,
        old_op: ttnn.DivideOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.divide_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            divide_module = Module.create()
            divide_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(divide_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="divide_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result, in0, in1, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    divide_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    divide_builder._set_golden_tensor(in0, input0)
                    divide_builder._set_golden_tensor(in1, input1)
                    divide_builder._annotate_presharded_arg(in0)
                    divide_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                divide_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return divide_module, divide_builder

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

    @split(ttnn.ClampTensorOp)
    def clamp_tensor_split(
        self,
        old_op: ttnn.ClampTensorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.clamp_tensor_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            clamp_tensor_module = Module.create()
            clamp_tensor_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [
                old_op.input.type,
                old_op.min.type,
                old_op.max.type,
            ]

            with InsertionPoint(clamp_tensor_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="clamp_tensor_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    min_tensor = inputs[1]
                    max_tensor = inputs[2]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result,
                        in0,
                        min_tensor,
                        max_tensor,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    clamp_tensor_builder._set_golden_tensor(
                        new_op_result, old_op_result
                    )

                    input0 = self._get_golden_tensor(old_op.input)
                    input1 = self._get_golden_tensor(old_op.min)
                    input2 = self._get_golden_tensor(old_op.max)
                    clamp_tensor_builder._set_golden_tensor(in0, input0)
                    clamp_tensor_builder._set_golden_tensor(min_tensor, input1)
                    clamp_tensor_builder._set_golden_tensor(max_tensor, input2)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(min_tensor)
                    ordered_inputs.append(max_tensor)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                clamp_tensor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return clamp_tensor_module, clamp_tensor_builder

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

        input_tensors = tuple([self._get_golden_tensor(in0) for in0 in inputs])
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input_tensors, dim_attr, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.ConcatOp)
    def concat_split(
        self,
        old_op: ttnn.ConcatOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.concat_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            concat_module = Module.create()
            concat_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [inp.type for inp in old_op.inputs]

            with InsertionPoint(concat_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="concat_module")
                def decorated_func(*inputs):
                    input_list = list(inputs)
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result,
                        input_list,
                        dim=old_op.dim,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    concat_builder._set_golden_tensor(new_op_result, old_op_result)

                    for i, old_inp in enumerate(old_op.inputs):
                        inp_golden = self._get_golden_tensor(old_inp)
                        concat_builder._set_golden_tensor(inputs[i], inp_golden)
                        ordered_inputs.append(inputs[i])

                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                concat_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return concat_module, concat_builder

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

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0, repeat_dims_attr, old_op.result.type.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.RepeatOp)
    def repeat_split(
        self,
        old_op: ttnn.RepeatOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.repeat_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            repeat_module = Module.create()
            repeat_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(repeat_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="repeat_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result,
                        in0,
                        old_op.repeat_dims,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    repeat_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    repeat_builder._set_golden_tensor(in0, input0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                repeat_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return repeat_module, repeat_builder

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

    @split(ttnn.WhereOp)
    def where_split(
        self,
        old_op: ttnn.WhereOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.where_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            where_module = Module.create()
            where_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [
                old_op.first.type,
                old_op.second.type,
                old_op.third.type,
            ]

            with InsertionPoint(where_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="where_module")
                def decorated_func(*inputs):
                    first = inputs[0]
                    second = inputs[1]
                    third = inputs[2]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result,
                        first,
                        second,
                        third,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    where_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.first)
                    input1 = self._get_golden_tensor(old_op.second)
                    input2 = self._get_golden_tensor(old_op.third)
                    where_builder._set_golden_tensor(first, input0)
                    where_builder._set_golden_tensor(second, input1)
                    where_builder._set_golden_tensor(third, input2)
                    where_builder._annotate_presharded_arg(first)
                    where_builder._annotate_presharded_arg(second)
                    where_builder._annotate_presharded_arg(third)

                    ordered_inputs.append(first)
                    ordered_inputs.append(second)
                    ordered_inputs.append(third)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                where_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return where_module, where_builder

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

    @split(ttnn.MatmulOp)
    def matmul_split(
        self,
        old_op: ttnn.MatmulOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.matmul_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            matmul_module = Module.create()
            matmul_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.a.type, old_op.b.type]

            with InsertionPoint(matmul_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="matmul_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result,
                        in0,
                        in1,
                        loc=old_op.location,
                        transpose_a=old_op.transpose_a,
                        transpose_b=old_op.transpose_b,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    matmul_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.a)
                    input1 = self._get_golden_tensor(old_op.b)
                    matmul_builder._set_golden_tensor(in0, input0)
                    matmul_builder._set_golden_tensor(in1, input1)
                    matmul_builder._annotate_presharded_arg(in0)
                    matmul_builder._annotate_presharded_arg(in1)

                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                matmul_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return matmul_module, matmul_builder

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

    @split(ttnn.LinearOp)
    def linear_split(
        self,
        old_op: ttnn.LinearOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.linear_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            linear_module = Module.create()
            linear_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.a.type, old_op.b.type]
            has_bias = old_op.bias is not None
            if has_bias:
                op_input_types.append(old_op.bias.type)

            with InsertionPoint(linear_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="linear_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    bias = inputs[2] if has_bias else None
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result,
                        in0,
                        in1,
                        loc=old_op.location,
                        bias=bias,
                        transpose_a=old_op.transpose_a,
                        transpose_b=old_op.transpose_b,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    linear_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.a)
                    input1 = self._get_golden_tensor(old_op.b)
                    linear_builder._set_golden_tensor(in0, input0)
                    linear_builder._set_golden_tensor(in1, input1)
                    linear_builder._annotate_presharded_arg(in0)
                    linear_builder._annotate_presharded_arg(in1)
                    ordered_inputs.append(in0)
                    ordered_inputs.append(in1)
                    if has_bias:
                        bias_golden = self._get_golden_tensor(old_op.bias)
                        linear_builder._set_golden_tensor(bias, bias_golden)
                        linear_builder._annotate_presharded_arg(bias)
                        ordered_inputs.append(bias)

                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                linear_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return linear_module, linear_builder

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

        op_golden_function = get_golden_function(ttnn_op)
        input0 = self._get_golden_tensor(in0)
        golden_output = op_golden_function(
            input0, old_op.min, old_op.max, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {old_op.result: new_op_result}
        return new_op, op_map_dictionary

    @split(ttnn.ClampScalarOp)
    def clamp_scalar_split(
        self,
        old_op: ttnn.ClampScalarOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.clamp_scalar_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            clamp_scalar_module = Module.create()
            clamp_scalar_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(clamp_scalar_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="clamp_scalar_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result,
                        in0,
                        old_op.min,
                        old_op.max,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    clamp_scalar_builder._set_golden_tensor(
                        new_op_result, old_op_result
                    )

                    input0 = self._get_golden_tensor(old_op.input)
                    clamp_scalar_builder._set_golden_tensor(in0, input0)
                    clamp_scalar_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                clamp_scalar_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return clamp_scalar_module, clamp_scalar_builder

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

        op_golden_function = get_golden_function(ttnn_op)
        input0 = self._get_golden_tensor(in0)
        golden_output = op_golden_function(
            input0, old_op.repeats, old_op.dim, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    @split(ttnn.RepeatInterleaveOp)
    def repeat_interleave_split(
        self,
        old_op: ttnn.RepeatInterleaveOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.repeat_interleave_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            repeat_interleave_module = Module.create()
            repeat_interleave_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(repeat_interleave_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="repeat_interleave_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(
                        result,
                        in0,
                        old_op.repeats,
                        old_op.dim,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    repeat_interleave_builder._set_golden_tensor(
                        new_op_result, old_op_result
                    )

                    input0 = self._get_golden_tensor(old_op.input)
                    repeat_interleave_builder._set_golden_tensor(in0, input0)
                    repeat_interleave_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                repeat_interleave_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return repeat_interleave_module, repeat_interleave_builder

    ############### ttnn.FullOp ###############

    @tag(ttnn.FullOp)
    def full(
        self,
        shape: List[int],
        fill_value: Union[int, float],
        device: Optional[Operand] = None,
        output_type: Optional[torch.dtype] = None,
        layout: Optional[ttnn.ir.LayoutAttr] = None,
        buffer_type: Optional[ttnn.ir.BufferTypeAttr] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.full)

        if output_type is None:
            mlir_output_type = self._get_type_from_torch_dtype(torch.float32)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        # Create attributes
        shape_attr = ttnn.ir.ShapeAttr.get(self._ctx, shape)
        if isinstance(fill_value, int):
            fill_value_attr = IntegerAttr.get(IntegerType.get_signless(32), fill_value)
        else:
            fill_value_attr = FloatAttr.get_f32(fill_value)

        layout_attr = ttnn.ir.LayoutAttr.get(self._ctx, layout)
        memory_config_attr = self._create_memory_config_attr(buffer_type)
        result = self.create_ttnn_tensor(
            shape, mlir_output_type, layout=layout, buffer_type=buffer_type
        )

        dtype = self._get_data_type_attribute(result)
        op_golden_function = get_golden_function(ttnn_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            shape_attr, fill_value_attr, mesh_shape_attr, mlir_output_type
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            shape=shape_attr,
            fill_value=fill_value_attr,
            device=device,
            dtype=dtype,
            layout=layout_attr,
            memory_config=memory_config_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.FullOp)
    def full_parser(
        self,
        old_op: ttnn.FullOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.full_parser)
        result = old_op.result.type
        device = global_dict[old_op.device] if old_op.device is not None else None

        new_op = ttnn_op(
            result,
            shape=old_op.shape,
            fill_value=old_op.fill_value,
            device=device,
            dtype=old_op.dtype,
            layout=old_op.layout,
            memory_config=old_op.memory_config,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        op_golden_function = get_golden_function(ttnn_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            old_op.shape, old_op.fill_value, mesh_shape_attr, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.FullOp)
    def full_split(
        self,
        old_op: ttnn.FullOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.full_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            full_module = Module.create()
            full_builder = TTNNBuilder(
                old_ctx, old_loc, self._mesh_shape, self._mesh_dict
            )

            with InsertionPoint(full_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(name="full_module")
                def decorated_func():
                    result = old_op.result.type

                    device = None
                    if old_op.device is not None:
                        mesh_shape_attr = ttnn.ir.MeshShapeAttr.get(
                            old_ctx, *self._mesh_shape
                        )
                        mesh_offset_attr = ttnn.ir.MeshOffsetAttr.get(
                            old_ctx, *self._mesh_offset
                        )
                        new_get_device_op = ttnn.GetDeviceOp(
                            mesh_shape=mesh_shape_attr,
                            mesh_offset=mesh_offset_attr,
                        )
                        device = new_get_device_op.device

                    result = old_op.result.type
                    memory_config_attr = old_op.memory_config

                    new_op = ttnn_op(
                        result,
                        shape=old_op.shape,
                        fill_value=old_op.fill_value,
                        device=device,
                        dtype=old_op.dtype,
                        layout=old_op.layout,
                        memory_config=old_op.memory_config,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    full_builder._set_golden_tensor(new_op_result, old_op_result)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                full_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return full_module, full_builder

    ############### ttnn.ConstantOp ###############

    @tag(ttnn.ConstantOp)
    def constant(
        self,
        value: Union[torch.Tensor, List, float, int],
        device: Optional[Operand] = None,
        output_type: Optional[torch.dtype] = None,
        layout: Optional[ttnn.ir.LayoutAttr] = None,
        buffer_type: Optional[ttnn.ir.BufferTypeAttr] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.constant)

        # Convert value to torch tensor if necessary
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)

        if output_type is None:
            mlir_output_type = self._get_type_from_torch_dtype(value.dtype)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)
            value = value.to(output_type)

        # Create DenseElementsAttr from the tensor
        value_shape = list(value.shape)
        mlir_value_type = RankedTensorType.get(value_shape, mlir_output_type)

        layout_attr = ttnn.ir.LayoutAttr.get(self._ctx, layout)
        memory_config_attr = self._create_memory_config_attr(buffer_type)
        result = self.create_ttnn_tensor(
            value_shape, mlir_output_type, layout=layout, buffer_type=buffer_type
        )

        if value.dtype == torch.bfloat16:
            u16 = value.detach().cpu().view(torch.int16).numpy().astype(np.uint16)
            value_attr = DenseElementsAttr.get(u16, type=result)
        else:
            value_attr = DenseElementsAttr.get(value.numpy())

        dtype = self._get_data_type_attribute(result)
        op_golden_function = get_golden_function(ttnn_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            value_attr, mesh_shape_attr, mlir_output_type
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            value=value_attr,
            device=device,
            dtype=dtype,
            layout=layout_attr,
            memory_config=memory_config_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.ConstantOp)
    def constant_parser(
        self,
        old_op: ttnn.ConstantOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.constant_parser)
        result = old_op.result.type
        device = global_dict[old_op.device] if old_op.device is not None else None

        new_op = ttnn_op(
            result,
            value=old_op.value,
            device=device,
            dtype=old_op.dtype,
            layout=old_op.layout,
            memory_config=old_op.memory_config,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        op_golden_function = get_golden_function(ttnn_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            old_op.value, mesh_shape_attr, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.ConstantOp)
    def constant_split(
        self,
        old_op: ttnn.ConstantOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.constant_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            constant_module = Module.create()
            constant_builder = TTNNBuilder(
                old_ctx, old_loc, self._mesh_shape, self._mesh_dict
            )

            with InsertionPoint(constant_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(name="constant_module")
                def decorated_func():
                    result = old_op.result.type

                    device = None
                    if old_op.device is not None:
                        mesh_shape_attr = ttnn.ir.MeshShapeAttr.get(
                            old_ctx, *self._mesh_shape
                        )
                        mesh_offset_attr = ttnn.ir.MeshOffsetAttr.get(
                            old_ctx, *self._mesh_offset
                        )
                        new_get_device_op = ttnn.GetDeviceOp(
                            mesh_shape=mesh_shape_attr,
                            mesh_offset=mesh_offset_attr,
                        )
                        device = new_get_device_op.device

                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        value=old_op.value,
                        device=device,
                        dtype=old_op.dtype,
                        layout=old_op.layout,
                        memory_config=old_op.memory_config,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    constant_builder._set_golden_tensor(new_op_result, old_op_result)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                constant_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return constant_module, constant_builder

    ############### ttnn.ReshapeOp ###############

    @tag(ttnn.ReshapeOp)
    def reshape(
        self,
        in0: Operand,
        shape: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.reshape)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        shape_attr = ArrayAttr.get(
            [IntegerAttr.get(IntegerType.get_signless(32), s) for s in shape]
        )
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, shape_attr, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            shape=shape_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.ReshapeOp)
    def reshape_parser(
        self,
        old_op: ttnn.ReshapeOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.reshape_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, shape=old_op.shape, loc=old_op.location)
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, old_op.shape, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.ReshapeOp)
    def reshape_split(
        self,
        old_op: ttnn.ReshapeOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.reshape_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            reshape_module = Module.create()
            reshape_builder = TTNNBuilder(
                old_ctx, old_loc, self._mesh_shape, self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(reshape_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="reshape_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, in0, shape=old_op.shape, loc=old_op.location
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    reshape_builder._set_golden_tensor(new_op_result, old_op_result)
                    input0 = self._get_golden_tensor(old_op.input)
                    reshape_builder._set_golden_tensor(in0, input0)
                    reshape_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                reshape_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return reshape_module, reshape_builder

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

        op_golden_function = get_golden_function(ttnn_op)
        input0 = self._get_golden_tensor(in0)
        golden_output = op_golden_function(
            input0, old_op.parameter, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    @split(ttnn.LeakyReluOp)
    def leaky_relu_split(
        self,
        old_op: ttnn.LeakyReluOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.leaky_relu_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            leaky_relu_module = Module.create()
            leaky_relu_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(leaky_relu_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="leaky_relu_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, old_op.parameter, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    leaky_relu_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    leaky_relu_builder._set_golden_tensor(in0, input0)
                    leaky_relu_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                leaky_relu_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return leaky_relu_module, leaky_relu_builder

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

        op_golden_function = get_golden_function(ttnn_op)
        input0 = self._get_golden_tensor(in0)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    @split(ttnn.MishOp)
    def mish_split(
        self,
        old_op: ttnn.MishOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.mish_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            mish_module = Module.create()
            mish_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(mish_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="mish_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    mish_builder._set_golden_tensor(new_op_result, old_op_result)

                    input0 = self._get_golden_tensor(old_op.input)
                    mish_builder._set_golden_tensor(in0, input0)
                    mish_builder._annotate_presharded_arg(in0)

                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                mish_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return mish_module, mish_builder

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

    @split(ttnn.DumpTensorOp)
    def dump_tensor_split(
        self,
        old_op: ttnn.DumpTensorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.dump_tensor_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            dump_tensor_module = Module.create()
            dump_tensor_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(dump_tensor_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="dump_tensor_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    ttnn_op(
                        old_op.file_path,
                        in0,
                        loc=old_op.location,
                    )

                    input0 = self._get_golden_tensor(old_op.input)
                    dump_tensor_builder._set_golden_tensor(in0, input0)
                    dump_tensor_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)

                    return

                new_func_op = decorated_func.func_op
                dump_tensor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return dump_tensor_module, dump_tensor_builder

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
            sharded_output_type = self.create_ttnn_tensor(
                inputs[0].type.shape,
                inputs[0].type.element_type,
                ttnn.Layout.Tile,
                ttnn.BufferType.L1,
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

            if not skip_golden:
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

    ############### ttnn.GetDeviceOp ###############

    @tag(ttnn.GetDeviceOp)
    def get_device(
        self,
        loc: Optional[str] = None,
    ) -> OpResult:
        mesh_shape_attr = ttnn.ir.MeshShapeAttr.get(
            self._ctx, self._mesh_shape[0], self._mesh_shape[1]
        )
        mesh_offset_attr = ttnn.ir.MeshOffsetAttr.get(self._ctx, 0, 0)
        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)
        device_op = ttnn.GetDeviceOp(
            mesh_shape=mesh_shape_attr, mesh_offset=mesh_offset_attr, loc=loc
        )
        return device_op.device

    @parse(ttnn.GetDeviceOp)
    def get_device_parser(
        self,
        old_op: ttnn.GetDeviceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.get_device_parser)
        mesh_shape_attr = old_op.mesh_shape
        mesh_offset_attr = old_op.mesh_offset
        new_op = ttnn_op(mesh_shape=mesh_shape_attr, mesh_offset=mesh_offset_attr)
        return new_op, {old_op.device: new_op.device}

    ############### ttnn.ToLayoutOp ###############

    @tag(ttnn.ToLayoutOp)
    def to_layout(
        self,
        input: Operand,
        layout: Optional[ttnn.ir.LayoutAttr] = None,
        buffer_type: Optional[ttnn.ir.BufferTypeAttr] = ttnn.BufferType.DRAM,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.to_layout)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input_golden = self._get_golden_tensor(input)
        shape = input.type.shape

        layout_attr = ttnn.ir.LayoutAttr.get(self._ctx, layout)
        memory_config_attr = self._create_memory_config_attr(buffer_type)
        result = self.create_ttnn_tensor(
            shape, mlir_output_type, layout=layout, buffer_type=buffer_type
        )
        dtype = (
            self._get_data_type_attribute(result)
            if layout != ttnn.Layout.RowMajor
            else None
        )

        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input_golden, layout_attr, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            input,
            layout=layout_attr,
            dtype=dtype,
            memory_config=memory_config_attr,
            loc=loc,
        )
        op_result = op.result

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.ToLayoutOp)
    def to_layout_parser(
        self,
        old_op: ttnn.ToLayoutOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.to_layout_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type
        layout_attr = old_op.layout

        new_op = ttnn_op(
            result,
            in0,
            layout=layout_attr,
            dtype=old_op.dtype,
            memory_config=old_op.memory_config,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, layout_attr, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    @split(ttnn.ToLayoutOp)
    def to_layout_split(
        self,
        old_op: ttnn.ToLayoutOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.to_layout_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            to_layout_module = Module.create()
            to_layout_builder = TTNNBuilder(
                old_ctx, old_loc, self._mesh_shape, self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(to_layout_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="to_layout_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    layout_attr = old_op.layout

                    new_op = ttnn_op(
                        result,
                        in0,
                        layout=layout_attr,
                        dtype=old_op.dtype,
                        memory_config=old_op.memory_config,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    to_layout_builder._set_golden_tensor(new_op_result, old_op_result)
                    input0 = self._get_golden_tensor(old_op.input)
                    to_layout_builder._set_golden_tensor(in0, input0)
                    to_layout_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                to_layout_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return to_layout_module, to_layout_builder

    ############### ttnn.ToDeviceOp ###############

    @tag(ttnn.ToDeviceOp)
    def to_device(
        self,
        input: Operand,
        device: Operand,
        memory_config=None,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """Move a tensor to device memory.

        If *memory_config* is ``None`` (the default), a DRAM Interleaved
        memory configuration is used.
        """
        ttnn_op = self.get_opview_from_method(TTNNBuilder.to_device)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input_golden = self._get_golden_tensor(input)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input_golden, mlir_output_type)
        shape = input.type.shape
        result = self.create_ttnn_tensor(
            shape, mlir_output_type, ttnn.Layout.RowMajor, ttnn.BufferType.DRAM
        )

        if memory_config is None:
            memory_config = self._create_memory_config_attr(ttnn.BufferType.DRAM)
        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(result, input, device, memory_config=memory_config, loc=loc)
        op_result = op.result

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.ToDeviceOp)
    def to_device_parser(
        self,
        old_op: ttnn.ToDeviceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.to_device_parser)
        in0 = global_dict[old_op.input]
        device = global_dict[old_op.device]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            in0,
            device,
            memory_config=old_op.memory_config,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    @split(ttnn.ToDeviceOp)
    def to_device_split(
        self,
        old_op: ttnn.ToDeviceOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.to_device_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            to_device_module = Module.create()
            to_device_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types: List[Type] = [old_op.input.type]

            with InsertionPoint(to_device_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="to_device_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    mesh_shape_attr = ttnn.ir.MeshShapeAttr.get(
                        old_ctx, *self._mesh_shape
                    )
                    mesh_offset_attr = ttnn.ir.MeshOffsetAttr.get(
                        old_ctx, *self._mesh_offset
                    )
                    new_get_device_op = ttnn.GetDeviceOp(
                        mesh_shape=mesh_shape_attr,
                        mesh_offset=mesh_offset_attr,
                    )

                    result = old_op.result.type
                    device = new_get_device_op.device
                    memory_config_attr = old_op.memory_config

                    new_op = ttnn_op(
                        result,
                        in0,
                        device,
                        memory_config=memory_config_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    to_device_builder._set_golden_tensor(new_op_result, old_op_result)
                    input0 = self._get_golden_tensor(old_op.input)
                    to_device_builder._set_golden_tensor(in0, input0)
                    to_device_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                to_device_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return to_device_module, to_device_builder

    ############### ttnn.FromDeviceOp ###############

    @tag(ttnn.FromDeviceOp)
    def from_device(
        self,
        input: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.from_device)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input_golden = self._get_golden_tensor(input)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input_golden, mlir_output_type)
        shape = input.type.shape
        result = self.create_ttnn_tensor(
            shape,
            mlir_output_type,
            layout=ttnn.Layout.RowMajor,
            buffer_type=ttnn.BufferType.SystemMemory,
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(result, input, loc=loc)
        op_result = op.result

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.FromDeviceOp)
    def from_device_parser(
        self,
        old_op: ttnn.FromDeviceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.from_device_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    @split(ttnn.FromDeviceOp)
    def from_device_split(
        self,
        old_op: ttnn.FromDeviceOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.from_device_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            from_device_module = Module.create()
            from_device_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(from_device_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="from_device_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    from_device_builder._set_golden_tensor(new_op_result, old_op_result)
                    input0 = self._get_golden_tensor(old_op.input)
                    from_device_builder._set_golden_tensor(in0, input0)
                    from_device_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                from_device_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return from_device_module, from_device_builder

    ############### ttnn.DeallocateOp ###############

    @tag(ttnn.DeallocateOp)
    def deallocate(
        self,
        input: Operand,
        force: bool = False,
        loc: Optional[str] = None,
    ) -> None:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.deallocate)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        ttnn_op(
            input,
            force=force,
            loc=loc,
        )

        return

    @parse(ttnn.DeallocateOp)
    def deallocate_parser(
        self,
        old_op: ttnn.DeallocateOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.deallocate_parser)
        in0 = global_dict[old_op.input]

        ttnn_op(
            in0,
            force=old_op.force,
            loc=old_op.location,
        )

        return None, {}

    @split(ttnn.DeallocateOp)
    def deallocate_split(
        self,
        old_op: ttnn.DeallocateOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.deallocate_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            deallocate_module = Module.create()
            deallocate_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(deallocate_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="deallocate_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    ttnn_op(
                        in0,
                        force=old_op.force,
                        loc=old_op.location,
                    )

                    input0 = self._get_golden_tensor(old_op.input)
                    deallocate_builder._set_golden_tensor(in0, input0)
                    deallocate_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)

                    return

                new_func_op = decorated_func.func_op
                deallocate_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return deallocate_module, deallocate_builder

    ############### ttnn.DistributeTensorOp ###############

    @tag(ttnn.DistributeTensorOp)
    def distribute_tensor(
        self,
        input: Operand,
        device: Operand,
        shard_dims: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.distribute_tensor)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        placements = []
        for dim in shard_dims:
            if dim >= 0:
                placements.append(f"<shard, {dim} : i64>")
            else:
                placements.append("<replicate>")
        placements_str = ", ".join(placements)

        mesh_x, mesh_y = self._mesh_shape[0], self._mesh_shape[1]
        config_str = (
            f"#ttnn.mesh_mapper_config<placements = [{placements_str}], "
            f"mesh_shape_override = [{mesh_x} : ui32, {mesh_y} : ui32]>"
        )
        config_attr = Attribute.parse(config_str)

        input0 = self._get_golden_tensor(input)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, config_attr, mlir_output_type)
        host_rm_result = self.create_ttnn_tensor(
            golden_output.shape,
            mlir_output_type,
            ttnn.Layout.RowMajor,
            ttnn.BufferType.SystemMemory,
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            host_rm_result,
            input,
            config_attr,
            device,
            loc=loc,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.DistributeTensorOp)
    def distribute_tensor_parser(
        self,
        old_op: ttnn.DistributeTensorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.distribute_tensor_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        device = global_dict[old_op.mesh_device]
        config_attr = old_op.mapper_config

        new_op = ttnn_op(
            result,
            in0,
            config_attr,
            device,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, config_attr, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    @split(ttnn.DistributeTensorOp)
    def distribute_tensor_split(
        self,
        old_op: ttnn.DistributeTensorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.distribute_tensor_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            distribute_tensor_module = Module.create()
            distribute_tensor_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types: List[Type] = [old_op.input.type]

            with InsertionPoint(distribute_tensor_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="distribute_tensor_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    mesh_shape_attr = ttnn.ir.MeshShapeAttr.get(
                        old_ctx, *self._mesh_shape
                    )
                    mesh_offset_attr = ttnn.ir.MeshOffsetAttr.get(
                        old_ctx, *self._mesh_offset
                    )
                    new_get_device_op = ttnn.GetDeviceOp(
                        mesh_shape=mesh_shape_attr,
                        mesh_offset=mesh_offset_attr,
                    )

                    result = old_op.result.type
                    device = new_get_device_op.device
                    config_attr = old_op.mapper_config

                    new_op = ttnn_op(
                        result,
                        in0,
                        config_attr,
                        device,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    distribute_tensor_builder._set_golden_tensor(
                        new_op_result, old_op_result
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    distribute_tensor_builder._set_golden_tensor(in0, input0)
                    distribute_tensor_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                distribute_tensor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return distribute_tensor_module, distribute_tensor_builder

    ############### ttnn.AggregateTensorOp ###############

    @tag(ttnn.AggregateTensorOp)
    def aggregate_tensor(
        self,
        input: Operand,
        device: Operand,
        shard_dims: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.aggregate_tensor)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input_type = input.type
        input_rank = len(input_type.shape)
        mesh_x, mesh_y = self._mesh_shape[0], self._mesh_shape[1]
        full_mesh_shape = [mesh_x, mesh_y]

        composer_dims = []
        target_mesh_shape = []
        for dim_idx, dim in enumerate(shard_dims):
            if dim >= 0:
                composer_dims.append(dim)
                target_mesh_shape.append(full_mesh_shape[dim_idx])
            else:
                non_overlapping = self._find_non_overlapping_dim(
                    input_rank, shard_dims, composer_dims
                )
                composer_dims.append(non_overlapping)
                target_mesh_shape.append(1)

        dims_str = ", ".join(f"{d} : i32" for d in composer_dims)
        mesh_str = ", ".join(f"{s} : ui32" for s in target_mesh_shape)
        config_str = (
            f"#ttnn.mesh_composer_config<dims = [{dims_str}], "
            f"mesh_shape_override = [{mesh_str}]>"
        )
        config_attr = Attribute.parse(config_str)

        input0 = self._get_golden_tensor(input)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, config_attr, mlir_output_type)

        host_result = self.create_ttnn_tensor(
            golden_output.shape,
            mlir_output_type,
            ttnn.Layout.RowMajor,
            ttnn.BufferType.SystemMemory,
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            host_result,
            input,
            config_attr,
            device,
            loc=loc,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.AggregateTensorOp)
    def aggregate_tensor_parser(
        self,
        old_op: ttnn.AggregateTensorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.aggregate_tensor_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        device = global_dict[old_op.mesh_device]
        config_attr = old_op.composer_config

        new_op = ttnn_op(
            result,
            in0,
            config_attr,
            device,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, config_attr, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    @split(ttnn.AggregateTensorOp)
    def aggregate_tensor_split(
        self,
        old_op: ttnn.AggregateTensorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.aggregate_tensor_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            aggregate_tensor_module = Module.create()
            aggregate_tensor_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types: List[Type] = [old_op.input.type]

            with InsertionPoint(aggregate_tensor_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="aggregate_tensor_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    mesh_shape_attr = ttnn.ir.MeshShapeAttr.get(
                        old_ctx, *self._mesh_shape
                    )
                    mesh_offset_attr = ttnn.ir.MeshOffsetAttr.get(
                        old_ctx, *self._mesh_offset
                    )
                    new_get_device_op = ttnn.GetDeviceOp(
                        mesh_shape=mesh_shape_attr,
                        mesh_offset=mesh_offset_attr,
                    )

                    result = old_op.result.type
                    device = new_get_device_op.device
                    config_attr = old_op.composer_config

                    new_op = ttnn_op(
                        result,
                        in0,
                        config_attr,
                        device,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    aggregate_tensor_builder._set_golden_tensor(
                        new_op_result, old_op_result
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    aggregate_tensor_builder._set_golden_tensor(in0, input0)
                    aggregate_tensor_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                aggregate_tensor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return aggregate_tensor_module, aggregate_tensor_builder

    @staticmethod
    def _find_non_overlapping_dim(
        input_rank: int, shard_dims: List[int], composer_dims: List[int]
    ) -> int:
        for d in range(input_rank - 1, -1, -1):
            if d not in shard_dims and d not in composer_dims:
                return d
        raise ValueError(
            f"No non-overlapping dimension found for input_rank={input_rank}, "
            f"shard_dims={shard_dims}, composer_dims={composer_dims}"
        )

    ############### ttnn.RMSNormPreAllGatherOp ###############

    @tag(ttnn.RMSNormPreAllGatherOp)
    def rms_norm_pre_all_gather(
        self,
        input: Operand,
        residual: Optional[Operand] = None,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.rms_norm_pre_all_gather)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input_golden = self._get_golden_tensor(input)
        residual_golden = (
            self._get_golden_tensor(residual) if residual is not None else None
        )
        op_golden_function = get_golden_function(ttnn_op)

        golden_output = op_golden_function(
            input_golden,
            residual_golden,
            mlir_output_type,
        )
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        output_dtype = self._get_data_type_attribute(input)

        op = ttnn_op(
            result,
            input,
            loc=loc,
            residual=residual,
            dtype=output_dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.RMSNormPreAllGatherOp)
    def rms_norm_pre_all_gather_parser(
        self,
        old_op: ttnn.RMSNormPreAllGatherOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(
            TTNNBuilder.rms_norm_pre_all_gather_parser
        )

        in0 = global_dict[old_op.input]
        residual = global_dict[old_op.residual] if old_op.residual is not None else None
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            in0,
            loc=old_op.location,
            residual=residual,
            dtype=old_op.dtype,
            memory_config=old_op.memory_config,
            compute_config=old_op.compute_config,
            program_config=old_op.program_config,
            use_2d_core_grid=old_op.use_2d_core_grid,
        )
        new_op_result = new_op.result

        input_golden = self._get_golden_tensor(in0)
        residual_golden = (
            self._get_golden_tensor(residual) if residual is not None else None
        )
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input_golden,
            residual_golden,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    @split(ttnn.RMSNormPreAllGatherOp)
    def rms_norm_pre_all_gather_split(
        self,
        old_op: ttnn.RMSNormPreAllGatherOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.rms_norm_pre_all_gather_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            rms_norm_pre_all_gather_module = Module.create()
            rms_norm_pre_all_gather_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]
            if old_op.residual is not None:
                op_input_types.append(old_op.residual.type)

            with InsertionPoint(rms_norm_pre_all_gather_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="rms_norm_pre_all_gather_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    input_idx = 1

                    residual = None
                    if old_op.residual is not None:
                        residual = inputs[input_idx]

                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        in0,
                        loc=old_op.location,
                        residual=residual,
                        dtype=old_op.dtype,
                        memory_config=old_op.memory_config,
                        compute_config=old_op.compute_config,
                        program_config=old_op.program_config,
                        use_2d_core_grid=old_op.use_2d_core_grid,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    rms_norm_pre_all_gather_builder._set_golden_tensor(
                        new_op_result, old_op_result
                    )

                    input_golden = self._get_golden_tensor(old_op.input)
                    rms_norm_pre_all_gather_builder._set_golden_tensor(
                        in0, input_golden
                    )
                    rms_norm_pre_all_gather_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)

                    if old_op.residual is not None:
                        residual_golden = self._get_golden_tensor(old_op.residual)
                        rms_norm_pre_all_gather_builder._set_golden_tensor(
                            residual, residual_golden
                        )
                        rms_norm_pre_all_gather_builder._annotate_presharded_arg(
                            residual
                        )
                        ordered_inputs.append(residual)

                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                rms_norm_pre_all_gather_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return rms_norm_pre_all_gather_module, rms_norm_pre_all_gather_builder

    ############### ttnn.LayerNormPreAllGatherOp ###############

    @tag(ttnn.LayerNormPreAllGatherOp)
    def layer_norm_pre_all_gather(
        self,
        input: Operand,
        residual_input: Optional[Operand] = None,
        recip: Optional[Operand] = None,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.layer_norm_pre_all_gather)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input_golden = self._get_golden_tensor(input)
        residual_golden = (
            self._get_golden_tensor(residual_input)
            if residual_input is not None
            else None
        )
        recip_golden = self._get_golden_tensor(recip) if recip is not None else None
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input_golden,
            residual_golden,
            recip_golden,
            mlir_output_type,
        )
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        output_dtype = self._get_data_type_attribute(input)

        op = ttnn_op(
            result,
            input,
            loc=loc,
            residual_input=residual_input,
            recip=recip,
            dtype=output_dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LayerNormPreAllGatherOp)
    def layer_norm_pre_all_gather_parser(
        self,
        old_op: ttnn.LayerNormPreAllGatherOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(
            TTNNBuilder.layer_norm_pre_all_gather_parser
        )

        in0 = global_dict[old_op.input]
        residual_input = (
            global_dict[old_op.residual_input]
            if old_op.residual_input is not None
            else None
        )
        recip = global_dict[old_op.recip] if old_op.recip is not None else None
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            in0,
            loc=old_op.location,
            residual_input=residual_input,
            recip=recip,
            dtype=old_op.dtype,
            memory_config=old_op.memory_config,
            compute_config=old_op.compute_config,
            program_config=old_op.program_config,
        )
        new_op_result = new_op.result

        input_golden = self._get_golden_tensor(in0)
        residual_golden = (
            self._get_golden_tensor(residual_input)
            if residual_input is not None
            else None
        )
        recip_golden = self._get_golden_tensor(recip) if recip is not None else None
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input_golden,
            residual_golden,
            recip_golden,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    @split(ttnn.LayerNormPreAllGatherOp)
    def layer_norm_pre_all_gather_split(
        self,
        old_op: ttnn.LayerNormPreAllGatherOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(
            TTNNBuilder.layer_norm_pre_all_gather_split
        )

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            layer_norm_pre_all_gather_module = Module.create()
            layer_norm_pre_all_gather_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]
            if old_op.residual_input is not None:
                op_input_types.append(old_op.residual_input.type)
            if old_op.recip is not None:
                op_input_types.append(old_op.recip.type)

            with InsertionPoint(layer_norm_pre_all_gather_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="layer_norm_pre_all_gather_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    input_idx = 1
                    residual_input = None
                    recip = None

                    if old_op.residual_input is not None:
                        residual_input = inputs[input_idx]
                        input_idx += 1

                    if old_op.recip is not None:
                        recip = inputs[input_idx]

                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        in0,
                        loc=old_op.location,
                        residual_input=residual_input,
                        recip=recip,
                        dtype=old_op.dtype,
                        memory_config=old_op.memory_config,
                        compute_config=old_op.compute_config,
                        program_config=old_op.program_config,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    layer_norm_pre_all_gather_builder._set_golden_tensor(
                        new_op_result, old_op_result
                    )

                    input0 = self._get_golden_tensor(old_op.input)
                    layer_norm_pre_all_gather_builder._set_golden_tensor(in0, input0)
                    layer_norm_pre_all_gather_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)

                    if old_op.residual_input is not None:
                        residual_golden = self._get_golden_tensor(old_op.residual_input)
                        layer_norm_pre_all_gather_builder._set_golden_tensor(
                            residual_input, residual_golden
                        )
                        layer_norm_pre_all_gather_builder._annotate_presharded_arg(
                            residual_input
                        )
                        ordered_inputs.append(residual_input)

                    if old_op.recip is not None:
                        recip_golden = self._get_golden_tensor(old_op.recip)
                        layer_norm_pre_all_gather_builder._set_golden_tensor(
                            recip, recip_golden
                        )
                        layer_norm_pre_all_gather_builder._annotate_presharded_arg(
                            recip
                        )
                        ordered_inputs.append(recip)

                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                layer_norm_pre_all_gather_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return layer_norm_pre_all_gather_module, layer_norm_pre_all_gather_builder

    ############### ttnn.LayerNormPostAllGatherOp ###############

    @tag(ttnn.LayerNormPostAllGatherOp)
    def layer_norm_post_all_gather(
        self,
        input: Operand,
        stats: Operand,
        weight: Optional[Operand] = None,
        bias: Optional[Operand] = None,
        epsilon: float = 1e-12,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.layer_norm_post_all_gather)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input_golden = self._get_golden_tensor(input)
        stats_golden = self._get_golden_tensor(stats)
        weight_golden = self._get_golden_tensor(weight) if weight is not None else None
        bias_golden = self._get_golden_tensor(bias) if bias is not None else None
        epsilon_attr = FloatAttr.get_f32(epsilon)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input_golden,
            stats_golden,
            weight_golden,
            bias_golden,
            epsilon_attr,
            mlir_output_type,
        )
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        output_dtype = self._get_data_type_attribute(input)

        op = ttnn_op(
            result,
            input,
            stats,
            loc=loc,
            weight=weight,
            bias=bias,
            epsilon=epsilon_attr,
            dtype=output_dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LayerNormPostAllGatherOp)
    def layer_norm_post_all_gather_parser(
        self,
        old_op: ttnn.LayerNormPostAllGatherOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(
            TTNNBuilder.layer_norm_post_all_gather_parser
        )

        in0 = global_dict[old_op.input]
        stats = global_dict[old_op.stats]
        weight = global_dict[old_op.weight] if old_op.weight is not None else None
        bias = global_dict[old_op.bias] if old_op.bias is not None else None
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            in0,
            stats,
            loc=old_op.location,
            weight=weight,
            bias=bias,
            epsilon=old_op.epsilon,
            dtype=old_op.dtype,
            memory_config=old_op.memory_config,
            compute_config=old_op.compute_config,
            program_config=old_op.program_config,
        )
        new_op_result = new_op.result

        input_golden = self._get_golden_tensor(in0)
        stats_golden = self._get_golden_tensor(stats)
        weight_golden = self._get_golden_tensor(weight) if weight is not None else None
        bias_golden = self._get_golden_tensor(bias) if bias is not None else None
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input_golden,
            stats_golden,
            weight_golden,
            bias_golden,
            old_op.epsilon,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    @split(ttnn.LayerNormPostAllGatherOp)
    def layer_norm_post_all_gather_split(
        self,
        old_op: ttnn.LayerNormPostAllGatherOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(
            TTNNBuilder.layer_norm_post_all_gather_split
        )

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            layer_norm_post_all_gather_module = Module.create()
            layer_norm_post_all_gather_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type, old_op.stats.type]
            if old_op.weight is not None:
                op_input_types.append(old_op.weight.type)
            if old_op.bias is not None:
                op_input_types.append(old_op.bias.type)

            with InsertionPoint(layer_norm_post_all_gather_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="layer_norm_post_all_gather_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    stats = inputs[1]
                    input_idx = 2
                    weight = None
                    bias = None

                    if old_op.weight is not None:
                        weight = inputs[input_idx]
                        input_idx += 1

                    if old_op.bias is not None:
                        bias = inputs[input_idx]

                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        in0,
                        stats,
                        loc=old_op.location,
                        weight=weight,
                        bias=bias,
                        epsilon=old_op.epsilon,
                        dtype=old_op.dtype,
                        memory_config=old_op.memory_config,
                        compute_config=old_op.compute_config,
                        program_config=old_op.program_config,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    layer_norm_post_all_gather_builder._set_golden_tensor(
                        new_op_result, old_op_result
                    )

                    input0 = self._get_golden_tensor(old_op.input)
                    layer_norm_post_all_gather_builder._set_golden_tensor(in0, input0)
                    layer_norm_post_all_gather_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)

                    stats_golden = self._get_golden_tensor(old_op.stats)
                    layer_norm_post_all_gather_builder._set_golden_tensor(
                        stats, stats_golden
                    )
                    layer_norm_post_all_gather_builder._annotate_presharded_arg(stats)
                    ordered_inputs.append(stats)

                    if old_op.weight is not None:
                        weight_golden = self._get_golden_tensor(old_op.weight)
                        layer_norm_post_all_gather_builder._set_golden_tensor(
                            weight, weight_golden
                        )
                        layer_norm_post_all_gather_builder._annotate_presharded_arg(
                            weight
                        )
                        ordered_inputs.append(weight)

                    if old_op.bias is not None:
                        bias_golden = self._get_golden_tensor(old_op.bias)
                        layer_norm_post_all_gather_builder._set_golden_tensor(
                            bias, bias_golden
                        )
                        layer_norm_post_all_gather_builder._annotate_presharded_arg(
                            bias
                        )
                        ordered_inputs.append(bias)

                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                layer_norm_post_all_gather_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return layer_norm_post_all_gather_module, layer_norm_post_all_gather_builder

    ############### ttnn.GatherOp ###############

    @tag(ttnn.GatherOp)
    def gather(
        self,
        in0: Operand,
        index: Operand,
        dim: int,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.gather)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        dim_attr = IntegerAttr.get(IntegerType.get_signless(32), dim)

        input0 = self._get_golden_tensor(in0)
        input_index = self._get_golden_tensor(index)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0,
            input_index,
            dim_attr,
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
            index,
            dim_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.GatherOp)
    def gather_parser(
        self,
        old_op: ttnn.GatherOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.gather_parser)

        in0 = global_dict[old_op.input]
        index = global_dict[old_op.index]
        result = old_op.result.type
        dim_attr = old_op.dim

        new_op = ttnn_op(
            result,
            in0,
            index,
            dim_attr,
            memory_config=old_op.memory_config,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input_index = self._get_golden_tensor(index)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0,
            input_index,
            dim_attr,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.GatherOp)
    def gather_split(
        self,
        old_op: ttnn.GatherOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.gather_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            gather_module = Module.create()
            gather_builder = TTNNBuilder(
                old_ctx, old_loc, self._mesh_shape, self._mesh_dict
            )
            op_input_types = [old_op.input.type, old_op.index.type]

            with InsertionPoint(gather_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="gather_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    index = inputs[1]
                    result = old_op.result.type
                    dim_attr = old_op.dim

                    new_op = ttnn_op(
                        result,
                        in0,
                        index,
                        dim_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    input_index = self._get_golden_tensor(old_op.index)
                    old_op_result = self._get_golden_tensor(old_op.result)
                    gather_builder._set_golden_tensor(new_op_result, old_op_result)
                    gather_builder._set_golden_tensor(in0, input0)
                    gather_builder._set_golden_tensor(index, input_index)
                    gather_builder._annotate_presharded_arg(in0)
                    gather_builder._annotate_presharded_arg(index)
                    ordered_inputs.extend([in0, index])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                gather_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return gather_module, gather_builder

    @tag(ttnn.SamplingOp)
    def sampling(
        self,
        input_values: Operand,
        input_indices: Operand,
        k: Operand,
        p: Operand,
        temp: Operand,
        seed: Optional[int] = None,
        loc: Optional[str] = None,
    ) -> OpResult:
        """Fused top-k + top-p + multinomial sampling on pre-filtered candidates.

        Args:
            input_values: Candidate logit values [batch=32, candidates] bf16.
            input_indices: Global vocab indices for candidates [batch=32, candidates] int32.
            k: Per-request top-k values [batch=32] uint32.
            p: Per-request top-p values [batch=32] bf16.
            temp: Per-request temperature values [batch=32] bf16 (1/temperature).
            seed: Optional random seed for reproducibility.
        Returns:
            Sampled global token indices [batch=32] int32.
        """
        ttnn_op = self.get_opview_from_method(TTNNBuilder.sampling)

        mlir_output_type = self._get_type_from_torch_dtype(torch.int32)

        vals_golden = self._get_golden_tensor(input_values)
        idx_golden = self._get_golden_tensor(input_indices)
        k_golden = self._get_golden_tensor(k)
        p_golden = self._get_golden_tensor(p)
        temp_golden = self._get_golden_tensor(temp)

        seed_attr = None
        if seed is not None:
            seed_attr = IntegerAttr.get(IntegerType.get_unsigned(32), seed)

        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            vals_golden,
            idx_golden,
            k_golden,
            p_golden,
            temp_golden,
            seed_attr,
            mlir_output_type,
        )

        batch = vals_golden.shape[0]
        result = self.create_ttnn_tensor([batch], mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            input_values,
            input_indices,
            k,
            p,
            temp,
            seed=seed_attr,
            loc=loc,
        )
        op_result = op.result
        self._set_golden_tensor(op_result, golden_output)
        return op_result

    ############### ttnn.AllGatherOp ###############

    @tag(ttnn.AllGatherOp)
    def all_gather(
        self,
        input: Operand,
        all_gather_dim: int,
        cluster_axis: int,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.all_gather)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(input)
        all_gather_dim_attr = IntegerAttr.get(
            IntegerType.get_signed(32), all_gather_dim
        )
        cluster_axis_attr = IntegerAttr.get(IntegerType.get_unsigned(32), cluster_axis)

        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0, all_gather_dim_attr, cluster_axis_attr, mlir_output_type
        )
        result = self.create_ttnn_tensor(
            golden_output.shape,
            mlir_output_type,
            ttnn.Layout.Tile,
            ttnn.BufferType.DRAM,
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            input,
            all_gather_dim_attr,
            cluster_axis_attr,
            loc=loc,
        )
        new_op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(new_op_result, golden_output)

        return new_op_result

    @parse(ttnn.AllGatherOp)
    def all_gather_parser(
        self,
        old_op: ttnn.AllGatherOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.all_gather_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        all_gather_dim_attr = old_op.all_gather_dim
        cluster_axis_attr = old_op.cluster_axis

        new_op = ttnn_op(
            result,
            in0,
            all_gather_dim_attr,
            cluster_axis_attr,
            sub_device_id=old_op.sub_device_id,
            memory_config=old_op.memory_config,
            num_links=old_op.num_links,
            topology=old_op.topology,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0,
            all_gather_dim_attr,
            cluster_axis_attr,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    @split(ttnn.AllGatherOp)
    def all_gather_split(
        self,
        old_op: ttnn.AllGatherOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.all_gather_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            all_gather_module = Module.create()
            all_gather_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(all_gather_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="all_gather_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    all_gather_dim_attr = old_op.all_gather_dim
                    cluster_axis_attr = old_op.cluster_axis

                    new_op = ttnn_op(
                        result,
                        in0,
                        all_gather_dim_attr,
                        cluster_axis_attr,
                        sub_device_id=old_op.sub_device_id,
                        memory_config=old_op.memory_config,
                        num_links=old_op.num_links,
                        topology=old_op.topology,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    all_gather_builder._set_golden_tensor(new_op_result, old_op_result)
                    input0 = self._get_golden_tensor(old_op.input)
                    all_gather_builder._set_golden_tensor(in0, input0)
                    all_gather_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                all_gather_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return all_gather_module, all_gather_builder

    ############### ttnn.ReduceScatterOp ###############

    @tag(ttnn.ReduceScatterOp)
    def reduce_scatter(
        self,
        input: Operand,
        reduce_type: ReduceType,
        scatter_dim: int,
        cluster_axis: int,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.reduce_scatter)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(input)
        reduce_type_attr = ttcore.ir.ReduceTypeAttr.get(self._ctx, reduce_type.value)
        scatter_dim_attr = IntegerAttr.get(IntegerType.get_signed(32), scatter_dim)
        cluster_axis_attr = IntegerAttr.get(IntegerType.get_unsigned(32), cluster_axis)

        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0,
            reduce_type_attr,
            scatter_dim_attr,
            cluster_axis_attr,
            mlir_output_type,
        )
        result = self.create_ttnn_tensor(
            golden_output.shape,
            mlir_output_type,
            ttnn.Layout.Tile,
            ttnn.BufferType.DRAM,
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            input,
            reduce_type_attr,
            scatter_dim_attr,
            cluster_axis_attr,
            loc=loc,
        )
        new_op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(new_op_result, golden_output)

        return new_op_result

    @parse(ttnn.ReduceScatterOp)
    def reduce_scatter_parser(
        self,
        old_op: ttnn.ReduceScatterOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.reduce_scatter_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        reduce_type_attr = old_op.reduce_type
        scatter_dim_attr = old_op.scatter_dim
        cluster_axis_attr = old_op.cluster_axis

        new_op = ttnn_op(
            result,
            in0,
            reduce_type_attr,
            scatter_dim_attr,
            cluster_axis_attr,
            sub_device_id=old_op.sub_device_id,
            memory_config=old_op.memory_config,
            num_links=old_op.num_links,
            topology=old_op.topology,
            compute_config=old_op.compute_config,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0,
            reduce_type_attr,
            scatter_dim_attr,
            cluster_axis_attr,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op, {old_op.result: new_op_result}

    @split(ttnn.ReduceScatterOp)
    def reduce_scatter_split(
        self,
        old_op: ttnn.ReduceScatterOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.reduce_scatter_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            reduce_scatter_module = Module.create()
            reduce_scatter_builder = TTNNBuilder(
                old_ctx, old_loc, mesh_name=self._mesh_name, mesh_dict=self._mesh_dict
            )
            op_input_types = [old_op.input.type]

            with InsertionPoint(reduce_scatter_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="reduce_scatter_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    reduce_type_attr = old_op.reduce_type
                    scatter_dim_attr = old_op.scatter_dim
                    cluster_axis_attr = old_op.cluster_axis

                    new_op = ttnn_op(
                        result,
                        in0,
                        reduce_type_attr,
                        scatter_dim_attr,
                        cluster_axis_attr,
                        sub_device_id=old_op.sub_device_id,
                        memory_config=old_op.memory_config,
                        num_links=old_op.num_links,
                        topology=old_op.topology,
                        compute_config=old_op.compute_config,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    old_op_result = self._get_golden_tensor(old_op.result)
                    reduce_scatter_builder._set_golden_tensor(
                        new_op_result, old_op_result
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    reduce_scatter_builder._set_golden_tensor(in0, input0)
                    reduce_scatter_builder._annotate_presharded_arg(in0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                reduce_scatter_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return reduce_scatter_module, reduce_scatter_builder

    # ----- Parse ttnn module ----

    @staticmethod
    def from_module(
        ctx: Context,
        mlir_text: str,
        golden_inputs: Dict[str, List[Dict[int, torch.tensor]]] = None,
    ) -> Tuple(Module, TTNNBuilder):
        if golden_inputs is None:
            golden_inputs = {}

        root_module = Module.parse(mlir_text, ctx)
        loc = Location.unknown(ctx)
        with ctx, loc:
            mesh_name = "mesh"
            mesh_dict = OrderedDict([("x", 1), ("y", 1)])

            meshes = None
            for named_attr in root_module.operation.attributes:
                if named_attr.name != "ttcore.meshes":
                    continue

                meshes = ttcore.ir.MeshesAttr.maybe_downcast(named_attr.attr)
                break

            if meshes:
                mesh = meshes.meshes[0]
                mesh_name = mesh.name
                shape = mesh.shape
                mesh_dict = OrderedDict(
                    x=1 if len(shape) == 1 else shape[0],
                    y=shape[0] if len(shape) == 1 else shape[1],
                )
            else:
                # Check if mesh information is stored in device attributes
                for entry in root_module.body.operations:
                    if isinstance(entry, ttcore.DeviceModuleOp):
                        for device_op_module in entry.regions[0].blocks[0].operations:
                            for device_module_op in (
                                device_op_module.regions[0].blocks[0].operations
                            ):
                                if isinstance(device_module_op, ttcore.DeviceOp):
                                    for attr in device_module_op.attributes:
                                        if attr.name == "sym_name":
                                            mesh_name = attr.attr.value
                                        if attr.name == "device_attr":
                                            device_attr = (
                                                ttcore.ir.DeviceAttr.maybe_downcast(
                                                    attr.attr
                                                )
                                            )
                                            mesh_dict = OrderedDict(
                                                [
                                                    ("x", device_attr.mesh_shape[0]),
                                                    ("y", device_attr.mesh_shape[1]),
                                                ]
                                            )

            ttnn_builder = TTNNBuilder(
                ctx, loc, mesh_name=mesh_name, mesh_dict=mesh_dict
            )
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
                                        if (
                                            isinstance(op, ttnn.GetDeviceOp)
                                            and op.mesh_offset is not None
                                        ):
                                            mesh_offset_attr = (
                                                ttnn.ir.MeshOffsetAttr.maybe_downcast(
                                                    op.mesh_offset
                                                )
                                            )
                                            builder._mesh_offset = [
                                                mesh_offset_attr.x,
                                                mesh_offset_attr.y,
                                            ]
                                        elif (
                                            isinstance(op, func.ReturnOp)
                                            or isinstance(op, ttnn.DeallocateOp)
                                            or isinstance(op, ttnn.GetDeviceOp)
                                        ):
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
                        sub_op_module_builder = None
                        for op in block.operations:
                            if (
                                isinstance(op, ttnn.GetDeviceOp)
                                and op.mesh_offset is not None
                            ):
                                mesh_offset_attr = (
                                    ttnn.ir.MeshOffsetAttr.maybe_downcast(
                                        op.mesh_offset
                                    )
                                )
                                builder._mesh_offset = [
                                    mesh_offset_attr.x,
                                    mesh_offset_attr.y,
                                ]
                            elif (
                                isinstance(op, func.ReturnOp)
                                or isinstance(op, ttnn.DeallocateOp)
                                or isinstance(op, ttnn.GetDeviceOp)
                            ):
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
