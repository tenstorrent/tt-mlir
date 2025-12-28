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

    @split(ttnn.AddOp)
    def add_split(
        self,
        old_op: ttnn.AddOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.add_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            add_module = Module.create()
            add_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(add_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="add_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        add_builder._set_golden_tensor(new_op_result, golden_output)
                        add_builder._set_golden_tensor(lhs, input0)
                        add_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
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

    @split(ttnn.AbsOp)
    def abs_split(
        self,
        old_op: ttnn.AbsOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.abs_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            abs_module = Module.create()
            abs_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(abs_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="abs_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        abs_builder._set_golden_tensor(new_op_result, golden_output)
                        abs_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                abs_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return abs_module, abs_builder

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

    @split(ttnn.CbrtOp)
    def cbrt_split(
        self,
        old_op: ttnn.CbrtOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.cbrt_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            cbrt_module = Module.create()
            cbrt_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(cbrt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="cbrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        cbrt_builder._set_golden_tensor(new_op_result, golden_output)
                        cbrt_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                cbrt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return cbrt_module, cbrt_builder

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

    @split(ttnn.CeilOp)
    def ceil_split(
        self,
        old_op: ttnn.CeilOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.ceil_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            ceil_module = Module.create()
            ceil_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(ceil_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="ceil_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        ceil_builder._set_golden_tensor(new_op_result, golden_output)
                        ceil_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                ceil_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return ceil_module, ceil_builder

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

    @split(ttnn.CosOp)
    def cos_split(
        self,
        old_op: ttnn.CosOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.cos_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            cos_module = Module.create()
            cos_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(cos_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="cos_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        cos_builder._set_golden_tensor(new_op_result, golden_output)
                        cos_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                cos_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return cos_module, cos_builder

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

    @split(ttnn.ErfOp)
    def erf_split(
        self,
        old_op: ttnn.ErfOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.erf_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            erf_module = Module.create()
            erf_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(erf_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="erf_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        erf_builder._set_golden_tensor(new_op_result, golden_output)
                        erf_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                erf_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return erf_module, erf_builder

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

    @split(ttnn.ErfcOp)
    def erfc_split(
        self,
        old_op: ttnn.ErfcOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.erfc_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            erfc_module = Module.create()
            erfc_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(erfc_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="erfc_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        erfc_builder._set_golden_tensor(new_op_result, golden_output)
                        erfc_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                erfc_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return erfc_module, erfc_builder

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

    @split(ttnn.ExpOp)
    def exp_split(
        self,
        old_op: ttnn.ExpOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.exp_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            exp_module = Module.create()
            exp_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(exp_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="exp_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        exp_builder._set_golden_tensor(new_op_result, golden_output)
                        exp_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                exp_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return exp_module, exp_builder

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

    @split(ttnn.FloorOp)
    def floor_split(
        self,
        old_op: ttnn.FloorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.floor_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            floor_module = Module.create()
            floor_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(floor_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="floor_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        floor_builder._set_golden_tensor(new_op_result, golden_output)
                        floor_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                floor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return floor_module, floor_builder

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

    @split(ttnn.GeluOp)
    def gelu_split(
        self,
        old_op: ttnn.GeluOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.gelu_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            gelu_module = Module.create()
            gelu_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(gelu_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="gelu_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        gelu_builder._set_golden_tensor(new_op_result, golden_output)
                        gelu_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                gelu_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return gelu_module, gelu_builder

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

    @split(ttnn.IsFiniteOp)
    def isfinite_split(
        self,
        old_op: ttnn.IsFiniteOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.isfinite_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            isfinite_module = Module.create()
            isfinite_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(isfinite_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="isfinite_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        isfinite_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        isfinite_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                isfinite_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return isfinite_module, isfinite_builder

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

    @split(ttnn.LogicalNotOp)
    def logical_not_split(
        self,
        old_op: ttnn.LogicalNotOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_not_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            not_module = Module.create()
            not_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(not_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_not_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        not_builder._set_golden_tensor(new_op_result, golden_output)
                        not_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                not_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return not_module, not_builder

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

    @split(ttnn.BitwiseNotOp)
    def bitwise_not_split(
        self,
        old_op: ttnn.BitwiseNotOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.bitwise_not_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            not_module = Module.create()
            not_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(not_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="bitwise_not_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        not_builder._set_golden_tensor(new_op_result, golden_output)
                        not_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                not_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return not_module, not_builder

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

    @split(ttnn.NegOp)
    def neg_split(
        self,
        old_op: ttnn.NegOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.neg_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            neg_module = Module.create()
            neg_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(neg_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="neg_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        neg_builder._set_golden_tensor(new_op_result, golden_output)
                        neg_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                neg_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return neg_module, neg_builder

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

    @split(ttnn.TanOp)
    def tan_split(
        self,
        old_op: ttnn.TanOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.tan_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            tan_module = Module.create()
            tan_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(tan_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="tan_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        tan_builder._set_golden_tensor(new_op_result, golden_output)
                        tan_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                tan_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return tan_module, tan_builder

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

    @split(ttnn.AtanOp)
    def atan_split(
        self,
        old_op: ttnn.AtanOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.atan_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            atan_module = Module.create()
            atan_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(atan_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="atan_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        atan_builder._set_golden_tensor(new_op_result, golden_output)
                        atan_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                atan_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return atan_module, atan_builder

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

    @split(ttnn.TanhOp)
    def tanh_split(
        self,
        old_op: ttnn.TanhOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.tanh_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            tanh_module = Module.create()
            tanh_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(tanh_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="tanh_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        tanh_builder._set_golden_tensor(new_op_result, golden_output)
                        tanh_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                tanh_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return tanh_module, tanh_builder

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

    @split(ttnn.ReciprocalOp)
    def reciprocal_split(
        self,
        old_op: ttnn.ReciprocalOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.reciprocal_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            reciprocal_module = Module.create()
            reciprocal_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(reciprocal_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="reciprocal_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        reciprocal_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        reciprocal_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                reciprocal_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return reciprocal_module, reciprocal_builder

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

    @split(ttnn.ReluOp)
    def relu_split(
        self,
        old_op: ttnn.ReluOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.relu_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            relu_module = Module.create()
            relu_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(relu_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="relu_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        relu_builder._set_golden_tensor(new_op_result, golden_output)
                        relu_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                relu_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return relu_module, relu_builder

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

    @split(ttnn.Relu6Op)
    def relu6_split(
        self,
        old_op: ttnn.Relu6Op,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.relu6_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            relu6_module = Module.create()
            relu6_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(relu6_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="relu6_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        relu6_builder._set_golden_tensor(new_op_result, golden_output)
                        relu6_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                relu6_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return relu6_module, relu6_builder

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

    @split(ttnn.RsqrtOp)
    def rsqrt_split(
        self,
        old_op: ttnn.RsqrtOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.rsqrt_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            rsqrt_module = Module.create()
            rsqrt_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(rsqrt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="rsqrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        rsqrt_builder._set_golden_tensor(new_op_result, golden_output)
                        rsqrt_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                rsqrt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return rsqrt_module, rsqrt_builder

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

    @split(ttnn.SigmoidOp)
    def sigmoid_split(
        self,
        old_op: ttnn.SigmoidOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.sigmoid_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            sigmoid_module = Module.create()
            sigmoid_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(sigmoid_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sigmoid_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        sigmoid_builder._set_golden_tensor(new_op_result, golden_output)
                        sigmoid_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                sigmoid_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sigmoid_module, sigmoid_builder

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

    @split(ttnn.SiluOp)
    def silu_split(
        self,
        old_op: ttnn.SiluOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.silu_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            silu_module = Module.create()
            silu_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(silu_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="silu_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        silu_builder._set_golden_tensor(new_op_result, golden_output)
                        silu_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                silu_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return silu_module, silu_builder

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

    @split(ttnn.SignOp)
    def sign_split(
        self,
        old_op: ttnn.SignOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.sign_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            sign_module = Module.create()
            sign_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(sign_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sign_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        sign_builder._set_golden_tensor(new_op_result, golden_output)
                        sign_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                sign_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sign_module, sign_builder

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

    @split(ttnn.SinOp)
    def sin_split(
        self,
        old_op: ttnn.SinOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.sin_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            sin_module = Module.create()
            sin_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(sin_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sin_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        sin_builder._set_golden_tensor(new_op_result, golden_output)
                        sin_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                sin_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sin_module, sin_builder

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

    @split(ttnn.SqrtOp)
    def sqrt_split(
        self,
        old_op: ttnn.SqrtOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.sqrt_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            sqrt_module = Module.create()
            sqrt_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(sqrt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sqrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        sqrt_builder._set_golden_tensor(new_op_result, golden_output)
                        sqrt_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                sqrt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sqrt_module, sqrt_builder

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

    @split(ttnn.TypecastOp)
    def typecast_split(
        self,
        old_op: ttnn.TypecastOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.typecast_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            typecast_module = Module.create()
            typecast_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(typecast_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="typecast_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, old_op.dtype, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(
                            input0,
                            result.element_type,
                        )
                        typecast_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        typecast_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                typecast_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return typecast_module, typecast_builder

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

    @split(ttnn.LogOp)
    def log_split(
        self,
        old_op: ttnn.LogOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.log_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            log_module = Module.create()
            log_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(log_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="log_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        log_builder._set_golden_tensor(new_op_result, golden_output)
                        log_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                log_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return log_module, log_builder

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

    @split(ttnn.Log1pOp)
    def log1p_split(
        self,
        old_op: ttnn.Log1pOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.log1p_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            log1p_module = Module.create()
            log1p_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(log1p_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="log1p_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        log1p_builder._set_golden_tensor(new_op_result, golden_output)
                        log1p_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                log1p_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return log1p_module, log1p_builder

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

    @split(ttnn.Expm1Op)
    def expm1_split(
        self,
        old_op: ttnn.Expm1Op,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.expm1_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            expm1_module = Module.create()
            expm1_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(expm1_module.body):

                ordered_inputs = []

                ordered_outputs = []

                @func.func(*op_input_types, name="expm1_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        expm1_builder._set_golden_tensor(new_op_result, golden_output)
                        expm1_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                expm1_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return expm1_module, expm1_builder

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

    @split(ttnn.EqualOp)
    def eq_split(
        self,
        old_op: ttnn.EqualOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.eq_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            eq_module = Module.create()
            eq_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(eq_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="eq_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        eq_builder._set_golden_tensor(new_op_result, golden_output)
                        eq_builder._set_golden_tensor(lhs, input0)
                        eq_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                eq_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return eq_module, eq_builder

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

    @split(ttnn.NotEqualOp)
    def ne_split(
        self,
        old_op: ttnn.NotEqualOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.ne_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            ne_module = Module.create()
            ne_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(ne_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="ne_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        ne_builder._set_golden_tensor(new_op_result, golden_output)
                        ne_builder._set_golden_tensor(lhs, input0)
                        ne_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                ne_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return ne_module, ne_builder

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

    @split(ttnn.GreaterEqualOp)
    def ge_split(
        self,
        old_op: ttnn.GreaterEqualOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.ge_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            ge_module = Module.create()
            ge_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(ge_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="ge_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        ge_builder._set_golden_tensor(new_op_result, golden_output)
                        ge_builder._set_golden_tensor(lhs, input0)
                        ge_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                ge_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return ge_module, ge_builder

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

    @split(ttnn.GreaterThanOp)
    def gt_split(
        self,
        old_op: ttnn.GreaterThanOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.gt_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            gt_module = Module.create()
            gt_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(gt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="gt_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        gt_builder._set_golden_tensor(new_op_result, golden_output)
                        gt_builder._set_golden_tensor(lhs, input0)
                        gt_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                gt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return gt_module, gt_builder

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

    @split(ttnn.LessEqualOp)
    def le_split(
        self,
        old_op: ttnn.LessEqualOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.le_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            le_module = Module.create()
            le_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(le_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="le_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        le_builder._set_golden_tensor(new_op_result, golden_output)
                        le_builder._set_golden_tensor(lhs, input0)
                        le_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                le_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return le_module, le_builder

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

    @split(ttnn.LessThanOp)
    def lt_split(
        self,
        old_op: ttnn.LessThanOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.lt_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            lt_module = Module.create()
            lt_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(lt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="lt_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        lt_builder._set_golden_tensor(new_op_result, golden_output)
                        lt_builder._set_golden_tensor(lhs, input0)
                        lt_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                lt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return lt_module, lt_builder

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

    @split(ttnn.LogicalAndOp)
    def logical_and_split(
        self,
        old_op: ttnn.LogicalAndOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_and_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            and_module = Module.create()
            and_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(and_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_and_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        and_builder._set_golden_tensor(new_op_result, golden_output)
                        and_builder._set_golden_tensor(lhs, input0)
                        and_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                and_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return and_module, and_builder

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

    @split(ttnn.LogicalLeftShiftOp)
    def logical_left_shift_split(
        self,
        old_op: ttnn.LogicalLeftShiftOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_left_shift_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            lshift_module = Module.create()
            lshift_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(lshift_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_left_shift_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        lshift_builder._set_golden_tensor(new_op_result, golden_output)
                        lshift_builder._set_golden_tensor(lhs, input0)
                        lshift_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                lshift_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return lshift_module, lshift_builder

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

    @split(ttnn.LogicalOrOp)
    def logical_or_split(
        self,
        old_op: ttnn.LogicalOrOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_or_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            or_module = Module.create()
            or_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(or_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_or_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        or_builder._set_golden_tensor(new_op_result, golden_output)
                        or_builder._set_golden_tensor(lhs, input0)
                        or_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                or_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return or_module, or_builder

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

    @split(ttnn.LogicalRightShiftOp)
    def logical_right_shift_split(
        self,
        old_op: ttnn.LogicalRightShiftOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_right_shift_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            rshift_module = Module.create()
            rshift_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(rshift_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_right_shift_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        rshift_builder._set_golden_tensor(new_op_result, golden_output)
                        rshift_builder._set_golden_tensor(lhs, input0)
                        rshift_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                rshift_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return rshift_module, rshift_builder

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

    @split(ttnn.LogicalXorOp)
    def logical_xor_split(
        self,
        old_op: ttnn.LogicalXorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_xor_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            xor_module = Module.create()
            xor_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(xor_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_xor_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        xor_builder._set_golden_tensor(new_op_result, golden_output)
                        xor_builder._set_golden_tensor(lhs, input0)
                        xor_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                xor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return xor_module, xor_builder

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

    @split(ttnn.BitwiseAndOp)
    def bitwise_and_split(
        self,
        old_op: ttnn.BitwiseAndOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.bitwise_and_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            and_module = Module.create()
            and_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(and_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="bitwise_and_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        and_builder._set_golden_tensor(new_op_result, golden_output)
                        and_builder._set_golden_tensor(lhs, input0)
                        and_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                and_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return and_module, and_builder

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

    @split(ttnn.BitwiseOrOp)
    def bitwise_or_split(
        self,
        old_op: ttnn.BitwiseOrOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.bitwise_or_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            or_module = Module.create()
            or_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(or_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="bitwise_or_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        or_builder._set_golden_tensor(new_op_result, golden_output)
                        or_builder._set_golden_tensor(lhs, input0)
                        or_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                or_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return or_module, or_builder

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

    @split(ttnn.BitwiseXorOp)
    def bitwise_xor_split(
        self,
        old_op: ttnn.BitwiseXorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.bitwise_xor_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            xor_module = Module.create()
            xor_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(xor_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="bitwise_xor_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        xor_builder._set_golden_tensor(new_op_result, golden_output)
                        xor_builder._set_golden_tensor(lhs, input0)
                        xor_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                xor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return xor_module, xor_builder

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

    @split(ttnn.MinimumOp)
    def minimum_split(
        self,
        old_op: ttnn.MinimumOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.minimum_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            min_module = Module.create()
            min_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(min_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="minimum_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        min_builder._set_golden_tensor(new_op_result, golden_output)
                        min_builder._set_golden_tensor(lhs, input0)
                        min_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                min_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return min_module, min_builder

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

    @split(ttnn.MaximumOp)
    def maximum_split(
        self,
        old_op: ttnn.MaximumOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.maximum_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            max_module = Module.create()
            max_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(max_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="maximum_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        max_builder._set_golden_tensor(new_op_result, golden_output)
                        max_builder._set_golden_tensor(lhs, input0)
                        max_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                max_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return max_module, max_builder

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

    @split(ttnn.SubtractOp)
    def subtract_split(
        self,
        old_op: ttnn.SubtractOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.subtract_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            sub_module = Module.create()
            sub_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(sub_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="subtract_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        sub_builder._set_golden_tensor(new_op_result, golden_output)
                        sub_builder._set_golden_tensor(lhs, input0)
                        sub_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                sub_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sub_module, sub_builder

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

    @split(ttnn.RemainderOp)
    def remainder_split(
        self,
        old_op: ttnn.RemainderOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.remainder_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            rem_module = Module.create()
            rem_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(rem_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="remainder_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        rem_builder._set_golden_tensor(new_op_result, golden_output)
                        rem_builder._set_golden_tensor(lhs, input0)
                        rem_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                rem_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return rem_module, rem_builder

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

    @split(ttnn.PowTensorOp)
    def pow_tensor_split(
        self,
        old_op: ttnn.PowTensorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.pow_tensor_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            pow_module = Module.create()
            pow_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(pow_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="pow_tensor_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        pow_builder._set_golden_tensor(new_op_result, golden_output)
                        pow_builder._set_golden_tensor(lhs, input0)
                        pow_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                pow_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return pow_module, pow_builder

    # class TTNN_GenericElementwiseBinaryOp

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

    @split(ttnn.Atan2Op)
    def atan2_split(
        self,
        old_op: ttnn.Atan2Op,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.atan2_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            atan2_module = Module.create()
            atan2_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(atan2_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="atan2_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        atan2_builder._set_golden_tensor(new_op_result, golden_output)
                        atan2_builder._set_golden_tensor(lhs, input0)
                        atan2_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                atan2_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return atan2_module, atan2_builder

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

    @split(ttnn.MultiplyOp)
    def multiply_split(
        self,
        old_op: ttnn.MultiplyOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.multiply_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            mul_module = Module.create()
            mul_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(mul_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="multiply_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        mul_builder._set_golden_tensor(new_op_result, golden_output)
                        mul_builder._set_golden_tensor(lhs, input0)
                        mul_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                mul_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return mul_module, mul_builder

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

    @split(ttnn.DivideOp)
    def divide_split(
        self,
        old_op: ttnn.DivideOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.divide_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            div_module = Module.create()
            div_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(div_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="divide_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        div_builder._set_golden_tensor(new_op_result, golden_output)
                        div_builder._set_golden_tensor(lhs, input0)
                        div_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                div_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return div_module, div_builder

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
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

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
            clamp_tensor_builder = TTNNBuilder(old_ctx, old_loc)
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
                        result, in0, min_tensor, max_tensor, loc=old_op.location
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.input)
                        min_tensor_golden = self._get_golden_tensor(old_op.min)
                        max_tensor_golden = self._get_golden_tensor(old_op.max)
                        op_golden_function = get_golden_function(ttnn_op)
                        golden_output = op_golden_function(
                            input0,
                            min_tensor_golden,
                            max_tensor_golden,
                            result.element_type,
                        )
                        clamp_tensor_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        clamp_tensor_builder._set_golden_tensor(in0, input0)
                        clamp_tensor_builder._set_golden_tensor(
                            min_tensor, min_tensor_golden
                        )
                        clamp_tensor_builder._set_golden_tensor(
                            max_tensor, max_tensor_golden
                        )
                        ordered_inputs.extend([in0, min_tensor, max_tensor])
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
        dim: int = 0,
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
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

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
            concat_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [in0.type for in0 in old_op.inputs]

            with InsertionPoint(concat_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="concat_module")
                def decorated_func(*inputs):
                    result = old_op.result.type
                    dim_attr = old_op.dim

                    new_op = ttnn_op(result, inputs, dim=dim_attr, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input_tensors = tuple(
                            [self._get_golden_tensor(in0) for in0 in old_op.inputs]
                        )
                        golden_output = op_golden_function(
                            input_tensors, dim_attr, result.element_type
                        )
                        concat_builder._set_golden_tensor(new_op_result, golden_output)
                        for input_operand, input0olden_tensor in zip(
                            old_op.inputs, input_tensors
                        ):
                            concat_builder._set_golden_tensor(
                                input_operand, input0olden_tensor
                            )
                        ordered_inputs.extend(inputs)
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
        golden_output = op_golden_function(input0, repeat_dims, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

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
            repeat_builder = TTNNBuilder(old_ctx, old_loc)
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

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.input)
                        op_golden_function = get_golden_function(ttnn_op)
                        golden_output = op_golden_function(
                            input0, old_op.repeat_dims, result.element_type
                        )
                        repeat_builder._set_golden_tensor(new_op_result, golden_output)
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
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

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
            where_builder = TTNNBuilder(old_ctx, old_loc)
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

                    new_op = ttnn_op(result, first, second, third, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        first_tensor = self._get_golden_tensor(old_op.first)
                        condition = first_tensor.apply_shardwise(
                            lambda shard: torch.where(
                                shard > 0,
                                torch.tensor(True, device=shard.device),
                                torch.tensor(False, device=shard.device),
                            )
                        )
                        input1 = self._get_golden_tensor(old_op.second)
                        input2 = self._get_golden_tensor(old_op.third)
                        op_golden_function = get_golden_function(ttnn_op)
                        golden_output = op_golden_function(
                            condition, input1, input2, result.element_type
                        )
                        where_builder._set_golden_tensor(new_op_result, golden_output)
                        where_builder._set_golden_tensor(first, first_tensor)
                        where_builder._set_golden_tensor(second, input1)
                        where_builder._set_golden_tensor(third, input2)
                        ordered_inputs.extend([first, second, third])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                where_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return where_module, where_builder

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

        # Determine output MLIR type from inputs or explicit dtype
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

    @split(ttnn.MatmulOp)
    def matmul_split(
        self,
        old_op: ttnn.MatmulOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.matmul_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            matmul_module = Module.create()
            matmul_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [old_op.a.type, old_op.b.type]

            with InsertionPoint(matmul_module.body):

                ordered_inputs: List[Operand] = []
                ordered_outputs: List[Operand] = []

                @func.func(*op_input_types, name="matmul_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
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
                        input0 = self._get_golden_tensor(old_op.a)
                        input1 = self._get_golden_tensor(old_op.b)
                        golden_output = op_golden_function(
                            input0,
                            input1,
                            old_op.transpose_a,
                            old_op.transpose_b,
                            result.element_type,
                        )
                        matmul_builder._set_golden_tensor(new_op_result, golden_output)
                        matmul_builder._set_golden_tensor(lhs, input0)
                        matmul_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                matmul_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return matmul_module, matmul_builder

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

    @split(ttnn.LinearOp)
    def linear_split(
        self,
        old_op: ttnn.LinearOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.linear_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            linear_module = Module.create()
            linear_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [old_op.a.type, old_op.b.type]
            if old_op.bias is not None:
                op_input_types.append(old_op.bias.type)

            with InsertionPoint(linear_module.body):

                ordered_inputs: List[Operand] = []
                ordered_outputs: List[Operand] = []

                @func.func(*op_input_types, name="linear_module")
                def decorated_func(*inputs):
                    a = inputs[0]
                    b = inputs[1]
                    bias = inputs[2] if len(inputs) > 2 else None
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
                        input0 = self._get_golden_tensor(old_op.a)
                        input1 = self._get_golden_tensor(old_op.b)
                        bias_golden = (
                            self._get_golden_tensor(old_op.bias)
                            if old_op.bias is not None
                            else None
                        )
                        golden_output = op_golden_function(
                            input0,
                            input1,
                            bias_golden,
                            old_op.transpose_a,
                            old_op.transpose_b,
                            result.element_type,
                        )
                        linear_builder._set_golden_tensor(new_op_result, golden_output)
                        linear_builder._set_golden_tensor(a, input0)
                        linear_builder._set_golden_tensor(b, input1)
                        if bias is not None and bias_golden is not None:
                            linear_builder._set_golden_tensor(bias, bias_golden)
                        ordered_inputs.extend([i for i in inputs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                linear_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return linear_module, linear_builder

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

    @split(ttnn.ClampScalarOp)
    def clamp_scalar_split(
        self,
        old_op: ttnn.ClampScalarOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.clamp_scalar_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            clamp_scalar_module = Module.create()
            clamp_scalar_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(clamp_scalar_module.body):

                ordered_inputs: List[Operand] = []
                ordered_outputs: List[Operand] = []

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

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(
                            input0, old_op.min, old_op.max, result.element_type
                        )
                        clamp_scalar_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        clamp_scalar_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                clamp_scalar_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return clamp_scalar_module, clamp_scalar_builder

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

    @split(ttnn.RepeatInterleaveOp)
    def repeat_interleave_split(
        self,
        old_op: ttnn.RepeatInterleaveOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.repeat_interleave_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            ri_module = Module.create()
            ri_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(ri_module.body):
                ordered_inputs: List[Operand] = []
                ordered_outputs: List[Operand] = []

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

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(
                            input0,
                            old_op.repeats,
                            old_op.dim,
                            result.element_type,
                        )
                        ri_builder._set_golden_tensor(new_op_result, golden_output)
                        ri_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                ri_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return ri_module, ri_builder

    @tag(ttnn.LeakyReluOp)
    def leaky_relu(
        self,
        in0: Operand,
        parameter: float = 0.01,
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

    @split(ttnn.LeakyReluOp)
    def leaky_relu_split(
        self,
        old_op: ttnn.LeakyReluOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.leaky_relu_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            lr_module = Module.create()
            lr_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(lr_module.body):
                ordered_inputs: List[Operand] = []
                ordered_outputs: List[Operand] = []

                @func.func(*op_input_types, name="leaky_relu_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        in0,
                        old_op.parameter,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(
                            input0,
                            old_op.parameter,
                            result.element_type,
                        )
                        lr_builder._set_golden_tensor(new_op_result, golden_output)
                        lr_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                lr_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return lr_module, lr_builder

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

    @split(ttnn.MishOp)
    def mish_split(
        self,
        old_op: ttnn.MishOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.mish_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            mish_module = Module.create()
            mish_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(mish_module.body):
                ordered_inputs: List[Operand] = []
                ordered_outputs: List[Operand] = []

                @func.func(*op_input_types, name="mish_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        mish_builder._set_golden_tensor(new_op_result, golden_output)
                        mish_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.extend([in0])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                mish_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return mish_module, mish_builder

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

        root_module = Module.parse(mlir_text, ctx)
        loc = Location.unknown(ctx)
        with ctx, loc:
            ttnn_builder = TTNNBuilder(ctx, loc)
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
