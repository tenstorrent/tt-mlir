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

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
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
            add_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(add_module.body):

                @func.func(*op_input_types, name="add_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        add_builder._set_golden_tensor(new_op.result, golden_output)
                        add_builder._set_output_ordering([new_op.result])
                        add_builder._set_golden_tensor(lhs, input0)
                        add_builder._set_golden_tensor(rhs, input1)
                        add_builder._set_input_ordering([lhs, rhs])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

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
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.AbsOp)
    def abs_parser(
        self,
        old_op: ttnn.AbsOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.abs_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, input_operand, loc=old_op.location, dtype=old_op.dtype)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
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
            abs_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(abs_module.body):

                @func.func(*op_input_types, name="abs_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, input_operand, loc=old_op.location, dtype=old_op.dtype
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        abs_builder._set_golden_tensor(new_op.result, golden_output)
                        abs_builder._set_output_ordering([new_op.result])
                        abs_builder._set_golden_tensor(input_operand, input0)
                        abs_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.CbrtOp)
    def cbrt_parser(
        self,
        old_op: ttnn.CbrtOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.cbrt_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            cbrt_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(cbrt_module.body):

                @func.func(*op_input_types, name="cbrt_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        cbrt_builder._set_golden_tensor(new_op.result, golden_output)
                        cbrt_builder._set_output_ordering([new_op.result])
                        cbrt_builder._set_golden_tensor(input_operand, input0)
                        cbrt_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.CeilOp)
    def ceil_parser(
        self,
        old_op: ttnn.CeilOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.ceil_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            ceil_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(ceil_module.body):

                @func.func(*op_input_types, name="ceil_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        ceil_builder._set_golden_tensor(new_op.result, golden_output)
                        ceil_builder._set_output_ordering([new_op.result])
                        ceil_builder._set_golden_tensor(input_operand, input0)
                        ceil_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.CosOp)
    def cos_parser(
        self,
        old_op: ttnn.CosOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.cos_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            cos_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(cos_module.body):

                @func.func(*op_input_types, name="cos_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        cos_builder._set_golden_tensor(new_op.result, golden_output)
                        cos_builder._set_output_ordering([new_op.result])
                        cos_builder._set_golden_tensor(input_operand, input0)
                        cos_builder._set_input_ordering([input_operand])

                    return new_op

        return cos_module, cos_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.ErfOp)
    def erf_parser(
        self,
        old_op: ttnn.ErfOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.erf_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            erf_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(erf_module.body):

                @func.func(*op_input_types, name="erf_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        erf_builder._set_golden_tensor(new_op.result, golden_output)
                        erf_builder._set_output_ordering([new_op.result])
                        erf_builder._set_golden_tensor(input_operand, input0)
                        erf_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.ErfcOp)
    def erfc_parser(
        self,
        old_op: ttnn.ErfcOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.erfc_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            erfc_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(erfc_module.body):

                @func.func(*op_input_types, name="erfc_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        erfc_builder._set_golden_tensor(new_op.result, golden_output)
                        erfc_builder._set_output_ordering([new_op.result])
                        erfc_builder._set_golden_tensor(input_operand, input0)
                        erfc_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.ExpOp)
    def exp_parser(
        self,
        old_op: ttnn.ExpOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.exp_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            exp_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(exp_module.body):

                @func.func(*op_input_types, name="exp_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        exp_builder._set_golden_tensor(new_op.result, golden_output)
                        exp_builder._set_output_ordering([new_op.result])
                        exp_builder._set_golden_tensor(input_operand, input0)
                        exp_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.FloorOp)
    def floor_parser(
        self,
        old_op: ttnn.FloorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.floor_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            floor_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(floor_module.body):

                @func.func(*op_input_types, name="floor_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        floor_builder._set_golden_tensor(new_op.result, golden_output)
                        floor_builder._set_output_ordering([new_op.result])
                        floor_builder._set_golden_tensor(input_operand, input0)
                        floor_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.GeluOp)
    def gelu_parser(
        self,
        old_op: ttnn.GeluOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.gelu_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            gelu_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(gelu_module.body):

                @func.func(*op_input_types, name="gelu_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        gelu_builder._set_golden_tensor(new_op.result, golden_output)
                        gelu_builder._set_output_ordering([new_op.result])
                        gelu_builder._set_golden_tensor(input_operand, input0)
                        gelu_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.IsFiniteOp)
    def isfinite_parser(
        self,
        old_op: ttnn.IsFiniteOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.isfinite_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            isfinite_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(isfinite_module.body):

                @func.func(*op_input_types, name="isfinite_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        isfinite_builder._set_golden_tensor(
                            new_op.result, golden_output
                        )
                        isfinite_builder._set_output_ordering([new_op.result])
                        isfinite_builder._set_golden_tensor(input_operand, input0)
                        isfinite_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.LogicalNotOp)
    def logical_not_parser(
        self,
        old_op: ttnn.LogicalNotOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.logical_not_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            not_module = Module.create()
            not_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(not_module.body):

                @func.func(*op_input_types, name="logical_not_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        not_builder._set_golden_tensor(new_op.result, golden_output)
                        not_builder._set_output_ordering([new_op.result])
                        not_builder._set_golden_tensor(input_operand, input0)
                        not_builder._set_input_ordering([input_operand])

                    return new_op

        return not_module, not_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.BitwiseNotOp)
    def bitwise_not_parser(
        self,
        old_op: ttnn.BitwiseNotOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.bitwise_not_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            not_module = Module.create()
            not_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(not_module.body):

                @func.func(*op_input_types, name="bitwise_not_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        not_builder._set_golden_tensor(new_op.result, golden_output)
                        not_builder._set_output_ordering([new_op.result])
                        not_builder._set_golden_tensor(input_operand, input0)
                        not_builder._set_input_ordering([input_operand])

                    return new_op

        return not_module, not_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.NegOp)
    def neg_parser(
        self,
        old_op: ttnn.NegOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.neg_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            neg_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(neg_module.body):

                @func.func(*op_input_types, name="neg_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        neg_builder._set_golden_tensor(new_op.result, golden_output)
                        neg_builder._set_output_ordering([new_op.result])
                        neg_builder._set_golden_tensor(input_operand, input0)
                        neg_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.TanOp)
    def tan_parser(
        self,
        old_op: ttnn.TanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.tan_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            tan_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(tan_module.body):

                @func.func(*op_input_types, name="tan_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        tan_builder._set_golden_tensor(new_op.result, golden_output)
                        tan_builder._set_output_ordering([new_op.result])
                        tan_builder._set_golden_tensor(input_operand, input0)
                        tan_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.AtanOp)
    def atan_parser(
        self,
        old_op: ttnn.AtanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.atan_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            atan_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(atan_module.body):

                @func.func(*op_input_types, name="atan_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        atan_builder._set_golden_tensor(new_op.result, golden_output)
                        atan_builder._set_output_ordering([new_op.result])
                        atan_builder._set_golden_tensor(input_operand, input0)
                        atan_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.TanhOp)
    def tanh_parser(
        self,
        old_op: ttnn.TanhOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.tanh_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            tanh_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(tanh_module.body):

                @func.func(*op_input_types, name="tanh_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        tanh_builder._set_golden_tensor(new_op.result, golden_output)
                        tanh_builder._set_output_ordering([new_op.result])
                        tanh_builder._set_golden_tensor(input_operand, input0)
                        tanh_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.ReciprocalOp)
    def reciprocal_parser(
        self,
        old_op: ttnn.ReciprocalOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.reciprocal_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            recip_module = Module.create()
            recip_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(recip_module.body):

                @func.func(*op_input_types, name="reciprocal_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        recip_builder._set_golden_tensor(new_op.result, golden_output)
                        recip_builder._set_output_ordering([new_op.result])
                        recip_builder._set_golden_tensor(input_operand, input0)
                        recip_builder._set_input_ordering([input_operand])

                    return new_op

        return recip_module, recip_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.ReluOp)
    def relu_parser(
        self,
        old_op: ttnn.ReluOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.relu_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            relu_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(relu_module.body):

                @func.func(*op_input_types, name="relu_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        relu_builder._set_golden_tensor(new_op.result, golden_output)
                        relu_builder._set_output_ordering([new_op.result])
                        relu_builder._set_golden_tensor(input_operand, input0)
                        relu_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.Relu6Op)
    def relu6_parser(
        self,
        old_op: ttnn.Relu6Op,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.relu6_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            relu6_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(relu6_module.body):

                @func.func(*op_input_types, name="relu6_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        relu6_builder._set_golden_tensor(new_op.result, golden_output)
                        relu6_builder._set_output_ordering([new_op.result])
                        relu6_builder._set_golden_tensor(input_operand, input0)
                        relu6_builder._set_input_ordering([input_operand])

                    return new_op

        return relu6_module, relu6_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.SiluOp)
    def silu_parser(
        self,
        old_op: ttnn.SiluOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.silu_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            silu_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(silu_module.body):

                @func.func(*op_input_types, name="silu_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        silu_builder._set_golden_tensor(new_op.result, golden_output)
                        silu_builder._set_output_ordering([new_op.result])
                        silu_builder._set_golden_tensor(input_operand, input0)
                        silu_builder._set_input_ordering([input_operand])

                    return new_op

        return silu_module, silu_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.RsqrtOp)
    def rsqrt_parser(
        self,
        old_op: ttnn.RsqrtOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.rsqrt_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            rsqrt_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(rsqrt_module.body):

                @func.func(*op_input_types, name="rsqrt_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        rsqrt_builder._set_golden_tensor(new_op.result, golden_output)
                        rsqrt_builder._set_output_ordering([new_op.result])
                        rsqrt_builder._set_golden_tensor(input_operand, input0)
                        rsqrt_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.SigmoidOp)
    def sigmoid_parser(
        self,
        old_op: ttnn.SigmoidOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.sigmoid_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            sig_module = Module.create()
            sig_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(sig_module.body):

                @func.func(*op_input_types, name="sigmoid_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        sig_builder._set_golden_tensor(new_op.result, golden_output)
                        sig_builder._set_output_ordering([new_op.result])
                        sig_builder._set_golden_tensor(input_operand, input0)
                        sig_builder._set_input_ordering([input_operand])

                    return new_op

        return sig_module, sig_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.SignOp)
    def sign_parser(
        self,
        old_op: ttnn.SignOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.sign_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            sign_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(sign_module.body):

                @func.func(*op_input_types, name="sign_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        sign_builder._set_golden_tensor(new_op.result, golden_output)
                        sign_builder._set_output_ordering([new_op.result])
                        sign_builder._set_golden_tensor(input_operand, input0)
                        sign_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.SinOp)
    def sin_parser(
        self,
        old_op: ttnn.SinOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.sin_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            sin_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(sin_module.body):

                @func.func(*op_input_types, name="sin_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        sin_builder._set_golden_tensor(new_op.result, golden_output)
                        sin_builder._set_output_ordering([new_op.result])
                        sin_builder._set_golden_tensor(input_operand, input0)
                        sin_builder._set_input_ordering([input_operand])

                    return new_op

        return sin_module, sin_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.SqrtOp)
    def sqrt_parser(
        self,
        old_op: ttnn.SqrtOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.sqrt_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            sqrt_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(sqrt_module.body):

                @func.func(*op_input_types, name="sqrt_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        sqrt_builder._set_golden_tensor(new_op.result, golden_output)
                        sqrt_builder._set_output_ordering([new_op.result])
                        sqrt_builder._set_golden_tensor(input_operand, input0)
                        sqrt_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.TypecastOp)
    def typecast_parser(
        self,
        old_op: ttnn.TypecastOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.typecast_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            tc_module = Module.create()
            tc_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(tc_module.body):

                @func.func(*op_input_types, name="typecast_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        tc_builder._set_golden_tensor(new_op.result, golden_output)
                        tc_builder._set_output_ordering([new_op.result])
                        tc_builder._set_golden_tensor(input_operand, input0)
                        tc_builder._set_input_ordering([input_operand])

                    return new_op

        return tc_module, tc_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.LogOp)
    def log_parser(
        self,
        old_op: ttnn.LogOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.log_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            log_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(log_module.body):

                @func.func(*op_input_types, name="log_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        log_builder._set_golden_tensor(new_op.result, golden_output)
                        log_builder._set_output_ordering([new_op.result])
                        log_builder._set_golden_tensor(input_operand, input0)
                        log_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.Log1pOp)
    def log1p_parser(
        self,
        old_op: ttnn.Log1pOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.log1p_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            log1p_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(log1p_module.body):

                @func.func(*op_input_types, name="log1p_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        log1p_builder._set_golden_tensor(new_op.result, golden_output)
                        log1p_builder._set_output_ordering([new_op.result])
                        log1p_builder._set_golden_tensor(input_operand, input0)
                        log1p_builder._set_input_ordering([input_operand])

                    return new_op

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.Expm1Op)
    def expm1_parser(
        self,
        old_op: ttnn.Expm1Op,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.expm1_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            expm1_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(expm1_module.body):

                @func.func(*op_input_types, name="expm1_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        expm1_builder._set_golden_tensor(new_op.result, golden_output)
                        expm1_builder._set_output_ordering([new_op.result])
                        expm1_builder._set_golden_tensor(input_operand, input0)
                        expm1_builder._set_input_ordering([input_operand])

                    return new_op

        return expm1_module, expm1_builder

    # class TTNN_ElementwiseUnaryWithFloatParameterOp

    ############### ttnn.LeakyReluOp ###############

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0, mlir_output_type, parameter=parameter
        )

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
            parameter=parameter,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.LeakyReluOp)
    def leaky_relu_parser(
        self,
        old_op: ttnn.LeakyReluOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.leaky_relu_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
            parameter=old_op.parameter,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(
                input0, result.element_type, parameter=float(old_op.parameter)
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
        return new_op, op_map_dictionary

    @split(ttnn.LeakyReluOp)
    def leaky_relu_split(
        self,
        old_op: ttnn.LeakyReluOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.leaky_relu_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            lrelu_module = Module.create()
            lrelu_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(lrelu_module.body):

                @func.func(*op_input_types, name="leaky_relu_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                        parameter=old_op.parameter,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(
                            input0,
                            result.element_type,
                            parameter=float(old_op.parameter),
                        )
                        lrelu_builder._set_golden_tensor(new_op.result, golden_output)
                        lrelu_builder._set_output_ordering([new_op.result])
                        lrelu_builder._set_golden_tensor(input_operand, input0)
                        lrelu_builder._set_input_ordering([input_operand])

                    return new_op

        return lrelu_module, lrelu_builder

    # class TTNN_ElementwiseBinaryOp

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            eq_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(eq_module.body):

                @func.func(*op_input_types, name="eq_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        eq_builder._set_golden_tensor(new_op.result, golden_output)
                        eq_builder._set_output_ordering([new_op.result])
                        eq_builder._set_golden_tensor(lhs, input0)
                        eq_builder._set_golden_tensor(rhs, input1)
                        eq_builder._set_input_ordering([lhs, rhs])

                    return new_op

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            ne_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(ne_module.body):

                @func.func(*op_input_types, name="ne_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        ne_builder._set_golden_tensor(new_op.result, golden_output)
                        ne_builder._set_output_ordering([new_op.result])
                        ne_builder._set_golden_tensor(lhs, input0)
                        ne_builder._set_golden_tensor(rhs, input1)
                        ne_builder._set_input_ordering([lhs, rhs])

                    return new_op

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            ge_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(ge_module.body):

                @func.func(*op_input_types, name="ge_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        ge_builder._set_golden_tensor(new_op.result, golden_output)
                        ge_builder._set_output_ordering([new_op.result])
                        ge_builder._set_golden_tensor(lhs, input0)
                        ge_builder._set_golden_tensor(rhs, input1)
                        ge_builder._set_input_ordering([lhs, rhs])

                    return new_op

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            gt_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(gt_module.body):

                @func.func(*op_input_types, name="gt_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        gt_builder._set_golden_tensor(new_op.result, golden_output)
                        gt_builder._set_output_ordering([new_op.result])
                        gt_builder._set_golden_tensor(lhs, input0)
                        gt_builder._set_golden_tensor(rhs, input1)
                        gt_builder._set_input_ordering([lhs, rhs])

                    return new_op

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            le_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(le_module.body):

                @func.func(*op_input_types, name="le_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        le_builder._set_golden_tensor(new_op.result, golden_output)
                        le_builder._set_output_ordering([new_op.result])
                        le_builder._set_golden_tensor(lhs, input0)
                        le_builder._set_golden_tensor(rhs, input1)
                        le_builder._set_input_ordering([lhs, rhs])

                    return new_op

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            lt_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(lt_module.body):

                @func.func(*op_input_types, name="lt_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        lt_builder._set_golden_tensor(new_op.result, golden_output)
                        lt_builder._set_output_ordering([new_op.result])
                        lt_builder._set_golden_tensor(lhs, input0)
                        lt_builder._set_golden_tensor(rhs, input1)
                        lt_builder._set_input_ordering([lhs, rhs])

                    return new_op

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            land_module = Module.create()
            land_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(land_module.body):

                @func.func(*op_input_types, name="logical_and_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        land_builder._set_golden_tensor(new_op.result, golden_output)
                        land_builder._set_output_ordering([new_op.result])
                        land_builder._set_golden_tensor(lhs, input0)
                        land_builder._set_golden_tensor(rhs, input1)
                        land_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return land_module, land_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            lshl_module = Module.create()
            lshl_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(lshl_module.body):

                @func.func(*op_input_types, name="logical_left_shift_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        lshl_builder._set_golden_tensor(new_op.result, golden_output)
                        lshl_builder._set_output_ordering([new_op.result])
                        lshl_builder._set_golden_tensor(lhs, input0)
                        lshl_builder._set_golden_tensor(rhs, input1)
                        lshl_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return lshl_module, lshl_builder

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            lor_module = Module.create()
            lor_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(lor_module.body):

                @func.func(*op_input_types, name="logical_or_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        lor_builder._set_golden_tensor(new_op.result, golden_output)
                        lor_builder._set_output_ordering([new_op.result])
                        lor_builder._set_golden_tensor(lhs, input0)
                        lor_builder._set_golden_tensor(rhs, input1)
                        lor_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return lor_module, lor_builder

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            lshr_module = Module.create()
            lshr_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(lshr_module.body):

                @func.func(*op_input_types, name="logical_right_shift_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        lshr_builder._set_golden_tensor(new_op.result, golden_output)
                        lshr_builder._set_output_ordering([new_op.result])
                        lshr_builder._set_golden_tensor(lhs, input0)
                        lshr_builder._set_golden_tensor(rhs, input1)
                        lshr_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return lshr_module, lshr_builder

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            lxor_module = Module.create()
            lxor_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(lxor_module.body):

                @func.func(*op_input_types, name="logical_xor_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        lxor_builder._set_golden_tensor(new_op.result, golden_output)
                        lxor_builder._set_output_ordering([new_op.result])
                        lxor_builder._set_golden_tensor(lhs, input0)
                        lxor_builder._set_golden_tensor(rhs, input1)
                        lxor_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return lxor_module, lxor_builder

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            band_module = Module.create()
            band_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(band_module.body):

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
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        band_builder._set_golden_tensor(new_op.result, golden_output)
                        band_builder._set_output_ordering([new_op.result])
                        band_builder._set_golden_tensor(lhs, input0)
                        band_builder._set_golden_tensor(rhs, input1)
                        band_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return band_module, band_builder

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            bor_module = Module.create()
            bor_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(bor_module.body):

                @func.func(*op_input_types, name="bitwise_or_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        bor_builder._set_golden_tensor(new_op.result, golden_output)
                        bor_builder._set_output_ordering([new_op.result])
                        bor_builder._set_golden_tensor(lhs, input0)
                        bor_builder._set_golden_tensor(rhs, input1)
                        bor_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return bor_module, bor_builder

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            bxor_module = Module.create()
            bxor_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(bxor_module.body):

                @func.func(*op_input_types, name="bitwise_xor_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        bxor_builder._set_golden_tensor(new_op.result, golden_output)
                        bxor_builder._set_output_ordering([new_op.result])
                        bxor_builder._set_golden_tensor(lhs, input0)
                        bxor_builder._set_golden_tensor(rhs, input1)
                        bxor_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return bxor_module, bxor_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            min_module = Module.create()
            min_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(min_module.body):

                @func.func(*op_input_types, name="minimum_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        min_builder._set_golden_tensor(new_op.result, golden_output)
                        min_builder._set_output_ordering([new_op.result])
                        min_builder._set_golden_tensor(lhs, input0)
                        min_builder._set_golden_tensor(rhs, input1)
                        min_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return min_module, min_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
        return new_op, op_map_dictionary

    @split(ttnn.MaximumOp)
    def maximum_split(
        self,
        old_op: ttnn.MaximumOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.maximum_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            max_module = Module.create()
            max_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(max_module.body):

                @func.func(*op_input_types, name="maximum_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        max_builder._set_golden_tensor(new_op.result, golden_output)
                        max_builder._set_output_ordering([new_op.result])
                        max_builder._set_golden_tensor(lhs, input0)
                        max_builder._set_golden_tensor(rhs, input1)
                        max_builder._set_input_ordering([lhs, rhs])

                    return new_op

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            sub_module = Module.create()
            sub_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(sub_module.body):

                @func.func(*op_input_types, name="subtract_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        sub_builder._set_golden_tensor(new_op.result, golden_output)
                        sub_builder._set_output_ordering([new_op.result])
                        sub_builder._set_golden_tensor(lhs, input0)
                        sub_builder._set_golden_tensor(rhs, input1)
                        sub_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return sub_module, sub_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            rem_module = Module.create()
            rem_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(rem_module.body):

                @func.func(*op_input_types, name="remainder_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        rem_builder._set_golden_tensor(new_op.result, golden_output)
                        rem_builder._set_output_ordering([new_op.result])
                        rem_builder._set_golden_tensor(lhs, input0)
                        rem_builder._set_golden_tensor(rhs, input1)
                        rem_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return rem_module, rem_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            pow_module = Module.create()
            pow_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(pow_module.body):

                @func.func(*op_input_types, name="pow_tensor_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        pow_builder._set_golden_tensor(new_op.result, golden_output)
                        pow_builder._set_output_ordering([new_op.result])
                        pow_builder._set_golden_tensor(lhs, input0)
                        pow_builder._set_golden_tensor(rhs, input1)
                        pow_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return pow_module, pow_builder

    # class TTNN_GenericElementwiseBinaryOp

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.AtanOp)
    def atan_parser(
        self,
        old_op: ttnn.AtanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.atan_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            atan_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(atan_module.body):

                @func.func(*op_input_types, name="atan_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        atan_builder._set_golden_tensor(new_op.result, golden_output)
                        atan_builder._set_output_ordering([new_op.result])
                        atan_builder._set_golden_tensor(input_operand, input0)
                        atan_builder._set_input_ordering([input_operand])

                    return new_op

        return atan_module, atan_builder

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            mul_module = Module.create()
            mul_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(mul_module.body):

                @func.func(*op_input_types, name="multiply_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        mul_builder._set_golden_tensor(new_op.result, golden_output)
                        mul_builder._set_output_ordering([new_op.result])
                        mul_builder._set_golden_tensor(lhs, input0)
                        mul_builder._set_golden_tensor(rhs, input1)
                        mul_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return mul_module, mul_builder

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

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

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

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            div_module = Module.create()
            div_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(div_module.body):

                @func.func(*op_input_types, name="divide_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        div_builder._set_golden_tensor(new_op.result, golden_output)
                        div_builder._set_output_ordering([new_op.result])
                        div_builder._set_golden_tensor(lhs, input0)
                        div_builder._set_golden_tensor(rhs, input1)
                        div_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return div_module, div_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0,
            input1,
            mlir_output_type,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.MatmulOp)
    def matmul_parser(
        self,
        old_op: ttnn.MatmulOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.matmul_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
            dtype=old_op.dtype,
            transpose_a=old_op.transpose_a,
            transpose_b=old_op.transpose_b,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(
                input0,
                input1,
                result.element_type,
                transpose_a=bool(old_op.transpose_a),
                transpose_b=bool(old_op.transpose_b),
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            mm_module = Module.create()
            mm_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(mm_module.body):

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
                        dtype=old_op.dtype,
                        transpose_a=old_op.transpose_a,
                        transpose_b=old_op.transpose_b,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0,
                            input1,
                            result.element_type,
                            transpose_a=bool(old_op.transpose_a),
                            transpose_b=bool(old_op.transpose_b),
                        )
                        mm_builder._set_golden_tensor(new_op.result, golden_output)
                        mm_builder._set_output_ordering([new_op.result])
                        mm_builder._set_golden_tensor(lhs, input0)
                        mm_builder._set_golden_tensor(rhs, input1)
                        mm_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return mm_module, mm_builder

    ############### ttnn.ClampScalarOp ###############

    @tag(ttnn.ClampScalarOp)
    def clamp_scalar(
        self,
        in0: Operand,
        min_arg: Optional[float] = None,
        max_arg: Optional[float] = None,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.clamp_scalar)
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0, mlir_output_type, min=min_arg, max=max_arg
        )

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
            min=min_arg,
            max=max_arg,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.ClampScalarOp)
    def clamp_scalar_parser(
        self,
        old_op: ttnn.ClampScalarOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.clamp_scalar_parser)
        inp = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            inp,
            loc=old_op.location,
            dtype=old_op.dtype,
            min=old_op.min,
            max=old_op.max,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(inp)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(
                input0,
                result.element_type,
                min=float(old_op.min),
                max=float(old_op.max),
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            csc_module = Module.create()
            csc_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(csc_module.body):

                @func.func(*op_input_types, name="clamp_scalar_module")
                def decorated_func(*inputs):
                    inp = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        inp,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                        min=old_op.min,
                        max=old_op.max,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(
                            input0,
                            result.element_type,
                            min=float(old_op.min),
                            max=float(old_op.max),
                        )
                        csc_builder._set_golden_tensor(new_op.result, golden_output)
                        csc_builder._set_output_ordering([new_op.result])
                        csc_builder._set_golden_tensor(inp, input0)
                        csc_builder._set_input_ordering([inp])

                    return new_op

        return csc_module, csc_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        golden_bias = self._get_golden_tensor(bias) if bias is not None else None
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0,
            input1,
            mlir_output_type,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            bias=golden_bias,
        )

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            bias=bias,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.LinearOp)
    def linear_parser(
        self,
        old_op: ttnn.LinearOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.linear_parser)
        inp = global_dict[old_op.input]
        weight = global_dict[old_op.weight]
        bias = global_dict[old_op.bias] if old_op.bias is not None else None
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            inp,
            weight,
            loc=old_op.location,
            dtype=old_op.dtype,
            transpose_a=old_op.transpose_a,
            transpose_b=old_op.transpose_b,
            bias=bias,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(inp)
            input1 = self._get_golden_tensor(weight)
            golden_bias = self._get_golden_tensor(bias) if bias is not None else None
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(
                input0,
                input1,
                result.element_type,
                transpose_a=bool(old_op.transpose_a),
                transpose_b=bool(old_op.transpose_b),
                bias=golden_bias,
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            lin_module = Module.create()
            lin_builder = TTNNBuilder(old_ctx, old_loc)
            # If bias exists, include it as an input type; else only input and weight
            op_input_types = (
                [old_op.input.type, old_op.weight.type, old_op.bias.type]
                if old_op.bias is not None
                else [old_op.input.type, old_op.weight.type]
            )

            with InsertionPoint(lin_module.body):

                @func.func(*op_input_types, name="linear_module")
                def decorated_func(*inputs):
                    inp = inputs[0]
                    weight = inputs[1]
                    bias = inputs[2] if len(inputs) > 2 else None
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        inp,
                        weight,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                        transpose_a=old_op.transpose_a,
                        transpose_b=old_op.transpose_b,
                        bias=bias,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        input1 = self._get_golden_tensor(old_op.weight)
                        golden_bias = (
                            self._get_golden_tensor(old_op.bias)
                            if old_op.bias is not None
                            else None
                        )
                        golden_output = op_golden_function(
                            input0,
                            input1,
                            result.element_type,
                            transpose_a=bool(old_op.transpose_a),
                            transpose_b=bool(old_op.transpose_b),
                            bias=golden_bias,
                        )
                        lin_builder._set_golden_tensor(new_op.result, golden_output)
                        lin_builder._set_output_ordering([new_op.result])
                        lin_builder._set_golden_tensor(inp, input0)
                        lin_builder._set_golden_tensor(weight, input1)
                        if bias is not None:
                            lin_builder._set_golden_tensor(bias, golden_bias)
                            lin_builder._set_input_ordering([inp, weight, bias])
                        else:
                            lin_builder._set_input_ordering([inp, weight])

                    return new_op

        return lin_module, lin_builder

    ############### ttnn.RepeatOp ###############

    @tag(ttnn.RepeatOp)
    def repeat(
        self,
        in0: Operand,
        dims: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.repeat)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        repeat_dimensions_attr = DenseI64ArrayAttr.get(repeat_dimensions)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0, mlir_output_type, repeat_dimensions=dims
        )

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            repeat_dimensions_attr,
            loc=loc,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.RepeatOp)
    def repeat_parser(
        self,
        old_op: ttnn.RepeatOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.repeat_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type
        repeat_dimensions_attr = old_op.repeat_dimensions

        new_op = ttnn_op(
            result,
            in0,
            repeat_dimensions_attr,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(inp)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(
                input0,
                repeat_dimensions_attr,
                result.element_type,
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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

                @func.func(*op_input_types, name="repeat_module")
                def decorated_func(*inputs):
                    inp = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        inp,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                        repeat_dims=old_op.repeat_dims,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(
                            input0,
                            result.element_type,
                            repeat_dimensions=list(old_op.repeat_dims),
                        )
                        repeat_builder._set_golden_tensor(new_op.result, golden_output)
                        repeat_builder._set_output_ordering([new_op.result])
                        repeat_builder._set_golden_tensor(inp, input0)
                        repeat_builder._set_input_ordering([inp])

                    return new_op

        return repeat_module, repeat_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(
            input0, mlir_output_type, repeats=repeats, dim=dim
        )

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
            repeats=repeats,
            dim=dim,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.RepeatInterleaveOp)
    def repeat_interleave_parser(
        self,
        old_op: ttnn.RepeatInterleaveOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.repeat_interleave_parser)
        inp = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            inp,
            loc=old_op.location,
            dtype=old_op.dtype,
            repeats=old_op.repeats,
            dim=old_op.dim,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(inp)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(
                input0,
                result.element_type,
                repeats=int(old_op.repeats),
                dim=int(old_op.dim),
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
        return new_op, op_map_dictionary

    @split(ttnn.RepeatInterleaveOp)
    def repeat_interleave_split(
        self,
        old_op: ttnn.RepeatInterleaveOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.repeat_interleave_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            rpi_module = Module.create()
            rpi_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(rpi_module.body):

                @func.func(*op_input_types, name="repeat_interleave_module")
                def decorated_func(*inputs):
                    inp = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        inp,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                        repeats=old_op.repeats,
                        dim=old_op.dim,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(
                            input0,
                            result.element_type,
                            repeats=int(old_op.repeats),
                            dim=int(old_op.dim),
                        )
                        rpi_builder._set_golden_tensor(new_op.result, golden_output)
                        rpi_builder._set_output_ordering([new_op.result])
                        rpi_builder._set_golden_tensor(inp, input0)
                        rpi_builder._set_input_ordering([inp])

                    return new_op

        return rpi_module, rpi_builder

    ############### ttnn.WhereOp ###############

    @tag(ttnn.WhereOp)
    def where(
        self,
        condition: Operand,
        true_value: Operand,
        false_value: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.where)
        dtype = self._get_data_type_attribute(true_value)

        mlir_output_type = (
            self.get_type(true_value)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        cond_g = self._get_golden_tensor(condition)
        true_g = self._get_golden_tensor(true_value)
        false_g = self._get_golden_tensor(false_value)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(cond_g, true_g, false_g, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            condition,
            true_value,
            false_value,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.WhereOp)
    def where_parser(
        self,
        old_op: ttnn.WhereOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.where_parser)
        condition = global_dict[old_op.condition]
        true_value = global_dict[old_op.true_value]
        false_value = global_dict[old_op.false_value]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            condition,
            true_value,
            false_value,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            cond_g = self._get_golden_tensor(condition)
            true_g = self._get_golden_tensor(true_value)
            false_g = self._get_golden_tensor(false_value)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(
                cond_g, true_g, false_g, result.element_type
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
                old_op.condition.type,
                old_op.true_value.type,
                old_op.false_value.type,
            ]

            with InsertionPoint(where_module.body):

                @func.func(*op_input_types, name="where_module")
                def decorated_func(*inputs):
                    condition = inputs[0]
                    true_value = inputs[1]
                    false_value = inputs[2]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        condition,
                        true_value,
                        false_value,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        cond_g = self._get_golden_tensor(old_op.condition)
                        true_g = self._get_golden_tensor(old_op.true_value)
                        false_g = self._get_golden_tensor(old_op.false_value)
                        golden_output = op_golden_function(
                            cond_g, true_g, false_g, result.element_type
                        )
                        where_builder._set_golden_tensor(new_op.result, golden_output)
                        where_builder._set_output_ordering([new_op.result])
                        where_builder._set_golden_tensor(condition, cond_g)
                        where_builder._set_golden_tensor(true_value, true_g)
                        where_builder._set_golden_tensor(false_value, false_g)
                        where_builder._set_input_ordering(
                            [condition, true_value, false_value]
                        )

                    return new_op

        return where_module, where_builder

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
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.MishOp)
    def mish_parser(
        self,
        old_op: ttnn.MishOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.mish_parser)
        input_operand = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            input_operand,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
        return new_op, op_map_dictionary

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
            mish_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(mish_module.body):

                @func.func(*op_input_types, name="mish_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        input_operand,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        mish_builder._set_golden_tensor(new_op.result, golden_output)
                        mish_builder._set_output_ordering([new_op.result])
                        mish_builder._set_golden_tensor(input_operand, input0)
                        mish_builder._set_input_ordering([input_operand])

                    return new_op

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

    ############### ttnn.ClampTensorOp ###############

    @tag(ttnn.ClampTensorOp)
    def clamp_tensor(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.clamp_tensor)
        dtype = self._get_data_type_attribute(in0)

        mlir_output_type = (
            self.get_type(in0)
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        input2 = self._get_golden_tensor(in2)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, input2, mlir_output_type)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            in2,
            loc=loc,
            dtype=dtype,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.ClampTensorOp)
    def clamp_tensor_parser(
        self,
        old_op: ttnn.ClampTensorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.clamp_tensor_parser)
        inp = global_dict[old_op.input]
        min_t = global_dict[old_op.min]
        max_t = global_dict[old_op.max]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            inp,
            min_t,
            max_t,
            loc=old_op.location,
            dtype=old_op.dtype,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(inp)
            input1 = self._get_golden_tensor(min_t)
            input2 = self._get_golden_tensor(max_t)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(
                input0, input1, input2, result.element_type
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            ctt_module = Module.create()
            ctt_builder = TTNNBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type, old_op.min.type, old_op.max.type]

            with InsertionPoint(ctt_module.body):

                @func.func(*op_input_types, name="clamp_tensor_module")
                def decorated_func(*inputs):
                    inp = inputs[0]
                    min_t = inputs[1]
                    max_t = inputs[2]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        inp,
                        min_t,
                        max_t,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        input1 = self._get_golden_tensor(old_op.min)
                        input2 = self._get_golden_tensor(old_op.max)
                        golden_output = op_golden_function(
                            input0, input1, input2, result.element_type
                        )
                        ctt_builder._set_golden_tensor(new_op.result, golden_output)
                        ctt_builder._set_output_ordering([new_op.result])
                        ctt_builder._set_golden_tensor(inp, input0)
                        ctt_builder._set_golden_tensor(min_t, input1)
                        ctt_builder._set_golden_tensor(max_t, input2)
                        ctt_builder._set_input_ordering([inp, min_t, max_t])

                    return new_op

        return ctt_module, ctt_builder

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
        dtype = self._get_data_type_attribute(ins[0])

        mlir_output_type = (
            self.get_type(ins[0])
            if output_type is None
            else self._get_type_from_torch_dtype(output_type)
        )

        input_tensors = tuple([self._get_golden_tensor(i) for i in ins])
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input_tensors, dim=dim)

        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        loc = self._get_location() if loc is None else Location.name(loc)

        op = ttnn_op(
            result,
            ins,
            loc=loc,
            dtype=dtype,
            dim=dim,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(ttnn.ConcatOp)
    def concat_parser(
        self,
        old_op: ttnn.ConcatOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.concat_parser)
        inputs = [global_dict[inp] for inp in old_op.inputs]
        result = old_op.result.type

        new_op = ttnn_op(
            result,
            inputs,
            loc=old_op.location,
            dtype=old_op.dtype,
            dim=old_op.dim,
        )

        if not self._disable_golden_check:
            input_tensors = tuple([self._get_golden_tensor(inp) for inp in inputs])
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input_tensors, dim=int(old_op.dim))
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {old_op.result: new_op.result}
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
            op_input_types = [inp.type for inp in old_op.inputs]

            with InsertionPoint(concat_module.body):

                @func.func(*op_input_types, name="concat_module")
                def decorated_func(*inputs):
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        inputs,
                        loc=old_op.location,
                        dtype=old_op.dtype,
                        dim=old_op.dim,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input_tensors = tuple(
                            [self._get_golden_tensor(inp) for inp in old_op.inputs]
                        )
                        golden_output = op_golden_function(
                            input_tensors, dim=int(old_op.dim)
                        )
                        concat_builder._set_golden_tensor(new_op.result, golden_output)
                        concat_builder._set_output_ordering([new_op.result])
                        for input_operand, input_golden_tensor in zip(
                            old_op.inputs, input_tensors
                        ):
                            concat_builder._set_golden_tensor(
                                input_operand, input_golden_tensor
                            )
                        concat_builder._set_input_ordering(old_op.inputs)

                    return new_op

        return concat_module, concat_builder

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
        ctx: Context, mlir_text: str, golden_inputs: List[torch.tensor] = None
    ) -> Tuple(Module, TTNNBuilder):
        if golden_inputs is None:
            golden_inputs = []

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
                                for block in device_module_op.body:
                                    for op in block.operations:
                                        if isinstance(op, func.ReturnOp):
                                            continue
                                        else:
                                            sub_op_module_builder = builder.split_op(op)
                                            sub_modules_and_builders.append(
                                                sub_op_module_builder
                                            )
                elif isinstance(entry, func.FuncOp):
                    for block in entry.body:
                        for op in block.operations:
                            if isinstance(op, func.ReturnOp):
                                continue
                            else:
                                sub_op_module_builder = builder.split_op(op)
                                sub_modules_and_builders.append(sub_op_module_builder)

        return sub_modules_and_builders
