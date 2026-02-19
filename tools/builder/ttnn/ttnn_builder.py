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
        split_on_demand: bool = False,
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
