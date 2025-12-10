# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Callable, Dict, Any, Sequence
import torch
from enum import Enum, auto
import re
from collections import OrderedDict
import math

from ttmlir.ir import *
from ttmlir.dialects import stablehlo, sdy, mpmd, func

from builder.base.builder import *
from builder.base.builder_utils import *

from golden import *


class StableHLOBuilder(Builder):

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

        self._arg_attrs: Dict[Operand, Dict[str, Attribute]] = {}

    # ----- Class helper methods -----

    @classmethod
    def build_opname_to_opview_map(cls):
        for name, obj in inspect.getmembers(stablehlo, inspect.isclass):
            if issubclass(obj, OpView) and obj is not OpView:
                op_name = getattr(obj, "OPERATION_NAME", None)

                if op_name is not None:
                    cls.opname_to_opview_map[op_name] = obj

    # ----- Public methods -----

    @property
    def arg_attrs(self) -> Dict[Operand, Dict[str, Attribute]]:
        return self._arg_attrs

    def get_arg_attrs(self, func_op: FuncOp) -> ArrayAttr:
        attrs = []
        for i, operand in enumerate(self._ordered_inputs):
            if operand in self._arg_attrs:
                attrs.append(DictAttr.get(self._arg_attrs[operand]))
            else:
                attrs.append(func_op.arg_attrs[i])

        return ArrayAttr.get(attrs)

    def create_sharding_attr_from_tuples(
        self,
        mesh_name: str,
        shardings: List[Tuple[str, bool]],
        replicated_axes: List[sdy.AxisRefAttr] = [],
        unreduced_axes: List[sdy.AxisRefAttr] = [],
    ) -> sdy.TensorShardingPerValueAttr:
        """
        Creates a tensor sharding per value attribute from a list of tuples.
        Each tuple contains a mesh name and a boolean indicating whether the sharding is closed.

        Parameters
        ----------
        mesh_name : str
            The name of the mesh to which the tensor sharding applies
        shardings : List[Tuple[str, bool]]
            A list of tuples, each containing a mesh name and a boolean indicating whether the sharding is closed

        Returns
        -------
        (*sdy.TensorShardingPerValueAttr*)
            A tensor sharding per value attribute that describes how tensors are distributed across the mesh
        """
        dimension_shardings = []
        for sharding in shardings:
            axis_ref_name, is_closed = sharding
            axes = []
            if axis_ref_name != "":
                axes = [self.axis_ref_attr(name=axis_ref_name)]
            dimension_sharding = self.dimension_sharding_attr(
                axes=axes, is_closed=is_closed
            )
            dimension_shardings.append(dimension_sharding)

        tensor_sharding = self.tensor_sharding_attr(
            mesh_name, dimension_shardings, replicated_axes, unreduced_axes
        )
        return self.tensor_sharding_per_value_attr([tensor_sharding])

    # ----- Private Methods ----

    def _create_mesh_attr_from_ordered_dict(
        self,
        mesh_dict: OrderedDict[str, int],
    ) -> sdy.MeshAttr:
        axes = [
            self.mesh_axis_attr(name=axis_name, size=size)
            for axis_name, size in mesh_dict.items()
        ]
        return self.mesh_attr(axes)

    def _get_mesh_attr(self, mesh_name: str) -> sdy.MeshAttr:
        if mesh_name not in self._meshes:
            raise ValueError(
                f"Mesh '{mesh_name}' not found. Available meshes: {list(self._meshes.keys())}"
            )

        mesh_dict = self._meshes[mesh_name]
        axes = [
            self.mesh_axis_attr(name=axis_name, size=size)
            for axis_name, size in mesh_dict.items()
        ]
        return self.mesh_attr(axes)

    def _get_mesh(self, mesh_name: str = "mesh") -> sdy.Mesh:
        return self.mesh(mesh_name, self._get_mesh_attr(mesh_name))

    def _get_output_shape_and_type(
        self,
        organize_golden_args: Callable,
        inputs: List[Operand],
        op_stablehlo_function: Callable,
        golden_kwargs: dict = {},
    ):
        op_golden_function = get_golden_function(op_stablehlo_function, **golden_kwargs)
        if op_golden_function is None:
            return None

        # If the op has no input, just call golden function with kwargs (e.g., zeros).
        if len(inputs) == 0:
            golden_output = op_golden_function(**golden_kwargs)
        else:
            golden_output = op_golden_function(
                *(organize_golden_args(inputs)), **golden_kwargs
            )

        return golden_output.shape, golden_output.dtype

    def _op_proxy(
        self,
        op_stablehlo_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
        organize_stablehlo_args: Optional[Callable] = None,
        organize_golden_args: Optional[Callable] = None,
        output_shape: Optional[Shape] = None,
        output_type: Optional[Type] = None,
        output_create_fn: Optional[Callable] = None,
        golden_kwargs: dict = {},
        stablehlo_kwargs: dict = {},
        loc: Optional[Union[str, Location]] = None,
        skip_golden: bool = False,
    ) -> Any:
        if not golden_kwargs:
            golden_kwargs = stablehlo_kwargs

        if organize_golden_args is None:
            organize_golden_args = self._organize_eltwise_golden

        with self._ctx, self._loc:
            id = self._get_next_global_id()
            loc = (
                self._get_loc_from_str(loc)
                if loc is not None
                else self._get_loc_of_extra_file_callee(id=id)
            )

            # Most StableHLO ops have MLIR type inference, so output is not needed.
            # Only create output if user explicitly provides output_shape, output_type, or output_create_fn
            # (e.g., for ops like broadcast_in_dim that don't have type inference)
            output = None
            if (
                output_shape is not None
                or output_type is not None
                or output_create_fn is not None
            ):
                # User explicitly requested output creation
                # Try to get shape/type from golden function if not fully provided
                output_shape_and_type = self._get_output_shape_and_type(
                    organize_golden_args, inputs, op_stablehlo_function, golden_kwargs
                )

                if not output_shape_and_type:
                    # No golden function - user must provide both shape and type
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
                    # Use provided values if available, otherwise use calculated
                    output_shape = (
                        calculated_output_shape
                        if output_shape is None
                        else output_shape
                    )
                    output_type = (
                        self._get_type_from_torch_dtype(calculated_output_type)
                        if output_type is None
                        else output_type
                    )

                # Create output tensor
                if output_create_fn is not None:
                    output = output_create_fn(output_shape, output_type)
                else:
                    output = self._create_ranked_tensor_type(output_shape, output_type)

            # Custom argument organization and create the stabelhlo op
            if organize_stablehlo_args is not None:
                stablehlo_args = organize_stablehlo_args(
                    inputs, output, stablehlo_kwargs
                )
                op = op_stablehlo_function(*stablehlo_args, loc=loc, **stablehlo_kwargs)
            else:
                # Default: elementwise binary operations
                op = op_stablehlo_function(*inputs, loc=loc, **stablehlo_kwargs)

            if unit_attrs is not None:
                for attr_name in unit_attrs:
                    op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

            if sharding_attr is not None:
                op.operation.attributes["sdy.sharding"] = sharding_attr

            if not skip_golden and not self._disable_golden_check:
                op_golden_function = get_golden_function(
                    op_stablehlo_function, **golden_kwargs
                )
                if op_golden_function is not None:
                    golden_output = op_golden_function(
                        *(organize_golden_args(inputs)), **golden_kwargs
                    )
                    self._set_golden_tensor(op.result, golden_output)

            return op.result

    def _eltwise_proxy(
        self,
        op_stablehlo_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        return self._op_proxy(op_stablehlo_function, inputs, unit_attrs, sharding_attr)

    def create_tensor_encoding(
        self, shape: Shape, element_type: Union[torch.dtype, TypeInfo]
    ) -> ttnn.ir.TTNNLayoutAttr:
        return None

    # ----- Public StableHLO Op Generators ----

    ################ stablehlo.AddOp ###############

    @tag(stablehlo.AddOp)
    def add(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.add)

        op = stablehlo_op(
            in0,
            in1,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, op.result.type.element_type
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.AddOp)
    def add_parser(
        self,
        old_op: stablehlo.AddOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.add_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op.result.type.element_type
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.AddOp)
    def add_split(
        self,
        old_op: stablehlo.AddOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.add_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            add_module = Module.create()
            add_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(add_module.body):

                @func.func(*op_input_types, name="add_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op.result.type.element_type
                        )
                        add_builder._set_golden_tensor(new_op, golden_output)
                        add_builder._set_output_ordering([new_op])
                        add_builder._set_golden_tensor(lhs, input0)
                        add_builder._set_golden_tensor(rhs, input1)
                        add_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return add_module, add_builder

    ################ stablehlo.AndOp ###############

    @tag(stablehlo.AndOp)
    def and_(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.and_)

        op = stablehlo_op(
            in0,
            in1,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, op.result.type.element_type
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.AndOp)
    def and_parser(
        self,
        old_op: stablehlo.AndOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.and_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op.result.type.element_type
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.AndOp)
    def and_split(
        self,
        old_op: stablehlo.AndOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.and_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            and_module = Module.create()
            and_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(and_module.body):

                @func.func(*op_input_types, name="and_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op.result.type.element_type
                        )
                        and_builder._set_golden_tensor(new_op, golden_output)
                        and_builder._set_output_ordering([new_op])
                        and_builder._set_golden_tensor(lhs, input0)
                        and_builder._set_golden_tensor(rhs, input1)
                        and_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return and_module, and_builder

    ################ stablehlo.AbsOp ###############

    @tag(stablehlo.AbsOp)
    def abs(
        self,
        in0: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.abs)

        op = stablehlo_op(
            in0,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op.result.type.element_type)
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.AbsOp)
    def abs_parser(
        self,
        old_op: stablehlo.AbsOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.abs_parser)
        operand = global_dict[old_op.operand]

        new_op = stablehlo_op(
            operand,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.AbsOp)
    def abs_split(
        self,
        old_op: stablehlo.AbsOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.abs_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            abs_module = Module.create()
            abs_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
            ]

            with InsertionPoint(abs_module.body):

                @func.func(*op_input_types, name="abs_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op.result.type.element_type
                        )
                        abs_builder._set_golden_tensor(new_op, golden_output)
                        abs_builder._set_output_ordering([new_op])
                        abs_builder._set_golden_tensor(operand, input0)
                        abs_builder._set_input_ordering([operand])

                    return new_op

        return abs_module, abs_builder

    ################ stablehlo.CompareOp ###############

    @tag(stablehlo.CompareOp)
    def compare(
        self,
        lhs: Operand,
        rhs: Operand,
        comparison_direction: str,
        compare_type: Optional[str] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.compare)

        # Get comparison direction attribute
        comparison_direction_attr = stablehlo.ComparisonDirectionAttr.get(
            comparison_direction, context=self._ctx
        )

        # Get optional compare type attribute
        compare_type_attr = None
        if compare_type is not None:
            compare_type_attr = stablehlo.ComparisonTypeAttr.get(
                compare_type, context=self._ctx
            )

        op = stablehlo_op(
            lhs,
            rhs,
            comparison_direction=comparison_direction_attr,
            compare_type=compare_type_attr,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            lhs_golden = self._get_golden_tensor(lhs)
            rhs_golden = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                lhs_golden,
                rhs_golden,
                comparison_direction,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.CompareOp)
    def compare_parser(
        self,
        old_op: stablehlo.CompareOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.compare_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        # Extract comparison direction from the attribute
        # The attribute string looks like "#stablehlo<comparison_direction LT>"
        # We need to extract just "LT"
        comparison_direction_attr = old_op.comparison_direction
        attr_str = str(comparison_direction_attr)
        comparison_direction = attr_str.split()[-1].rstrip(">")

        # Extract compare type if present
        compare_type_attr = old_op.compare_type if old_op.compare_type else None

        new_op = stablehlo_op(
            lhs,
            rhs,
            comparison_direction=comparison_direction_attr,
            compare_type=compare_type_attr,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            lhs_golden = self._get_golden_tensor(lhs)
            rhs_golden = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                lhs_golden,
                rhs_golden,
                comparison_direction,
                new_op.result.type.element_type,
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.CompareOp)
    def compare_split(
        self,
        old_op: stablehlo.CompareOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.compare_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)

        # Extract comparison direction from the attribute
        # The attribute string looks like "#stablehlo<comparison_direction LT>"
        comparison_direction_attr = old_op.comparison_direction
        attr_str = str(comparison_direction_attr)
        comparison_direction = attr_str.split()[-1].rstrip(">")
        compare_type_attr = old_op.compare_type if old_op.compare_type else None

        with old_context, old_loc:
            compare_module = Module.create()
            compare_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(compare_module.body):

                @func.func(*op_input_types, name="compare_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(
                        lhs,
                        rhs,
                        comparison_direction=comparison_direction_attr,
                        compare_type=compare_type_attr,
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        lhs_golden = self._get_golden_tensor(old_op.lhs)
                        rhs_golden = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            lhs_golden,
                            rhs_golden,
                            comparison_direction,
                            new_op.result.type.element_type,
                        )
                        compare_builder._set_golden_tensor(new_op, golden_output)
                        compare_builder._set_output_ordering([new_op])
                        compare_builder._set_golden_tensor(lhs, lhs_golden)
                        compare_builder._set_golden_tensor(rhs, rhs_golden)
                        compare_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return compare_module, compare_builder

    ################ stablehlo.CeilOp ###############

    @tag(stablehlo.CeilOp)
    def ceil(
        self,
        in0: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.ceil)

        op = stablehlo_op(
            in0,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op.result.type.element_type)
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.CeilOp)
    def ceil_parser(
        self,
        old_op: stablehlo.CeilOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.ceil_parser)
        operand = global_dict[old_op.operand]

        new_op = stablehlo_op(
            operand,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.CeilOp)
    def ceil_split(
        self,
        old_op: stablehlo.CeilOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.ceil_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            ceil_module = Module.create()
            ceil_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
            ]

            with InsertionPoint(ceil_module.body):

                @func.func(*op_input_types, name="ceil_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op.result.type.element_type
                        )
                        ceil_builder._set_golden_tensor(new_op, golden_output)
                        ceil_builder._set_output_ordering([new_op])
                        ceil_builder._set_golden_tensor(operand, input0)
                        ceil_builder._set_input_ordering([operand])

                    return new_op

        return ceil_module, ceil_builder

    ################ stablehlo.DivOp ###############

    @tag(stablehlo.DivOp)
    def divide(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.divide)

        op = stablehlo_op(
            in0,
            in1,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, op.result.type.element_type
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.DivOp)
    def divide_parser(
        self,
        old_op: stablehlo.DivOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.divide_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op.result.type.element_type
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.DivOp)
    def divide_split(
        self,
        old_op: stablehlo.DivOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.divide_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            divide_module = Module.create()
            divide_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(divide_module.body):

                @func.func(*op_input_types, name="divide_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op.result.type.element_type
                        )
                        divide_builder._set_golden_tensor(new_op, golden_output)
                        divide_builder._set_output_ordering([new_op])
                        divide_builder._set_golden_tensor(lhs, input0)
                        divide_builder._set_golden_tensor(rhs, input1)
                        divide_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return divide_module, divide_builder

    ################ stablehlo.CosineOp ###############

    @tag(stablehlo.CosineOp)
    def cosine(
        self,
        in0: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.cosine)

        op = stablehlo_op(
            in0,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op.result.type.element_type)
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.CosineOp)
    def cosine_parser(
        self,
        old_op: stablehlo.CosineOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.cosine_parser)
        operand = global_dict[old_op.operand]

        new_op = stablehlo_op(
            operand,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.CosineOp)
    def cosine_split(
        self,
        old_op: stablehlo.CosineOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.cosine_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            cosine_module = Module.create()
            cosine_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
            ]

            with InsertionPoint(cosine_module.body):

                @func.func(*op_input_types, name="cosine_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op.result.type.element_type
                        )
                        cosine_builder._set_golden_tensor(new_op, golden_output)
                        cosine_builder._set_output_ordering([new_op])
                        cosine_builder._set_golden_tensor(operand, input0)
                        cosine_builder._set_input_ordering([operand])

                    return new_op

        return cosine_module, cosine_builder

    ################ stablehlo.ExpOp ###############

    @tag(stablehlo.ExpOp)
    def exp(
        self,
        in0: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.exp)

        op = stablehlo_op(
            in0,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op.result.type.element_type)
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.ExpOp)
    def exp_parser(
        self,
        old_op: stablehlo.ExpOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.exp_parser)
        operand = global_dict[old_op.operand]

        new_op = stablehlo_op(
            operand,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.ExpOp)
    def exp_split(
        self,
        old_op: stablehlo.ExpOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.exp_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            exp_module = Module.create()
            exp_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
            ]

            with InsertionPoint(exp_module.body):

                @func.func(*op_input_types, name="exp_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op.result.type.element_type
                        )
                        exp_builder._set_golden_tensor(new_op, golden_output)
                        exp_builder._set_output_ordering([new_op])
                        exp_builder._set_golden_tensor(operand, input0)
                        exp_builder._set_input_ordering([operand])

                    return new_op

        return exp_module, exp_builder

    ################ stablehlo.FloorOp ###############

    @tag(stablehlo.FloorOp)
    def floor(
        self,
        in0: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.floor)

        op = stablehlo_op(
            in0,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op.result.type.element_type)
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.FloorOp)
    def floor_parser(
        self,
        old_op: stablehlo.FloorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.floor_parser)
        operand = global_dict[old_op.operand]

        new_op = stablehlo_op(
            operand,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.FloorOp)
    def floor_split(
        self,
        old_op: stablehlo.FloorOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.floor_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            floor_module = Module.create()
            floor_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
            ]

            with InsertionPoint(floor_module.body):

                @func.func(*op_input_types, name="floor_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op.result.type.element_type
                        )
                        floor_builder._set_golden_tensor(new_op, golden_output)
                        floor_builder._set_output_ordering([new_op])
                        floor_builder._set_golden_tensor(operand, input0)
                        floor_builder._set_input_ordering([operand])

                    return new_op

        return floor_module, floor_builder

    ################ stablehlo.ClampOp ###############

    @tag(stablehlo.ClampOp)
    def clamp(
        self,
        min: Operand,
        operand: Operand,
        max: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.clamp)

        op = stablehlo_op(
            min,
            operand,
            max,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            min_golden = self._get_golden_tensor(min)
            operand_golden = self._get_golden_tensor(operand)
            max_golden = self._get_golden_tensor(max)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                min_golden, operand_golden, max_golden, op.result.type.element_type
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.ClampOp)
    def clamp_parser(
        self,
        old_op: stablehlo.ClampOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.clamp_parser)
        min_val = global_dict[old_op.min]
        operand = global_dict[old_op.operand]
        max_val = global_dict[old_op.max]

        new_op = stablehlo_op(
            min_val,
            operand,
            max_val,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            min_golden = self._get_golden_tensor(min_val)
            operand_golden = self._get_golden_tensor(operand)
            max_golden = self._get_golden_tensor(max_val)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                min_golden, operand_golden, max_golden, new_op.result.type.element_type
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.ClampOp)
    def clamp_split(
        self,
        old_op: stablehlo.ClampOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.clamp_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            clamp_module = Module.create()
            clamp_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.min.type,
                old_op.operand.type,
                old_op.max.type,
            ]

            with InsertionPoint(clamp_module.body):

                @func.func(*op_input_types, name="clamp_module")
                def decorated_func(*inputs):
                    min_val = inputs[0]
                    operand = inputs[1]
                    max_val = inputs[2]

                    new_op = stablehlo_op(
                        min_val, operand, max_val, loc=old_op.location
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        min_golden = self._get_golden_tensor(old_op.min)
                        operand_golden = self._get_golden_tensor(old_op.operand)
                        max_golden = self._get_golden_tensor(old_op.max)
                        golden_output = op_golden_function(
                            min_golden,
                            operand_golden,
                            max_golden,
                            new_op.result.type.element_type,
                        )
                        clamp_builder._set_golden_tensor(new_op, golden_output)
                        clamp_builder._set_output_ordering([new_op])
                        clamp_builder._set_golden_tensor(min_val, min_golden)
                        clamp_builder._set_golden_tensor(operand, operand_golden)
                        clamp_builder._set_golden_tensor(max_val, max_golden)
                        clamp_builder._set_input_ordering([min_val, operand, max_val])

                    return new_op

        return clamp_module, clamp_builder

    ################ stablehlo.ConcatenateOp ###############

    @tag(stablehlo.ConcatenateOp)
    def concatenate(
        self,
        inputs: List[Operand],
        dim: int = 0,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.concatenate)

        op = stablehlo_op(
            inputs,
            dimension=dim,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input_goldens = tuple(self._get_golden_tensor(inp) for inp in inputs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input_goldens, dim, op.result.type.element_type
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.ConcatenateOp)
    def concatenate_parser(
        self,
        old_op: stablehlo.ConcatenateOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.concatenate_parser)
        inputs = [global_dict[inp] for inp in old_op.inputs]
        dim = old_op.dimension.value

        new_op = stablehlo_op(
            inputs,
            dimension=dim,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input_goldens = tuple(self._get_golden_tensor(inp) for inp in inputs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input_goldens, dim, new_op.result.type.element_type
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.ConcatenateOp)
    def concatenate_split(
        self,
        old_op: stablehlo.ConcatenateOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.concatenate_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        dim = old_op.dimension.value

        with old_context, old_loc:
            concatenate_module = Module.create()
            concatenate_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [inp.type for inp in old_op.inputs]

            with InsertionPoint(concatenate_module.body):

                @func.func(*op_input_types, name="concatenate_module")
                def decorated_func(*inputs):
                    new_op = stablehlo_op(
                        list(inputs), dimension=dim, loc=old_op.location
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input_goldens = tuple(
                            self._get_golden_tensor(inp) for inp in old_op.inputs
                        )
                        golden_output = op_golden_function(
                            input_goldens, dim, new_op.result.type.element_type
                        )
                        concatenate_builder._set_golden_tensor(new_op, golden_output)
                        concatenate_builder._set_output_ordering([new_op])
                        for i, inp in enumerate(inputs):
                            concatenate_builder._set_golden_tensor(
                                inp, input_goldens[i]
                            )
                        concatenate_builder._set_input_ordering(list(inputs))

                    return new_op

        return concatenate_module, concatenate_builder

    ################ stablehlo.ConvertOp ###############

    @tag(stablehlo.ConvertOp)
    def convert(
        self,
        in0: Operand,
        element_type: Type,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.convert)

        input_type = RankedTensorType(in0.type)
        output_type = RankedTensorType.get(list(input_type.shape), element_type)

        op = stablehlo_op(
            output_type,
            in0,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op.result.type.element_type)
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.ConvertOp)
    def convert_parser(
        self,
        old_op: stablehlo.ConvertOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.convert_parser)
        operand = global_dict[old_op.operand]
        output_type = old_op.result.type

        new_op = stablehlo_op(
            output_type,
            operand,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.ConvertOp)
    def convert_split(
        self,
        old_op: stablehlo.ConvertOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.convert_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        output_type = old_op.result.type

        with old_context, old_loc:
            convert_module = Module.create()
            convert_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
            ]

            with InsertionPoint(convert_module.body):

                @func.func(*op_input_types, name="convert_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(output_type, operand, loc=old_op.location)

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op.result.type.element_type
                        )
                        convert_builder._set_golden_tensor(new_op, golden_output)
                        convert_builder._set_output_ordering([new_op])
                        convert_builder._set_golden_tensor(operand, input0)
                        convert_builder._set_input_ordering([operand])

                    return new_op

        return convert_module, convert_builder

    ################ stablehlo.ConvolutionOp ###############

    @tag(stablehlo.ConvolutionOp)
    def convolution(
        self,
        lhs: Operand,
        rhs: Operand,
        window_strides: List[int],
        padding: List[List[int]],
        input_batch_dimension: int,
        input_feature_dimension: int,
        input_spatial_dimensions: List[int],
        kernel_input_feature_dimension: int,
        kernel_output_feature_dimension: int,
        kernel_spatial_dimensions: List[int],
        output_batch_dimension: int,
        output_feature_dimension: int,
        output_spatial_dimensions: List[int],
        feature_group_count: int = 1,
        batch_group_count: int = 1,
        lhs_dilation: Optional[List[int]] = None,
        rhs_dilation: Optional[List[int]] = None,
        window_reversal: Optional[List[bool]] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.convolution)

        num_spatial_dims = len(input_spatial_dimensions)
        if lhs_dilation is None:
            lhs_dilation = [1] * num_spatial_dims
        if rhs_dilation is None:
            rhs_dilation = [1] * num_spatial_dims
        if window_reversal is None:
            window_reversal = [False] * num_spatial_dims

        dimension_numbers = stablehlo.ConvDimensionNumbers.get(
            context=self._ctx,
            input_batch_dimension=input_batch_dimension,
            input_feature_dimension=input_feature_dimension,
            input_spatial_dimensions=input_spatial_dimensions,
            kernel_input_feature_dimension=kernel_input_feature_dimension,
            kernel_output_feature_dimension=kernel_output_feature_dimension,
            kernel_spatial_dimensions=kernel_spatial_dimensions,
            output_batch_dimension=output_batch_dimension,
            output_feature_dimension=output_feature_dimension,
            output_spatial_dimensions=output_spatial_dimensions,
        )

        op = stablehlo_op(
            lhs=lhs,
            rhs=rhs,
            dimension_numbers=dimension_numbers,
            window_strides=window_strides,
            padding=padding,
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            window_reversal=window_reversal,
            feature_group_count=feature_group_count,
            batch_group_count=batch_group_count,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            lhs_golden = self._get_golden_tensor(lhs)
            rhs_golden = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                lhs_golden,
                rhs_golden,
                window_strides,
                padding,
                lhs_dilation,
                rhs_dilation,
                window_reversal,
                input_batch_dimension,
                input_feature_dimension,
                input_spatial_dimensions,
                kernel_input_feature_dimension,
                kernel_output_feature_dimension,
                kernel_spatial_dimensions,
                output_batch_dimension,
                output_feature_dimension,
                output_spatial_dimensions,
                feature_group_count,
                batch_group_count,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.ConvolutionOp)
    def convolution_parser(
        self,
        old_op: stablehlo.ConvolutionOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.convolution_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        dim_numbers_attr = old_op.dimension_numbers
        dim_numbers = stablehlo.ConvDimensionNumbers(dim_numbers_attr)

        input_batch_dimension = dim_numbers.input_batch_dimension
        input_feature_dimension = dim_numbers.input_feature_dimension
        input_spatial_dimensions = list(dim_numbers.input_spatial_dimensions)
        kernel_input_feature_dimension = dim_numbers.kernel_input_feature_dimension
        kernel_output_feature_dimension = dim_numbers.kernel_output_feature_dimension
        kernel_spatial_dimensions = list(dim_numbers.kernel_spatial_dimensions)
        output_batch_dimension = dim_numbers.output_batch_dimension
        output_feature_dimension = dim_numbers.output_feature_dimension
        output_spatial_dimensions = list(dim_numbers.output_spatial_dimensions)

        num_spatial_dims = len(input_spatial_dimensions)

        window_strides = (
            list(old_op.window_strides)
            if old_op.window_strides is not None
            else [1] * num_spatial_dims
        )
        # Padding is a DenseIntElementsAttr with shape [num_spatial_dims, 2]
        # We need to convert it to [[low, high], ...] format
        if old_op.padding is not None:
            padding_flat = [int(p) for p in old_op.padding]
            padding = [
                [padding_flat[i * 2], padding_flat[i * 2 + 1]]
                for i in range(num_spatial_dims)
            ]
        else:
            padding = [[0, 0]] * num_spatial_dims
        lhs_dilation = (
            list(old_op.lhs_dilation)
            if old_op.lhs_dilation is not None
            else [1] * num_spatial_dims
        )
        rhs_dilation = (
            list(old_op.rhs_dilation)
            if old_op.rhs_dilation is not None
            else [1] * num_spatial_dims
        )
        window_reversal = (
            list(old_op.window_reversal)
            if old_op.window_reversal is not None
            else [False] * num_spatial_dims
        )
        feature_group_count = old_op.feature_group_count.value
        batch_group_count = old_op.batch_group_count.value

        # Get the result type from the original op
        result_type = old_op.result.type

        new_op = stablehlo_op(
            result_type,
            lhs,
            rhs,
            dimension_numbers=dim_numbers_attr,
            window_strides=window_strides,
            padding=padding,
            lhs_dilation=lhs_dilation,
            rhs_dilation=rhs_dilation,
            window_reversal=window_reversal,
            feature_group_count=feature_group_count,
            batch_group_count=batch_group_count,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            lhs_golden = self._get_golden_tensor(lhs)
            rhs_golden = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                lhs_golden,
                rhs_golden,
                window_strides,
                padding,
                lhs_dilation,
                rhs_dilation,
                window_reversal,
                input_batch_dimension,
                input_feature_dimension,
                input_spatial_dimensions,
                kernel_input_feature_dimension,
                kernel_output_feature_dimension,
                kernel_spatial_dimensions,
                output_batch_dimension,
                output_feature_dimension,
                output_spatial_dimensions,
                feature_group_count,
                batch_group_count,
                new_op.result.type.element_type,
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.ConvolutionOp)
    def convolution_split(
        self,
        old_op: stablehlo.ConvolutionOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.convolution_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)

        dim_numbers_attr = old_op.dimension_numbers
        dim_numbers = stablehlo.ConvDimensionNumbers(dim_numbers_attr)

        input_batch_dimension = dim_numbers.input_batch_dimension
        input_feature_dimension = dim_numbers.input_feature_dimension
        input_spatial_dimensions = list(dim_numbers.input_spatial_dimensions)
        kernel_input_feature_dimension = dim_numbers.kernel_input_feature_dimension
        kernel_output_feature_dimension = dim_numbers.kernel_output_feature_dimension
        kernel_spatial_dimensions = list(dim_numbers.kernel_spatial_dimensions)
        output_batch_dimension = dim_numbers.output_batch_dimension
        output_feature_dimension = dim_numbers.output_feature_dimension
        output_spatial_dimensions = list(dim_numbers.output_spatial_dimensions)

        num_spatial_dims = len(input_spatial_dimensions)

        window_strides = (
            list(old_op.window_strides)
            if old_op.window_strides is not None
            else [1] * num_spatial_dims
        )
        # Padding is a DenseIntElementsAttr with shape [num_spatial_dims, 2]
        # We need to convert it to [[low, high], ...] format
        if old_op.padding is not None:
            padding_flat = [int(p) for p in old_op.padding]
            padding = [
                [padding_flat[i * 2], padding_flat[i * 2 + 1]]
                for i in range(num_spatial_dims)
            ]
        else:
            padding = [[0, 0]] * num_spatial_dims
        lhs_dilation = (
            list(old_op.lhs_dilation)
            if old_op.lhs_dilation is not None
            else [1] * num_spatial_dims
        )
        rhs_dilation = (
            list(old_op.rhs_dilation)
            if old_op.rhs_dilation is not None
            else [1] * num_spatial_dims
        )
        window_reversal = (
            list(old_op.window_reversal)
            if old_op.window_reversal is not None
            else [False] * num_spatial_dims
        )
        feature_group_count = old_op.feature_group_count.value
        batch_group_count = old_op.batch_group_count.value
        result_type = old_op.result.type

        with old_context, old_loc:
            convolution_module = Module.create()
            convolution_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(convolution_module.body):

                @func.func(*op_input_types, name="convolution_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(
                        result_type,
                        lhs,
                        rhs,
                        dimension_numbers=dim_numbers_attr,
                        window_strides=window_strides,
                        padding=padding,
                        lhs_dilation=lhs_dilation,
                        rhs_dilation=rhs_dilation,
                        window_reversal=window_reversal,
                        feature_group_count=feature_group_count,
                        batch_group_count=batch_group_count,
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        lhs_golden = self._get_golden_tensor(old_op.lhs)
                        rhs_golden = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            lhs_golden,
                            rhs_golden,
                            window_strides,
                            padding,
                            lhs_dilation,
                            rhs_dilation,
                            window_reversal,
                            input_batch_dimension,
                            input_feature_dimension,
                            input_spatial_dimensions,
                            kernel_input_feature_dimension,
                            kernel_output_feature_dimension,
                            kernel_spatial_dimensions,
                            output_batch_dimension,
                            output_feature_dimension,
                            output_spatial_dimensions,
                            feature_group_count,
                            batch_group_count,
                            new_op.result.type.element_type,
                        )
                        convolution_builder._set_golden_tensor(new_op, golden_output)
                        convolution_builder._set_output_ordering([new_op])
                        convolution_builder._set_golden_tensor(lhs, lhs_golden)
                        convolution_builder._set_golden_tensor(rhs, rhs_golden)
                        convolution_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return convolution_module, convolution_builder

    ################ stablehlo.DotGeneralOp ###############

    def _compute_dot_general_result_shape(
        self,
        lhs_shape: List[int],
        rhs_shape: List[int],
        batch_dims_lhs: List[int],
        contract_dims_lhs: List[int],
        batch_dims_rhs: List[int],
        contract_dims_rhs: List[int],
    ) -> List[int]:
        result_shape = []
        # Add batch dimensions
        for dim in batch_dims_lhs:
            result_shape.append(lhs_shape[dim])
        # Add non-batch, non-contract dimensions from lhs and rhs
        for i, dim_size in enumerate(lhs_shape):
            if i not in batch_dims_lhs and i not in contract_dims_lhs:
                result_shape.append(dim_size)
        for i, dim_size in enumerate(rhs_shape):
            if i not in batch_dims_rhs and i not in contract_dims_rhs:
                result_shape.append(dim_size)
        return result_shape

    @tag(stablehlo.DotGeneralOp)
    def dot_general(
        self,
        in0: Operand,
        in1: Operand,
        batch_dims_lhs: List[int],
        contract_dims_lhs: List[int],
        batch_dims_rhs: List[int],
        contract_dims_rhs: List[int],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.dot_general)

        dot_dimension_numbers = stablehlo.DotDimensionNumbers.get(
            context=self._ctx,
            lhs_batching_dimensions=batch_dims_lhs,
            lhs_contracting_dimensions=contract_dims_lhs,
            rhs_batching_dimensions=batch_dims_rhs,
            rhs_contracting_dimensions=contract_dims_rhs,
        )

        lhs_shape = list(in0.type.shape)
        rhs_shape = list(in1.type.shape)
        result_shape = self._compute_dot_general_result_shape(
            lhs_shape,
            rhs_shape,
            batch_dims_lhs,
            contract_dims_lhs,
            batch_dims_rhs,
            contract_dims_rhs,
        )
        result_type = RankedTensorType.get(result_shape, in0.type.element_type)

        op = stablehlo_op(
            result_type,
            in0,
            in1,
            dot_dimension_numbers=dot_dimension_numbers,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            lhs_golden = self._get_golden_tensor(in0)
            rhs_golden = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                lhs_golden,
                rhs_golden,
                batch_dims_lhs,
                contract_dims_lhs,
                batch_dims_rhs,
                contract_dims_rhs,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.DotGeneralOp)
    def dot_general_parser(
        self,
        old_op: stablehlo.DotGeneralOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.dot_general_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        dot_dim_numbers_attr = old_op.dot_dimension_numbers
        dot_dim_numbers = stablehlo.DotDimensionNumbers(dot_dim_numbers_attr)
        batch_dims_lhs = list(dot_dim_numbers.lhs_batching_dimensions)
        contract_dims_lhs = list(dot_dim_numbers.lhs_contracting_dimensions)
        batch_dims_rhs = list(dot_dim_numbers.rhs_batching_dimensions)
        contract_dims_rhs = list(dot_dim_numbers.rhs_contracting_dimensions)

        lhs_shape = list(lhs.type.shape)
        rhs_shape = list(rhs.type.shape)
        result_shape = self._compute_dot_general_result_shape(
            lhs_shape,
            rhs_shape,
            batch_dims_lhs,
            contract_dims_lhs,
            batch_dims_rhs,
            contract_dims_rhs,
        )
        result_type = RankedTensorType.get(result_shape, lhs.type.element_type)

        new_op = stablehlo_op(
            result_type,
            lhs,
            rhs,
            dot_dimension_numbers=dot_dim_numbers_attr,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            lhs_golden = self._get_golden_tensor(lhs)
            rhs_golden = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                lhs_golden,
                rhs_golden,
                batch_dims_lhs,
                contract_dims_lhs,
                batch_dims_rhs,
                contract_dims_rhs,
                new_op.result.type.element_type,
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.DotGeneralOp)
    def dot_general_split(
        self,
        old_op: stablehlo.DotGeneralOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.dot_general_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)

        dot_dim_numbers_attr = old_op.dot_dimension_numbers
        dot_dim_numbers = stablehlo.DotDimensionNumbers(dot_dim_numbers_attr)
        batch_dims_lhs = list(dot_dim_numbers.lhs_batching_dimensions)
        contract_dims_lhs = list(dot_dim_numbers.lhs_contracting_dimensions)
        batch_dims_rhs = list(dot_dim_numbers.rhs_batching_dimensions)
        contract_dims_rhs = list(dot_dim_numbers.rhs_contracting_dimensions)

        with old_context, old_loc:
            dot_general_module = Module.create()
            dot_general_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(dot_general_module.body):

                @func.func(*op_input_types, name="dot_general_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    lhs_shape = list(lhs.type.shape)
                    rhs_shape = list(rhs.type.shape)
                    result_shape = self._compute_dot_general_result_shape(
                        lhs_shape,
                        rhs_shape,
                        batch_dims_lhs,
                        contract_dims_lhs,
                        batch_dims_rhs,
                        contract_dims_rhs,
                    )
                    result_type = RankedTensorType.get(
                        result_shape, lhs.type.element_type
                    )

                    new_op = stablehlo_op(
                        result_type,
                        lhs,
                        rhs,
                        dot_dimension_numbers=dot_dim_numbers_attr,
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        lhs_golden = self._get_golden_tensor(old_op.lhs)
                        rhs_golden = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            lhs_golden,
                            rhs_golden,
                            batch_dims_lhs,
                            contract_dims_lhs,
                            batch_dims_rhs,
                            contract_dims_rhs,
                            new_op.result.type.element_type,
                        )
                        dot_general_builder._set_golden_tensor(new_op, golden_output)
                        dot_general_builder._set_output_ordering([new_op])
                        dot_general_builder._set_golden_tensor(lhs, lhs_golden)
                        dot_general_builder._set_golden_tensor(rhs, rhs_golden)
                        dot_general_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return dot_general_module, dot_general_builder

    ################ stablehlo.ConstantOp ###############

    @tag(stablehlo.ConstantOp)
    def constant(
        self,
        value: torch.Tensor,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.constant)

        value_attr = DenseElementsAttr.get(value.numpy())

        op = stablehlo_op(
            value_attr,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(value_attr)
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.ConstantOp)
    def constant_parser(
        self,
        old_op: stablehlo.ConstantOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.constant_parser)
        value_attr = old_op.value

        new_op = stablehlo_op(
            value_attr,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(value_attr)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.ConstantOp)
    def constant_split(
        self,
        old_op: stablehlo.ConstantOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.constant_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        value_attr = old_op.value

        with old_context, old_loc:
            constant_module = Module.create()
            constant_builder = StableHLOBuilder(old_context, old_loc)

            with InsertionPoint(constant_module.body):

                @func.func(name="constant_module")
                def decorated_func():
                    new_op = stablehlo_op(value_attr, loc=old_op.location)

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(value_attr)
                        constant_builder._set_golden_tensor(new_op, golden_output)
                        constant_builder._set_output_ordering([new_op])
                        constant_builder._set_input_ordering([])

                    return new_op

        return constant_module, constant_builder

    ################ stablehlo.IotaOp ###############

    @tag(stablehlo.IotaOp)
    def iota(
        self,
        shape: List[int],
        iota_dimension: int,
        element_type: Type,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.iota)

        output_type = RankedTensorType.get(shape, element_type)

        op = stablehlo_op(
            output_type,
            iota_dimension=iota_dimension,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(iota_dimension, op.result.type)
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.IotaOp)
    def iota_parser(
        self,
        old_op: stablehlo.IotaOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.iota_parser)
        iota_dimension = old_op.iota_dimension.value
        output_type = old_op.result.type

        new_op = stablehlo_op(
            output_type,
            iota_dimension=iota_dimension,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(iota_dimension, new_op.result.type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.IotaOp)
    def iota_split(
        self,
        old_op: stablehlo.IotaOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.iota_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        iota_dimension = old_op.iota_dimension.value
        output_type = old_op.result.type

        with old_context, old_loc:
            iota_module = Module.create()
            iota_builder = StableHLOBuilder(old_context, old_loc)

            with InsertionPoint(iota_module.body):

                @func.func(name="iota_module")
                def decorated_func():
                    new_op = stablehlo_op(
                        output_type,
                        iota_dimension=iota_dimension,
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            iota_dimension, new_op.result.type
                        )
                        iota_builder._set_golden_tensor(new_op, golden_output)
                        iota_builder._set_output_ordering([new_op])
                        iota_builder._set_input_ordering([])

                    return new_op

        return iota_module, iota_builder

    ################ stablehlo.GatherOp ###############

    @tag(stablehlo.GatherOp)
    def gather(
        self,
        operand: Operand,
        start_indices: Operand,
        offset_dims: List[int],
        collapsed_slice_dims: List[int],
        start_index_map: List[int],
        index_vector_dim: int,
        slice_sizes: List[int],
        operand_batching_dims: Optional[List[int]] = None,
        start_indices_batching_dims: Optional[List[int]] = None,
        indices_are_sorted: bool = False,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        """
        Creates ``stablehlo.gather``.

        *Gather operation.*

        Gathers slices from the operand tensor at positions specified by start_indices.

        Parameters
        ----------
        operand : Operand
            Input tensor to gather from
        start_indices : Operand
            Tensor containing starting indices
        offset_dims : List[int]
            Dimensions in the output that correspond to the offset dimensions
        collapsed_slice_dims : List[int]
            Dimensions of the operand that are collapsed (have slice size 1)
        start_index_map : List[int]
            Maps index vector elements to operand dimensions
        index_vector_dim : int
            Dimension of start_indices that contains the index vectors
        slice_sizes : List[int]
            Size of slices to gather along each dimension
        operand_batching_dims : Optional[List[int]]
            Operand dimensions participating in batching (default: [])
        start_indices_batching_dims : Optional[List[int]]
            Start indices dimensions participating in batching (default: [])
        indices_are_sorted : bool
            Whether indices are guaranteed to be sorted (default: False)
        loc : Optional[str]
            Location for the operation
        unit_attrs : Optional[List[str]]
            Optional list of unit attributes
        sharding_attr : Optional[sdy.TensorShardingPerValueAttr]
            Optional tensor sharding attribute

        Returns
        -------
        (*OpResult*)
            Gathered tensor
        """
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.gather)

        if operand_batching_dims is None:
            operand_batching_dims = []
        if start_indices_batching_dims is None:
            start_indices_batching_dims = []

        dimension_numbers = stablehlo.GatherDimensionNumbers.get(
            context=self._ctx,
            offset_dims=offset_dims,
            collapsed_slice_dims=collapsed_slice_dims,
            operand_batching_dims=operand_batching_dims,
            start_indices_batching_dims=start_indices_batching_dims,
            start_index_map=start_index_map,
            index_vector_dim=index_vector_dim,
        )

        op = stablehlo_op(
            operand,
            start_indices,
            dimension_numbers=dimension_numbers,
            slice_sizes=slice_sizes,
            indices_are_sorted=indices_are_sorted,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            operand_golden = self._get_golden_tensor(operand)
            start_indices_golden = self._get_golden_tensor(start_indices)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                operand_golden,
                start_indices_golden,
                offset_dims,
                collapsed_slice_dims,
                operand_batching_dims,
                start_indices_batching_dims,
                start_index_map,
                index_vector_dim,
                slice_sizes,
                indices_are_sorted,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.GatherOp)
    def gather_parser(
        self,
        old_op: stablehlo.GatherOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.gather_parser)
        operand = global_dict[old_op.operand]
        start_indices = global_dict[old_op.start_indices]

        dimension_numbers_attr = old_op.dimension_numbers
        dimension_numbers = stablehlo.GatherDimensionNumbers(dimension_numbers_attr)
        offset_dims = list(dimension_numbers.offset_dims)
        collapsed_slice_dims = list(dimension_numbers.collapsed_slice_dims)
        operand_batching_dims = list(dimension_numbers.operand_batching_dims)
        start_indices_batching_dims = list(
            dimension_numbers.start_indices_batching_dims
        )
        start_index_map = list(dimension_numbers.start_index_map)
        index_vector_dim = dimension_numbers.index_vector_dim

        slice_sizes = [s for s in old_op.slice_sizes]
        indices_are_sorted = old_op.indices_are_sorted.value

        # Recreate dimension_numbers attribute to ensure proper type
        new_dimension_numbers = stablehlo.GatherDimensionNumbers.get(
            context=self._ctx,
            offset_dims=offset_dims,
            collapsed_slice_dims=collapsed_slice_dims,
            operand_batching_dims=operand_batching_dims,
            start_indices_batching_dims=start_indices_batching_dims,
            start_index_map=start_index_map,
            index_vector_dim=index_vector_dim,
        )

        new_op = stablehlo_op(
            operand,
            start_indices,
            dimension_numbers=new_dimension_numbers,
            slice_sizes=slice_sizes,
            indices_are_sorted=indices_are_sorted,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            operand_golden = self._get_golden_tensor(operand)
            start_indices_golden = self._get_golden_tensor(start_indices)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                operand_golden,
                start_indices_golden,
                offset_dims,
                collapsed_slice_dims,
                operand_batching_dims,
                start_indices_batching_dims,
                start_index_map,
                index_vector_dim,
                slice_sizes,
                indices_are_sorted,
                new_op.result.type.element_type,
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.GatherOp)
    def gather_split(
        self,
        old_op: stablehlo.GatherOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.gather_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)

        dimension_numbers_attr = old_op.dimension_numbers
        dimension_numbers = stablehlo.GatherDimensionNumbers(dimension_numbers_attr)
        offset_dims = list(dimension_numbers.offset_dims)
        collapsed_slice_dims = list(dimension_numbers.collapsed_slice_dims)
        operand_batching_dims = list(dimension_numbers.operand_batching_dims)
        start_indices_batching_dims = list(
            dimension_numbers.start_indices_batching_dims
        )
        start_index_map = list(dimension_numbers.start_index_map)
        index_vector_dim = dimension_numbers.index_vector_dim

        slice_sizes = [s for s in old_op.slice_sizes]
        indices_are_sorted = old_op.indices_are_sorted.value

        # Recreate dimension_numbers attribute to ensure proper type
        new_dimension_numbers = stablehlo.GatherDimensionNumbers.get(
            context=old_context,
            offset_dims=offset_dims,
            collapsed_slice_dims=collapsed_slice_dims,
            operand_batching_dims=operand_batching_dims,
            start_indices_batching_dims=start_indices_batching_dims,
            start_index_map=start_index_map,
            index_vector_dim=index_vector_dim,
        )

        with old_context, old_loc:
            gather_module = Module.create()
            gather_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
                old_op.start_indices.type,
            ]

            with InsertionPoint(gather_module.body):

                @func.func(*op_input_types, name="gather_module")
                def decorated_func(*inputs):
                    operand = inputs[0]
                    start_indices = inputs[1]

                    new_op = stablehlo_op(
                        operand,
                        start_indices,
                        dimension_numbers=new_dimension_numbers,
                        slice_sizes=slice_sizes,
                        indices_are_sorted=indices_are_sorted,
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        operand_golden = self._get_golden_tensor(old_op.operand)
                        start_indices_golden = self._get_golden_tensor(
                            old_op.start_indices
                        )
                        golden_output = op_golden_function(
                            operand_golden,
                            start_indices_golden,
                            offset_dims,
                            collapsed_slice_dims,
                            operand_batching_dims,
                            start_indices_batching_dims,
                            start_index_map,
                            index_vector_dim,
                            slice_sizes,
                            indices_are_sorted,
                            new_op.result.type.element_type,
                        )
                        gather_builder._set_golden_tensor(new_op, golden_output)
                        gather_builder._set_output_ordering([new_op])
                        gather_builder._set_golden_tensor(operand, operand_golden)
                        gather_builder._set_golden_tensor(
                            start_indices, start_indices_golden
                        )
                        gather_builder._set_input_ordering([operand, start_indices])

                    return new_op

        return gather_module, gather_builder

    ################ stablehlo.DynamicUpdateSliceOp ###############

    @tag(stablehlo.DynamicUpdateSliceOp)
    def dynamic_update_slice(
        self,
        operand: Operand,
        update: Operand,
        start_indices: List[Operand],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(
            StableHLOBuilder.dynamic_update_slice
        )

        op = stablehlo_op(
            operand,
            update,
            start_indices,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            operand_golden = self._get_golden_tensor(operand)
            update_golden = self._get_golden_tensor(update)
            start_indices_goldens = [
                self._get_golden_tensor(idx) for idx in start_indices
            ]
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                operand_golden,
                update_golden,
                start_indices_goldens,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.DynamicUpdateSliceOp)
    def dynamic_update_slice_parser(
        self,
        old_op: stablehlo.DynamicUpdateSliceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.dynamic_update_slice_parser
        )
        operand = global_dict[old_op.operand]
        update = global_dict[old_op.update]
        start_indices = [global_dict[idx] for idx in old_op.start_indices]

        new_op = stablehlo_op(
            operand,
            update,
            start_indices,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            operand_golden = self._get_golden_tensor(operand)
            update_golden = self._get_golden_tensor(update)
            start_indices_goldens = [
                self._get_golden_tensor(idx) for idx in start_indices
            ]
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                operand_golden,
                update_golden,
                start_indices_goldens,
                new_op.result.type.element_type,
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.DynamicUpdateSliceOp)
    def dynamic_update_slice_split(
        self,
        old_op: stablehlo.DynamicUpdateSliceOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(
            StableHLOBuilder.dynamic_update_slice_split
        )

        old_context = old_op.context
        old_loc = Location.unknown(old_context)

        with old_context, old_loc:
            dynamic_update_slice_module = Module.create()
            dynamic_update_slice_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
                old_op.update.type,
            ] + [idx.type for idx in old_op.start_indices]

            with InsertionPoint(dynamic_update_slice_module.body):

                @func.func(*op_input_types, name="dynamic_update_slice_module")
                def decorated_func(*inputs):
                    operand = inputs[0]
                    update = inputs[1]
                    start_indices = list(inputs[2:])

                    new_op = stablehlo_op(
                        operand,
                        update,
                        start_indices,
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        operand_golden = self._get_golden_tensor(old_op.operand)
                        update_golden = self._get_golden_tensor(old_op.update)
                        start_indices_goldens = [
                            self._get_golden_tensor(idx) for idx in old_op.start_indices
                        ]
                        golden_output = op_golden_function(
                            operand_golden,
                            update_golden,
                            start_indices_goldens,
                            new_op.result.type.element_type,
                        )
                        dynamic_update_slice_builder._set_golden_tensor(
                            new_op, golden_output
                        )
                        dynamic_update_slice_builder._set_output_ordering([new_op])
                        dynamic_update_slice_builder._set_golden_tensor(
                            operand, operand_golden
                        )
                        dynamic_update_slice_builder._set_golden_tensor(
                            update, update_golden
                        )
                        for i, idx in enumerate(start_indices):
                            dynamic_update_slice_builder._set_golden_tensor(
                                idx, start_indices_goldens[i]
                            )
                        dynamic_update_slice_builder._set_input_ordering(
                            [operand, update] + start_indices
                        )

                    return new_op

        return dynamic_update_slice_module, dynamic_update_slice_builder

    ################ stablehlo.BroadcastInDimOp ###############

    @tag(stablehlo.BroadcastInDimOp)
    def broadcast_in_dim(
        self,
        operand: Operand,
        shape: List[int],
        broadcast_dimensions: List[int],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.broadcast_in_dim)

        input_type = self._get_type(operand)
        element_type = input_type.element_type
        output_type = RankedTensorType.get(shape, element_type)

        broadcast_dimensions_attr = DenseI64ArrayAttr.get(
            broadcast_dimensions, context=self._ctx
        )

        op = stablehlo_op(
            output_type,
            operand,
            broadcast_dimensions=broadcast_dimensions_attr,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            operand_golden = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                operand_golden,
                shape,
                broadcast_dimensions,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.BroadcastInDimOp)
    def broadcast_in_dim_parser(
        self,
        old_op: stablehlo.BroadcastInDimOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.broadcast_in_dim_parser
        )
        operand = global_dict[old_op.operand]

        output_type = old_op.result.type
        broadcast_dimensions = [int(d) for d in old_op.broadcast_dimensions]

        broadcast_dimensions_attr = DenseI64ArrayAttr.get(
            broadcast_dimensions, context=self._ctx
        )

        new_op = stablehlo_op(
            output_type,
            operand,
            broadcast_dimensions=broadcast_dimensions_attr,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            operand_golden = self._get_golden_tensor(operand)
            output_shape = list(RankedTensorType(output_type).shape)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                operand_golden,
                output_shape,
                broadcast_dimensions,
                new_op.result.type.element_type,
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.BroadcastInDimOp)
    def broadcast_in_dim_split(
        self,
        old_op: stablehlo.BroadcastInDimOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(
            StableHLOBuilder.broadcast_in_dim_split
        )

        old_context = old_op.context
        old_loc = Location.unknown(old_context)

        output_type = old_op.result.type
        output_shape = list(RankedTensorType(output_type).shape)
        broadcast_dimensions = [int(d) for d in old_op.broadcast_dimensions]

        broadcast_dimensions_attr = DenseI64ArrayAttr.get(
            broadcast_dimensions, context=old_context
        )

        with old_context, old_loc:
            broadcast_in_dim_module = Module.create()
            broadcast_in_dim_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(broadcast_in_dim_module.body):

                @func.func(*op_input_types, name="broadcast_in_dim_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(
                        output_type,
                        operand,
                        broadcast_dimensions=broadcast_dimensions_attr,
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        operand_golden = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            operand_golden,
                            output_shape,
                            broadcast_dimensions,
                            new_op.result.type.element_type,
                        )
                        broadcast_in_dim_builder._set_golden_tensor(
                            new_op, golden_output
                        )
                        broadcast_in_dim_builder._set_output_ordering([new_op])
                        broadcast_in_dim_builder._set_golden_tensor(
                            operand, operand_golden
                        )
                        broadcast_in_dim_builder._set_input_ordering([operand])

                    return new_op

        return broadcast_in_dim_module, broadcast_in_dim_builder

    ################ stablehlo.BatchNormGradOp ###############

    @tag(stablehlo.BatchNormGradOp)
    def batch_norm_grad(
        self,
        operand: Operand,
        scale: Operand,
        mean: Operand,
        variance: Operand,
        grad_output: Operand,
        epsilon: float,
        feature_index: int,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> Tuple[OpResult, OpResult, OpResult]:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.batch_norm_grad)

        op = stablehlo_op(
            operand,
            scale,
            mean,
            variance,
            grad_output,
            epsilon=FloatAttr.get(F32Type.get(self._ctx), epsilon),
            feature_index=IntegerAttr.get(
                IntegerType.get_signless(64, self._ctx), feature_index
            ),
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            operand_golden = self._get_golden_tensor(operand)
            scale_golden = self._get_golden_tensor(scale)
            mean_golden = self._get_golden_tensor(mean)
            variance_golden = self._get_golden_tensor(variance)
            grad_output_golden = self._get_golden_tensor(grad_output)
            op_golden_function = get_golden_function(stablehlo_op)
            (
                grad_operand_golden,
                grad_scale_golden,
                grad_offset_golden,
            ) = op_golden_function(
                operand_golden,
                scale_golden,
                mean_golden,
                variance_golden,
                grad_output_golden,
                epsilon,
                feature_index,
            )
            self._set_golden_tensor(op.grad_operand, grad_operand_golden)
            self._set_golden_tensor(op.grad_scale, grad_scale_golden)
            self._set_golden_tensor(op.grad_offset, grad_offset_golden)

        return op.grad_operand, op.grad_scale, op.grad_offset

    @parse(stablehlo.BatchNormGradOp)
    def batch_norm_grad_parser(
        self,
        old_op: stablehlo.BatchNormGradOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.batch_norm_grad_parser
        )
        operand = global_dict[old_op.operand]
        scale = global_dict[old_op.scale]
        mean = global_dict[old_op.mean]
        variance = global_dict[old_op.variance]
        grad_output = global_dict[old_op.grad_output]

        epsilon = float(old_op.epsilon.value)
        feature_index = int(old_op.feature_index.value)

        new_op = stablehlo_op(
            operand,
            scale,
            mean,
            variance,
            grad_output,
            epsilon=FloatAttr.get(F32Type.get(self._ctx), epsilon),
            feature_index=IntegerAttr.get(
                IntegerType.get_signless(64, self._ctx), feature_index
            ),
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            operand_golden = self._get_golden_tensor(operand)
            scale_golden = self._get_golden_tensor(scale)
            mean_golden = self._get_golden_tensor(mean)
            variance_golden = self._get_golden_tensor(variance)
            grad_output_golden = self._get_golden_tensor(grad_output)
            op_golden_function = get_golden_function(stablehlo_op)
            (
                grad_operand_golden,
                grad_scale_golden,
                grad_offset_golden,
            ) = op_golden_function(
                operand_golden,
                scale_golden,
                mean_golden,
                variance_golden,
                grad_output_golden,
                epsilon,
                feature_index,
            )
            self._set_golden_tensor(new_op.grad_operand, grad_operand_golden)
            self._set_golden_tensor(new_op.grad_scale, grad_scale_golden)
            self._set_golden_tensor(new_op.grad_offset, grad_offset_golden)

        op_map_dictionary = {}
        op_map_dictionary[old_op.grad_operand] = new_op.grad_operand
        op_map_dictionary[old_op.grad_scale] = new_op.grad_scale
        op_map_dictionary[old_op.grad_offset] = new_op.grad_offset
        return new_op, op_map_dictionary

    @split(stablehlo.BatchNormGradOp)
    def batch_norm_grad_split(
        self,
        old_op: stablehlo.BatchNormGradOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(
            StableHLOBuilder.batch_norm_grad_split
        )

        old_context = old_op.context
        old_loc = Location.unknown(old_context)

        epsilon = float(old_op.epsilon.value)
        feature_index = int(old_op.feature_index.value)

        with old_context, old_loc:
            batch_norm_grad_module = Module.create()
            batch_norm_grad_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
                old_op.scale.type,
                old_op.mean.type,
                old_op.variance.type,
                old_op.grad_output.type,
            ]

            with InsertionPoint(batch_norm_grad_module.body):

                @func.func(*op_input_types, name="batch_norm_grad_module")
                def decorated_func(*inputs):
                    operand = inputs[0]
                    scale = inputs[1]
                    mean = inputs[2]
                    variance = inputs[3]
                    grad_output = inputs[4]

                    new_op = stablehlo_op(
                        operand,
                        scale,
                        mean,
                        variance,
                        grad_output,
                        epsilon=FloatAttr.get(F32Type.get(old_context), epsilon),
                        feature_index=IntegerAttr.get(
                            IntegerType.get_signless(64, old_context), feature_index
                        ),
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        operand_golden = self._get_golden_tensor(old_op.operand)
                        scale_golden = self._get_golden_tensor(old_op.scale)
                        mean_golden = self._get_golden_tensor(old_op.mean)
                        variance_golden = self._get_golden_tensor(old_op.variance)
                        grad_output_golden = self._get_golden_tensor(old_op.grad_output)
                        (
                            grad_operand_golden,
                            grad_scale_golden,
                            grad_offset_golden,
                        ) = op_golden_function(
                            operand_golden,
                            scale_golden,
                            mean_golden,
                            variance_golden,
                            grad_output_golden,
                            epsilon,
                            feature_index,
                        )
                        batch_norm_grad_builder._set_golden_tensor(
                            new_op.grad_operand, grad_operand_golden
                        )
                        batch_norm_grad_builder._set_golden_tensor(
                            new_op.grad_scale, grad_scale_golden
                        )
                        batch_norm_grad_builder._set_golden_tensor(
                            new_op.grad_offset, grad_offset_golden
                        )
                        batch_norm_grad_builder._set_output_ordering(
                            [new_op.grad_operand, new_op.grad_scale, new_op.grad_offset]
                        )
                        batch_norm_grad_builder._set_golden_tensor(
                            operand, operand_golden
                        )
                        batch_norm_grad_builder._set_golden_tensor(scale, scale_golden)
                        batch_norm_grad_builder._set_golden_tensor(mean, mean_golden)
                        batch_norm_grad_builder._set_golden_tensor(
                            variance, variance_golden
                        )
                        batch_norm_grad_builder._set_golden_tensor(
                            grad_output, grad_output_golden
                        )
                        batch_norm_grad_builder._set_input_ordering(
                            [operand, scale, mean, variance, grad_output]
                        )

                    return new_op.grad_operand, new_op.grad_scale, new_op.grad_offset

        return batch_norm_grad_module, batch_norm_grad_builder

    ################ stablehlo.BatchNormTrainingOp ###############

    @tag(stablehlo.BatchNormTrainingOp)
    def batch_norm_training(
        self,
        operand: Operand,
        scale: Operand,
        offset: Operand,
        epsilon: float,
        feature_index: int,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> Tuple[OpResult, OpResult, OpResult]:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.batch_norm_training)

        op = stablehlo_op(
            operand,
            scale,
            offset,
            epsilon=FloatAttr.get(F32Type.get(self._ctx), epsilon),
            feature_index=IntegerAttr.get(
                IntegerType.get_signless(64, self._ctx), feature_index
            ),
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            operand_golden = self._get_golden_tensor(operand)
            scale_golden = self._get_golden_tensor(scale)
            offset_golden = self._get_golden_tensor(offset)
            op_golden_function = get_golden_function(stablehlo_op)
            output_golden, batch_mean_golden, batch_var_golden = op_golden_function(
                operand_golden,
                scale_golden,
                offset_golden,
                epsilon,
                feature_index,
            )
            self._set_golden_tensor(op.output, output_golden)
            self._set_golden_tensor(op.batch_mean, batch_mean_golden)
            self._set_golden_tensor(op.batch_var, batch_var_golden)

        return op.output, op.batch_mean, op.batch_var

    @parse(stablehlo.BatchNormTrainingOp)
    def batch_norm_training_parser(
        self,
        old_op: stablehlo.BatchNormTrainingOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.batch_norm_training_parser
        )
        operand = global_dict[old_op.operand]
        scale = global_dict[old_op.scale]
        offset = global_dict[old_op.offset]

        epsilon = float(old_op.epsilon.value)
        feature_index = int(old_op.feature_index.value)

        new_op = stablehlo_op(
            operand,
            scale,
            offset,
            epsilon=FloatAttr.get(F32Type.get(self._ctx), epsilon),
            feature_index=IntegerAttr.get(
                IntegerType.get_signless(64, self._ctx), feature_index
            ),
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            operand_golden = self._get_golden_tensor(operand)
            scale_golden = self._get_golden_tensor(scale)
            offset_golden = self._get_golden_tensor(offset)
            op_golden_function = get_golden_function(stablehlo_op)
            output_golden, batch_mean_golden, batch_var_golden = op_golden_function(
                operand_golden,
                scale_golden,
                offset_golden,
                epsilon,
                feature_index,
            )
            self._set_golden_tensor(new_op.output, output_golden)
            self._set_golden_tensor(new_op.batch_mean, batch_mean_golden)
            self._set_golden_tensor(new_op.batch_var, batch_var_golden)

        op_map_dictionary = {}
        op_map_dictionary[old_op.output] = new_op.output
        op_map_dictionary[old_op.batch_mean] = new_op.batch_mean
        op_map_dictionary[old_op.batch_var] = new_op.batch_var
        return new_op, op_map_dictionary

    @split(stablehlo.BatchNormTrainingOp)
    def batch_norm_training_split(
        self,
        old_op: stablehlo.BatchNormTrainingOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(
            StableHLOBuilder.batch_norm_training_split
        )

        old_context = old_op.context
        old_loc = Location.unknown(old_context)

        epsilon = float(old_op.epsilon.value)
        feature_index = int(old_op.feature_index.value)

        with old_context, old_loc:
            batch_norm_training_module = Module.create()
            batch_norm_training_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
                old_op.scale.type,
                old_op.offset.type,
            ]

            with InsertionPoint(batch_norm_training_module.body):

                @func.func(*op_input_types, name="batch_norm_training_module")
                def decorated_func(*inputs):
                    operand = inputs[0]
                    scale = inputs[1]
                    offset = inputs[2]

                    new_op = stablehlo_op(
                        operand,
                        scale,
                        offset,
                        epsilon=FloatAttr.get(F32Type.get(old_context), epsilon),
                        feature_index=IntegerAttr.get(
                            IntegerType.get_signless(64, old_context), feature_index
                        ),
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        operand_golden = self._get_golden_tensor(old_op.operand)
                        scale_golden = self._get_golden_tensor(old_op.scale)
                        offset_golden = self._get_golden_tensor(old_op.offset)
                        (
                            output_golden,
                            batch_mean_golden,
                            batch_var_golden,
                        ) = op_golden_function(
                            operand_golden,
                            scale_golden,
                            offset_golden,
                            epsilon,
                            feature_index,
                        )
                        batch_norm_training_builder._set_golden_tensor(
                            new_op.output, output_golden
                        )
                        batch_norm_training_builder._set_golden_tensor(
                            new_op.batch_mean, batch_mean_golden
                        )
                        batch_norm_training_builder._set_golden_tensor(
                            new_op.batch_var, batch_var_golden
                        )
                        batch_norm_training_builder._set_output_ordering(
                            [new_op.output, new_op.batch_mean, new_op.batch_var]
                        )
                        batch_norm_training_builder._set_golden_tensor(
                            operand, operand_golden
                        )
                        batch_norm_training_builder._set_golden_tensor(
                            scale, scale_golden
                        )
                        batch_norm_training_builder._set_golden_tensor(
                            offset, offset_golden
                        )
                        batch_norm_training_builder._set_input_ordering(
                            [operand, scale, offset]
                        )

                    return new_op.output, new_op.batch_mean, new_op.batch_var

        return batch_norm_training_module, batch_norm_training_builder

    # ----- Logical and Bitwise Operations -----

    def or_(
        self,
        in0: Operand,
        in1: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.or``.

        *Elementwise OR operation.*

        Performs elementwise OR operation between two tensors.
        For booleans, performs logical OR.
        For integers, performs bitwise OR.

        Mathematical definition:
        - Logical: or(x, y) = x OR y
        - Bitwise: or(x, y) = x | y

        .. code-block:: mlir

            // Logical OR for booleans
            %result = stablehlo.or(%lhs, %rhs) : tensor<3xi1>, tensor<3xi1> -> tensor<3xi1>
            // Input tensors:
            // lhs: [true, false, true]
            // rhs: [true, true, false]
            // Output tensor:
            // [true, true, true]

            // Bitwise OR for integers
            %result = stablehlo.or(%lhs, %rhs) : tensor<3xi32>, tensor<3xi32> -> tensor<3xi32>
            // Input tensors:
            // lhs: [5, 6, 7]  // Binary: 101, 110, 111
            // rhs: [3, 3, 3]  // Binary: 011, 011, 011
            // Output tensor:
            // [7, 7, 7]       // Binary: 111, 111, 111

        Parameters
        ----------
        in0 : Operand
            First input tensor (boolean or integer type)
        in1 : Operand
            Second input tensor (boolean or integer type)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise OR of the inputs
        """

        return self._eltwise_proxy(
            stablehlo.OrOp,
            [in0, in1],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def xor(
        self,
        in0: Operand,
        in1: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.xor``.

        *Elementwise XOR operation.*

        Performs elementwise XOR operation between two tensors.
        For booleans, performs logical XOR.
        For integers, performs bitwise XOR.

        Mathematical definition:
        - Logical: xor(x, y) = x XOR y
        - Bitwise: xor(x, y) = x ^ y

        .. code-block:: mlir

            // Logical XOR for booleans
            %result = stablehlo.xor(%lhs, %rhs) : tensor<3xi1>, tensor<3xi1> -> tensor<3xi1>
            // Input tensors:
            // lhs: [true, false, true]
            // rhs: [true, true, false]
            // Output tensor:
            // [false, true, true]

            // Bitwise XOR for integers
            %result = stablehlo.xor(%lhs, %rhs) : tensor<3xi32>, tensor<3xi32> -> tensor<3xi32>
            // Input tensors:
            // lhs: [5, 6, 7]  // Binary: 101, 110, 111
            // rhs: [3, 3, 3]  // Binary: 011, 011, 011
            // Output tensor:
            // [6, 5, 4]       // Binary: 110, 101, 100

        Parameters
        ----------
        in0 : Operand
            First input tensor (boolean or integer type)
        in1 : Operand
            Second input tensor (boolean or integer type)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise XOR of the inputs
        """

        return self._eltwise_proxy(
            stablehlo.XorOp,
            [in0, in1],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def not_(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.not``.

        *Elementwise NOT operation.*

        Performs elementwise NOT operation on a tensor.
        For booleans, performs logical NOT.
        For integers, performs bitwise NOT.

        Mathematical definition:
        - Logical: not(x) = NOT x
        - Bitwise: not(x) = ~x

        .. code-block:: mlir

            // Logical NOT for booleans
            %result = stablehlo.not(%input) : tensor<3xi1> -> tensor<3xi1>
            // Input tensor:
            // input: [true, false, true]
            // Output tensor:
            // [false, true, false]

            // Bitwise NOT for integers
            %result = stablehlo.not(%input) : tensor<3xi32> -> tensor<3xi32>
            // Input tensor:
            // input: [0, 1, 2]  // Binary: 000, 001, 010
            // Output tensor:
            // [-1, -2, -3]      // Binary (two's complement): 111...111, 111...110, 111...101

        Parameters
        ----------
        in0 : Operand
            Input tensor (boolean or integer type)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise NOT of the input
        """

        return self._eltwise_proxy(
            stablehlo.NotOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    # ----- Elementwise Unary Operations -----

    def neg(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.negate``.

        *Elementwise negation operation.*

        Computes the element-wise negation of the input tensor.

        Mathematical definition: neg(x) = -x

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise negated values of the input
        """
        return self._eltwise_proxy(
            stablehlo.NegOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def rsqrt(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.rsqrt``.

        *Elementwise reciprocal square root operation.*

        Computes the element-wise reciprocal square root of the input tensor.

        Mathematical definition: rsqrt(x) = 1/√x

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise reciprocal square root values of the input
        """
        return self._eltwise_proxy(
            stablehlo.RsqrtOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def sine(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.sine``.

        *Elementwise sine operation.*

        Computes the element-wise sine of the input tensor.

        Mathematical definition: sine(x) = sin(x)

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise sine values of the input
        """
        return self._eltwise_proxy(
            stablehlo.SineOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def sqrt(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.sqrt``.

        *Elementwise square root operation.*

        Computes the element-wise square root of the input tensor.

        Mathematical definition: sqrt(x) = √x

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise square root values of the input
        """
        return self._eltwise_proxy(
            stablehlo.SqrtOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def logistic(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.logistic``.

        *Elementwise logistic (sigmoid) operation.*

        Computes the element-wise logistic function of the input tensor.

        Mathematical definition: logistic(x) = 1 / (1 + exp(-x))

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise logistic values of the input
        """
        return self._eltwise_proxy(
            stablehlo.LogisticOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def tan(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.tan``.

        *Elementwise tangent operation.*

        Computes the element-wise tangent of the input tensor.

        Mathematical definition: tan(x) = sin(x) / cos(x)

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise tangent values of the input
        """
        return self._eltwise_proxy(
            stablehlo.TanOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def log(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.log``.

        *Elementwise natural logarithm operation.*

        Computes the element-wise natural logarithm of the input tensor.

        Mathematical definition: log(x) = ln(x)

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise natural logarithm values of the input
        """
        return self._eltwise_proxy(
            stablehlo.LogOp,
            [in0],
            unit_attrs=unit_attrs,
        )

    def slice(
        self,
        in0: Operand,
        start_indices: List[int],
        limit_indices: List[int],
        strides: Optional[List[int]] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.slice``.

        *Slice operation.*

        Extracts a slice from the operand using statically-computed starting indices
        and produces a result tensor. start_indices contain the starting indices of
        the slice for each dimension, limit_indices contain the ending indices
        (exclusive) for the slice for each dimension, and strides contain the
        strides for each dimension.

        More formally: result[result_index] = operand[operand_index] where
        operand_index = start_indices + result_index * strides.

        .. code-block:: mlir

            // %operand: [
            //            [0, 0, 0, 0],
            //            [0, 0, 1, 1],
            //            [0, 0, 1, 1]
            //           ]
            %result = "stablehlo.slice"(%operand) {
              start_indices = array<i64: 1, 2>,
              limit_indices = array<i64: 3, 4>,
              strides = array<i64: 1, 1>
            } : (tensor<3x4xi64>) -> tensor<2x2xi64>
            // %result: [
            //            [1, 1],
            //            [1, 1]
            //           ]

        Parameters
        ----------
        in0 : Operand
            Input tensor to slice
        start_indices : List[int]
            Starting indices of the slice for each dimension
        limit_indices : List[int]
            Ending indices (exclusive) of the slice for each dimension
        strides : *Optional[List[int]]*
            Strides for each dimension (default: [1, 1, ...])
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the extracted slice
        """
        if strides is None:
            strides = [1] * len(start_indices)

        if not (len(start_indices) == len(limit_indices) == len(strides)):
            raise ValueError(
                "start_indices, limit_indices, and strides must have the same length"
            )

        start_indices_attr = DenseI64ArrayAttr.get(start_indices, context=self._ctx)
        limit_indices_attr = DenseI64ArrayAttr.get(limit_indices, context=self._ctx)
        strides_attr = DenseI64ArrayAttr.get(strides, context=self._ctx)

        return self._op_proxy(
            stablehlo.SliceOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
            stablehlo_kwargs={
                "start_indices": start_indices_attr,
                "limit_indices": limit_indices_attr,
                "strides": strides_attr,
            },
            golden_kwargs={
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            },
        )

    # ----- Tensor Manipulation Operations -----

    def transpose(
        self,
        in0: Operand,
        permutation: List[int],
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.transpose``.

        *Tensor transpose operation.*

        Permutes the dimensions of the input tensor according to the given permutation.
        This operation rearranges the axes of the tensor without changing the data.

        Mathematical definition: For a tensor with dimensions [d0, d1, ..., dn-1] and
        permutation [p0, p1, ..., pn-1], the output tensor has dimensions
        [d_p0, d_p1, ..., d_pn-1].

        .. code-block:: mlir
            // Transpose a 2x3 tensor by swapping dimensions 0 and 1
            %result = stablehlo.transpose(%input) {permutation = array<i64: 1, 0>} :
                tensor<2x3xf32> -> tensor<3x2xf32>
            // Input tensor:
            // [[1.0, 2.0, 3.0],
            //  [4.0, 5.0, 6.0]]
            // Output tensor:
            // [[1.0, 4.0],
            //  [2.0, 5.0],
            //  [3.0, 6.0]]

        Parameters
        ----------
        in0 : Operand
            Input tensor to transpose
        permutation : List[int]
            The desired ordering of dimensions (0-indexed)
        unit_attrs : Optional[List[str]]
            Optional list of unit attributes
        sharding_attr : Optional[sdy.TensorShardingPerValueAttr]
            Optional tensor sharding attribute for distributed execution

        Returns
        -------
        (*OpView*)
            A tensor with permuted dimensions according to the permutation
        """
        return self._op_proxy(
            stablehlo.TransposeOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
            stablehlo_kwargs={"permutation": permutation},
        )

    def reshape(
        self,
        in0: Operand,
        shape: Sequence[int],
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.reshape``.

        *Tensor reshape operation.*

        Reinterprets the dimensions of ``in0`` without changing the data layout.
        Constraint: product of dimensions must be equal; no -1 or dynamic dims here.

        .. code-block:: mlir

            %input = ... : tensor<2x3xf32>
            %result = stablehlo.reshape %input : (tensor<2x3xf32>) -> tensor<1x6xf32>

        Parameters
        ----------
        in0 : Operand
            Input tensor to reshape
        shape : Sequence[int]
            The desired output shape
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes
        sharding_attr : *Optional[sdy.TensorShardingPerValueAttr]*
            Optional sharding attribute for the output tensor

        Returns
        -------
        (*OpView*)
            The reshaped tensor
        """

        input_type = self._get_type(in0)
        shape_tuple = tuple(int(d) for d in shape)

        # static-only: allow zero, forbid negatives / inferred (-1)
        if any((not isinstance(d, int)) or (d < 0) for d in shape_tuple):
            raise ValueError(
                f"stablehlo.reshape expects a static non-negative-int shape, got {shape_tuple}"
            )

        if math.prod(input_type.shape) != math.prod(shape_tuple):
            raise ValueError(
                "number of elements must be the same for reshape: "
                f"{math.prod(input_type.shape)} != {math.prod(shape_tuple)}"
            )

        out_elem_type = input_type.element_type

        return self._op_proxy(
            stablehlo.ReshapeOp,
            inputs=[in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
            # create result type inside _op_proxy using these hints:
            output_shape=shape_tuple,
            output_type=out_elem_type,
            # tell how to call the op: ReshapeOp(result_type, in0)
            organize_stablehlo_args=lambda i, o, k: (o, i[0]),
            # golden wiring: provide input tensor(s) and pass `shape`
            organize_golden_args=lambda i: (self._get_golden_tensor(i[0]),),
            golden_kwargs={"shape": shape_tuple},
        )

    # ----- Reduce Operations -----

    def _reduce_op_proxy(
        self,
        in0: Operand,
        dimensions: List[int],
        init_attr: Attribute,
        reduce_op_creator: Callable,
        loc: Optional[Location] = None,
    ) -> OpView:
        """
        Helper method to create a StableHLO reduce operation.

        Parameters
        ----------
        in0 : Operand
            Input tensor to reduce
        dimensions : List[int]
            Dimensions along which to reduce
        init_attr : Attribute
            Initial value attribute
        reduce_op_creator : Callable
            Function that creates the reduce operation in the body
        loc : Optional[Location]
            Location for the operation

        Returns
        -------
        OpView
            The reduce operation result
        """
        with self._ctx, self._loc:
            if loc is None:
                id = self._get_next_global_id()
                loc = self._get_loc_of_extra_file_callee(id=id)

            input_type = RankedTensorType(in0.type)
            element_type = input_type.element_type

            input_shape = list(input_type.shape)
            dimensions_set = set(dimensions)
            output_shape = [
                input_shape[i]
                for i in range(len(input_shape))
                if i not in dimensions_set
            ]

            output_type = RankedTensorType.get(output_shape, element_type)

            init_value = stablehlo.ConstantOp(init_attr, loc=loc).result

            reduce_op = stablehlo.ReduceOp(
                [output_type],
                inputs=[in0],
                init_values=[init_value],
                dimensions=dimensions,
                loc=loc,
            )

            reduction_type = RankedTensorType.get([], element_type)
            block = Block.create_at_start(
                reduce_op.regions[0], [reduction_type, reduction_type]
            )

            with InsertionPoint(block):
                reduce_result = reduce_op_creator(
                    block.arguments[0], block.arguments[1], loc
                )
                stablehlo.ReturnOp([reduce_result], loc=loc)

            return reduce_op.result

    def reduce_sum(
        self,
        in0: Operand,
        dimensions: List[int],
        keep_dims: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.reduce`` with sum reduction.

        *Sum reduction operation.*

        Reduces the input tensor by summing elements along the specified dimensions.

        Mathematical definition: For each output element, sum all input elements
        along the specified reduction dimensions.

        .. code-block:: mlir

            // Sum along dimension 0
            %result = stablehlo.reduce(%input init: %init) applies stablehlo.add across dimensions = [0] :
                (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
            // Input tensor:
            // [[1.0, 2.0, 3.0],
            //  [4.0, 5.0, 6.0]]
            // Output tensor:
            // [5.0, 7.0, 9.0]

        Parameters
        ----------
        in0 : Operand
            Input tensor to reduce
        dimensions : List[int]
            Dimensions along which to reduce (0-indexed)
        keep_dims : bool, optional
            Whether to keep the reduced dimensions with size 1. Default is False.
        unit_attrs : Optional[List[str]]
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor with reduced dimensions
        """
        input_type = RankedTensorType(in0.type)
        element_type = input_type.element_type
        zero_attr = self._get_zero_attr(element_type)

        def add_creator(arg0, arg1, loc):
            return stablehlo.AddOp(arg0, arg1, loc=loc).result

        result = self._reduce_op_proxy(in0, dimensions, zero_attr, add_creator)

        if not self._disable_golden_check:
            input_golden = self._get_golden_tensor(in0)
            output_golden = torch.sum(input_golden, dim=dimensions, keepdim=keep_dims)
            self._set_golden_tensor(result, output_golden)

        return result

    def reduce_max(
        self,
        in0: Operand,
        dimensions: List[int],
        keep_dims: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.reduce`` with max reduction.

        *Max reduction operation.*

        Reduces the input tensor by taking the maximum element along the specified dimensions.

        Mathematical definition: For each output element, find the maximum of all input
        elements along the specified reduction dimensions.

        .. code-block:: mlir

            // Max along dimension 0
            %result = stablehlo.reduce(%input init: %init) applies stablehlo.maximum across dimensions = [0] :
                (tensor<2x3xf32>, tensor<f32>) -> tensor<3xf32>
            // Input tensor:
            // [[1.0, 5.0, 3.0],
            //  [4.0, 2.0, 6.0]]
            // Output tensor:
            // [4.0, 5.0, 6.0]

        Parameters
        ----------
        in0 : Operand
            Input tensor to reduce
        dimensions : List[int]
            Dimensions along which to reduce (0-indexed)
        keep_dims : bool, optional
            Whether to keep the reduced dimensions with size 1. Default is False.
        unit_attrs : Optional[List[str]]
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor with reduced dimensions
        """
        input_type = RankedTensorType(in0.type)
        element_type = input_type.element_type
        neg_inf_attr = self._get_neg_inf_attr(element_type)

        def max_creator(arg0, arg1, loc):
            return stablehlo.MaxOp(arg0, arg1, loc=loc).result

        result = self._reduce_op_proxy(in0, dimensions, neg_inf_attr, max_creator)

        if not self._disable_golden_check:
            input_golden = self._get_golden_tensor(in0)
            if dimensions:
                output_golden = torch.amax(
                    input_golden, dim=dimensions, keepdim=keep_dims
                )
            else:
                output_golden = torch.amax(input_golden, keepdim=keep_dims)
            self._set_golden_tensor(result, output_golden)

        return result

    def reduce_min(
        self,
        in0: Operand,
        dimensions: List[int],
        keep_dims: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.reduce`` with min reduction.

        *Min reduction operation.*

        Reduces the input tensor by taking the minimum element along the specified dimensions.

        Mathematical definition: For each output element, find the minimum of all input
        elements along the specified reduction dimensions.

        Parameters
        ----------
        in0 : Operand
            Input tensor to reduce
        dimensions : List[int]
            Dimensions along which to reduce (0-indexed)
        keep_dims : bool, optional
            Whether to keep the reduced dimensions with size 1. Default is False.
        unit_attrs : Optional[List[str]]
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor with reduced dimensions
        """
        input_type = RankedTensorType(in0.type)
        element_type = input_type.element_type
        pos_inf_attr = self._get_pos_inf_attr(element_type)

        def min_creator(arg0, arg1, loc):
            return stablehlo.MinOp(arg0, arg1, loc=loc).result

        result = self._reduce_op_proxy(in0, dimensions, pos_inf_attr, min_creator)

        if not self._disable_golden_check:
            input_golden = self._get_golden_tensor(in0)
            if dimensions:
                output_golden = torch.amin(
                    input_golden, dim=dimensions, keepdim=keep_dims
                )
            else:
                output_golden = torch.amin(input_golden, keepdim=keep_dims)
            self._set_golden_tensor(result, output_golden)

        return result

    def pool_2d(
        self,
        in0: Operand,
        kernel_size: Sequence[int],
        stride: Optional[Sequence[int]] = None,
        padding: Optional[Sequence[int]] = None,
        pool_type: str = "max",
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.reduce_window`` with configurable pooling operation.

        *Pooling operation.*

        Performs pooling on the input tensor. Accepts both 2D and 4D inputs.
        For 2D inputs (H, W), they will be automatically reshaped to 4D (1, 1, H, W)
        during conversion to TTIR, pooling will be applied, and then reshaped back to 2D.

        .. code-block:: mlir

            // Rank-2: Max pool with 3x3 kernel, stride 2x2, padding 1x1
            %init = stablehlo.constant dense<0xFF80> : tensor<bf16>
            %result = "stablehlo.reduce_window"(%input, %init) <{
                padding = dense<[[1, 1], [1, 1]]> : tensor<2x2xi64>,
                window_dimensions = array<i64: 3, 3>,
                window_strides = array<i64: 2, 2>
            }> ({...}) : (tensor<32x32xbf16>, tensor<bf16>) -> tensor<16x16xbf16>

            // Rank-4: Max pool with batch=1, channel=1, 3x3 spatial kernel
            %result = "stablehlo.reduce_window"(%input, %init) <{
                padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>,
                window_dimensions = array<i64: 1, 1, 3, 3>,
                window_strides = array<i64: 1, 1, 2, 2>
            }> ({...}) : (tensor<1x32x64x64xbf16>, tensor<bf16>) -> tensor<1x32x32x32xbf16>

        Parameters
        ----------
        in0 : Operand
            Input tensor to pool. Can be 2D (H, W) or 4D (N, C, H, W).
        kernel_size : Sequence[int]
            Size of the pooling window. Must match input rank.
            - Rank-2: [kernel_h, kernel_w]
            - Rank-4: [kernel_n, kernel_c, kernel_h, kernel_w]
        stride : Optional[Sequence[int]]
            Stride of the pooling window. Must match input rank.
            If None, defaults to all 1s.
            - Rank-2: [stride_h, stride_w]
            - Rank-4: [stride_n, stride_c, stride_h, stride_w]
        padding : Optional[Sequence[int]]
            Padding to apply as [left, right] pairs for each dimension.
            Length must be 2*rank. If None, defaults to all 0s.
            - Rank-2: [pad_h_left, pad_h_right, pad_w_left, pad_w_right]
            - Rank-4: [pad_n_l, pad_n_r, pad_c_l, pad_c_r, pad_h_l, pad_h_r, pad_w_l, pad_w_r]
        pool_type : str
            Type of pooling operation. Options: "max" or "avg".
            Default is "max".
        unit_attrs : Optional[List[str]]
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Pooled tensor
        """
        with self._ctx, self._loc:
            id = self._get_next_global_id()
            loc = self._get_loc_of_extra_file_callee(id=id)

            input_type = RankedTensorType(in0.type)
            element_type = input_type.element_type
            input_shape = list(input_type.shape)
            rank = len(input_shape)

            window_dimensions = list(kernel_size)
            window_strides = list(stride) if stride is not None else [1] * rank
            padding_flat = list(padding) if padding is not None else [0] * (2 * rank)

            output_shape = []
            for i in range(rank):
                pad_low = padding_flat[2 * i]
                pad_high = padding_flat[2 * i + 1]
                output_dim = (
                    input_shape[i] + pad_low + pad_high - kernel_size[i]
                ) // window_strides[i] + 1
                output_shape.append(output_dim)

            output_type = RankedTensorType.get(output_shape, element_type)

            if pool_type == "max":
                init_attr = self._get_neg_inf_attr(element_type)
            elif pool_type == "avg":
                init_attr = self._get_zero_attr(element_type)

            init_value = stablehlo.ConstantOp(init_attr, loc=loc).result

            # Convert padding from flat list to nested list [[low, high], [low, high], ...].
            # This is required for the padding attribute of the reduce_window op.
            padding_2d = [
                [padding_flat[2 * i], padding_flat[2 * i + 1]] for i in range(rank)
            ]

            reduce_window_op = stablehlo.ReduceWindowOp(
                [output_type],
                inputs=[in0],
                init_values=[init_value],
                window_dimensions=window_dimensions,
                window_strides=window_strides,
                base_dilations=None,
                window_dilations=None,
                padding=padding_2d,
                loc=loc,
            )

            reduction_type = RankedTensorType.get([], element_type)
            block = Block.create_at_start(
                reduce_window_op.regions[0], [reduction_type, reduction_type]
            )

            with InsertionPoint(block):
                if pool_type == "max":
                    reduction_result = stablehlo.MaxOp(
                        block.arguments[0], block.arguments[1], loc=loc
                    ).result
                elif pool_type == "avg":
                    reduction_result = stablehlo.AddOp(
                        block.arguments[0], block.arguments[1], loc=loc
                    ).result
                stablehlo.ReturnOp([reduction_result], loc=loc)

            result = reduce_window_op.result
            if pool_type == "avg":
                window_size = 1
                for dim_size in kernel_size:
                    window_size *= dim_size
                divisor_attr = DenseElementsAttr.get_splat(
                    output_type, FloatAttr.get(element_type, float(window_size))
                )
                divisor = stablehlo.ConstantOp(divisor_attr, loc=loc).result
                result = stablehlo.DivOp(result, divisor, loc=loc).result

            if not self._disable_golden_check:
                input_golden = self._get_golden_tensor(in0)

                if rank == 2:
                    input_golden_4d = input_golden.unsqueeze(0).unsqueeze(0)
                    torch_kernel = (kernel_size[0], kernel_size[1])
                    torch_stride = (window_strides[0], window_strides[1])
                    torch_padding = (padding_flat[0], padding_flat[2])
                elif rank == 4:
                    input_golden_4d = input_golden
                    torch_kernel = (kernel_size[2], kernel_size[3])
                    torch_stride = (window_strides[2], window_strides[3])
                    torch_padding = (padding_flat[4], padding_flat[6])

                if pool_type == "max":
                    output_golden = torch.nn.functional.max_pool2d(
                        input_golden_4d,
                        kernel_size=torch_kernel,
                        stride=torch_stride,
                        padding=torch_padding,
                    )
                elif pool_type == "avg":
                    output_golden = torch.nn.functional.avg_pool2d(
                        input_golden_4d,
                        kernel_size=torch_kernel,
                        stride=torch_stride,
                        padding=torch_padding,
                        count_include_pad=True,
                    )

                if rank == 2:
                    output_golden = output_golden.squeeze(0).squeeze(0)

                self._set_golden_tensor(result, output_golden)

            return result

    def _get_zero_attr(self, element_type: Type) -> Attribute:
        """Create a zero constant attribute for the given element type."""
        if IntegerType.isinstance(element_type):
            return DenseElementsAttr.get_splat(
                RankedTensorType.get([], element_type), IntegerAttr.get(element_type, 0)
            )
        else:
            return DenseElementsAttr.get_splat(
                RankedTensorType.get([], element_type), FloatAttr.get(element_type, 0.0)
            )

    def _get_one_attr(self, element_type: Type) -> Attribute:
        """Create a one constant attribute for the given element type."""
        if IntegerType.isinstance(element_type):
            return DenseElementsAttr.get_splat(
                RankedTensorType.get([], element_type), IntegerAttr.get(element_type, 1)
            )
        else:
            return DenseElementsAttr.get_splat(
                RankedTensorType.get([], element_type), FloatAttr.get(element_type, 1.0)
            )

    def _get_neg_inf_attr(self, element_type: Type) -> Attribute:
        """Create a negative infinity constant attribute for the given element type."""
        if IntegerType.isinstance(element_type):
            int_type = IntegerType(element_type)
            width = int_type.width
            min_val = -(2 ** (width - 1))
            return DenseElementsAttr.get_splat(
                RankedTensorType.get([], element_type),
                IntegerAttr.get(element_type, min_val),
            )
        else:
            return DenseElementsAttr.get_splat(
                RankedTensorType.get([], element_type),
                FloatAttr.get(element_type, -math.inf),
            )

    def _get_pos_inf_attr(self, element_type: Type) -> Attribute:
        """Create a positive infinity constant attribute for the given element type."""
        if IntegerType.isinstance(element_type):
            int_type = IntegerType(element_type)
            width = int_type.width
            max_val = (2 ** (width - 1)) - 1
            return DenseElementsAttr.get_splat(
                RankedTensorType.get([], element_type),
                IntegerAttr.get(element_type, max_val),
            )
        else:
            return DenseElementsAttr.get_splat(
                RankedTensorType.get([], element_type),
                FloatAttr.get(element_type, math.inf),
            )

    # ----- Public Shardy Attribute Generators ----

    def mesh_axis_attr(
        self,
        name: str,
        size: int,
    ) -> sdy.MeshAxisAttr:
        """
        Creates a mesh axis attribute.
        This attribute represents a single axis in a mesh, defined by its name and size.

        Parameters
        ----------
        name : str
            The name of the mesh axis
        size : int
            The size of the mesh axis, indicating how many elements are along this axis

        Returns
        -------
        (*sdy.MeshAxisAttr*)
            A mesh axis attribute representing the specified axis with its name and size
        """
        return sdy.MeshAxisAttr.get(name, size)

    def mesh_attr(
        self,
        axes: List[sdy.MeshAxisAttr],
    ) -> MeshAttr:
        """
        Creates a mesh attribute from a list of mesh axis attributes.
        This attribute represents a mesh, which is a collection of axes that can be used
        to define the layout of tensors across multiple devices or processing units.

        Parameters
        ----------
        axes : List[sdy.MeshAxisAttr]
            A list of mesh axis attributes that define the axes of the mesh

        Returns
        -------
        (*sdy.MeshAttr*)
            A mesh attribute representing the collection of axes in the mesh
        """
        return sdy.MeshAttr.get(axes)

    def axis_ref_attr(
        self,
        name: str,
        sub_axis_info_attr: Optional[sdy.AxisRefAttr] = None,
    ) -> sdy.AxisRefAttr:
        """
        Creates an axis reference attribute.
        This attribute is used to reference a specific axis in a mesh, optionally with additional
        sub-axis information.

        Parameters
        ----------
        name : str
            The name of the axis reference
        sub_axis_info_attr : *Optional[sdy.AxisRefAttr]*
            An optional sub-axis reference attribute that provides additional information about the axis

        Returns
        -------
        (*sdy.AxisRefAttr*)
            An axis reference attribute that can be used to refer to a specific axis in a mesh
        """
        return sdy.AxisRefAttr.get(name, sub_axis_info_attr)

    def dimension_sharding_attr(
        self,
        axes: List[sdy.AxisRefAttr],
        is_closed: bool,
        priority: Optional[int] = None,
    ) -> sdy.DimensionShardingAttr:
        """
        Creates a dimension sharding attribute.
        This attribute defines how a tensor is sharded across multiple devices or processing units
        based on the specified axes. It can also indicate whether the sharding is closed and an optional priority for the sharding.

        Parameters
        ----------
        axes : List[sdy.AxisRefAttr]
            A list of axis reference attributes that define how the tensor is sharded across the mesh
        is_closed : bool
            A boolean indicating whether the sharding is closed
        priority : *Optional[int]*
            An optional integer that specifies the priority of the sharding. If not provided, defaults to None.

        Returns
        -------
        (*sdy.DimensionShardingAttr*)
            A dimension sharding attribute that describes how a tensor is distributed across the mesh
        """
        return sdy.DimensionShardingAttr.get(axes, is_closed, priority)

    def tensor_sharding_attr(
        self,
        mesh_name: str,
        dimension_shardings: List[sdy.DimensionShardingAttr],
        replicated_axes: List[sdy.AxisRefAttr] = [],
        unreduced_axes: List[sdy.AxisRefAttr] = [],
    ) -> sdy.TensorShardingAttr:
        """
        Creates a tensor sharding attribute.
        This attribute describes how a tensor is sharded across a mesh, including the mesh name,
        the dimension shardings, and any replicated or unreduced axes.

        Parameters
        ----------
        mesh_name : str
            The name of the mesh to which the tensor sharding applies
        dimension_shardings : List[sdy.DimensionShardingAttr]
            A list of dimension sharding attributes that define how the tensor is sharded across the mesh
        replicated_axes : List[sdy.AxisRefAttr]
            A list of axis reference attributes that are replicated across the mesh. Defaults to an empty list
        unreduced_axes : List[sdy.AxisRefAttr]
            A list of axis reference attributes that are not reduced in the sharding. Defaults to an empty list

        Returns
        -------
        (*sdy.TensorShardingAttr*)
            A tensor sharding attribute that describes how a tensor is distributed across the mesh
        """
        return sdy.TensorShardingAttr.get(
            mesh_name,
            dimension_shardings,
            replicated_axes,
            unreduced_axes,
        )

    def tensor_sharding_per_value_attr(
        self,
        shardings: List[sdy.TensorShardingAttr],
    ) -> sdy.TensorShardingPerValueAttr:
        """
        Creates a tensor sharding per value attribute from a list of tensor sharding attributes.
        This attribute allows for specifying different sharding strategies for different tensors.

        Parameters
        ----------
        shardings : List[sdy.TensorShardingAttr]
            A list of tensor sharding attributes, each defining a sharding strategy for a tensor

        Returns
        -------
        (*sdy.TensorShardingPerValueAttr*)
            A tensor sharding per value attribute that describes how multiple tensors are distributed across the mesh
        """
        return sdy.TensorShardingPerValueAttr.get(
            shardings,
        )

    # ----- Public Shardy Op Generators ----

    def mesh(self, mesh_name: str, mesh_attr: sdy.MeshAttr) -> sdy.MeshOp:
        """
        Creates a mesh operation.
        This operation defines a mesh in the system, which can be used to distribute tensors
        across multiple devices or processing units. The mesh is identified by its name and
        defined by the provided mesh attribute.

        Parameters
        ----------
        mesh_name : str
            The name of the mesh to be created
        mesh_attr : sdy.MeshAttr
            The mesh attribute that defines the axes and properties of the mesh

        Returns
        -------
        (*sdy.MeshOp*)
            A mesh operation that represents the defined mesh in the system
        """
        return sdy.MeshOp(sym_name=mesh_name, mesh=mesh_attr)

    def sharding_constraint(
        self,
        in0: Operand,
        tensor_sharding_attr: sdy.TensorShardingAttr,
    ) -> sdy.ShardingConstraintOp:
        """
        Creates a sharding constraint operation.
        This operation applies a sharding constraint to a tensor, specifying how the tensor should be distributed
        across a mesh based on the provided tensor sharding attribute.

        Parameters
        ----------
        in0 : Operand
            The input tensor to which the sharding constraint will be applied
        tensor_sharding_attr : sdy.TensorShardingAttr
            The tensor sharding attribute that defines how the tensor should be sharded across the mesh

        Returns
        -------
        (*sdy.ShardingConstraintOp*)
            A sharding constraint operation that applies the specified sharding to the input tensor
        """
        return sdy.ShardingConstraintOp(in0, tensor_sharding_attr)

    # ----- Experimental Mpmd Attribute Generators ----

    def experimental_named_mesh_attr(
        self,
        name: str,
        mesh_attr: sdy.MeshAttr,
    ) -> mpmd.NamedMeshAttr:
        return mpmd.NamedMeshAttr.get(name, mesh_attr)

    def experimental_topology_attr(
        self,
        meshes: List[mpmd.NamedMeshAttr],
    ) -> mpmd.TopologyAttr:
        return mpmd.TopologyAttr.get(meshes)

    def experimental_user_origin_attr(
        self,
        user_name: str,
        transpose_count: int = 0,
    ) -> mpmd.UserOriginAttr:
        return mpmd.UserOriginAttr.get(
            user_name=user_name, transpose_count=transpose_count
        )

    def experimental_origin_attr(
        self,
        origin_label: str,
    ) -> mpmd.OriginAttr:
        return mpmd.OriginAttr.get(origin_label=origin_label)

    # ----- Parse stablehlo module ----

    @staticmethod
    def from_module(
        ctx: Context, mlir_text: str, golden_inputs: List[torch.tensor] = None
    ) -> Tuple(Module, StableHLOBuilder):
        if golden_inputs is None:
            golden_inputs = []

        parsed_module = Module.parse(mlir_text, ctx)
        loc = Location.unknown(ctx)
        with ctx, loc:
            stablehlo_builder = StableHLOBuilder(ctx, loc)
            new_module = Module.create()
            fn_input_types = stablehlo_builder.get_input_types(parsed_module)

            if len(golden_inputs) == 0:
                for ttype in fn_input_types:
                    shape = ttype.shape
                    dtype = stablehlo_builder._get_torch_dtype_from_type(
                        ttype.element_type
                    )
                    # Handle scalar tensors (empty shape)
                    if len(shape) == 0:
                        if dtype in (
                            torch.int8,
                            torch.int16,
                            torch.int32,
                            torch.int64,
                            torch.uint8,
                            torch.uint16,
                            torch.uint32,
                            torch.uint64,
                        ):
                            golden_input = torch.zeros(1, dtype=dtype).squeeze()
                        else:
                            golden_input = torch.randn(1, dtype=dtype).squeeze()
                    else:
                        if dtype in (
                            torch.int8,
                            torch.int16,
                            torch.int32,
                            torch.int64,
                            torch.uint8,
                            torch.uint16,
                            torch.uint32,
                            torch.uint64,
                        ):
                            # Use zeros for integer types (safe default for indices)
                            golden_input = torch.zeros(*shape, dtype=dtype)
                        else:
                            golden_input = torch.randn(*shape, dtype=dtype)
                    golden_inputs.append(golden_input)

            with InsertionPoint(new_module.body):

                @func.func(*fn_input_types, name="parsed_module")
                def decorated_func(*inputs):
                    golden_dict = {}
                    for operand, torch_golden in zip(inputs, golden_inputs):
                        golden_dict[operand] = torch_golden

                    input_goldens: Dict[
                        Operand, GoldenMapTensor
                    ] = stablehlo_builder._create_builder_golden_from_torch_tensor(
                        golden_dict
                    )
                    stablehlo_builder._set_goldens(input_goldens)
                    stablehlo_builder._set_input_ordering(inputs)

                    global_dict = {}
                    for entry in parsed_module.body.operations:
                        if isinstance(entry, func.FuncOp):
                            for i, arg in enumerate(entry.arguments):
                                global_dict[arg] = inputs[i]

                    global_result = None
                    for entry in parsed_module.body.operations:
                        for block in entry.body:
                            for op in block.operations:
                                if isinstance(op, func.ReturnOp):
                                    global_result = tuple(
                                        global_dict[operand] for operand in op.operands
                                    )
                                else:
                                    (
                                        parsed_op,
                                        op_golden_dictionary,
                                    ) = stablehlo_builder._build_op_from_parsed_op(
                                        op, global_dict
                                    )
                                    global_dict.update(op_golden_dictionary)

                    outputs = (
                        global_result
                        if hasattr(global_result, "__iter__")
                        else (global_result,)
                    )
                    output_goldens: Dict[Operand, GoldenMapTensor] = {}
                    for op in outputs:
                        output_goldens[op] = stablehlo_builder._get_golden_tensor(op)
                    stablehlo_builder._set_goldens(output_goldens)
                    stablehlo_builder._set_output_ordering(list(outputs))

                    return process_multi_return_result(global_result)

        return new_module, stablehlo_builder

    # ----- Split stablehlo module ----

    def split_op(
        self,
        parsed_op: Operation,
    ) -> Tuple[Module, StableHLOBuilder]:
        split_function = self.get_split_from_opview(type(parsed_op))
        return split_function(self, parsed_op)

    @staticmethod
    def split_module(
        module: Module,
        builder: StableHLOBuilder,
    ) -> List[Tuple[Module, StableHLOBuilder]]:
        sub_modules_and_builders = []
        old_ctx = module.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            for entry in module.body.operations:
                for block in entry.body:
                    for op in block.operations:
                        if isinstance(op, func.ReturnOp):
                            continue
                        else:
                            sub_op_module_builder = builder.split_op(op)
                            sub_modules_and_builders.append(sub_op_module_builder)

        return sub_modules_and_builders
