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

    ############### stablehlo.LogOp ###############

    @tag(stablehlo.LogOp)
    def log(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.log)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

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
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.LogOp)
    def log_parser(
        self,
        old_op: stablehlo.LogOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.log_parser)
        in0 = global_dict[old_op.operand]

        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.LogOp)
    def log_split(
        self,
        old_op: stablehlo.LogOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.log_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            log_module = Module.create()
            log_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(log_module.body):

                @func.func(*op_input_types, name="log_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        log_builder._set_golden_tensor(new_op.result, golden_output)
                        log_builder._set_output_ordering([new_op.result])
                        log_builder._set_golden_tensor(in0, input0)
                        log_builder._set_input_ordering([in0])

                    return new_op

        return log_module, log_builder

    ############### stablehlo.NegOp ###############

    @tag(stablehlo.NegOp)
    def neg(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.neg)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

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
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.NegOp)
    def neg_parser(
        self,
        old_op: stablehlo.NegOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.neg_parser)
        in0 = global_dict[old_op.operand]
        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.NegOp)
    def neg_split(
        self,
        old_op: stablehlo.NegOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.neg_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            neg_module = Module.create()
            neg_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(neg_module.body):

                @func.func(*op_input_types, name="neg_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    new_op = stablehlo_op(in0, loc=old_op.location)

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        neg_builder._set_golden_tensor(new_op.result, golden_output)
                        neg_builder._set_output_ordering([new_op.result])
                        neg_builder._set_golden_tensor(in0, input0)
                        neg_builder._set_input_ordering([in0])

                    return new_op

        return neg_module, neg_builder

    ############### stablehlo.RsqrtOp ###############

    @tag(stablehlo.RsqrtOp)
    def rsqrt(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.rsqrt)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

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
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.RsqrtOp)
    def rsqrt_parser(
        self,
        old_op: stablehlo.RsqrtOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.rsqrt_parser)
        in0 = global_dict[old_op.operand]

        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.RsqrtOp)
    def rsqrt_split(
        self,
        old_op: stablehlo.RsqrtOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.rsqrt_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            rsqrt_module = Module.create()
            rsqrt_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(rsqrt_module.body):

                @func.func(*op_input_types, name="rsqrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    new_op = stablehlo_op(in0, loc=old_op.location)

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        rsqrt_builder._set_golden_tensor(new_op.result, golden_output)
                        rsqrt_builder._set_output_ordering([new_op.result])
                        rsqrt_builder._set_golden_tensor(in0, input0)
                        rsqrt_builder._set_input_ordering([in0])

                    return new_op

        return rsqrt_module, rsqrt_builder

    ############### stablehlo.SineOp ###############

    @tag(stablehlo.SineOp)
    def sine(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.sine)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

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
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.SineOp)
    def sine_parser(
        self,
        old_op: stablehlo.SineOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.sine_parser)
        in0 = global_dict[old_op.operand]
        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.SineOp)
    def sine_split(
        self,
        old_op: stablehlo.SineOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.sine_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            sine_module = Module.create()
            sine_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(sine_module.body):

                @func.func(*op_input_types, name="sine_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    new_op = stablehlo_op(in0, loc=old_op.location)

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        sine_builder._set_golden_tensor(new_op.result, golden_output)
                        sine_builder._set_output_ordering([new_op.result])
                        sine_builder._set_golden_tensor(in0, input0)
                        sine_builder._set_input_ordering([in0])

                    return new_op

        return sine_module, sine_builder

    ############### stablehlo.SqrtOp ###############

    @tag(stablehlo.SqrtOp)
    def sqrt(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.sqrt)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

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
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.SqrtOp)
    def sqrt_parser(
        self,
        old_op: stablehlo.SqrtOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.sqrt_parser)
        in0 = global_dict[old_op.operand]
        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.SqrtOp)
    def sqrt_split(
        self,
        old_op: stablehlo.SqrtOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.sqrt_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            sqrt_module = Module.create()
            sqrt_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(sqrt_module.body):

                @func.func(*op_input_types, name="sqrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    new_op = stablehlo_op(in0, loc=old_op.location)

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        sqrt_builder._set_golden_tensor(new_op.result, golden_output)
                        sqrt_builder._set_output_ordering([new_op.result])
                        sqrt_builder._set_golden_tensor(in0, input0)
                        sqrt_builder._set_input_ordering([in0])

                    return new_op

        return sqrt_module, sqrt_builder

    ############### stablehlo.TanOp ###############

    @tag(stablehlo.TanOp)
    def tan(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.tan)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

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
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.TanOp)
    def tan_parser(
        self,
        old_op: stablehlo.TanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.tan_parser)
        in0 = global_dict[old_op.operand]
        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.TanOp)
    def tan_split(
        self,
        old_op: stablehlo.TanOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.tan_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            tan_module = Module.create()
            tan_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(tan_module.body):

                @func.func(*op_input_types, name="tan_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    new_op = stablehlo_op(in0, loc=old_op.location)

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        tan_builder._set_golden_tensor(new_op.result, golden_output)
                        tan_builder._set_output_ordering([new_op.result])
                        tan_builder._set_golden_tensor(in0, input0)
                        tan_builder._set_input_ordering([in0])

                    return new_op

        return tan_module, tan_builder

    ############### stablehlo.TanhOp ###############

    @tag(stablehlo.TanhOp)
    def tanh(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.tanh)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

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
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.TanhOp)
    def tanh_parser(
        self,
        old_op: stablehlo.TanhOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.tanh_parser)
        in0 = global_dict[old_op.operand]

        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.TanhOp)
    def tanh_split(
        self,
        old_op: stablehlo.TanhOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.tanh_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            tanh_module = Module.create()
            tanh_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(tanh_module.body):

                @func.func(*op_input_types, name="tanh_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    new_op = stablehlo_op(in0, loc=old_op.location)

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        tanh_builder._set_golden_tensor(new_op.result, golden_output)
                        tanh_builder._set_output_ordering([new_op.result])
                        tanh_builder._set_golden_tensor(in0, input0)
                        tanh_builder._set_input_ordering([in0])

                    return new_op

        return tanh_module, tanh_builder

    ############### stablehlo.Log1pOp ###############

    @tag(stablehlo.Log1pOp)
    def log1p(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.log1p)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

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
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.Log1pOp)
    def log1p_parser(
        self,
        old_op: stablehlo.Log1pOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.log1p_parser)
        in0 = global_dict[old_op.operand]

        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.Log1pOp)
    def log1p_split(
        self,
        old_op: stablehlo.Log1pOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.log1p_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            log1p_module = Module.create()
            log1p_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(log1p_module.body):

                @func.func(*op_input_types, name="log1p_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    new_op = stablehlo_op(in0, loc=old_op.location)

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        log1p_builder._set_golden_tensor(new_op.result, golden_output)
                        log1p_builder._set_output_ordering([new_op.result])
                        log1p_builder._set_golden_tensor(in0, input0)
                        log1p_builder._set_input_ordering([in0])

                    return new_op

        return log1p_module, log1p_builder

    ############### stablehlo.LogisticOp ###############

    @tag(stablehlo.LogisticOp)
    def logistic(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.logistic)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

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
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.LogisticOp)
    def logistic_parser(
        self,
        old_op: stablehlo.LogisticOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.logistic_parser)
        in0 = global_dict[old_op.operand]
        new_op = stablehlo_op(
            in0,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.LogisticOp)
    def logistic_split(
        self,
        old_op: stablehlo.LogisticOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.logistic_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            logistic_module = Module.create()
            logistic_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(logistic_module.body):

                @func.func(*op_input_types, name="logistic_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    new_op = stablehlo_op(in0, loc=old_op.location)

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        logistic_builder._set_golden_tensor(
                            new_op.result, golden_output
                        )
                        logistic_builder._set_output_ordering([new_op.result])
                        logistic_builder._set_golden_tensor(in0, input0)
                        logistic_builder._set_input_ordering([in0])

                    return new_op

        return logistic_module, logistic_builder

    ############### stablehlo.SliceOp ###############

    @tag(stablehlo.SliceOp)
    def slice(
        self,
        in0: Operand,
        start_indices: List[int],
        limit_indices: List[int],
        strides: Optional[List[int]] = None,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.slice)

        if strides is None:
            strides = [1] * len(start_indices)

        start_indices_attr = DenseI64ArrayAttr.get(start_indices)
        limit_indices_attr = DenseI64ArrayAttr.get(limit_indices)
        strides_attr = DenseI64ArrayAttr.get(strides)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            start_indices_attr,
            limit_indices_attr,
            strides_attr,
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
            golden_output = op_golden_function(
                input0,
                start_indices_attr,
                limit_indices_attr,
                strides_attr,
                mlir_output_type,
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.SliceOp)
    def slice_parser(
        self,
        old_op: stablehlo.SliceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.slice_parser)
        in0 = global_dict[old_op.operand]
        start_indices_attr = old_op.start_indices
        limit_indices_attr = old_op.limit_indices
        strides_attr = old_op.strides

        new_op = stablehlo_op(
            in0,
            start_indices_attr,
            limit_indices_attr,
            strides_attr,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                start_indices_attr,
                limit_indices_attr,
                strides_attr,
                old_op.result.type.element_type,
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.SliceOp)
    def slice_split(
        self,
        old_op: stablehlo.SliceOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.slice_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            slice_module = Module.create()
            slice_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(slice_module.body):

                @func.func(*op_input_types, name="slice_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    start_indices_attr = old_op.start_indices
                    limit_indices_attr = old_op.limit_indices
                    strides_attr = old_op.strides

                    new_op = stablehlo_op(
                        in0,
                        start_indices_attr,
                        limit_indices_attr,
                        strides_attr,
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0,
                            start_indices_attr,
                            limit_indices_attr,
                            strides_attr,
                            old_op.result.type.element_type,
                        )
                        slice_builder._set_golden_tensor(new_op.result, golden_output)
                        slice_builder._set_output_ordering([new_op.result])
                        slice_builder._set_golden_tensor(in0, input0)
                        slice_builder._set_input_ordering([in0])

                    return new_op

        return slice_module, slice_builder

    ############### stablehlo.TransposeOp ###############

    @tag(stablehlo.TransposeOp)
    def transpose(
        self,
        in0: Operand,
        permutation: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.transpose)
        permutation_attr = DenseI64ArrayAttr.get(permutation)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            permutation_attr,
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
            golden_output = op_golden_function(
                input0, permutation_attr, op.result.type.element_type
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.TransposeOp)
    def transpose_parser(
        self,
        old_op: stablehlo.TransposeOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.transpose_parser)
        in0 = global_dict[old_op.operand]
        permutation_attr = old_op.permutation

        new_op = stablehlo_op(
            in0,
            permutation_attr,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, permutation_attr, old_op.result.type.element_type
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.TransposeOp)
    def transpose_split(
        self,
        old_op: stablehlo.TransposeOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.transpose_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            transpose_module = Module.create()
            transpose_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(transpose_module.body):

                @func.func(*op_input_types, name="transpose_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    permutation_attr = old_op.permutation

                    new_op = stablehlo_op(
                        in0,
                        permutation_attr,
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, permutation_attr, old_op.result.type.element_type
                        )
                        transpose_builder._set_golden_tensor(
                            new_op.result, golden_output
                        )
                        transpose_builder._set_output_ordering([new_op.result])
                        transpose_builder._set_golden_tensor(in0, input0)
                        transpose_builder._set_input_ordering([in0])

                    return new_op

        return transpose_module, transpose_builder

    # Helper used by refactored ops to produce a meaningful Location
    def _get_location(self) -> Location:
        stack = inspect.stack()
        caller_frame = stack[2]
        filename = caller_frame.filename
        lineno = caller_frame.lineno
        return Location.name(f"{filename}:{lineno}")

    ############### stablehlo.ReshapeOp ###############

    @tag(stablehlo.ReshapeOp)
    def reshape(
        self,
        in0: Operand,
        shape: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.reshape)
        shape_attr = ArrayAttr.get(
            [IntegerAttr.get(IntegerType.get_signless(32), s) for s in shape]
        )

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        result = self._create_ranked_tensor_type(shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            result,
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
            golden_output = op_golden_function(
                input0, result.shape, op.result.type.element_type
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.ReshapeOp)
    def reshape_parser(
        self,
        old_op: stablehlo.ReshapeOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.reshape_parser)
        in0 = global_dict[old_op.operand]
        shape_attr = old_op.result.type.shape
        result = old_op.result.type
        new_op = stablehlo_op(old_op.result.type, in0, loc=old_op.location)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, shape_attr, result.element_type)
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.ReshapeOp)
    def reshape_split(
        self,
        old_op: stablehlo.ReshapeOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.reshape_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            reshape_module = Module.create()
            reshape_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(reshape_module.body):

                @func.func(*op_input_types, name="reshape_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    shape_attr = old_op.result.type.shape
                    result = old_op.result.type

                    new_op = stablehlo_op(result, in0, loc=old_op.location)

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, shape_attr, result.element_type
                        )
                        reshape_builder._set_golden_tensor(new_op.result, golden_output)
                        reshape_builder._set_output_ordering([new_op.result])
                        reshape_builder._set_golden_tensor(in0, input0)
                        reshape_builder._set_input_ordering([in0])

                    return new_op

        return reshape_module, reshape_builder

    ############### stablehlo.SortOp ###############

    @tag(stablehlo.SortOp)
    def sort(
        self,
        inputs: List[Operand],
        dimension: int = -1,
        is_stable: bool = False,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> Tuple[OpResult, ...]:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.sort)

        # Determine if this is single input mode (return values + indices)
        single_input = len(inputs) == 1

        input_type = RankedTensorType(inputs[0].type)
        input_shape = list(input_type.shape)

        # Handle negative dimension
        if dimension < 0:
            dimension = len(input_shape) + dimension

        dimension_attr = IntegerAttr.get(IntegerType.get_signless(64), dimension)
        is_stable_attr = BoolAttr.get(is_stable)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        # Determine result types based on input configuration
        if single_input:
            # Single input: return sorted tensor + indices
            result = [
                inputs[0].type,
                RankedTensorType.get(input_shape, IntegerType.get_signless(32)),
            ]
        else:
            # Multiple inputs: return sorted versions of all inputs
            result = [inp.type for inp in inputs]

        op = stablehlo_op(
            result,
            inputs,
            dimension=dimension_attr,
            is_stable=is_stable_attr,
            loc=loc,
        )

        # Create default comparator region for ascending sort
        # For multiple inputs, we compare elements from the first input
        first_input_type = RankedTensorType(inputs[0].type)
        element_type = first_input_type.element_type
        scalar_type = RankedTensorType.get([], element_type)

        # The comparator takes 2*num_inputs arguments (pairs from each input)
        comparator_arg_types = [scalar_type] * (2 * len(inputs))

        block = Block.create_at_start(op.comparator, comparator_arg_types)

        with InsertionPoint(block):
            # For default ascending sort, compare first two arguments (from first input)
            # Return true if lhs < rhs
            lhs = block.arguments[0]  # First element from first input
            rhs = block.arguments[1]  # Second element from first input

            # Create comparison: lhs < rhs for ascending sort
            compare_result = stablehlo.CompareOp(
                lhs, rhs, stablehlo.ComparisonDirection.LT, loc=loc
            ).result

            stablehlo.ReturnOp([compare_result], loc=loc)

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input_tensors = tuple([self._get_golden_tensor(inp) for inp in inputs])
            op_golden_function = get_golden_function(stablehlo_op)
            golden_outputs = op_golden_function(
                input_tensors, dimension, is_stable, single_input
            )
            for i, result in enumerate(op.results):
                self._set_golden_tensor(result, golden_outputs[i])

        return tuple(op.results)

    @parse(stablehlo.SortOp)
    def sort_parser(
        self,
        old_op: stablehlo.SortOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.sort_parser)
        inputs = [global_dict[operand] for operand in old_op.inputs]
        dimension_attr = old_op.dimension
        is_stable_attr = old_op.is_stable

        result_types = [result.type for result in old_op.results]

        new_op = stablehlo_op(
            result_types,
            inputs,
            dimension=dimension_attr,
            is_stable=is_stable_attr,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input_tensors = tuple([self._get_golden_tensor(inp) for inp in inputs])
            op_golden_function = get_golden_function(stablehlo_op)
            single_input = len(inputs) == 1 and len(old_op.results) == 2
            golden_outputs = op_golden_function(
                input_tensors, dimension_attr.value, is_stable_attr.value, single_input
            )
            for i, result in enumerate(new_op.results):
                self._set_golden_tensor(result, golden_outputs[i])

        op_map_dictionary = {}
        for i, result in enumerate(old_op.results):
            op_map_dictionary[result] = new_op.results[i]
        return new_op, op_map_dictionary

    @split(stablehlo.SortOp)
    def sort_split(
        self,
        old_op: stablehlo.SortOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.sort_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            sort_module = Module.create()
            sort_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [operand.type for operand in old_op.inputs]

            with InsertionPoint(sort_module.body):

                @func.func(*op_input_types, name="sort_module")
                def decorated_func(*inputs):
                    dimension_attr = old_op.dimension
                    is_stable_attr = old_op.is_stable
                    result_types = [result.type for result in old_op.results]

                    new_op = stablehlo_op(
                        result_types,
                        inputs,
                        dimension=dimension_attr,
                        is_stable=is_stable_attr,
                        loc=old_loc,
                    )

                    if not sort_builder._disable_golden_check:
                        input_tensors = tuple(
                            [sort_builder._get_golden_tensor(inp) for inp in inputs]
                        )
                        op_golden_function = get_golden_function(stablehlo_op)
                        single_input = len(inputs) == 1 and len(old_op.results) == 2
                        golden_outputs = op_golden_function(
                            input_tensors,
                            dimension_attr.value,
                            is_stable_attr.value,
                            single_input,
                        )
                        for i, result in enumerate(new_op.results):
                            sort_builder._set_golden_tensor(result, golden_outputs[i])
                        sort_builder._set_output_ordering(list(new_op.results))
                        for i, inp in enumerate(inputs):
                            sort_builder._set_golden_tensor(inp, input_tensors[i])
                        sort_builder._set_input_ordering(list(inputs))

                    return tuple(new_op.results)

        return sort_module, sort_builder

    # ----- Random Number Generation Operations -----

    ############### stablehlo.RngBitGeneratorOp ###############

    @tag(stablehlo.RngBitGeneratorOp)
    def rng_bit_generator(
        self,
        initial_state: Operand,
        output_shape: List[int],
        algorithm: str = "DEFAULT",
        output_dtype: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> Tuple[OpResult, OpResult]:
        """
        Creates ``stablehlo.rng_bit_generator``.

        Returns an output filled with uniform random data and an updated output
        state given an initial state using the pseudorandom number generator algorithm.

        .. code-block:: mlir

            // Generate random bits using THREE_FRY algorithm
            %output_state, %output = stablehlo.rng_bit_generator %initial_state,
                algorithm = THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x2xui64>)

        Parameters
        ----------
        initial_state : Operand
            Initial state tensor for the random number generator.
        output_shape : List[int]
            Shape of the output random tensor.
        algorithm : str, optional
            RNG algorithm to use. Options: "THREE_FRY", "PHILOX", "DEFAULT". Default is "DEFAULT".
        output_dtype : Optional[torch.dtype], optional
            Data type for the output tensor. If None, uses ui64.
        loc : Optional[str]
            Location for MLIR debugging
        unit_attrs : Optional[List[str]]
            Unit attributes to add to the operation
        sharding_attr : Optional[sdy.TensorShardingPerValueAttr]
            Tensor sharding attribute for distributed execution

        Returns
        -------
        Tuple[OpResult, OpResult]
            A tuple containing (output_state, output) where output_state is the
            updated RNG state and output is the generated random data.
        """
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.rng_bit_generator)

        # Get input state type information
        input_state_type = RankedTensorType(initial_state.type)

        if output_dtype is None:
            # Default to ui64 for random bit generation
            output_element_type = IntegerType.get_unsigned(64)
        else:
            output_element_type = self._get_type_from_torch_dtype(output_dtype)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        # Create output types
        output_state_type = input_state_type  # Same as input state
        output_tensor_type = RankedTensorType.get(output_shape, output_element_type)

        # Create algorithm attribute
        # The algorithm parameter should be passed as a string that gets converted to the appropriate attribute
        algorithm_attr = StringAttr.get(algorithm)

        op = stablehlo_op(
            output_state_type,
            output_tensor_type,
            algorithm_attr,
            initial_state,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input_state_golden = self._get_golden_tensor(initial_state)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output_state, golden_output = op_golden_function(
                input_state_golden, output_shape, algorithm
            )
            self._set_golden_tensor(op.output_state, golden_output_state)
            self._set_golden_tensor(op.output, golden_output)

        return op.output_state, op.output

    @parse(stablehlo.RngBitGeneratorOp)
    def rng_bit_generator_parser(
        self,
        old_op: stablehlo.RngBitGeneratorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.rng_bit_generator_parser
        )
        initial_state = global_dict[old_op.initial_state]
        algorithm_attr = old_op.rng_algorithm

        new_op = stablehlo_op(
            old_op.output_state.type,
            old_op.output.type,
            algorithm_attr,
            initial_state,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input_state_golden = self._get_golden_tensor(initial_state)
            op_golden_function = get_golden_function(stablehlo_op)
            output_shape = list(RankedTensorType(old_op.output.type).shape)
            algorithm_str = str(algorithm_attr).replace(
                '"', ""
            )  # Convert attr to string
            golden_output_state, golden_output = op_golden_function(
                input_state_golden, output_shape, algorithm_str
            )
            self._set_golden_tensor(new_op.output_state, golden_output_state)
            self._set_golden_tensor(new_op.output, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.output_state] = new_op.output_state
        op_map_dictionary[old_op.output] = new_op.output
        return new_op, op_map_dictionary

    @split(stablehlo.RngBitGeneratorOp)
    def rng_bit_generator_split(
        self,
        old_op: stablehlo.RngBitGeneratorOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(
            StableHLOBuilder.rng_bit_generator_split
        )

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            rng_bit_generator_module = Module.create()
            rng_bit_generator_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.initial_state.type]

            with InsertionPoint(rng_bit_generator_module.body):

                @func.func(*op_input_types, name="rng_bit_generator_module")
                def decorated_func(*inputs):
                    algorithm_attr = old_op.rng_algorithm

                    new_op = stablehlo_op(
                        old_op.output_state.type,
                        old_op.output.type,
                        algorithm_attr,
                        inputs[0],
                        loc=old_loc,
                    )

                    if not rng_bit_generator_builder._disable_golden_check:
                        input_state_golden = (
                            rng_bit_generator_builder._get_golden_tensor(inputs[0])
                        )
                        op_golden_function = get_golden_function(stablehlo_op)
                        output_shape = list(RankedTensorType(old_op.output.type).shape)
                        algorithm_str = str(algorithm_attr).replace(
                            '"', ""
                        )  # Convert attr to string
                        golden_output_state, golden_output = op_golden_function(
                            input_state_golden, output_shape, algorithm_str
                        )
                        rng_bit_generator_builder._set_golden_tensor(
                            new_op.output_state, golden_output_state
                        )
                        rng_bit_generator_builder._set_golden_tensor(
                            new_op.output, golden_output
                        )
                        rng_bit_generator_builder._set_output_ordering(
                            [new_op.output_state, new_op.output]
                        )
                        rng_bit_generator_builder._set_golden_tensor(
                            inputs[0], input_state_golden
                        )
                        rng_bit_generator_builder._set_input_ordering([inputs[0]])

                    return tuple([new_op.output_state, new_op.output])

        return rng_bit_generator_module, rng_bit_generator_builder

    # ----- Tensor Manipulation Operations -----

    def concatenate(
        self,
        inputs: List[Operand],
        dim: int = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.concatenate``.

        *Tensor concatenation operation.*

        Concatenates a variadic number of tensors in `inputs` along `dim`
        dimension in the same order as the given arguments. All input tensors
        must have the same shape except in the concatenating dimension.

        .. code-block:: mlir

            // Concatenate two tensors along dimension 0
            %result = stablehlo.concatenate %input0, %input1, dim = 0 : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
            // Input tensors:
            // input0: [[1.0, 2.0, 3.0],
            //          [4.0, 5.0, 6.0]]
            // input1: [[7.0, 8.0, 9.0]]
            // Output tensor:
            // [[1.0, 2.0, 3.0],
            //  [4.0, 5.0, 6.0],
            //  [7.0, 8.0, 9.0]]

        Parameters
        ----------
        inputs : List[Operand]
            List of input tensors to concatenate. All tensors must have the same
            rank and matching dimensions except along the concatenation dimension.
        dim : int, optional
            Dimension along which to concatenate. Must be in range [0, rank).
            Default is 0.
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing all input tensors concatenated along the specified dimension
        """
        return self._op_proxy(
            stablehlo.ConcatenateOp,
            inputs,
            organize_stablehlo_args=lambda i, o, k: (i,),
            stablehlo_kwargs={"dimension": dim},
            organize_golden_args=lambda i: (
                tuple([self._get_golden_tensor(inp) for inp in i]),
            ),
            golden_kwargs={"dim": dim},
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
        during conversion to StableHLO, pooling will be applied, and then reshaped back to 2D.

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
                    dtype = stablehlo_builder._get_datatype_from_torch_dtype(
                        ttype.element_type
                    )
                    # Handle scalar tensors (empty shape)
                    if len(shape) == 0:
                        golden_input = torch.randn(1, dtype=dtype).squeeze()
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

    ################ stablehlo.MaximumOp ###############

    @tag(stablehlo.MaxOp)
    def maximum(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.maximum)

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

    @parse(stablehlo.MaxOp)
    def maximum_parser(
        self,
        old_op: stablehlo.MaxOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.maximum_parser)
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

    @split(stablehlo.MaxOp)
    def maximum_split(
        self,
        old_op: stablehlo.MaxOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.maximum_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            max_module = Module.create()
            max_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(max_module.body):

                @func.func(*op_input_types, name="maximum_module")
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
                        max_builder._set_golden_tensor(new_op, golden_output)
                        max_builder._set_output_ordering([new_op])
                        max_builder._set_golden_tensor(lhs, input0)
                        max_builder._set_golden_tensor(rhs, input1)
                        max_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return max_module, max_builder

    ################ stablehlo.MinimumOp ###############

    @tag(stablehlo.MinOp)
    def minimum(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.minimum)

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

    @parse(stablehlo.MinOp)
    def minimum_parser(
        self,
        old_op: stablehlo.MinOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.minimum_parser)
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

    @split(stablehlo.MinOp)
    def minimum_split(
        self,
        old_op: stablehlo.MinOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.minimum_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            min_module = Module.create()
            min_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(min_module.body):

                @func.func(*op_input_types, name="minimum_module")
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
                        min_builder._set_golden_tensor(new_op, golden_output)
                        min_builder._set_output_ordering([new_op])
                        min_builder._set_golden_tensor(lhs, input0)
                        min_builder._set_golden_tensor(rhs, input1)
                        min_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return min_module, min_builder

    ################ stablehlo.MulOp ###############

    @tag(stablehlo.MulOp)
    def multiply(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.multiply)

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

    @parse(stablehlo.MulOp)
    def multiply_parser(
        self,
        old_op: stablehlo.MulOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.multiply_parser)
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

    @split(stablehlo.MulOp)
    def multiply_split(
        self,
        old_op: stablehlo.MulOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.multiply_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            mul_module = Module.create()
            mul_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(mul_module.body):

                @func.func(*op_input_types, name="multiply_module")
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
                        mul_builder._set_golden_tensor(new_op, golden_output)
                        mul_builder._set_output_ordering([new_op])
                        mul_builder._set_golden_tensor(lhs, input0)
                        mul_builder._set_golden_tensor(rhs, input1)
                        mul_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return mul_module, mul_builder

    ################ stablehlo.SubtractOp ###############

    @tag(stablehlo.SubtractOp)
    def subtract(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.subtract)

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

    @parse(stablehlo.SubtractOp)
    def subtract_parser(
        self,
        old_op: stablehlo.SubtractOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.subtract_parser)
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

    @split(stablehlo.SubtractOp)
    def subtract_split(
        self,
        old_op: stablehlo.SubtractOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.subtract_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            sub_module = Module.create()
            sub_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(sub_module.body):

                @func.func(*op_input_types, name="subtract_module")
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
                        sub_builder._set_golden_tensor(new_op, golden_output)
                        sub_builder._set_output_ordering([new_op])
                        sub_builder._set_golden_tensor(lhs, input0)
                        sub_builder._set_golden_tensor(rhs, input1)
                        sub_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return sub_module, sub_builder

    ################ stablehlo.PowOp ###############

    @tag(stablehlo.PowOp)
    def pow(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.pow)

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

    @parse(stablehlo.PowOp)
    def pow_parser(
        self,
        old_op: stablehlo.PowOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.pow_parser)
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

    @split(stablehlo.PowOp)
    def pow_split(
        self,
        old_op: stablehlo.PowOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.pow_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            pow_module = Module.create()
            pow_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(pow_module.body):

                @func.func(*op_input_types, name="pow_module")
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
                        pow_builder._set_golden_tensor(new_op, golden_output)
                        pow_builder._set_output_ordering([new_op])
                        pow_builder._set_golden_tensor(lhs, input0)
                        pow_builder._set_golden_tensor(rhs, input1)
                        pow_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return pow_module, pow_builder

    ################ stablehlo.ShiftRightLogicalOp ###############

    @tag(stablehlo.ShiftRightLogicalOp)
    def shift_right_logical(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.shift_right_logical)

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

    @parse(stablehlo.ShiftRightLogicalOp)
    def shift_right_logical_parser(
        self,
        old_op: stablehlo.ShiftRightLogicalOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.shift_right_logical_parser
        )
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

    @split(stablehlo.ShiftRightLogicalOp)
    def shift_right_logical_split(
        self,
        old_op: stablehlo.ShiftRightLogicalOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(
            StableHLOBuilder.shift_right_logical_split
        )

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            srl_module = Module.create()
            srl_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(srl_module.body):

                @func.func(*op_input_types, name="shift_right_logical_module")
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
                        srl_builder._set_golden_tensor(new_op, golden_output)
                        srl_builder._set_output_ordering([new_op])
                        srl_builder._set_golden_tensor(lhs, input0)
                        srl_builder._set_golden_tensor(rhs, input1)
                        srl_builder._set_input_ordering([lhs, rhs])

                    return new_op

        return srl_module, srl_builder

    ############### stablehlo.ReverseOp ###############

    @tag(stablehlo.ReverseOp)
    def reverse(
        self,
        in0: Operand,
        dimensions: List[int],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.reverse)

        dimensions_attr = DenseI64ArrayAttr.get(dimensions, context=self._ctx)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            dimensions_attr,
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
            golden_output = op_golden_function(
                input0, dimensions_attr, op.result.type.element_type
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.ReverseOp)
    def reverse_parser(
        self,
        old_op: stablehlo.ReverseOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.reverse_parser)

        in0 = global_dict[old_op.operand]
        dimensions_attr = old_op.dimensions

        new_op = stablehlo_op(
            in0,
            dimensions_attr,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, dimensions_attr, old_op.result.type.element_type
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.ReverseOp)
    def reverse_split(
        self,
        old_op: stablehlo.ReverseOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.reverse_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            reverse_module = Module.create()
            reverse_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(reverse_module.body):

                @func.func(*op_input_types, name="reverse_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    new_op = stablehlo_op(
                        in0,
                        old_op.dimensions,
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.dimensions, old_op.result.type.element_type
                        )
                        reverse_builder._set_golden_tensor(new_op.result, golden_output)
                        reverse_builder._set_output_ordering([new_op.result])
                        reverse_builder._set_golden_tensor(in0, input0)
                        reverse_builder._set_input_ordering([in0])

                    return new_op

        return reverse_module, reverse_builder

    ############### stablehlo.PadOp ###############

    @tag(stablehlo.PadOp)
    def pad(
        self,
        in0: Operand,
        padding_value: int,
        edge_padding_low: List[int],
        edge_padding_high: List[int],
        interior_padding: List[int],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.pad)

        edge_low_attr = DenseI64ArrayAttr.get(edge_padding_low, context=self._ctx)
        edge_high_attr = DenseI64ArrayAttr.get(edge_padding_high, context=self._ctx)
        interior_attr = DenseI64ArrayAttr.get(interior_padding, context=self._ctx)
        padding_value_attr = IntegerAttr.get(self.get_type(in0), padding_value)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        padding_value = DenseElementsAttr.get_splat(
            self.get_type(in0), padding_value_attr
        )
        padding_value = stablehlo.ConstantOp(padding_value, loc=loc).result

        op = stablehlo_op(
            in0,
            padding_value,
            edge_low_attr,
            edge_high_attr,
            interior_attr,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            pad_val = self._get_golden_tensor(padding_value)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                pad_val,
                edge_low_attr,
                edge_high_attr,
                interior_attr,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.PadOp)
    def pad_parser(
        self,
        old_op: stablehlo.PadOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.pad_parser)

        in0 = global_dict[old_op.operand]
        padding_value_operand = global_dict[old_op.padding_value]
        edge_low_attr = old_op.edge_padding_low
        edge_high_attr = old_op.edge_padding_high
        interior_attr = old_op.interior_padding

        new_op = stablehlo_op(
            in0,
            padding_value_operand,
            edge_low_attr,
            edge_high_attr,
            interior_attr,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            pad_val = self._get_golden_tensor(padding_value_operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                pad_val,
                edge_low_attr,
                edge_high_attr,
                interior_attr,
                old_op.result.type.element_type,
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.PadOp)
    def pad_split(
        self,
        old_op: stablehlo.PadOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.pad_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            pad_module = Module.create()
            pad_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [old_op.operand.type]

            with InsertionPoint(pad_module.body):

                @func.func(*op_input_types, name="pad_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    pad_val = inputs[1] if len(inputs) > 1 else None
                    new_op = stablehlo_op(
                        in0,
                        pad_val if pad_val is not None else old_op.padding_value,
                        old_op.edge_padding_low,
                        old_op.edge_padding_high,
                        old_op.interior_padding,
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        pad_val_golden = self._get_golden_tensor(
                            pad_val if pad_val is not None else old_op.padding_value
                        )
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0,
                            pad_val_golden,
                            old_op.edge_padding_low,
                            old_op.edge_padding_high,
                            old_op.interior_padding,
                            old_op.result.type.element_type,
                        )
                        pad_builder._set_golden_tensor(new_op.result, golden_output)
                        pad_builder._set_output_ordering([new_op.result])
                        pad_builder._set_golden_tensor(in0, input0)
                        pad_builder._set_input_ordering([in0])

                    return new_op

        return pad_module, pad_builder

    ############### stablehlo.SelectOp ###############

    @tag(stablehlo.SelectOp)
    def select(
        self,
        pred: Operand,
        on_true: Operand,
        on_false: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.select)

        op = stablehlo_op(
            pred,
            on_true,
            on_false,
            loc=loc,
        )

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            pred_g = self._get_golden_tensor(pred)
            true_g = self._get_golden_tensor(on_true)
            false_g = self._get_golden_tensor(on_false)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                pred_g, true_g, false_g, op.result.type.element_type
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.SelectOp)
    def select_parser(
        self,
        old_op: stablehlo.SelectOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.select_parser)

        pred = global_dict[old_op.pred]
        on_true = global_dict[old_op.on_true]
        on_false = global_dict[old_op.on_false]

        new_op = stablehlo_op(
            pred,
            on_true,
            on_false,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            pred_g = self._get_golden_tensor(pred)
            true_g = self._get_golden_tensor(on_true)
            false_g = self._get_golden_tensor(on_false)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                pred_g, true_g, false_g, old_op.result.type.element_type
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.SelectOp)
    def select_split(
        self,
        old_op: stablehlo.SelectOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.select_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            sel_module = Module.create()
            sel_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [
                old_op.pred.type,
                old_op.on_true.type,
                old_op.on_false.type,
            ]

            with InsertionPoint(sel_module.body):

                @func.func(*op_input_types, name="select_module")
                def decorated_func(*inputs):
                    pred, on_true, on_false = inputs[0], inputs[1], inputs[2]
                    new_op = stablehlo_op(pred, on_true, on_false, loc=old_op.location)

                    if not self._disable_golden_check:
                        pred_g = self._get_golden_tensor(pred_old)
                        true_g = self._get_golden_tensor(true_old)
                        false_g = self._get_golden_tensor(false_old)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            pred_g, true_g, false_g, old_op.result.type.element_type
                        )
                        sel_builder._set_golden_tensor(new_op.result, golden_output)
                        sel_builder._set_output_ordering([new_op.result])
                        sel_builder._set_golden_tensor(pred, pred_g)
                        sel_builder._set_golden_tensor(on_true, true_g)
                        sel_builder._set_golden_tensor(on_false, false_g)
                        sel_builder._set_input_ordering([pred, on_true, on_false])

                    return new_op

        return sel_module, sel_builder

    ############### stablehlo.ScatterOp ###############

    @tag(stablehlo.ScatterOp)
    def scatter(
        self,
        inputs: List[Operand],
        scatter_indices: Operand,
        updates: List[Operand],
        update_window_dims: List[int],
        inserted_window_dims: List[int],
        input_batching_dims: List[int],
        scatter_indices_batching_dims: List[int],
        scatter_dims_to_operand_dims: List[int],
        index_vector_dim: int,
        indices_are_sorted: bool = False,
        unique_indices: bool = False,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> Tuple[OpResult, ...]:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.scatter)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        # Create scatter dimension numbers attribute
        scatter_dimension_numbers_attr = stablehlo.ScatterDimensionNumbers.get(
            update_window_dims,
            inserted_window_dims,
            input_batching_dims,
            scatter_indices_batching_dims,
            scatter_dims_to_operand_dims,
            index_vector_dim,
        )
        indices_are_sorted_attr = BoolAttr.get(indices_are_sorted)
        unique_indices_attr = BoolAttr.get(unique_indices)

        # Result types are the same as input types
        results = [inp.type for inp in inputs]

        op = stablehlo_op(
            results,
            inputs,
            scatter_indices,
            updates,
            scatter_dimension_numbers_attr,
            indices_are_sorted=indices_are_sorted_attr,
            unique_indices=unique_indices_attr,
            loc=loc,
        )

        # Create default update computation (addition)
        # The computation takes 2*len(inputs) arguments (pairs from each input)
        element_types = [RankedTensorType(inp.type).element_type for inp in inputs]
        scalar_types = [
            RankedTensorType.get([], elem_type) for elem_type in element_types
        ]

        # The update computation takes pairs: (old_value, update_value) for each input
        computation_arg_types = []
        for scalar_type in scalar_types:
            computation_arg_types.extend([scalar_type, scalar_type])

        block = Block.create_at_start(op.update_computation, computation_arg_types)

        with InsertionPoint(block):
            results = []
            for i in range(len(inputs)):
                old_val = block.arguments[2 * i]
                update_val = block.arguments[2 * i + 1]

                add_result = stablehlo.AddOp(old_val, update_val, loc=loc).result
                results.append(add_result)

            stablehlo.ReturnOp(results, loc=loc)

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input_tensors = [self._get_golden_tensor(input) for input in inputs]
            indices_tensor = self._get_golden_tensor(scatter_indices)
            update_tensors = [self._get_golden_tensor(update) for update in updates]

            op_golden_function = get_golden_function(stablehlo_op)
            golden_outputs = op_golden_function(
                input_tensors,
                indices_tensor,
                update_tensors,
                scatter_dimension_numbers_attr,
                indices_are_sorted=indices_are_sorted_attr,
                unique_indices=unique_indices_attr,
            )

            for i, result in enumerate(op.results):
                self._set_golden_tensor(result, golden_outputs[i])

        return tuple(op.results)

    @parse(stablehlo.ScatterOp)
    def scatter_parser(
        self,
        old_op: stablehlo.ScatterOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.scatter_parser)
        inputs = [global_dict[operand] for operand in old_op.inputs]
        scatter_indices = global_dict[old_op.scatter_indices]
        updates = [global_dict[operand] for operand in old_op.updates]
        scatter_dimension_numbers_attr = old_op.scatter_dimension_numbers
        indices_are_sorted_attr = old_op.indices_are_sorted
        unique_indices_attr = old_op.unique_indices

        results = [result.type for result in old_op.results]

        new_op = stablehlo_op(
            results,
            inputs,
            scatter_indices,
            updates,
            scatter_dimension_numbers_attr,
            indices_are_sorted=indices_are_sorted_attr,
            unique_indices=unique_indices_attr,
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            input_tensors = [self._get_golden_tensor(inp) for inp in inputs]
            indices_tensor = self._get_golden_tensor(scatter_indices)
            update_tensors = [self._get_golden_tensor(upd) for upd in updates]

            op_golden_function = get_golden_function(stablehlo_op)
            golden_outputs = op_golden_function(
                input_tensors,
                indices_tensor,
                update_tensors,
                scatter_dimension_numbers_attr,
                indices_are_sorted=indices_are_sorted_attr,
                unique_indices=unique_indices_attr,
            )

            for i, result in enumerate(new_op.results):
                self._set_golden_tensor(result, golden_outputs[i])

        op_map_dictionary = {}
        for i, result in enumerate(old_op.results):
            op_map_dictionary[result] = new_op.results[i]
        return new_op, op_map_dictionary

    @split(stablehlo.ScatterOp)
    def scatter_split(
        self,
        old_op: stablehlo.ScatterOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.scatter_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            scatter_module = Module.create()
            scatter_builder = StableHLOBuilder(old_ctx, old_loc)
            input_types = [operand.type for operand in old_op.inputs]
            indices_type = [old_op.scatter_indices.type]
            updates_types = [operand.type for operand in old_op.updates]
            op_input_types = input_types + indices_type + updates_types

            with InsertionPoint(scatter_module.body):

                @func.func(*op_input_types, name="scatter_module")
                def decorated_func(*all_inputs):
                    n_inputs = len(input_types)
                    n_updates = len(updates_types)

                    inputs = all_inputs[:n_inputs]
                    scatter_indices = all_inputs[n_inputs]
                    updates = all_inputs[n_inputs + 1 : n_inputs + 1 + n_updates]

                    scatter_dimension_numbers_attr = old_op.scatter_dimension_numbers
                    indices_are_sorted_attr = old_op.indices_are_sorted
                    unique_indices_attr = old_op.unique_indices
                    results = [result.type for result in old_op.results]

                    new_op = stablehlo_op(
                        results,
                        inputs,
                        scatter_indices,
                        updates,
                        scatter_dimension_numbers_attr,
                        indices_are_sorted=indices_are_sorted_attr,
                        unique_indices=unique_indices_attr,
                        loc=old_loc,
                    )

                    if not scatter_builder._disable_golden_check:
                        input_tensors = [
                            scatter_builder._get_golden_tensor(inp) for inp in inputs
                        ]
                        indices_tensor = scatter_builder._get_golden_tensor(
                            scatter_indices
                        )
                        update_tensors = [
                            scatter_builder._get_golden_tensor(upd) for upd in updates
                        ]

                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_outputs = op_golden_function(
                            input_tensors,
                            indices_tensor,
                            update_tensors,
                            scatter_dimension_numbers_attr,
                            indices_are_sorted=indices_are_sorted_attr,
                            unique_indices=unique_indices_attr,
                        )

                        for i, result in enumerate(new_op.results):
                            scatter_builder._set_golden_tensor(
                                result, golden_outputs[i]
                            )
                        scatter_builder._set_output_ordering(list(new_op.results))
                        for i, inp in enumerate(all_inputs):
                            if i < n_inputs:
                                scatter_builder._set_golden_tensor(
                                    inp, input_tensors[i]
                                )
                            elif i == n_inputs:
                                scatter_builder._set_golden_tensor(inp, indices_tensor)
                            else:
                                scatter_builder._set_golden_tensor(
                                    inp, update_tensors[i - n_inputs - 1]
                                )
                        scatter_builder._set_input_ordering(list(all_inputs))

                    return tuple(new_op.results)

        return scatter_module, scatter_builder

    ############### stablehlo.SelectAndScatterOp ###############

    @tag(stablehlo.SelectAndScatterOp)
    def select_and_scatter(
        self,
        operand: Operand,
        source: Operand,
        init_value: Operand,
        window_dimensions: List[int],
        window_strides: List[int],
        padding: Optional[List[List[int]]] = None,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.select_and_scatter)

        if output_type is None:
            mlir_output_elem_type = self.get_type(operand)
        else:
            mlir_output_elem_type = self._get_type_from_torch_dtype(output_type)

        # Result shape equals operand shape
        operand_type = RankedTensorType(operand.type)
        result = self._create_ranked_tensor_type(
            operand_type.shape,
            mlir_output_elem_type,
        )

        window_dimensions_attr = DenseI64ArrayAttr.get(window_dimensions)
        window_strides_attr = DenseI64ArrayAttr.get(window_strides)
        padding_attr = None
        if padding is not None:
            # Convert [[low, high], ...] to tensor<rankx2xi64> DenseElementsAttr
            flat = []
            for pair in padding:
                flat.extend([int(pair[0]), int(pair[1])])
            padding_attr = DenseElementsAttr.get(
                RankedTensorType.get(
                    [len(padding), 2], IntegerType.get_signless(64, self._ctx)
                ),
                flat,
            )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        stablehlo_kwargs = {
            "window_dimensions": window_dimensions_attr,
            "window_strides": window_strides_attr,
        }
        if padding_attr is not None:
            stablehlo_kwargs["padding"] = padding_attr

        op = stablehlo_op(
            result,
            operand,
            source,
            init_value,
            loc=loc,
            **stablehlo_kwargs,
        )

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            operand_g = self._get_golden_tensor(operand)
            source_g = self._get_golden_tensor(source)
            init_g = self._get_golden_tensor(init_value)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                operand_g,
                source_g,
                init_g,
                window_dimensions_attr,
                window_strides_attr,
                padding_attr,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op.result, golden_output)

        return op.result

    @parse(stablehlo.SelectAndScatterOp)
    def select_and_scatter_parser(
        self,
        old_op: stablehlo.SelectAndScatterOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.select_and_scatter_parser
        )

        operand = global_dict[old_op.operand]
        source = global_dict[old_op.source]
        init_value = global_dict[old_op.init_value]
        result_type = old_op.result.type

        new_op = stablehlo_op(
            result_type,
            operand,
            source,
            init_value,
            window_dimensions=old_op.window_dimensions,
            window_strides=old_op.window_strides,
            padding=getattr(old_op, "padding", None),
            loc=old_op.location,
        )

        if not self._disable_golden_check:
            operand_g = self._get_golden_tensor(operand)
            source_g = self._get_golden_tensor(source)
            init_g = self._get_golden_tensor(init_value)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                operand_g,
                source_g,
                init_g,
                old_op.window_dimensions,
                old_op.window_strides,
                getattr(old_op, "padding", None),
                result_type.element_type,
            )
            self._set_golden_tensor(new_op.result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op.result
        return new_op, op_map_dictionary

    @split(stablehlo.SelectAndScatterOp)
    def select_and_scatter_split(
        self,
        old_op: stablehlo.SelectAndScatterOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(
            StableHLOBuilder.select_and_scatter_split
        )

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            sas_module = Module.create()
            sas_builder = StableHLOBuilder(old_ctx, old_loc)
            op_input_types = [
                old_op.operand.type,
                old_op.source.type,
                old_op.init_value.type,
            ]

            with InsertionPoint(sas_module.body):

                @func.func(*op_input_types, name="select_and_scatter_module")
                def decorated_func(*inputs):
                    operand, source, init_value = inputs[0], inputs[1], inputs[2]
                    result_type = old_op.result.type

                    new_op = stablehlo_op(
                        result_type,
                        operand,
                        source,
                        init_value,
                        window_dimensions=old_op.window_dimensions,
                        window_strides=old_op.window_strides,
                        padding=getattr(old_op, "padding", None),
                        loc=old_op.location,
                    )

                    if not self._disable_golden_check:
                        operand_g = self._get_golden_tensor(old_op.operand)
                        source_g = self._get_golden_tensor(old_op.source)
                        init_g = self._get_golden_tensor(old_op.init_value)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            operand_g,
                            source_g,
                            init_g,
                            old_op.window_dimensions,
                            old_op.window_strides,
                            getattr(old_op, "padding", None),
                            result_type.element_type,
                        )
                        sas_builder._set_golden_tensor(new_op.result, golden_output)
                        sas_builder._set_output_ordering([new_op.result])
                        sas_builder._set_golden_tensor(operand, operand_g)
                        sas_builder._set_golden_tensor(source, source_g)
                        sas_builder._set_golden_tensor(init_value, init_g)
                        sas_builder._set_input_ordering([operand, source, init_value])

                    return new_op

        return sas_module, sas_builder

    def clamp(
        self,
        min: Operand,
        operand: Operand,
        max: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.clamp``.

        *Elementwise clamp operation.*

        Clamps each element of the operand tensor between a minimum and maximum value.
        For each element, returns min if element < min, max if element > max, otherwise element.

        Mathematical definition: clamp(min, x, max) = min(max(x, min), max)

        .. code-block:: mlir

            // Clamp elements between min and max
            %result = stablehlo.clamp(%min, %operand, %max) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // min: [5, 10, 15]
            // operand: [3, 13, 23]
            // max: [10, 15, 20]
            // Output tensor:
            // [5, 13, 20]

        Parameters
        ----------
        min : Operand
            Minimum value tensor (can be scalar or tensor)
        operand : Operand
            Input tensor to be clamped
        max : Operand
            Maximum value tensor (can be scalar or tensor)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes
        sharding_attr : *Optional[sdy.TensorShardingPerValueAttr]*
            Optional sharding attribute

        Returns
        -------
        (*OpView*)
            A tensor containing the clamped values
        """
        return self._eltwise_proxy(
            stablehlo.ClampOp,
            [min, operand, max],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    # ----- Logical and Bitwise Operations -----

    def and_(
        self,
        in0: Operand,
        in1: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.and``.

        *Elementwise AND operation.*

        Performs elementwise AND operation between two tensors.
        For booleans, performs logical AND.
        For integers, performs bitwise AND.

        Mathematical definition:
        - Logical: and(x, y) = x AND y
        - Bitwise: and(x, y) = x & y

        .. code-block:: mlir

            // Logical AND for booleans
            %result = stablehlo.and(%lhs, %rhs) : tensor<3xi1>, tensor<3xi1> -> tensor<3xi1>
            // Input tensors:
            // lhs: [true, false, true]
            // rhs: [true, true, false]
            // Output tensor:
            // [true, false, false]

            // Bitwise AND for integers
            %result = stablehlo.and(%lhs, %rhs) : tensor<3xi32>, tensor<3xi32> -> tensor<3xi32>
            // Input tensors:
            // lhs: [5, 6, 7]  // Binary: 101, 110, 111
            // rhs: [3, 3, 3]  // Binary: 011, 011, 011
            // Output tensor:
            // [1, 2, 3]       // Binary: 001, 010, 011

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
            A tensor containing the elementwise AND of the inputs
        """

        return self._eltwise_proxy(
            stablehlo.AndOp,
            [in0, in1],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

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

    def abs(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.abs``.

        *Elementwise absolute value operation.*

        Computes the element-wise absolute value of the input tensor.

        Mathematical definition: abs(x) = |x|

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise absolute values of the input
        """
        return self._eltwise_proxy(
            stablehlo.AbsOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def ceil(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.ceil``.

        *Elementwise ceiling operation.*

        Computes the element-wise ceiling of the input tensor.

        Mathematical definition: ceil(x) = ⌈x⌉

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise ceiling values of the input
        """
        return self._eltwise_proxy(
            stablehlo.CeilOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def cosine(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.cosine``.

        *Elementwise cosine operation.*

        Computes the element-wise cosine of the input tensor.

        Mathematical definition: cosine(x) = cos(x)

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise cosine values of the input
        """
        return self._eltwise_proxy(
            stablehlo.CosineOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def dot_general(
        self,
        in0: Operand,
        in1: Operand,
        batch_dims_lhs: List[int],
        contract_dims_lhs: List[int],
        batch_dims_rhs: List[int],
        contract_dims_rhs: List[int],
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.dot_general``.

        *Generalized dot product operation.*

        Flexible tensor operation that generalizes matrix multiplication by allowing user to specify which
        dimensions of two tensors to contract. Matrix multiplication is a special case of this operation,
        where the contraction happens along the last axis of the first tensor and the second-to-last axis of the second tensor.
        From StableHLO DotGeneral Op https://openxla.org/stablehlo/spec#dot_general

        Parameters
        ----------
        in0 : Operand
            Left-hand side input tensor
        in1 : Operand
            Right-hand side input tensor
        batch_dims_lhs : *List[int]*
            Batch dimensions for the left-hand side tensor
        contract_dims_lhs : *List[int]*
            Contracting dimensions for the left-hand side tensor
        batch_dims_rhs : *List[int]*
            Batch dimensions for the right-hand side tensor
        contract_dims_rhs : *List[int]*
            Contracting dimensions for the right-hand side tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        from ttmlir.ir import ArrayAttr, IntegerAttr, IntegerType

        # Create dimension numbers attribute using proper MLIR attribute construction
        dot_dimension_numbers = stablehlo.DotDimensionNumbers.get(
            context=self._ctx,
            lhs_batching_dimensions=batch_dims_lhs,
            lhs_contracting_dimensions=contract_dims_lhs,
            rhs_batching_dimensions=batch_dims_rhs,
            rhs_contracting_dimensions=contract_dims_rhs,
        )

        lhs_rhape = in0.type.shape
        rhs_shape = in1.type.shape

        result_shape = []
        # Add batch dimensions
        for dim in batch_dims_lhs:
            result_shape.append(lhs_rhape[dim])

        # add non-batch, non-contract dimensions from lhs and rhs
        for i, dim_size in enumerate(lhs_rhape):
            if i not in batch_dims_lhs and i not in contract_dims_lhs:
                result_shape.append(dim_size)
        for i, dim_size in enumerate(rhs_shape):
            if i not in batch_dims_rhs and i not in contract_dims_rhs:
                result_shape.append(dim_size)

        result_type = RankedTensorType.get(result_shape, in0.type.element_type)
        return self._op_proxy(
            stablehlo.DotGeneralOp,
            [in0, in1],
            organize_stablehlo_args=lambda inputs, *_: (
                result_type,
                inputs[0],
                inputs[1],
            ),
            organize_golden_args=lambda inputs: (
                self._get_golden_tensor(inputs[0]),
                self._get_golden_tensor(inputs[1]),
            ),
            stablehlo_kwargs={"dot_dimension_numbers": dot_dimension_numbers},
            golden_kwargs={
                "batch_dims_lhs": batch_dims_lhs,
                "contract_dims_lhs": contract_dims_lhs,
                "batch_dims_rhs": batch_dims_rhs,
                "contract_dims_rhs": contract_dims_rhs,
            },
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def exp(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.exponential``.

        *Elementwise exponential operation.*

        Computes the element-wise exponential of the input tensor.

        Mathematical definition: exp(x) = e^x

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise exponential values of the input
        """
        return self._eltwise_proxy(
            stablehlo.ExpOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def floor(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.floor``.

        *Elementwise floor operation.*

        Computes the element-wise floor of the input tensor.

        Mathematical definition: floor(x) = ⌊x⌋

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise floor values of the input
        """
        return self._eltwise_proxy(
            stablehlo.FloorOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )
