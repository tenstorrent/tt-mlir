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

from golden import get_golden_function, apply_sharding, apply_unsharding


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

    # ----- Class helper methods -----

    @classmethod
    def build_opname_to_opview_map(cls):
        for name, obj in inspect.getmembers(stablehlo, inspect.isclass):
            if issubclass(obj, OpView) and obj is not OpView:
                op_name = getattr(obj, "OPERATION_NAME", None)

                if op_name is not None:
                    cls.opname_to_opview_map[op_name] = obj

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

    def _get_location(self) -> Location:
        stack = inspect.stack()
        caller_frame = stack[2]
        filename = caller_frame.filename
        lineno = caller_frame.lineno
        return Location.name(f"{filename}:{lineno}")

    # ----- Public StableHLO Op Generators ----

    ############### stablehlo.ReduceScatterOp ###############

    @tag(stablehlo.ReduceScatterOp)
    def reduce_scatter(
        self,
        input: Operand,
        scatter_dimensions: int,
        replica_groups: List[List[int]],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.reduce_scatter)

        scatter_dimensions_attr = IntegerAttr.get(
            IntegerType.get_signless(64), scatter_dimensions
        )
        replica_groups_attr = DenseElementsAttr.get(np.array(replica_groups))
        input0 = self._get_golden_tensor(input)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(
            input0, scatter_dimensions_attr, replica_groups_attr
        )
        result = self._create_ranked_tensor_type(
            golden_output.shape, self.get_type(input)
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            result,
            input,
            scatter_dimensions_attr,
            replica_groups_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.ReduceScatterOp)
    def reduce_scatter_parser(
        self,
        old_op: stablehlo.ReduceScatterOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.reduce_scatter_parser
        )

        input = global_dict[old_op.operand]
        scatter_dimensions_attr = old_op.scatter_dimensions
        replica_groups_attr = old_op.replica_groups

        new_op = stablehlo_op(
            old_op.result.type,
            input,
            scatter_dimensions_attr,
            replica_groups_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, scatter_dimensions_attr, replica_groups_attr
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### stablehlo.CollectivePermuteOp ###############

    @tag(stablehlo.CollectivePermuteOp)
    def collective_permute(
        self,
        input: Operand,
        source_target_pairs: List[Tuple[int, int]],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.collective_permute)

        source_target_pairs_attr = DenseElementsAttr.get(np.array(source_target_pairs))

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            input,
            source_target_pairs_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, source_target_pairs_attr)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.CollectivePermuteOp)
    def collective_permute_parser(
        self,
        old_op: stablehlo.CollectivePermuteOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.collective_permute_parser
        )

        input = global_dict[old_op.operand]
        source_target_pairs_attr = old_op.source_target_pairs

        new_op = stablehlo_op(
            old_op.result.type,
            input,
            source_target_pairs_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, source_target_pairs_attr)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### stablehlo.CollectiveBroadcastOp ###############

    @tag(stablehlo.CollectiveBroadcastOp)
    def collective_broadcast(
        self,
        input: Operand,
        replica_groups: List[List[int]],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(
            StableHLOBuilder.collective_broadcast
        )

        replica_groups_attr = DenseElementsAttr.get(np.array(replica_groups))

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            input,
            replica_groups_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, replica_groups_attr)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.CollectiveBroadcastOp)
    def collective_broadcast_parser(
        self,
        old_op: stablehlo.CollectiveBroadcastOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.collective_broadcast_parser
        )

        input = global_dict[old_op.operand]
        replica_groups_attr = old_op.replica_groups

        new_op = stablehlo_op(
            old_op.result.type,
            input,
            replica_groups_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, replica_groups_attr)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### stablehlo.AllToAllOp ###############

    @tag(stablehlo.AllToAllOp)
    def all_to_all(
        self,
        input: Operand,
        split_dim: int,
        concat_dim: int,
        split_count: int,
        replica_groups: List[List[int]],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.all_to_all)

        split_dim_attr = IntegerAttr.get(IntegerType.get_signless(64), split_dim)
        concat_dim_attr = IntegerAttr.get(IntegerType.get_signless(64), concat_dim)
        split_count_attr = IntegerAttr.get(IntegerType.get_signless(64), split_count)
        replica_groups_attr = DenseElementsAttr.get(np.array(replica_groups))

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            input,
            split_dim_attr,
            concat_dim_attr,
            split_count_attr,
            replica_groups_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                split_dim_attr,
                concat_dim_attr,
                split_count_attr,
                replica_groups_attr,
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.AllToAllOp)
    def all_to_all_parser(
        self,
        old_op: stablehlo.AllToAllOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.all_to_all_parser)

        input = global_dict[old_op.operand]
        split_dim_attr = old_op.split_dim
        concat_dim_attr = old_op.concat_dim
        split_count_attr = old_op.split_count
        replica_groups_attr = old_op.replica_groups

        new_op = stablehlo_op(
            input,
            split_dim_attr,
            concat_dim_attr,
            split_count_attr,
            replica_groups_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                split_dim_attr,
                concat_dim_attr,
                split_count_attr,
                replica_groups_attr,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### stablehlo.AllReduceOp ###############

    @tag(stablehlo.AllReduceOp)
    def all_reduce(
        input: Operand,
        replica_groups: Optional[List[List[int]]],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.all_reduce)

        replica_groups_attr = DenseElementsAttr.get(np.array(replica_groups))
        input0 = self._get_golden_tensor(input)
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(input0, replica_groups_attr)
        result = self._create_ranked_tensor_type(
            golden_output.shape, self.get_type(input)
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            result,
            input,
            replica_groups_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.AllReduceOp)
    def all_reduce_parser(
        self,
        old_op: stablehlo.AllReduceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.all_reduce_parser)

        input = global_dict[old_op.operand]
        replica_groups_attr = old_op.replica_groups

        new_op = stablehlo_op(
            old_op.result.type,
            input,
            replica_groups_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, replica_groups_attr)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### stablehlo.AllGatherOp ###############

    def infer_all_gather_output_shape(
        self,
        input: Operand,
        all_gather_dim: int,
        replica_groups: Optional[List[List[int]]],
    ) -> List[int]:
        input_type = input.type
        input_shape = input_type.shape

        if replica_groups is None:
            return input_shape

        group_size = len(replica_groups[0])
        output_shape = list(input_shape)
        output_shape[all_gather_dim] = input_shape[all_gather_dim] * group_size
        return output_shape

    @tag(stablehlo.AllGatherOp)
    def all_gather(
        self,
        input: Operand,
        all_gather_dim: int,
        replica_groups: Optional[List[List[int]]],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.all_gather)

        all_gather_dim_attr = IntegerAttr.get(
            IntegerType.get_signless(64), all_gather_dim
        )
        replica_groups_attr = DenseElementsAttr.get(np.array(replica_groups))
        output_shape = self.infer_all_gather_output_shape(
            input, all_gather_dim, replica_groups
        )
        result = self._create_ranked_tensor_type(output_shape, self.get_type(input))

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            [result],
            [input],
            all_gather_dim_attr,
            replica_groups_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, all_gather_dim_attr, replica_groups_attr
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.AllGatherOp)
    def all_gather_parser(
        self,
        old_op: stablehlo.AllGatherOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.all_gather_parser)

        input = global_dict[old_op.operand]
        all_gather_dim_attr = old_op.all_gather_dim
        replica_groups_attr = old_op.replica_groups

        new_op = stablehlo_op(
            [old_op.result.type],
            [input],
            all_gather_dim_attr,
            replica_groups_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, all_gather_dim_attr, replica_groups_attr
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### stablehlo.DynamicUpdateSliceOp ###############

    @tag(stablehlo.DynamicUpdateSliceOp)
    def dynamic_update_slice(
        self,
        input: Operand,
        update: Operand,
        start_indices: List[Operand],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(
            StableHLOBuilder.dynamic_update_slice
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            input,
            update,
            start_indices,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input_tensor = self._get_golden_tensor(input)
            update_tensor = self._get_golden_tensor(update)
            start_indices_tensors = [
                self._get_golden_tensor(idx) for idx in start_indices
            ]
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input_tensor,
                update_tensor,
                start_indices_tensors,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.DynamicUpdateSliceOp)
    def dynamic_update_slice_parser(
        self,
        old_op: stablehlo.DynamicUpdateSliceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.dynamic_update_slice_parser
        )

        input = global_dict[old_op.operand]
        update = global_dict[old_op.update]
        start_indices = [global_dict[idx] for idx in old_op.start_indices]

        new_op = stablehlo_op(
            input,
            update,
            start_indices,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input_tensor = self._get_golden_tensor(input)
            update_tensor = self._get_golden_tensor(update)
            start_indices_tensors = [
                self._get_golden_tensor(idx) for idx in start_indices
            ]
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input_tensor,
                update_tensor,
                start_indices_tensors,
                new_op_result.type.element_type,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="dynamic_update_slice_module")
                def decorated_func(*inputs):
                    input = inputs[0]
                    update = inputs[1]
                    start_indices = inputs[2 : 2 + len(old_op.start_indices)]

                    new_op = stablehlo_op(
                        input, update, start_indices, loc=old_op.location
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input_tensor = self._get_golden_tensor(old_op.operand)
                        update_tensor = self._get_golden_tensor(old_op.update)
                        start_indices_tensors = [
                            self._get_golden_tensor(idx) for idx in old_op.start_indices
                        ]
                        golden_output = op_golden_function(
                            input_tensor,
                            update_tensor,
                            start_indices_tensors,
                            new_op_result.type.element_type,
                        )
                        dynamic_update_slice_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        dynamic_update_slice_builder._set_golden_tensor(
                            input, input_tensor
                        )
                        dynamic_update_slice_builder._set_golden_tensor(
                            update, update_tensor
                        )
                        ordered_inputs.extend([input, update] + list(start_indices))
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                dynamic_update_slice_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return dynamic_update_slice_module, dynamic_update_slice_builder

    ############### stablehlo.AddOp ###############

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

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

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
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="add_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
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

    ################ stablehlo.AndOp ###############

    @tag(stablehlo.AndOp)
    def logical_and(
        self,
        in0: Operand,
        in1: Operand,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.logical_and)

        op = stablehlo_op(
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

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
                input0, input1, op_result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.AndOp)
    def logical_and_parser(
        self,
        old_op: stablehlo.AndOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.logical_and_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.AndOp)
    def logical_and_split(
        self,
        old_op: stablehlo.AndOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.logical_and_split)

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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="and_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op_result.type.element_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op_result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="abs_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op_result.type.element_type
                        )
                        abs_builder._set_golden_tensor(new_op_result, golden_output)
                        abs_builder._set_golden_tensor(operand, input0)
                        ordered_inputs.append(operand)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                abs_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return abs_module, abs_builder

    ################ stablehlo.SortOp ###############

    @tag(stablehlo.SortOp)
    def sort(
        self,
        in0: Operand,
        dimension: int = -1,
        is_stable: bool = False,
        descending: bool = False,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.sort)

        dimension_attr = IntegerAttr.get(IntegerType.get_signless(64), dimension)
        is_stable_attr = BoolAttr.get(is_stable)

        op = stablehlo_op(
            [in0.type],
            [in0],
            dimension=dimension_attr,
            is_stable=is_stable_attr,
            loc=loc,
        )

        element_type = RankedTensorType(in0.type).element_type
        scalar_type = RankedTensorType.get([], element_type)
        compare_direction = stablehlo.ComparisonDirectionAttr.get(
            "GT" if descending else "LT", self._ctx
        )
        compare_type = stablehlo.ComparisonTypeAttr.get("TOTALORDER", self._ctx)

        comparator_region = op.comparator
        comparator_block = Block.create_at_start(
            comparator_region, [scalar_type, scalar_type]
        )
        with InsertionPoint(comparator_block):
            compare_result = stablehlo.CompareOp(
                comparator_block.arguments[0],
                comparator_block.arguments[1],
                compare_direction,
                compare_type=compare_type,
            ).result
            stablehlo.ReturnOp([compare_result], loc=op.location)

        op_result = op.results[0]

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
                dimension_attr,
                is_stable_attr,
                BoolAttr.get(descending),
                op_result.type.element_type,
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.SortOp)
    def sort_parser(
        self,
        old_op: stablehlo.SortOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.sort_parser)

        input_operand = global_dict[old_op.inputs[0]]
        descending = False
        for block in old_op.comparator.blocks:
            for op in block.operations:
                if isinstance(op, stablehlo.CompareOp):
                    direction_attr = getattr(op, "comparison_direction", None)
                    if direction_attr is None:
                        direction_attr = op.operation.attributes.get(
                            "comparison_direction"
                        )
                    if direction_attr is None:
                        direction_attr = op.operation.attributes.get(
                            "compare_direction"
                        )
                    if direction_attr is not None:
                        if hasattr(direction_attr, "value"):
                            direction_str = str(direction_attr.value)
                        else:
                            direction_str = str(direction_attr)
                        if "GT" in direction_str or "GE" in direction_str:
                            descending = True
                    break
            if descending:
                break

        new_op = stablehlo_op(
            [old_op.result.type],
            [input_operand],
            dimension=old_op.dimension,
            is_stable=old_op.is_stable,
            loc=old_op.location,
        )
        element_type = RankedTensorType(input_operand.type).element_type
        scalar_type = RankedTensorType.get([], element_type)
        compare_direction = stablehlo.ComparisonDirectionAttr.get(
            "GT" if descending else "LT", self._ctx
        )
        compare_type = stablehlo.ComparisonTypeAttr.get("TOTALORDER", self._ctx)

        comparator_region = new_op.comparator
        comparator_block = Block.create_at_start(
            comparator_region, [scalar_type, scalar_type]
        )
        with InsertionPoint(comparator_block):
            compare_result = stablehlo.CompareOp(
                comparator_block.arguments[0],
                comparator_block.arguments[1],
                compare_direction,
                compare_type=compare_type,
            ).result
            stablehlo.ReturnOp([compare_result], loc=new_op.location)

        new_op_result = new_op.results[0]

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(input_operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                old_op.dimension,
                old_op.is_stable,
                BoolAttr.get(descending),
                new_op_result.type.element_type,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.SortOp)
    def sort_split(
        self,
        old_op: stablehlo.SortOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.sort_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            sort_module = Module.create()
            sort_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.inputs[0].type,
            ]

            with InsertionPoint(sort_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sort_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    descending = False
                    for block in old_op.comparator.blocks:
                        for op in block.operations:
                            if isinstance(op, stablehlo.CompareOp):
                                direction_attr = getattr(
                                    op, "comparison_direction", None
                                )
                                if direction_attr is None:
                                    direction_attr = op.operation.attributes.get(
                                        "comparison_direction"
                                    )
                                if direction_attr is None:
                                    direction_attr = op.operation.attributes.get(
                                        "compare_direction"
                                    )
                                if direction_attr is not None:
                                    if hasattr(direction_attr, "value"):
                                        direction_str = str(direction_attr.value)
                                    else:
                                        direction_str = str(direction_attr)
                                    if "GT" in direction_str or "GE" in direction_str:
                                        descending = True
                                break
                        if descending:
                            break

                    new_op = stablehlo_op(
                        [input_operand.type],
                        [input_operand],
                        dimension=old_op.dimension,
                        is_stable=old_op.is_stable,
                        loc=old_op.location,
                    )
                    element_type = RankedTensorType(input_operand.type).element_type
                    scalar_type = RankedTensorType.get([], element_type)
                    compare_direction = stablehlo.ComparisonDirectionAttr.get(
                        "GT" if descending else "LT", sort_builder._ctx
                    )
                    compare_type = stablehlo.ComparisonTypeAttr.get(
                        "TOTALORDER", sort_builder._ctx
                    )

                    comparator_region = new_op.comparator
                    comparator_block = Block.create_at_start(
                        comparator_region, [scalar_type, scalar_type]
                    )
                    with InsertionPoint(comparator_block):
                        compare_result = stablehlo.CompareOp(
                            comparator_block.arguments[0],
                            comparator_block.arguments[1],
                            compare_direction,
                            compare_type=compare_type,
                        ).result
                        stablehlo.ReturnOp([compare_result], loc=new_op.location)

                    new_op_result = new_op.results[0]

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.inputs[0])
                        golden_output = op_golden_function(
                            input0,
                            old_op.dimension,
                            old_op.is_stable,
                            BoolAttr.get(descending),
                            new_op_result.type.element_type,
                        )
                        sort_builder._set_golden_tensor(new_op_result, golden_output)
                        sort_builder._set_golden_tensor(input_operand, input0)
                        ordered_inputs.append(input_operand)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                sort_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sort_module, sort_builder

    ################ stablehlo.GetDimensionSizeOp ###############

    @tag(stablehlo.GetDimensionSizeOp)
    def get_dimension_size(
        self,
        in0: Operand,
        dimension: int = 0,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.get_dimension_size)

        dimension_attr = IntegerAttr.get(
            IntegerType.get_signless(64, self._ctx), dimension
        )

        op = stablehlo_op(
            in0,
            dimension=dimension_attr,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, dimension_attr, op_result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.GetDimensionSizeOp)
    def get_dimension_size_parser(
        self,
        old_op: stablehlo.GetDimensionSizeOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.get_dimension_size_parser
        )
        operand = global_dict[old_op.operand]

        new_op = stablehlo_op(
            operand,
            dimension=old_op.dimension,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, old_op.dimension, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.GetDimensionSizeOp)
    def get_dimension_size_split(
        self,
        old_op: stablehlo.GetDimensionSizeOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(
            StableHLOBuilder.get_dimension_size_split
        )

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            get_dimension_size_module = Module.create()
            get_dimension_size_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
            ]

            with InsertionPoint(get_dimension_size_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="get_dimension_size_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(
                        operand, dimension=old_op.dimension, loc=old_op.location
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, old_op.dimension, new_op_result.type.element_type
                        )
                        get_dimension_size_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        get_dimension_size_builder._set_golden_tensor(operand, input0)
                        ordered_inputs.append(operand)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                get_dimension_size_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return get_dimension_size_module, get_dimension_size_builder

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op_result.type.element_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op_result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="ceil_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op_result.type.element_type
                        )
                        ceil_builder._set_golden_tensor(new_op_result, golden_output)
                        ceil_builder._set_golden_tensor(operand, input0)
                        ordered_inputs.append(operand)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                ceil_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

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
        op_result = op.result

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
                input0, input1, op_result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="divide_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
                        )
                        divide_builder._set_golden_tensor(new_op_result, golden_output)
                        divide_builder._set_golden_tensor(lhs, input0)
                        divide_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                divide_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op_result.type.element_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op_result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="cosine_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op_result.type.element_type
                        )
                        cosine_builder._set_golden_tensor(new_op_result, golden_output)
                        cosine_builder._set_golden_tensor(operand, input0)
                        ordered_inputs.append(operand)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                cosine_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op_result.type.element_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op_result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="exp_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op_result.type.element_type
                        )
                        exp_builder._set_golden_tensor(new_op_result, golden_output)
                        exp_builder._set_golden_tensor(operand, input0)
                        ordered_inputs.append(operand)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                exp_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, op_result.type.element_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, new_op_result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="floor_module")
                def decorated_func(*inputs):
                    operand = inputs[0]

                    new_op = stablehlo_op(operand, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.operand)
                        golden_output = op_golden_function(
                            input0, new_op_result.type.element_type
                        )
                        floor_builder._set_golden_tensor(new_op_result, golden_output)
                        floor_builder._set_golden_tensor(operand, input0)
                        ordered_inputs.append(operand)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                floor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return floor_module, floor_builder

    def broadcast_in_dim(
        self,
        in0: Operand,
        broadcast_dimensions: List[int],
        output_shape: List[int],
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.broadcast_in_dim``.
        *Tensor broadcast operation.*
        Broadcasts a tensor to a new shape by replicating its values along specified dimensions.
        The broadcast_dimensions parameter specifies how dimensions of the input map to
        dimensions of the output.
        .. code-block:: mlir
            // Broadcast a 1D tensor to 2D
            %result = stablehlo.broadcast_in_dim(%input) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xf32>) -> tensor<2x3xf32>
            // Input tensor:
            // [1.0, 2.0, 3.0]
            // Output tensor:
            // [[1.0, 2.0, 3.0],
            //  [1.0, 2.0, 3.0]]
        Parameters
        ----------
        in0 : Operand
            Input tensor to broadcast
        broadcast_dimensions : *List[int]*
            List of dimension mappings from input to output
        output_shape : *List[int]*
            Target shape for the broadcasted tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes
        sharding_attr : *Optional[sdy.TensorShardingPerValueAttr]*, optional
            Optional sharding attribute for distributed execution
        Returns
        -------
        (*OpView*)
            The broadcasted tensor
        """
        output_type = self._get_type(in0).element_type
        return self._op_proxy(
            stablehlo.BroadcastInDimOp,
            [in0],
            organize_golden_args=self._organize_eltwise_golden,
            organize_stablehlo_args=lambda inputs, output, _: (output, inputs[0]),
            output_shape=output_shape,
            output_type=output_type,
            golden_kwargs={"size": output_shape},
            stablehlo_kwargs={
                "broadcast_dimensions": broadcast_dimensions,
            },
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    ################ stablehlo.LogOp ###############

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="log_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        log_builder._set_golden_tensor(new_op_result, golden_output)
                        log_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                log_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="neg_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        neg_builder._set_golden_tensor(new_op_result, golden_output)
                        neg_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                neg_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="rsqrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        rsqrt_builder._set_golden_tensor(new_op_result, golden_output)
                        rsqrt_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                rsqrt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

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

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, mlir_output_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sine_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        sine_builder._set_golden_tensor(new_op_result, golden_output)
                        sine_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                sine_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sqrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        sqrt_builder._set_golden_tensor(new_op_result, golden_output)
                        sqrt_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="tan_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        tan_builder._set_golden_tensor(new_op_result, golden_output)
                        tan_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                tan_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="tanh_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        tanh_builder._set_golden_tensor(new_op_result, golden_output)
                        tanh_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                tanh_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="log1p_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        log1p_builder._set_golden_tensor(new_op_result, golden_output)
                        log1p_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                log1p_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, old_op.result.type.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logistic_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.result.type.element_type
                        )
                        logistic_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        logistic_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                logistic_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return logistic_module, logistic_builder

    ############### stablehlo.SliceOp ###############

    @tag(stablehlo.SliceOp)
    def slice(
        self,
        in0: Operand,
        start_indices: List[int],
        limit_indices: List[int],
        strides: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.slice)

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
        op_result = op.result

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
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

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
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

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
                    new_op_result = new_op.result

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
                        slice_builder._set_golden_tensor(new_op_result, golden_output)
                        slice_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                slice_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return slice_module, slice_builder

    ############### stablehlo.DynamicSliceOp ###############

    @tag(stablehlo.DynamicSliceOp)
    def dynamic_slice(
        self,
        operand: Operand,
        start_indices: List[Operand],
        slice_sizes: List[int],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        # If all start indices are Python ints, use static slice
        if all(isinstance(i, int) for i in start_indices):
            limit_indices = [s + sz for s, sz in zip(start_indices, slice_sizes)]
            strides = [1] * len(start_indices)
            return self.slice(
                operand,
                start_indices,
                limit_indices,
                strides,
                loc=loc,
                unit_attrs=unit_attrs,
                sharding_attr=sharding_attr,
            )

        # Otherwise, ensure every start index is a Value (wrap ints as constants)
        start_indices_vals = [
            (
                self.constant(torch.tensor(i, dtype=torch.int64)).result
                if isinstance(i, int)
                else i
            )
            for i in start_indices
        ]

        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.dynamic_slice)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            operand,
            start_indices_vals,
            slice_sizes=slice_sizes,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            operand_tensor = self._get_golden_tensor(operand)
            start_indices_tensors = [
                self._get_golden_tensor(idx) for idx in start_indices_vals
            ]
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                operand_tensor,
                start_indices_tensors,
                slice_sizes,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.DynamicSliceOp)
    def dynamic_slice_parser(
        self,
        old_op: stablehlo.DynamicSliceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.dynamic_slice_parser
        )

        operand = global_dict[old_op.operand]
        start_indices = [global_dict[idx] for idx in old_op.start_indices]
        slice_sizes = list(old_op.slice_sizes)

        new_op = stablehlo_op(
            operand,
            start_indices,
            slice_sizes=slice_sizes,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            operand_tensor = self._get_golden_tensor(operand)
            start_indices_tensors = [
                self._get_golden_tensor(idx) for idx in start_indices
            ]
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                operand_tensor,
                start_indices_tensors,
                slice_sizes,
                new_op_result.type.element_type,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {old_op.result: new_op_result}
        return new_op, op_map_dictionary

    @split(stablehlo.DynamicSliceOp)
    def dynamic_slice_split(
        self,
        old_op: stablehlo.DynamicSliceOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.dynamic_slice_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            dynamic_slice_module = Module.create()
            dynamic_slice_builder = StableHLOBuilder(old_context, old_loc)

            op_input_types = [old_op.operand.type] + [
                idx.type for idx in old_op.start_indices
            ]
            slice_sizes = list(old_op.slice_sizes)

            with InsertionPoint(dynamic_slice_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="dynamic_slice_module")
                def decorated_func(*inputs):
                    operand = inputs[0]
                    start_indices = inputs[1:]

                    new_op = stablehlo_op(
                        operand,
                        start_indices,
                        slice_sizes=slice_sizes,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        operand_tensor = self._get_golden_tensor(old_op.operand)
                        start_indices_tensors = [
                            self._get_golden_tensor(idx) for idx in old_op.start_indices
                        ]
                        golden_output = op_golden_function(
                            operand_tensor,
                            start_indices_tensors,
                            slice_sizes,
                            new_op_result.type.element_type,
                        )
                        dynamic_slice_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        dynamic_slice_builder._set_golden_tensor(
                            operand, operand_tensor
                        )

                        ordered_inputs.extend([operand] + list(start_indices))
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                dynamic_slice_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return dynamic_slice_module, dynamic_slice_builder

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, permutation_attr, op_result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, permutation_attr, old_op.result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="transpose_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    permutation_attr = old_op.permutation

                    new_op = stablehlo_op(
                        in0,
                        permutation_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, permutation_attr, old_op.result.type.element_type
                        )
                        transpose_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        transpose_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                transpose_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return transpose_module, transpose_builder

    ############### stablehlo.PadOp ###############

    @tag(stablehlo.PadOp)
    def pad(
        self,
        in0: Operand,
        padding_value: Operand,
        padding: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.pad)
        element_type = in0.type.element_type

        rank = len(in0.type.shape)
        edge_low = [padding[2 * i] for i in range(rank)]
        edge_high = [padding[2 * i + 1] for i in range(rank)]
        # Currently adding PadOp to support interior padding with [0,0..0] only
        interior = [0] * rank

        edge_low_attr = DenseI64ArrayAttr.get(edge_low)
        edge_high_attr = DenseI64ArrayAttr.get(edge_high)
        interior_attr = DenseI64ArrayAttr.get(interior)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            in0,
            padding_value,
            edge_low_attr,
            edge_high_attr,
            interior_attr,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            padding_value_golden = self._get_golden_tensor(padding_value)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                padding_value_golden,
                edge_low_attr,
                edge_high_attr,
                interior_attr,
                op_result.type.element_type,
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.PadOp)
    def pad_parser(
        self,
        old_op: stablehlo.PadOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.pad_parser)
        in0 = global_dict[old_op.operand]
        padding_value = global_dict[old_op.padding_value]
        edge_low_attr = old_op.edge_padding_low
        edge_high_attr = old_op.edge_padding_high
        interior_attr = old_op.interior_padding

        new_op = stablehlo_op(
            in0,
            padding_value,
            edge_low_attr,
            edge_high_attr,
            interior_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            padding_value_golden = self._get_golden_tensor(padding_value)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                padding_value_golden,
                edge_low_attr,
                edge_high_attr,
                interior_attr,
                old_op.result.type.element_type,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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
            op_input_types = [old_op.operand.type, old_op.padding_value.type]

            with InsertionPoint(pad_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="pad_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    padding_value = inputs[1]
                    edge_low_attr = old_op.edge_padding_low
                    edge_high_attr = old_op.edge_padding_high
                    interior_attr = old_op.interior_padding

                    new_op = stablehlo_op(
                        in0,
                        padding_value,
                        edge_low_attr,
                        edge_high_attr,
                        interior_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        padding_value_golden = self._get_golden_tensor(
                            old_op.padding_value
                        )
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0,
                            padding_value_golden,
                            edge_low_attr,
                            edge_high_attr,
                            interior_attr,
                            old_op.result.type.element_type,
                        )
                        pad_builder._set_golden_tensor(new_op_result, golden_output)
                        pad_builder._set_golden_tensor(in0, input0)
                        pad_builder._set_golden_tensor(
                            padding_value, padding_value_golden
                        )
                        ordered_inputs.append(in0)
                        ordered_inputs.append(padding_value)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                pad_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return pad_module, pad_builder

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, result.shape, op_result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, shape_attr, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="reshape_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    shape_attr = old_op.result.type.shape
                    result = old_op.result.type

                    new_op = stablehlo_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, shape_attr, result.element_type
                        )
                        reshape_builder._set_golden_tensor(new_op_result, golden_output)
                        reshape_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                reshape_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return reshape_module, reshape_builder

    ############### stablehlo.MaxOp ###############

    @tag(stablehlo.MaxOp)
    def max(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.max)

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
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, input1, mlir_output_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.MaxOp)
    def max_parser(
        self,
        old_op: stablehlo.MaxOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.max_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.MaxOp)
    def max_split(
        self,
        old_op: stablehlo.MaxOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.max_split)

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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="max_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
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

    ############### stablehlo.MinOp ###############

    @tag(stablehlo.MinOp)
    def min(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.min)

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
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, input1, mlir_output_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.MinOp)
    def min_parser(
        self,
        old_op: stablehlo.MinOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.min_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.MinOp)
    def min_split(
        self,
        old_op: stablehlo.MinOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.min_split)

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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="min_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
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

    ############### stablehlo.MulOp ###############

    @tag(stablehlo.MulOp)
    def mul(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.mul)

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
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, input1, mlir_output_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.MulOp)
    def mul_parser(
        self,
        old_op: stablehlo.MulOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.mul_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]

        new_op = stablehlo_op(
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.MulOp)
    def mul_split(
        self,
        old_op: stablehlo.MulOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.mul_split)

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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="mul_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
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

    ############### stablehlo.SubtractOp ###############

    @tag(stablehlo.SubtractOp)
    def subtract(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.subtract)

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
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, input1, mlir_output_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="subtract_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
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

    ############### stablehlo.PowOp ###############

    @tag(stablehlo.PowOp)
    def pow(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.pow)

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
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, input1, mlir_output_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="pow_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op_result.type.element_type
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

    ############### stablehlo.ShiftRightLogicalOp ###############

    @tag(stablehlo.ShiftRightLogicalOp)
    def shift_right_logical(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.shift_right_logical)

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
            in1,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            input1 = self._get_golden_tensor(in1)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(input0, input1, mlir_output_type)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, input1, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="shift_right_logical_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]

                    new_op = stablehlo_op(lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, new_op.result.type.element_type
                        )
                        srl_builder._set_golden_tensor(new_op_result, golden_output)
                        srl_builder._set_golden_tensor(lhs, input0)
                        srl_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                srl_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return srl_module, srl_builder

    ############### stablehlo.ReverseOp ###############

    @tag(stablehlo.ReverseOp)
    def reverse(
        self,
        in0: Operand,
        dimensions: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.reverse)

        dimensions_attr = DenseI64ArrayAttr.get(dimensions, context=self._ctx)

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
            dimensions_attr,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, dimensions_attr, mlir_output_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0, dimensions_attr, old_op.result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="reverse_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]

                    new_op = stablehlo_op(
                        in0,
                        old_op.dimensions,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.operand)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0, old_op.dimensions, old_op.result.type.element_type
                        )
                        reverse_builder._set_golden_tensor(new_op_result, golden_output)
                        reverse_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                reverse_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return reverse_module, reverse_builder

    ############### stablehlo.SelectOp ###############

    @tag(stablehlo.SelectOp)
    def select(
        self,
        pred: Operand,
        on_true: Operand,
        on_false: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.select)

        if output_type is None:
            mlir_output_type = self.get_type(on_true)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            pred,
            on_true,
            on_false,
            loc=loc,
        )
        op_result = op.result

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
                pred_g, true_g, false_g, mlir_output_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            pred_g = self._get_golden_tensor(pred)
            true_g = self._get_golden_tensor(on_true)
            false_g = self._get_golden_tensor(on_false)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                pred_g, true_g, false_g, old_op.result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="select_module")
                def decorated_func(*inputs):
                    pred, on_true, on_false = inputs[0], inputs[1], inputs[2]

                    new_op = stablehlo_op(pred, on_true, on_false, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        pred_g = self._get_golden_tensor(old_op.pred)
                        true_g = self._get_golden_tensor(old_op.on_true)
                        false_g = self._get_golden_tensor(old_op.on_false)
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            pred_g, true_g, false_g, old_op.result.type.element_type
                        )
                        sel_builder._set_golden_tensor(new_op_result, golden_output)
                        sel_builder._set_golden_tensor(pred, pred_g)
                        sel_builder._set_golden_tensor(on_true, true_g)
                        sel_builder._set_golden_tensor(on_false, false_g)
                        ordered_inputs.extend([pred, on_true, on_false])
                        ordered_outputs.append(new_op_result)

                    return new_op

        return sel_module, sel_builder

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
        op_result = op.result

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
                min_golden, operand_golden, max_golden, op_result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            min_golden = self._get_golden_tensor(min_val)
            operand_golden = self._get_golden_tensor(operand)
            max_golden = self._get_golden_tensor(max_val)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                min_golden, operand_golden, max_golden, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="clamp_module")
                def decorated_func(*inputs):
                    min_val = inputs[0]
                    operand = inputs[1]
                    max_val = inputs[2]

                    new_op = stablehlo_op(
                        min_val, operand, max_val, loc=old_op.location
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        min_golden = self._get_golden_tensor(old_op.min)
                        operand_golden = self._get_golden_tensor(old_op.operand)
                        max_golden = self._get_golden_tensor(old_op.max)
                        golden_output = op_golden_function(
                            min_golden,
                            operand_golden,
                            max_golden,
                            new_op_result.type.element_type,
                        )
                        clamp_builder._set_golden_tensor(new_op_result, golden_output)
                        clamp_builder._set_golden_tensor(min_val, min_golden)
                        clamp_builder._set_golden_tensor(operand, operand_golden)
                        clamp_builder._set_golden_tensor(max_val, max_golden)
                        ordered_inputs.extend([min_val, operand, max_val])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                clamp_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return clamp_module, clamp_builder

    ################ stablehlo.ConcatenateOp ###############

    @tag(stablehlo.ConcatenateOp)
    def concatenate(
        self,
        inputs: List[Operand],
        dim: int,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.concatenate)

        dim_attr = IntegerAttr.get(IntegerType.get_signless(64), dim)

        op = stablehlo_op(
            inputs,
            dim_attr,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            input_goldens = tuple(self._get_golden_tensor(inp) for inp in inputs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input_goldens, dim_attr, op_result.type.element_type
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.ConcatenateOp)
    def concatenate_parser(
        self,
        old_op: stablehlo.ConcatenateOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.concatenate_parser)
        inputs = [global_dict[inp] for inp in old_op.inputs]
        dim_attr = old_op.dimension

        new_op = stablehlo_op(
            inputs,
            dim_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input_goldens = tuple(self._get_golden_tensor(inp) for inp in inputs)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input_goldens, dim_attr, new_op_result.type.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.ConcatenateOp)
    def concatenate_split(
        self,
        old_op: stablehlo.ConcatenateOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.concatenate_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)

        with old_context, old_loc:
            concatenate_module = Module.create()
            concatenate_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [inp.type for inp in old_op.inputs]

            with InsertionPoint(concatenate_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="concatenate_module")
                def decorated_func(*inputs):
                    dim_attr = old_op.dimension

                    new_op = stablehlo_op(inputs, dim_attr, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        input_goldens = tuple(
                            self._get_golden_tensor(inp) for inp in old_op.inputs
                        )
                        golden_output = op_golden_function(
                            input_goldens, dim_attr, new_op_result.type.element_type
                        )
                        concatenate_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        for i, inp in enumerate(inputs):
                            concatenate_builder._set_golden_tensor(
                                inp, input_goldens[i]
                            )
                        ordered_inputs.extend(inputs)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                concatenate_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return concatenate_module, concatenate_builder

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
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(stablehlo_op)
            mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
            golden_output = op_golden_function(value_attr, mesh_shape_attr)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        new_op_result = new_op.result

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(stablehlo_op)
            mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
            golden_output = op_golden_function(value_attr, mesh_shape_attr)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(name="constant_module")
                def decorated_func():
                    new_op = stablehlo_op(value_attr, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        mesh_shape_attr = DenseI32ArrayAttr.get(
                            constant_builder._mesh_shape
                        )
                        golden_output = op_golden_function(value_attr, mesh_shape_attr)
                        constant_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                constant_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return constant_module, constant_builder

    ################ stablehlo.IotaOp ###############

    @tag(stablehlo.IotaOp)
    def iota(
        self,
        output: Operand,
        iota_dimension: int,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.iota)

        iota_dimension_attr = IntegerAttr.get(
            IntegerType.get_signless(64, self._ctx), iota_dimension
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            output,
            iota_dimension_attr,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(stablehlo_op)
            mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
            golden_output = op_golden_function(
                iota_dimension_attr,
                DenseI64ArrayAttr.get(list(op.result.type.shape)),
                mesh_shape_attr,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.IotaOp)
    def iota_parser(
        self,
        old_op: stablehlo.IotaOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.iota_parser)

        result = old_op.result.type
        iota_dimension_attr = old_op.iota_dimension

        new_op = stablehlo_op(
            result,
            iota_dimension_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(stablehlo_op)
            mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
            golden_output = op_golden_function(
                iota_dimension_attr,
                DenseI64ArrayAttr.get(list(new_op_result.type.shape)),
                mesh_shape_attr,
                new_op_result.type.element_type,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.IotaOp)
    def iota_split(
        self,
        old_op: stablehlo.IotaOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.iota_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        iota_dimension_attr = old_op.iota_dimension
        result = old_op.result.type

        with old_context, old_loc:
            iota_module = Module.create()
            iota_builder = StableHLOBuilder(old_context, old_loc)

            with InsertionPoint(iota_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(name="iota_module")
                def decorated_func():
                    new_op = stablehlo_op(
                        result,
                        iota_dimension_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        mesh_shape_attr = DenseI32ArrayAttr.get(
                            iota_builder._mesh_shape
                        )
                        golden_output = op_golden_function(
                            iota_dimension_attr,
                            DenseI64ArrayAttr.get(list(new_op_result.type.shape)),
                            mesh_shape_attr,
                            new_op_result.type.element_type,
                        )
                        iota_builder._set_golden_tensor(new_op_result, golden_output)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                iota_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return iota_module, iota_builder

    ################ stablehlo.DynamicIotaOp ###############

    @tag(stablehlo.DynamicIotaOp)
    def dynamic_iota(
        self,
        output: Operand,
        output_shape: Operand,
        iota_dimension: int,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.dynamic_iota)

        iota_dimension_attr = IntegerAttr.get(
            IntegerType.get_signless(64, self._ctx), iota_dimension
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = stablehlo_op(
            output,
            output_shape,
            iota_dimension_attr,
            loc=loc,
        )
        op_result = op.result

        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(stablehlo_op)
            output_shape_golden = self._get_golden_tensor(output_shape)
            mesh_shape_attr = DenseI64ArrayAttr.get(self._mesh_shape)
            golden_output = op_golden_function(
                output_shape_golden,
                iota_dimension_attr,
                mesh_shape_attr,
                op.result.type.element_type,
            )
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.DynamicIotaOp)
    def dynamic_iota_parser(
        self,
        old_op: stablehlo.DynamicIotaOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.dynamic_iota_parser)

        result = old_op.result.type
        iota_dimension_attr = old_op.iota_dimension
        output_shape = global_dict[old_op.output_shape]

        new_op = stablehlo_op(
            result,
            output_shape,
            iota_dimension_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            op_golden_function = get_golden_function(stablehlo_op)
            output_shape_golden = self._get_golden_tensor(output_shape)
            mesh_shape_attr = DenseI64ArrayAttr.get(self._mesh_shape)
            golden_output = op_golden_function(
                output_shape_golden,
                iota_dimension_attr,
                mesh_shape_attr,
                new_op_result.type.element_type,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.DynamicIotaOp)
    def dynamic_iota_split(
        self,
        old_op: stablehlo.DynamicIotaOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.dynamic_iota_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        iota_dimension_attr = old_op.iota_dimension
        result = old_op.result.type
        old_output_shape = old_op.output_shape

        with old_context, old_loc:
            dynamic_iota_module = Module.create()
            dynamic_iota_builder = StableHLOBuilder(old_context, old_loc)

            with InsertionPoint(dynamic_iota_module.body):

                ordered_inputs = []
                ordered_outputs = []

                op_input_types = [old_output_shape.type]

                @func.func(*op_input_types, name="dynamic_iota_module")
                def decorated_func(output_shape):
                    ordered_inputs.append(output_shape)

                    new_op = stablehlo_op(
                        result,
                        output_shape,
                        iota_dimension_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        output_shape_golden = dynamic_iota_builder._get_golden_tensor(
                            output_shape
                        )
                        mesh_shape_attr = DenseI64ArrayAttr.get(
                            dynamic_iota_builder._mesh_shape
                        )
                        golden_output = op_golden_function(
                            output_shape_golden,
                            iota_dimension_attr,
                            mesh_shape_attr,
                            new_op_result.type.element_type,
                        )
                        dynamic_iota_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                dynamic_iota_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return dynamic_iota_module, dynamic_iota_builder

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

        epsilon_attr = FloatAttr.get(F32Type.get(self._ctx), epsilon)
        feature_index_attr = IntegerAttr.get(
            IntegerType.get_signless(64, self._ctx), feature_index
        )

        op = stablehlo_op(
            operand,
            scale,
            mean,
            variance,
            grad_output,
            epsilon=epsilon_attr,
            feature_index=feature_index_attr,
            loc=loc,
        )
        op_grad_operand = op.grad_operand
        op_grad_scale = op.grad_scale
        op_grad_offset = op.grad_offset

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
                epsilon_attr,
                feature_index_attr,
                op_grad_operand.type.element_type,
                op_grad_scale.type.element_type,
                op_grad_offset.type.element_type,
            )
            self._set_golden_tensor(op_grad_operand, grad_operand_golden)
            self._set_golden_tensor(op_grad_scale, grad_scale_golden)
            self._set_golden_tensor(op_grad_offset, grad_offset_golden)

        return op_grad_operand, op_grad_scale, op_grad_offset

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
        epsilon_attr = old_op.epsilon
        feature_index_attr = old_op.feature_index

        new_op = stablehlo_op(
            operand,
            scale,
            mean,
            variance,
            grad_output,
            epsilon=epsilon_attr,
            feature_index=feature_index_attr,
            loc=old_op.location,
        )
        new_op_grad_operand = new_op.grad_operand
        new_op_grad_scale = new_op.grad_scale
        new_op_grad_offset = new_op.grad_offset

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
                epsilon_attr,
                feature_index_attr,
                new_op_grad_operand.type.element_type,
                new_op_grad_scale.type.element_type,
                new_op_grad_offset.type.element_type,
            )
            self._set_golden_tensor(new_op_grad_operand, grad_operand_golden)
            self._set_golden_tensor(new_op_grad_scale, grad_scale_golden)
            self._set_golden_tensor(new_op_grad_offset, grad_offset_golden)

        op_map_dictionary = {}
        op_map_dictionary[old_op.grad_operand] = new_op_grad_operand
        op_map_dictionary[old_op.grad_scale] = new_op_grad_scale
        op_map_dictionary[old_op.grad_offset] = new_op_grad_offset
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

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="batch_norm_grad_module")
                def decorated_func(*inputs):
                    operand = inputs[0]
                    scale = inputs[1]
                    mean = inputs[2]
                    variance = inputs[3]
                    grad_output = inputs[4]
                    epsilon_attr = old_op.epsilon
                    feature_index_attr = old_op.feature_index

                    new_op = stablehlo_op(
                        operand,
                        scale,
                        mean,
                        variance,
                        grad_output,
                        epsilon=epsilon_attr,
                        feature_index=feature_index_attr,
                        loc=old_op.location,
                    )
                    new_op_grad_operand = new_op.grad_operand
                    new_op_grad_scale = new_op.grad_scale
                    new_op_grad_offset = new_op.grad_offset

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
                            epsilon_attr,
                            feature_index_attr,
                            new_op_grad_operand.type.element_type,
                            new_op_grad_scale.type.element_type,
                            new_op_grad_offset.type.element_type,
                        )
                        batch_norm_grad_builder._set_golden_tensor(
                            new_op_grad_operand, grad_operand_golden
                        )
                        batch_norm_grad_builder._set_golden_tensor(
                            new_op_grad_scale, grad_scale_golden
                        )
                        batch_norm_grad_builder._set_golden_tensor(
                            new_op_grad_offset, grad_offset_golden
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
                        ordered_inputs.extend(
                            [operand, scale, mean, variance, grad_output]
                        )
                        ordered_outputs.extend(
                            [new_op_grad_operand, new_op_grad_scale, new_op_grad_offset]
                        )

                    return new_op

                new_func_op = decorated_func.func_op
                batch_norm_grad_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

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

        epsilon_attr = FloatAttr.get(F32Type.get(self._ctx), epsilon)
        feature_index_attr = IntegerAttr.get(
            IntegerType.get_signless(64, self._ctx), feature_index
        )

        op = stablehlo_op(
            operand,
            scale,
            offset,
            epsilon=epsilon_attr,
            feature_index=feature_index_attr,
            loc=loc,
        )
        op_output = op.output
        op_batch_mean = op.batch_mean
        op_batch_var = op.batch_var

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
                epsilon_attr,
                feature_index_attr,
                op_output.type.element_type,
                op_batch_mean.type.element_type,
                op_batch_var.type.element_type,
            )
            self._set_golden_tensor(op_output, output_golden)
            self._set_golden_tensor(op_batch_mean, batch_mean_golden)
            self._set_golden_tensor(op_batch_var, batch_var_golden)

        return op_output, op_batch_mean, op_batch_var

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
        epsilon_attr = old_op.epsilon
        feature_index_attr = old_op.feature_index

        new_op = stablehlo_op(
            operand,
            scale,
            offset,
            epsilon=epsilon_attr,
            feature_index=feature_index_attr,
            loc=old_op.location,
        )
        new_op_output = new_op.output
        new_op_batch_mean = new_op.batch_mean
        new_op_batch_var = new_op.batch_var

        if not self._disable_golden_check:
            operand_golden = self._get_golden_tensor(operand)
            scale_golden = self._get_golden_tensor(scale)
            offset_golden = self._get_golden_tensor(offset)
            op_golden_function = get_golden_function(stablehlo_op)
            output_golden, batch_mean_golden, batch_var_golden = op_golden_function(
                operand_golden,
                scale_golden,
                offset_golden,
                epsilon_attr,
                feature_index_attr,
                new_op_output.type.element_type,
                new_op_batch_mean.type.element_type,
                new_op_batch_var.type.element_type,
            )
            self._set_golden_tensor(new_op_output, output_golden)
            self._set_golden_tensor(new_op_batch_mean, batch_mean_golden)
            self._set_golden_tensor(new_op_batch_var, batch_var_golden)

        op_map_dictionary = {}
        op_map_dictionary[old_op.output] = new_op_output
        op_map_dictionary[old_op.batch_mean] = new_op_batch_mean
        op_map_dictionary[old_op.batch_var] = new_op_batch_var
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

        with old_context, old_loc:
            batch_norm_training_module = Module.create()
            batch_norm_training_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
                old_op.scale.type,
                old_op.offset.type,
            ]

            with InsertionPoint(batch_norm_training_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="batch_norm_training_module")
                def decorated_func(*inputs):
                    operand = inputs[0]
                    scale = inputs[1]
                    offset = inputs[2]
                    epsilon_attr = old_op.epsilon
                    feature_index_attr = old_op.feature_index

                    new_op = stablehlo_op(
                        operand,
                        scale,
                        offset,
                        epsilon=epsilon_attr,
                        feature_index=feature_index_attr,
                        loc=old_op.location,
                    )
                    new_op_output = new_op.output
                    new_op_batch_mean = new_op.batch_mean
                    new_op_batch_var = new_op.batch_var

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
                            epsilon_attr,
                            feature_index_attr,
                            new_op_output.type.element_type,
                            new_op_batch_mean.type.element_type,
                            new_op_batch_var.type.element_type,
                        )
                        batch_norm_training_builder._set_golden_tensor(
                            new_op_output, output_golden
                        )
                        batch_norm_training_builder._set_golden_tensor(
                            new_op_batch_mean, batch_mean_golden
                        )
                        batch_norm_training_builder._set_golden_tensor(
                            new_op_batch_var, batch_var_golden
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
                        ordered_inputs.extend([operand, scale, offset])
                        ordered_outputs.extend(
                            [new_op_output, new_op_batch_mean, new_op_batch_var]
                        )

                    return new_op

                new_func_op = decorated_func.func_op
                batch_norm_training_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return batch_norm_training_module, batch_norm_training_builder

    ################ stablehlo.BatchNormInferenceOp ###############

    @tag(stablehlo.BatchNormInferenceOp)
    def batch_norm_inference(
        self,
        operand: Operand,
        scale: Operand,
        offset: Operand,
        mean: Operand,
        variance: Operand,
        epsilon: float,
        feature_index: int,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(
            StableHLOBuilder.batch_norm_inference
        )
        epsilon_attr = FloatAttr.get(F32Type.get(self._ctx), epsilon)
        feature_index_attr = IntegerAttr.get(
            IntegerType.get_signless(64, self._ctx), feature_index
        )
        op = stablehlo_op(
            operand,
            scale,
            offset,
            mean,
            variance,
            epsilon=epsilon_attr,
            feature_index=feature_index_attr,
            loc=loc,
        )
        op_output = op.result
        if sharding_attr is not None:
            op.operation.attributes["sdy.sharding"] = sharding_attr
        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)
        if not self._disable_golden_check:
            operand_golden = self._get_golden_tensor(operand)
            scale_golden = self._get_golden_tensor(scale)
            offset_golden = self._get_golden_tensor(offset)
            mean_golden = self._get_golden_tensor(mean)
            variance_golden = self._get_golden_tensor(variance)
            op_golden_function = get_golden_function(stablehlo_op)
            output_golden = op_golden_function(
                operand_golden,
                scale_golden,
                offset_golden,
                mean_golden,
                variance_golden,
                epsilon_attr,
                feature_index_attr,
                op_output.type.element_type,
            )
            self._set_golden_tensor(op_output, output_golden)
        return op_output

    @parse(stablehlo.BatchNormInferenceOp)
    def batch_norm_inference_parser(
        self,
        old_op: stablehlo.BatchNormInferenceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.batch_norm_inference_parser
        )
        operand = global_dict[old_op.operand]
        scale = global_dict[old_op.scale]
        offset = global_dict[old_op.offset]
        mean = global_dict[old_op.mean]
        variance = global_dict[old_op.variance]
        epsilon_attr = old_op.epsilon
        feature_index_attr = old_op.feature_index
        new_op = stablehlo_op(
            operand,
            scale,
            offset,
            mean,
            variance,
            epsilon=epsilon_attr,
            feature_index=feature_index_attr,
            loc=old_op.location,
        )
        new_op_output = new_op.result
        if not self._disable_golden_check:
            operand_golden = self._get_golden_tensor(operand)
            scale_golden = self._get_golden_tensor(scale)
            offset_golden = self._get_golden_tensor(offset)
            mean_golden = self._get_golden_tensor(mean)
            variance_golden = self._get_golden_tensor(variance)
            op_golden_function = get_golden_function(stablehlo_op)
            output_golden = op_golden_function(
                operand_golden,
                scale_golden,
                offset_golden,
                mean_golden,
                variance_golden,
                epsilon_attr,
                feature_index_attr,
                new_op_output.type.element_type,
            )
            self._set_golden_tensor(new_op_output, output_golden)
        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_output
        return new_op, op_map_dictionary

    @split(stablehlo.BatchNormInferenceOp)
    def batch_norm_inference_split(
        self,
        old_op: stablehlo.BatchNormInferenceOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(
            StableHLOBuilder.batch_norm_inference_split
        )
        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            batch_norm_inference_module = Module.create()
            batch_norm_inference_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.operand.type,
                old_op.scale.type,
                old_op.offset.type,
                old_op.mean.type,
                old_op.variance.type,
            ]
            with InsertionPoint(batch_norm_inference_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="batch_norm_inference_module")
                def decorated_func(*inputs):
                    operand = inputs[0]
                    scale = inputs[1]
                    offset = inputs[2]
                    mean = inputs[3]
                    variance = inputs[4]
                    epsilon_attr = old_op.epsilon
                    feature_index_attr = old_op.feature_index
                    new_op = stablehlo_op(
                        operand,
                        scale,
                        offset,
                        mean,
                        variance,
                        epsilon=epsilon_attr,
                        feature_index=feature_index_attr,
                        loc=old_op.location,
                    )
                    new_op_output = new_op.result
                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(stablehlo_op)
                        operand_golden = self._get_golden_tensor(old_op.operand)
                        scale_golden = self._get_golden_tensor(old_op.scale)
                        offset_golden = self._get_golden_tensor(old_op.offset)
                        mean_golden = self._get_golden_tensor(old_op.mean)
                        variance_golden = self._get_golden_tensor(old_op.variance)
                        output_golden = op_golden_function(
                            operand_golden,
                            scale_golden,
                            offset_golden,
                            mean_golden,
                            variance_golden,
                            epsilon_attr,
                            feature_index_attr,
                            new_op_output.type.element_type,
                        )
                        batch_norm_inference_builder._set_golden_tensor(
                            new_op_output, output_golden
                        )
                        batch_norm_inference_builder._set_golden_tensor(
                            operand, operand_golden
                        )
                        batch_norm_inference_builder._set_golden_tensor(
                            scale, scale_golden
                        )
                        batch_norm_inference_builder._set_golden_tensor(
                            offset, offset_golden
                        )
                        batch_norm_inference_builder._set_golden_tensor(
                            mean, mean_golden
                        )
                        batch_norm_inference_builder._set_golden_tensor(
                            variance, variance_golden
                        )
                        ordered_inputs.extend([operand, scale, offset, mean, variance])
                        ordered_outputs.append(new_op_output)
                    return new_op

                new_func_op = decorated_func.func_op
                batch_norm_inference_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]
        return batch_norm_inference_module, batch_norm_inference_builder

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

    ############### stablehlo.ReduceWindowOp ###############

    @tag(stablehlo.ReduceWindowOp)
    def reduce_window(
        self,
        in0: Operand,
        init_value: Union[Operand, float, int],
        window_dimensions: Sequence[int],
        window_strides: Optional[Sequence[int]] = None,
        base_dilations: Optional[Sequence[int]] = None,
        window_dilations: Optional[Sequence[int]] = None,
        padding: Optional[Sequence[Sequence[int]]] = None,
        body: str = "add",
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.reduce_window)

        with self._ctx, self._loc:
            id = self._get_next_global_id()
            loc = self._get_loc_of_extra_file_callee(id=id)

            input_type = RankedTensorType(in0.type)
            element_type = input_type.element_type
            input_shape = list(input_type.shape)
            rank = len(input_shape)

            window_dimensions = list(window_dimensions)
            window_strides = (
                list(window_strides) if window_strides is not None else [1] * rank
            )
            base_dilations = (
                list(base_dilations) if base_dilations is not None else [1] * rank
            )
            window_dilations = (
                list(window_dilations) if window_dilations is not None else [1] * rank
            )
            if padding is None:
                padding_2d = [[0, 0] for _ in range(rank)]
            else:
                padding_2d = [list(p) for p in padding]

            output_shape = []
            for i in range(rank):
                dilated_input = (
                    (input_shape[i] - 1) * base_dilations[i] + 1
                    if input_shape[i] > 0
                    else 0
                )
                padded_input = padding_2d[i][0] + dilated_input + padding_2d[i][1]
                dilated_window = (
                    (window_dimensions[i] - 1) * window_dilations[i] + 1
                    if window_dimensions[i] > 0
                    else 0
                )
                if padded_input == 0 or dilated_window > padded_input:
                    output_dim = 0
                else:
                    output_dim = (
                        (padded_input - dilated_window) // window_strides[i]
                    ) + 1
                output_shape.append(output_dim)

            output_type = RankedTensorType.get(output_shape, element_type)

            if isinstance(init_value, (int, float)):
                if isinstance(init_value, float):
                    init_attr = DenseElementsAttr.get_splat(
                        RankedTensorType.get([], element_type),
                        FloatAttr.get(element_type, init_value),
                    )
                else:
                    init_attr = DenseElementsAttr.get_splat(
                        RankedTensorType.get([], element_type),
                        IntegerAttr.get(element_type, init_value),
                    )
                init_value_op = stablehlo.ConstantOp(init_attr, loc=loc).result
                if not self._disable_golden_check:
                    init_golden_function = get_golden_function(stablehlo.ConstantOp)
                    mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
                    init_golden = init_golden_function(init_attr, mesh_shape_attr)
                    self._set_golden_tensor(init_value_op, init_golden)
            else:
                init_value_op = init_value

            reduce_window_op = stablehlo_op(
                [output_type],
                inputs=[in0],
                init_values=[init_value_op],
                window_dimensions=window_dimensions,
                window_strides=window_strides,
                base_dilations=base_dilations,
                window_dilations=window_dilations,
                padding=padding_2d,
                loc=loc,
            )

            reduction_type = RankedTensorType.get([], element_type)
            block = Block.create_at_start(
                reduce_window_op.regions[0], [reduction_type, reduction_type]
            )

            with InsertionPoint(block):
                if body == "add":
                    reduction_result = stablehlo.AddOp(
                        block.arguments[0], block.arguments[1], loc=loc
                    ).result
                elif body == "max":
                    reduction_result = stablehlo.MaxOp(
                        block.arguments[0], block.arguments[1], loc=loc
                    ).result
                else:
                    raise ValueError(
                        f"Unsupported reduction body: {body}. "
                        "Supported options: 'add', 'max'"
                    )
                stablehlo.ReturnOp([reduction_result], loc=loc)

            result = reduce_window_op.result

            if not self._disable_golden_check:
                input_golden = self._get_golden_tensor(in0)
                init_golden = self._get_golden_tensor(init_value_op)
                op_golden_function = get_golden_function(stablehlo_op)
                golden_output = op_golden_function(
                    input_golden,
                    init_golden,
                    reduce_window_op.window_dimensions,
                    reduce_window_op.window_strides,
                    reduce_window_op.base_dilations,
                    reduce_window_op.window_dilations,
                    reduce_window_op.padding,
                    result.type,
                    body,
                )
                self._set_golden_tensor(result, golden_output)

            return result

    @parse(stablehlo.ReduceWindowOp)
    def reduce_window_parser(
        self,
        old_op: stablehlo.ReduceWindowOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(
            StableHLOBuilder.reduce_window_parser
        )

        input_operand = global_dict[old_op.inputs[0]]
        init_value_operand = global_dict[old_op.init_values[0]]

        new_op = stablehlo_op(
            [old_op.result.type],
            inputs=[input_operand],
            init_values=[init_value_operand],
            window_dimensions=old_op.window_dimensions,
            window_strides=old_op.window_strides,
            base_dilations=old_op.base_dilations,
            window_dilations=old_op.window_dilations,
            padding=old_op.padding,
            loc=old_op.location,
        )

        old_region = old_op.body
        new_region = new_op.regions[0]
        element_type = RankedTensorType(old_op.inputs[0].type).element_type
        reduction_type = RankedTensorType.get([], element_type)
        block = Block.create_at_start(new_region, [reduction_type, reduction_type])

        body = "add"
        for op in old_region.blocks[0].operations:
            if isinstance(op, stablehlo.AddOp):
                body = "add"
                break
            elif isinstance(op, stablehlo.MaxOp):
                body = "max"
                break

        with InsertionPoint(block):
            if body == "add":
                reduction_result = stablehlo.AddOp(
                    block.arguments[0], block.arguments[1], loc=old_op.location
                ).result
            elif body == "max":
                reduction_result = stablehlo.MaxOp(
                    block.arguments[0], block.arguments[1], loc=old_op.location
                ).result
            stablehlo.ReturnOp([reduction_result], loc=old_op.location)

        new_op_result = new_op.result

        if not self._disable_golden_check:
            input_golden = self._get_golden_tensor(input_operand)
            init_golden = self._get_golden_tensor(init_value_operand)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input_golden,
                init_golden,
                new_op.window_dimensions,
                new_op.window_strides,
                new_op.base_dilations,
                new_op.window_dilations,
                new_op.padding,
                new_op_result.type,
                body,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.ReduceWindowOp)
    def reduce_window_split(
        self,
        old_op: stablehlo.ReduceWindowOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.reduce_window_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)

        body = "add"
        for op in old_op.body.blocks[0].operations:
            if isinstance(op, stablehlo.AddOp):
                body = "add"
                break
            elif isinstance(op, stablehlo.MaxOp):
                body = "max"
                break

        with old_context, old_loc:
            reduce_window_module = Module.create()
            reduce_window_builder = StableHLOBuilder(old_context, old_loc)
            op_input_types = [
                old_op.inputs[0].type,
                old_op.init_values[0].type,
            ]

            with InsertionPoint(reduce_window_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="reduce_window_module")
                def decorated_func(*inputs):
                    input_operand = inputs[0]
                    init_value_operand = inputs[1]

                    new_op = stablehlo_op(
                        [old_op.result.type],
                        inputs=[input_operand],
                        init_values=[init_value_operand],
                        window_dimensions=old_op.window_dimensions,
                        window_strides=old_op.window_strides,
                        base_dilations=old_op.base_dilations,
                        window_dilations=old_op.window_dilations,
                        padding=old_op.padding,
                        loc=old_op.location,
                    )

                    element_type = RankedTensorType(old_op.inputs[0].type).element_type
                    reduction_type = RankedTensorType.get([], element_type)
                    block = Block.create_at_start(
                        new_op.regions[0], [reduction_type, reduction_type]
                    )

                    with InsertionPoint(block):
                        if body == "add":
                            reduction_result = stablehlo.AddOp(
                                block.arguments[0],
                                block.arguments[1],
                                loc=old_op.location,
                            ).result
                        elif body == "max":
                            reduction_result = stablehlo.MaxOp(
                                block.arguments[0],
                                block.arguments[1],
                                loc=old_op.location,
                            ).result
                        stablehlo.ReturnOp([reduction_result], loc=old_op.location)

                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input_golden = self._get_golden_tensor(old_op.inputs[0])
                        init_golden = self._get_golden_tensor(old_op.init_values[0])
                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input_golden,
                            init_golden,
                            new_op.window_dimensions,
                            new_op.window_strides,
                            new_op.base_dilations,
                            new_op.window_dilations,
                            new_op.padding,
                            new_op_result.type,
                            body,
                        )
                        reduce_window_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        reduce_window_builder._set_golden_tensor(
                            input_operand, input_golden
                        )
                        reduce_window_builder._set_golden_tensor(
                            init_value_operand, init_golden
                        )
                        ordered_inputs.extend([input_operand, init_value_operand])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                reduce_window_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return reduce_window_module, reduce_window_builder

    ############### stablehlo.ConvolutionOp ###############

    @tag(stablehlo.ConvolutionOp)
    def convolution(
        self,
        in0: Operand,
        weight: Operand,
        window_strides: List[int],
        padding: List[int],
        lhs_dilation: List[int],
        rhs_dilation: List[int],
        input_batch_dimension: int,
        input_feature_dimension: int,
        input_spatial_dimensions: List[int],
        kernel_output_feature_dimension: int,
        kernel_input_feature_dimension: int,
        kernel_spatial_dimensions: List[int],
        output_batch_dimension: int,
        output_feature_dimension: int,
        output_spatial_dimensions: List[int],
        feature_group_count: int,
        batch_group_count: int,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        stablehlo_op = self.get_opview_from_method(StableHLOBuilder.convolution)

        # Create StableHLO ConvDimensionNumbers
        dimension_numbers = stablehlo.ConvDimensionNumbers.get(
            input_batch_dimension=input_batch_dimension,
            input_feature_dimension=input_feature_dimension,
            input_spatial_dimensions=input_spatial_dimensions,
            kernel_input_feature_dimension=kernel_input_feature_dimension,
            kernel_output_feature_dimension=kernel_output_feature_dimension,
            kernel_spatial_dimensions=kernel_spatial_dimensions,
            output_batch_dimension=output_batch_dimension,
            output_feature_dimension=output_feature_dimension,
            output_spatial_dimensions=output_spatial_dimensions,
            ctx=self._ctx,
        )

        window_strides_attr = DenseI64ArrayAttr.get(window_strides)
        num_spatial_dims = len(input_spatial_dimensions)
        if len(padding) == num_spatial_dims * 2:
            padding_2d = [
                [padding[i * 2], padding[i * 2 + 1]] for i in range(num_spatial_dims)
            ]
        elif len(padding) == num_spatial_dims:
            padding_2d = [[p, p] for p in padding]
        else:
            padding_2d = [[0, 0]] * num_spatial_dims
        padding_attr = DenseElementsAttr.get(
            torch.tensor(padding_2d, dtype=torch.int64).numpy()
        )
        lhs_dilation_attr = DenseI64ArrayAttr.get(lhs_dilation)
        rhs_dilation_attr = DenseI64ArrayAttr.get(rhs_dilation)
        window_reversal_attr = DenseBoolArrayAttr.get(
            [False] * len(input_spatial_dimensions)
        )

        # Get golden tensors
        input0 = self._get_golden_tensor(in0)
        weight0 = self._get_golden_tensor(weight)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        # Compute golden output
        op_golden_function = get_golden_function(stablehlo_op)
        golden_output = op_golden_function(
            input0,
            weight0,
            window_strides_attr,
            padding_attr,
            lhs_dilation_attr,
            rhs_dilation_attr,
            window_reversal_attr,
            dimension_numbers,
            IntegerAttr.get(IntegerType.get_signless(64), feature_group_count),
            IntegerAttr.get(IntegerType.get_signless(64), batch_group_count),
            mlir_output_type,
        )
        result_type = self._create_ranked_tensor_type(
            golden_output.shape, mlir_output_type
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        # Create the StableHLO convolution op
        op = stablehlo_op(
            result_type,
            in0,
            weight,
            dimension_numbers,
            feature_group_count,
            batch_group_count,
            window_strides=window_strides_attr,
            padding=padding_attr,
            lhs_dilation=lhs_dilation_attr,
            rhs_dilation=rhs_dilation_attr,
            window_reversal=window_reversal_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(stablehlo.ConvolutionOp)
    def convolution_parser(
        self,
        old_op: stablehlo.ConvolutionOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        stablehlo_op = self.get_opview_from_parser(StableHLOBuilder.convolution_parser)
        in0 = global_dict[old_op.lhs]
        weight = global_dict[old_op.rhs]

        result_type = old_op.result.type

        new_op = stablehlo_op(
            result_type,
            in0,
            weight,
            old_op.dimension_numbers,
            old_op.feature_group_count.value,
            old_op.batch_group_count.value,
            window_strides=old_op.window_strides,
            padding=old_op.padding,
            lhs_dilation=old_op.lhs_dilation,
            rhs_dilation=old_op.rhs_dilation,
            window_reversal=old_op.window_reversal,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(in0)
            weight0 = self._get_golden_tensor(weight)
            op_golden_function = get_golden_function(stablehlo_op)
            golden_output = op_golden_function(
                input0,
                weight0,
                old_op.window_strides,
                old_op.padding,
                old_op.lhs_dilation,
                old_op.rhs_dilation,
                old_op.window_reversal,
                old_op.dimension_numbers,
                old_op.feature_group_count,
                old_op.batch_group_count,
                result_type.element_type,
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(stablehlo.ConvolutionOp)
    def convolution_split(
        self,
        old_op: stablehlo.ConvolutionOp,
    ) -> Tuple[Module, StableHLOBuilder]:
        stablehlo_op = self.get_opview_from_split(StableHLOBuilder.convolution_split)

        old_context = old_op.context
        old_location = Location.unknown(old_context)
        with old_context, old_location:

            convolution_module = Module.create()
            convolution_builder = StableHLOBuilder(old_context, old_location)
            op_input_types = []

            conv_input = old_op.lhs
            conv_weight = old_op.rhs
            op_input_types.append(self._get_type(conv_input))
            op_input_types.append(self._get_type(conv_weight))

            with InsertionPoint(convolution_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="convolution_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    weight = inputs[1]
                    result_type = old_op.result.type

                    new_op = stablehlo_op(
                        result_type,
                        in0,
                        weight,
                        old_op.dimension_numbers,
                        old_op.feature_group_count.value,
                        old_op.batch_group_count.value,
                        window_strides=old_op.window_strides,
                        padding=old_op.padding,
                        lhs_dilation=old_op.lhs_dilation,
                        rhs_dilation=old_op.rhs_dilation,
                        window_reversal=old_op.window_reversal,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        input0 = self._get_golden_tensor(old_op.lhs)
                        weight0 = self._get_golden_tensor(old_op.rhs)

                        op_golden_function = get_golden_function(stablehlo_op)
                        golden_output = op_golden_function(
                            input0,
                            weight0,
                            old_op.window_strides,
                            old_op.padding,
                            old_op.lhs_dilation,
                            old_op.rhs_dilation,
                            old_op.window_reversal,
                            old_op.dimension_numbers,
                            old_op.feature_group_count,
                            old_op.batch_group_count,
                            result_type.element_type,
                        )
                        convolution_builder._set_golden_tensor(
                            new_op_result, golden_output
                        )
                        convolution_builder._set_golden_tensor(in0, input0)
                        convolution_builder._set_golden_tensor(weight, weight0)
                        ordered_inputs.extend([in0, weight])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                convolution_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return convolution_module, convolution_builder

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

    def _get_zero_attr(self, element_type: Type) -> Attribute:
        if IntegerType.isinstance(element_type):
            return DenseElementsAttr.get_splat(
                RankedTensorType.get([], element_type), IntegerAttr.get(element_type, 0)
            )
        else:
            return DenseElementsAttr.get_splat(
                RankedTensorType.get([], element_type), FloatAttr.get(element_type, 0.0)
            )

    def _get_one_attr(self, element_type: Type) -> Attribute:
        if IntegerType.isinstance(element_type):
            return DenseElementsAttr.get_splat(
                RankedTensorType.get([], element_type), IntegerAttr.get(element_type, 1)
            )
        else:
            return DenseElementsAttr.get_splat(
                RankedTensorType.get([], element_type), FloatAttr.get(element_type, 1.0)
            )

    def _get_neg_inf_attr(self, element_type: Type) -> Attribute:
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

    def _apply_sharding_to_shape(
        self, shape: List[int], sharding: sdy.TensorShardingAttr
    ) -> List[int]:
        new_shape = list(shape)

        for i in range(len(new_shape)):
            dim_sharding_attr = sharding.dimension_shardings[i]
            dim_sharding = sdy.DimensionShardingAttr.maybe_downcast(dim_sharding_attr)

            # Mesh encoding in sdy.TensorShardingAttr starts with @ (ex. @mesh).
            mesh_name = str(sharding.mesh_or_ref)[1:]
            mesh_dict = self._meshes[mesh_name]

            for axis_ref_attr in dim_sharding.axes:
                axis_ref = sdy.AxisRefAttr.maybe_downcast(axis_ref_attr)
                axis_name = axis_ref.name
                new_shape[i] = new_shape[i] // mesh_dict[axis_name]

        return new_shape

    def _apply_sharding_to_type(
        self, tensor_type: RankedTensorType, sharding: sdy.TensorShardingAttr
    ) -> RankedTensorType:
        new_shape = self._apply_sharding_to_shape(list(tensor_type.shape), sharding)
        return RankedTensorType.get(new_shape, tensor_type.element_type)

    def _apply_sharding_to_golden(
        self,
        golden_tensor: GoldenMapTensor,
        sharding: sdy.TensorShardingAttr,
        is_shard: bool,
    ) -> GoldenMapTensor:
        all_replicated = True
        for dim_sharding_attr in sharding.dimension_shardings:
            dim_sharding = sdy.DimensionShardingAttr.maybe_downcast(dim_sharding_attr)
            if len(dim_sharding.axes) != 0:
                all_replicated = False
                break

        # Mesh encoding in sdy.TensorShardingAttr starts with @ (ex. @mesh).
        mesh_name = str(sharding.mesh_or_ref)[1:]
        mesh_dict = self._meshes[mesh_name]
        mesh_shape = []
        for some_length in mesh_dict.values():
            mesh_shape.append(some_length)

        shard_dims = []
        if not all_replicated:
            for dim_num, dim_sharding_attr in enumerate(sharding.dimension_shardings):
                dim_sharding = sdy.DimensionShardingAttr.maybe_downcast(
                    dim_sharding_attr
                )
                if len(dim_sharding.axes) != 0:
                    shard_dims.append(dim_num)
        elif all_replicated:
            shard_dims = [None] * len(mesh_shape)

        if is_shard:
            return apply_sharding(golden_tensor, mesh_shape, shard_dims)
        return apply_unsharding(golden_tensor, mesh_shape, shard_dims)

    def _run_dummy_func(
        self,
        nested_func: Callable,
        original_inputs: List[Operand],
        in_shardings: List[sdy.TensorShardingAttr],
        out_shardings: List[sdy.TensorShardingAttr],
    ) -> List[RankedTensorType]:
        new_ctx = Context()
        new_loc = Location.unknown(new_ctx)
        dumy_output_types = []

        with new_ctx, new_loc:
            new_module = Module.create()
            dummy_builder = StableHLOBuilder(
                new_ctx, new_loc, disable_golden_check=True
            )
            dummy_builder._root_module_insertion_point = new_module.body
            dummy_builder._current_module_insertion_point = new_module.body
            new_module.body.append(dummy_builder._get_mesh())

            with InsertionPoint(new_module.body):

                dummy_input_shapes = []
                dummy_input_types = []
                for index, inp in enumerate(original_inputs):
                    inp_type = inp.type
                    new_shape = dummy_builder._apply_sharding_to_shape(
                        list(inp_type.shape), in_shardings[index]
                    )
                    dummy_input_shapes.append(new_shape)
                    dummy_input_types.append(
                        self._get_torch_dtype_from_type(inp_type.element_type)
                    )

                new_func_op = dummy_builder.func(dummy_input_shapes, dummy_input_types)(
                    nested_func
                )

                for block in new_func_op.body:
                    for op in block.operations:
                        if isinstance(op, func.ReturnOp):
                            for result in op.operands:
                                result_type = result.type
                                result_shape = list(result_type.shape)
                                result_torch_element_type = (
                                    dummy_builder._get_torch_dtype_from_type(
                                        result_type.element_type
                                    )
                                )
                                dumy_output_types.append(
                                    (result_shape, result_torch_element_type)
                                )
                            break

        output_types_in_self_ctx = []
        for shape, torch_element_type in dumy_output_types:
            new_result = self._create_ranked_tensor_type(
                shape,
                self._get_type_from_torch_dtype(torch_element_type),
            )
            output_types_in_self_ctx.append(new_result)

        return output_types_in_self_ctx

    # ----- Public Shardy Attribute Generators ----

    def mesh_axis_attr(
        self,
        name: str,
        size: int,
    ) -> sdy.MeshAxisAttr:
        return sdy.MeshAxisAttr.get(name, size)

    def mesh_attr(
        self,
        axes: List[sdy.MeshAxisAttr],
    ) -> MeshAttr:
        return sdy.MeshAttr.get(axes)

    def axis_ref_attr(
        self,
        name: str,
        sub_axis_info_attr: Optional[sdy.AxisRefAttr] = None,
    ) -> sdy.AxisRefAttr:
        return sdy.AxisRefAttr.get(name, sub_axis_info_attr)

    def dimension_sharding_attr(
        self,
        axes: List[sdy.AxisRefAttr],
        is_closed: bool,
        priority: Optional[int] = None,
    ) -> sdy.DimensionShardingAttr:
        return sdy.DimensionShardingAttr.get(axes, is_closed, priority)

    def tensor_sharding_attr(
        self,
        mesh_name: str,
        dimension_shardings: List[sdy.DimensionShardingAttr],
        replicated_axes: List[sdy.AxisRefAttr] = [],
        unreduced_axes: List[sdy.AxisRefAttr] = [],
    ) -> sdy.TensorShardingAttr:
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
        return sdy.TensorShardingPerValueAttr.get(
            shardings,
        )

    def manual_axes_attr(
        self,
        manual_axes: List[str],
    ) -> sdy.ManualAxesAttr:
        manual_axes_attr = []
        for axis in manual_axes:
            manual_axes_attr.append(StringAttr.get(axis))

        return sdy.ManualAxesAttr.get(
            manual_axes_attr,
        )

    def axes_ref_list_attr(
        self,
        axis_ref_list: List[sdy.AxisRefAttr],
    ) -> sdy.AxisRefListAttr:
        return sdy.AxisRefListAttr.get(axis_ref_list)

    def list_of_axis_ref_lists_attr(
        self, list_of_axis_ref_lists: List[sdy.AxisRefListAttr]
    ) -> sdy.ListOfAxisRefListsAttr:
        return sdy.ListOfAxisRefListsAttr.get(list_of_axis_ref_lists)

    # ----- Public Shardy Op Generators ----

    ################ sdy.MeshOp ###############

    @tag(sdy.MeshOp)
    def mesh(self, mesh_name: str, mesh_attr: sdy.MeshAttr) -> sdy.MeshOp:
        sdy_op = self.get_opview_from_method(StableHLOBuilder.mesh)
        return sdy_op(sym_name=mesh_name, mesh=mesh_attr)

    ################ sdy.ShardingConstraintOp ###############

    @tag(sdy.ShardingConstraintOp)
    def sharding_constraint(
        self,
        in0: Operand,
        tensor_sharding_attr: sdy.TensorShardingAttr,
    ) -> sdy.ShardingConstraintOp:
        sdy_op = self.get_opview_from_method(StableHLOBuilder.sharding_constraint)

        op = sdy_op(in0, tensor_sharding_attr)
        op_result = op.results[0]

        if not self._disable_golden_check:
            input_golden = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(sdy_op)
            golden_output = op_golden_function(input_golden)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    ################ sdy.ReshardOp ###############

    @tag(sdy.ReshardOp)
    def reshard(
        self,
        in0: Operand,
        tensor_sharding_attr: sdy.TensorShardingAttr,
    ) -> sdy.ReshardOp:
        sdy_op = self.get_opview_from_method(StableHLOBuilder.reshard)

        op = sdy_op(in0, tensor_sharding_attr)
        op_result = op.results[0]

        if not self._disable_golden_check:
            input_golden = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(sdy_op)
            golden_output = op_golden_function(input_golden)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    ################ sdy.ManualComputationOp ###############

    @tag(sdy.ManualComputationOp)
    def manual_computation(
        self,
        nested_func: Callable,
        original_inputs: List[Operand],
        in_shardings: List[sdy.TensorShardingAttr],
        out_shardings: List[sdy.TensorShardingAttr],
        manual_axes: List[str],
        loc: Optional[str] = None,
    ) -> sdy.ManualComputationOp:
        sdy_op = self.get_opview_from_method(StableHLOBuilder.manual_computation)

        original_input_goldens = [
            self._get_golden_tensor(inp) for inp in original_inputs
        ]
        original_input_types = [inp.type for inp in original_inputs]
        original_output_types = self._run_dummy_func(
            nested_func, original_inputs, in_shardings, out_shardings
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        new_manual_computation_op = sdy_op(
            original_output_types,
            original_inputs,
            self.tensor_sharding_per_value_attr(in_shardings),
            self.tensor_sharding_per_value_attr(out_shardings),
            self.manual_axes_attr(manual_axes),
            loc=loc,
        )
        new_manual_computation_op_results = new_manual_computation_op.results

        region = new_manual_computation_op.body
        block = Block.create_at_start(region)

        for inp_type, inp_golden, in_sharding in zip(
            original_input_types, original_input_goldens, in_shardings
        ):
            new_inp_type = self._apply_sharding_to_type(inp_type, in_sharding)
            new_arg = block.add_argument(new_inp_type, loc=Location.unknown(self._ctx))
            new_inp_golden = self._apply_sharding_to_golden(
                inp_golden, in_sharding, is_shard=True
            )
            self._set_golden_tensor(new_arg, new_inp_golden)

        output_goldens = []
        with InsertionPoint(block):
            result = nested_func(*block.arguments, self)
            outputs = result if hasattr(result, "__iter__") else [result]
            sdy_return0 = sdy.ReturnOp(outputs)

            for i, output in enumerate(outputs):
                output_golden = self._get_golden_tensor(output)
                output_goldens.append(output_golden)

        for i, (out_sharding, output_golden) in enumerate(
            zip(out_shardings, output_goldens)
        ):
            new_output_golden = self._apply_sharding_to_golden(
                output_golden, out_sharding, is_shard=False
            )
            self._set_golden_tensor(
                new_manual_computation_op_results[i], new_output_golden
            )

        return (
            new_manual_computation_op_results[0]
            if len(new_manual_computation_op_results) == 1
            else tuple(new_manual_computation_op_results)
        )

    @parse(sdy.ManualComputationOp)
    def manual_computation_parser(
        self,
        old_op: stablehlo.ManualComputationOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        sdy_op = self.get_opview_from_parser(StableHLOBuilder.manual_computation_parser)

        tensors = []
        for old_tensor in old_op.tensors:
            tensors.append(global_dict[old_tensor])
        old_op_result_types = [result.type for result in old_op.results_]

        new_op = sdy_op(
            old_op_result_types,
            tensors,
            old_op.in_shardings,
            old_op.out_shardings,
            old_op.manual_axes,
            loc=old_op.location,
        )
        new_op_results = new_op.results_

        old_block = old_op.body.blocks[0]
        new_region = new_op.body
        new_block = Block.create_at_start(new_region)
        new_in_shardings = sdy.TensorShardingPerValueAttr.maybe_downcast(
            new_op.in_shardings
        )
        new_out_shardings = sdy.TensorShardingPerValueAttr.maybe_downcast(
            new_op.out_shardings
        )

        for old_arg in old_block.arguments:
            new_arg = new_block.add_argument(
                old_arg.type, loc=Location.unknown(self._ctx)
            )
            global_dict[old_arg] = new_arg

        if not self._disable_golden_check:
            original_input_goldens = []
            for old_tensor in old_op.tensors:
                original_input_goldens.append(
                    self._get_golden_tensor(global_dict[old_tensor])
                )

            for new_arg, inp_golden, in_sharding in zip(
                new_block.arguments,
                original_input_goldens,
                new_in_shardings.shardings,
            ):
                tensor_sharding = sdy.TensorShardingAttr.maybe_downcast(in_sharding)
                new_inp_golden = self._apply_sharding_to_golden(
                    inp_golden, tensor_sharding, is_shard=True
                )
                self._set_golden_tensor(new_arg, new_inp_golden)

        local_results = []
        with InsertionPoint(new_block):
            for old_inner_op in old_block.operations:
                if isinstance(old_inner_op, sdy.ReturnOp):
                    global_result = tuple(
                        global_dict[result] for result in old_inner_op.results_
                    )
                    local_results.extend(old_inner_op.results_)
                    sdy.ReturnOp(global_result)
                else:
                    parsed_op, op_golden_dictionary = self._build_op_from_parsed_op(
                        old_inner_op, global_dict
                    )
                    global_dict.update(op_golden_dictionary)

        if not self._disable_golden_check:
            for i, (out_sharding, old_result) in enumerate(
                zip(new_out_shardings.shardings, local_results)
            ):
                tensor_sharding = sdy.TensorShardingAttr.maybe_downcast(out_sharding)
                output_golden = self._get_golden_tensor(global_dict[old_result])
                new_output_golden = self._apply_sharding_to_golden(
                    output_golden, tensor_sharding, is_shard=False
                )
                self._set_golden_tensor(new_op_results[i], new_output_golden)

        op_map_dictionary = {}
        for old_op, new_op_result in zip(old_op.results_, new_op_results):
            op_map_dictionary[old_op] = new_op_result
        return new_op, op_map_dictionary

    ################ sdy.AllGatherOp ###############

    @tag(sdy.AllGatherOp)
    def sdy_all_gather(
        self,
        in0: Operand,
        gathering_axes: sdy.ListOfAxisRefListsAttr,
        out_sharding: sdy.TensorShardingAttr,
    ) -> sdy.AllGatherOp:
        sdy_op = self.get_opview_from_method(StableHLOBuilder.sdy_all_gather)

        op = sdy_op(in0, gathering_axes, out_sharding)
        op_result = op.results[0]

        if not self._disable_golden_check:
            input_golden = self._get_golden_tensor(in0)
            op_golden_function = get_golden_function(sdy_op)
            golden_output = op_golden_function(input_golden)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

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
        ctx: Context,
        mlir_text: str,
        golden_inputs: Dict[str, List[torch.tensor]] = None,
    ) -> Tuple(Module, StableHLOBuilder):
        if golden_inputs is None:
            golden_inputs = {}

        root_module = Module.parse(mlir_text, ctx)
        loc = Location.unknown(ctx)
        with ctx, loc:
            mesh_name = "mesh"
            mesh_shape = OrderedDict([("x", 1), ("y", 1)])

            for op in root_module.body.operations:
                if not isinstance(op, sdy.MeshOp):
                    continue

                mesh_name = op.sym_name.value
                mesh_attr = sdy.MeshAttr.maybe_downcast(op.mesh)
                shape = []
                for axis_attr in mesh_attr.axes:
                    axis = sdy.MeshAxisAttr.maybe_downcast(axis_attr)
                    shape.append(axis.size)
                mesh_shape = OrderedDict(
                    x=1 if len(shape) == 1 else shape[0],
                    y=shape[0] if len(shape) == 1 else shape[1],
                )
                break

            stablehlo_builder = StableHLOBuilder(ctx, loc, mesh_name, mesh_shape)
            new_module = stablehlo_builder.parse_root_module(root_module, golden_inputs)
            new_module.body.append(stablehlo_builder._get_mesh())

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
            for func_op in module.body.operations:
                if (
                    not isinstance(func_op, func.FuncOp)
                    or func_op.name.value in builder._nested_funcs
                ):
                    continue

                for block in func_op.body:
                    for op in block.operations:
                        if isinstance(op, func.ReturnOp):
                            continue
                        elif isinstance(op, func.CallOp):
                            sub_op_module_builder = builder.split_call_op(op)
                            if len(sub_op_module_builder) != 0:
                                sub_modules_and_builders.append(sub_op_module_builder)
                        else:
                            sub_op_module_builder = builder.split_op(op)
                            if len(sub_op_module_builder) != 0:
                                sub_modules_and_builders.append(sub_op_module_builder)

        return sub_modules_and_builders
