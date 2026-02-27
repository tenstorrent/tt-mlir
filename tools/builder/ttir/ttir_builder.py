# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Callable, Dict, Any, Sequence
import torch
from enum import Enum, auto
import re
from contextvars import ContextVar

from ttmlir.ir import *
from ttmlir.dialects import ttir, ttcore, tensor, quant, func
from ttmlir.passes import GoldenTensor, DataType

from builder.base.builder import *
from builder.base.builder_utils import *
from builder.base.builder_enums import *

from golden import *


class TTIRBuilder(Builder):

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

    # ----- Private methods ----

    def _get_empty_op(self, tensor_type: RankedTensorType) -> OpResult:
        """Get TTIR-specific empty operation."""
        return ttir.EmptyOp(tensor_type).result

    def _organize_eltwise_ttir(
        self, inputs: List[Operand], output_type: RankedTensorType
    ):
        return (output_type, *inputs)

    def _op_proxy(
        self,
        op_ttir_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
        organize_ttir_args: Optional[Callable] = None,
        organize_golden_args: Optional[Callable] = None,
        output_shape: Optional[Shape] = None,
        output_type: Optional[Type] = None,
        output_create_fn: Optional[Callable] = None,
        golden_kwargs: dict = {},
        ttir_kwargs: dict = {},
        loc: Optional[Union[str, Location]] = None,
        skip_golden: bool = False,
    ) -> Any:
        if not golden_kwargs:
            golden_kwargs = ttir_kwargs

        if organize_ttir_args is None:
            organize_ttir_args = self._organize_eltwise_ttir

        if organize_golden_args is None:
            organize_golden_args = self._organize_eltwise_golden

        with self._ctx, self._loc:
            # If output shape or type is not provided, calculate it using golden function.
            # This is needed because TTIR ops do not have shape or type MLIR inference trait.
            output_shape_and_type = self._get_output_shape_and_type(
                organize_golden_args, inputs, op_ttir_function, golden_kwargs
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
            output_element_type = (
                self._get_type_from_torch_dtype(calculated_output_type)
                if not output_type
                else output_type
            )

            # Create output tensor using provided function or create empty tensor.
            if output_create_fn:
                output_type = output_create_fn(output_shape, output_element_type)
            else:
                output_type = RankedTensorType.get(output_shape, output_element_type)

            # Prepare location for the op.
            id = self._get_next_global_id()
            loc = (
                self._get_loc_from_str(loc)
                if loc is not None
                else self._get_loc_of_extra_file_callee(id=id)
            )

            # Organize arguments and create the TTIR op.
            if organize_ttir_args(inputs, output_type) == 0:
                op = op_ttir_function(
                    loc=loc,
                    **ttir_kwargs,
                )
            else:
                op = op_ttir_function(
                    *organize_ttir_args(inputs, output_type),
                    loc=loc,
                    **ttir_kwargs,
                )

            # Set unit attributes if provided.
            if unit_attrs is not None:
                for attr_name in unit_attrs:
                    op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

            if not skip_golden:
                op_golden_function = get_golden_function(
                    op_ttir_function, **golden_kwargs
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

    def _get_location(self) -> Location:
        stack = inspect.stack()
        caller_frame = stack[2]
        filename = caller_frame.filename
        lineno = caller_frame.lineno
        return Location.name(f"{filename}:{lineno}")

    def create_tensor_encoding(
        self, shape: Shape, element_type: Union[torch.dtype, TypeInfo]
    ) -> ttnn.ir.TTNNLayoutAttr:
        return None

    # ----- Public Op Generators ----

    ############### ttir.AllToAllOp ###############

    @tag(ttir.AllToAllOp)
    def all_to_all(
        self,
        input: Operand,
        split_dim: int,
        concat_dim: int,
        split_count: int,
        replica_groups: List[List[int]],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.all_to_all)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(input)
        split_dim_attr = IntegerAttr.get(IntegerType.get_signed(32), split_dim)
        concat_dim_attr = IntegerAttr.get(IntegerType.get_signed(32), concat_dim)
        split_count_attr = IntegerAttr.get(IntegerType.get_signed(32), split_count)
        replica_groups_attr = DenseElementsAttr.get(np.array(replica_groups))
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            split_dim_attr,
            concat_dim_attr,
            split_count_attr,
            replica_groups_attr,
            mlir_output_type,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
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

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.AllToAllOp)
    def all_to_all_parser(
        self,
        old_op: ttir.AllToAllOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.all_to_all_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        split_dim_attr = old_op.split_dim
        concat_dim_attr = old_op.concat_dim
        split_count_attr = old_op.split_count
        replica_groups_attr = old_op.replica_groups

        new_op = ttir_op(
            result,
            in0,
            split_dim_attr,
            concat_dim_attr,
            split_count_attr,
            replica_groups_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            split_dim_attr,
            concat_dim_attr,
            split_count_attr,
            replica_groups_attr,
            old_op.result.type.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttir.CollectiveBroadcastOp ###############

    @tag(ttir.CollectiveBroadcastOp)
    def collective_broadcast(
        self,
        input: Operand,
        replica_groups: List[Tuple[int, int]],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.collective_broadcast)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(input)
        replica_groups_attr = DenseElementsAttr.get(np.array(replica_groups))
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, replica_groups_attr, mlir_output_type
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            input,
            replica_groups_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.CollectiveBroadcastOp)
    def collective_broadcast_parser(
        self,
        old_op: ttir.CollectiveBroadcastOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.collective_broadcast_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        replica_groups_attr = old_op.replica_groups

        new_op = ttir_op(
            result,
            in0,
            replica_groups_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, replica_groups_attr, old_op.result.type.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttir.CollectivePermuteOp ###############

    @tag(ttir.CollectivePermuteOp)
    def collective_permute(
        self,
        input: Operand,
        source_target_pairs: List[Tuple[int, int]],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.collective_permute)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(input)
        source_target_pairs_attr = DenseElementsAttr.get(np.array(source_target_pairs))
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, source_target_pairs_attr, mlir_output_type
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            input,
            source_target_pairs_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.CollectivePermuteOp)
    def collective_permute_parser(
        self,
        old_op: ttir.CollectivePermuteOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.collective_permute_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        source_target_pairs_attr = old_op.source_target_pairs
        new_op = ttir_op(
            result,
            in0,
            source_target_pairs_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, source_target_pairs_attr, old_op.result.type.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttir.ReduceScatterOp ###############

    @tag(ttir.ReduceScatterOp)
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
        ttir_op = self.get_opview_from_method(TTIRBuilder.reduce_scatter)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(input)
        reduce_type_attr = ttcore.ir.ReduceTypeAttr.get(self._ctx, reduce_type.value)
        scatter_dim_attr = IntegerAttr.get(IntegerType.get_signed(32), scatter_dim)
        cluster_axis_attr = IntegerAttr.get(IntegerType.get_unsigned(32), cluster_axis)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            reduce_type_attr,
            scatter_dim_attr,
            cluster_axis_attr,
            mlir_output_type,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            input,
            reduce_type_attr,
            scatter_dim_attr,
            cluster_axis_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.ReduceScatterOp)
    def reduce_scatter_parser(
        self,
        old_op: ttir.ReduceScatterOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.reduce_scatter_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        reduce_type_attr = old_op.reduce_type
        scatter_dim_attr = old_op.scatter_dim
        cluster_axis_attr = old_op.cluster_axis

        new_op = ttir_op(
            result,
            in0,
            reduce_type_attr,
            scatter_dim_attr,
            cluster_axis_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            reduce_type_attr,
            scatter_dim_attr,
            cluster_axis_attr,
            old_op.result.type.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttir.AllReduceOp ###############

    @tag(ttir.AllReduceOp)
    def all_reduce(
        self,
        input: Operand,
        cluster_axis: int,
        reduce_type: ReduceType,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.all_reduce)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(input)
        reduce_type_attr = ttcore.ir.ReduceTypeAttr.get(self._ctx, reduce_type.value)
        cluster_axis_attr = IntegerAttr.get(IntegerType.get_unsigned(32), cluster_axis)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, reduce_type_attr, cluster_axis_attr, mlir_output_type
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            input,
            reduce_type_attr,
            cluster_axis_attr,
            loc=loc,
        )
        new_op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(new_op_result, golden_output)

        return new_op_result

    @parse(ttir.AllReduceOp)
    def all_reduce_parser(
        self,
        old_op: ttir.AllReduceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.all_reduce_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        reduce_type_attr = old_op.reduce_type
        cluster_axis_attr = old_op.cluster_axis

        new_op = ttir_op(
            result,
            in0,
            reduce_type_attr,
            cluster_axis_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            reduce_type_attr,
            cluster_axis_attr,
            old_op.result.type.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttir.MeshShardOp ###############

    @tag(ttir.MeshShardOp)
    def mesh_shard(
        self,
        input: Operand,
        shard_type: MeshShardType,
        shard_direction: MeshShardDirection,
        shard_shape: List[int],
        shard_dims: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.mesh_shard)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(input)
        shard_type_attr = ttcore.ir.MeshShardTypeAttr.get(self._ctx, shard_type.value)
        shard_direction_attr = ttcore.ir.MeshShardDirectionAttr.get(
            self._ctx, shard_direction.value
        )
        shard_shape_attr = DenseI64ArrayAttr.get(shard_shape)
        shard_dims_attr = DenseI64ArrayAttr.get(shard_dims)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            shard_type_attr,
            shard_direction_attr,
            shard_shape_attr,
            shard_dims_attr,
            mlir_output_type,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            input,
            shard_type_attr,
            shard_direction_attr,
            shard_shape_attr,
            shard_dims_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.MeshShardOp)
    def mesh_shard_parser(
        self,
        old_op: ttir.MeshShardOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.mesh_shard_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type
        shard_type_attr = old_op.shard_type
        shard_direction_attr = old_op.shard_direction
        shard_shape_attr = old_op.shard_shape
        shard_dims_attr = old_op.shard_dims

        new_op = ttir_op(
            result,
            in0,
            shard_type_attr,
            shard_direction_attr,
            shard_shape_attr,
            shard_dims_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            shard_type_attr,
            shard_direction_attr,
            shard_shape_attr,
            shard_dims_attr,
            old_op.result.type.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttir.AllGatherOp ###############

    @tag(ttir.AllGatherOp)
    def all_gather(
        self,
        input: Operand,
        all_gather_dim: int,
        cluster_axis: int,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.all_gather)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(input)
        all_gather_dim_attr = IntegerAttr.get(
            IntegerType.get_signed(32), all_gather_dim
        )
        cluster_axis_attr = IntegerAttr.get(IntegerType.get_unsigned(32), cluster_axis)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, all_gather_dim_attr, cluster_axis_attr, mlir_output_type
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.AllGatherOp)
    def all_gather_parser(
        self,
        old_op: ttir.AllGatherOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.all_gather_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        all_gather_dim_attr = old_op.all_gather_dim
        cluster_axis_attr = old_op.cluster_axis

        new_op = ttir_op(
            result,
            in0,
            all_gather_dim_attr,
            cluster_axis_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            all_gather_dim_attr,
            cluster_axis_attr,
            old_op.result.type.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    ############### ttir.ToLayoutOp ###############

    @tag(ttir.ToLayoutOp)
    def to_layout(
        self,
        in0: Operand,
        output_type: RankedTensorType,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.to_layout)

        output = self._get_empty_op(output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            [output_type],
            in0,
            output,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, output_type)
        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.ToLayoutOp)
    def to_layout_parser(
        self,
        old_op: ttir.ToLayoutOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.to_layout_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        output = self._get_empty_op(result)

        new_op = ttir_op(
            [result],
            in0,
            output,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.ToLayoutOp)
    def to_layout_split(
        self,
        old_op: ttir.ToLayoutOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.to_layout_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            to_layout_module = Module.create()
            to_layout_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(to_layout_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="to_layout_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    output = self._get_empty_op(result)

                    new_op = ttir_op(
                        [result],
                        in0,
                        output,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    to_layout_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    to_layout_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                to_layout_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return to_layout_module, to_layout_builder

    ############### ttir.RearrangeOp ###############

    @tag(ttir.RearrangeOp)
    def rearrange(
        self,
        in0: Operand,
        pattern: str,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.rearrange)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        pattern_attr = StringAttr.get(pattern)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, pattern_attr, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            pattern_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.RearrangeOp)
    def rearrange_parser(
        self,
        old_op: ttir.RearrangeOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.rearrange_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        pattern_attr = old_op.pattern

        new_op = ttir_op(
            result,
            in0,
            pattern_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, pattern_attr, old_op.result.type.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.RearrangeOp)
    def rearrange_split(
        self,
        old_op: ttir.RearrangeOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.rearrange_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            rearrange_module = Module.create()
            rearrange_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(rearrange_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="rearrange_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        old_op.pattern,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    rearrange_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    rearrange_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                rearrange_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return rearrange_module, rearrange_builder

    ############### ttir.ReduceAndOp ###############

    @tag(ttir.ReduceAndOp)
    def reduce_and(
        self,
        in0: Operand,
        keep_dim: bool,
        dim_arg: List[int] = None,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.reduce_and)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        if dim_arg is None:
            dim_arg = list(range(len(self.get_shape(in0))))
        dim_arg_attr = ArrayAttr.get(
            [IntegerAttr.get(IntegerType.get_signless(32), d) for d in dim_arg]
        )
        keep_dim_attr = BoolAttr.get(keep_dim, self._ctx)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, dim_arg_attr, keep_dim_attr, mlir_output_type
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            keep_dim_attr,
            dim_arg=dim_arg_attr,
            loc=loc,
        )
        new_op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(new_op_result, golden_output)

        return new_op_result

    @parse(ttir.ReduceAndOp)
    def reduce_and_parser(
        self,
        old_op: ttir.ReduceAndOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.reduce_and_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type
        keep_dim_attr = old_op.keep_dim
        dim_arg_attr = old_op.dim_arg

        new_op = ttir_op(
            result,
            in0,
            keep_dim_attr,
            dim_arg=dim_arg_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, dim_arg_attr, keep_dim_attr, old_op.result.type.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.ReduceAndOp)
    def reduce_and_split(
        self,
        old_op: ttir.ReduceAndOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.reduce_and_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            reduce_module = Module.create()
            reduce_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(reduce_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="reduce_and_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        old_op.keep_dim,
                        dim_arg=old_op.dim_arg,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    reduce_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    reduce_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                reduce_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return reduce_module, reduce_builder

    ############### ttir.RepeatOp ###############

    @tag(ttir.RepeatOp)
    def repeat(
        self,
        in0: Operand,
        repeat_dimensions: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.repeat)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        repeat_dimensions_attr = DenseI64ArrayAttr.get(repeat_dimensions)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, repeat_dimensions_attr, mlir_output_type
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            repeat_dimensions_attr,
            loc=loc,
        )
        new_op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(new_op_result, golden_output)

        return new_op_result

    @parse(ttir.RepeatOp)
    def repeat_parser(
        self,
        old_op: ttir.RepeatOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.repeat_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type
        repeat_dimensions_attr = old_op.repeat_dimensions

        new_op = ttir_op(
            result,
            in0,
            repeat_dimensions_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, repeat_dimensions_attr, old_op.result.type.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.RepeatOp)
    def repeat_split(
        self,
        old_op: ttir.RepeatOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.repeat_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            repeat_module = Module.create()
            repeat_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(repeat_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="repeat_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        old_op.repeat_dimensions,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    repeat_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
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

    ############### ttir.ArangeOp ###############

    @tag(ttir.ArangeOp)
    def arange(
        self,
        shape: List[int],
        dtype: torch.dtype,
        start: int,
        end: int,
        step: int,
        arange_dimension: int,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.arange)
        shape_attr = DenseI64ArrayAttr.get(shape)
        start_attr = IntegerAttr.get(IntegerType.get_signed(64), start)
        end_attr = IntegerAttr.get(IntegerType.get_signed(64), end)
        step_attr = IntegerAttr.get(IntegerType.get_signed(64), step)
        arange_dimension_attr = IntegerAttr.get(
            IntegerType.get_signless(64), arange_dimension
        )
        mlir_output_type = self._get_type_from_torch_dtype(dtype)
        result = self._create_ranked_tensor_type(shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            start_attr,
            end_attr,
            step_attr,
            arange_dimension_attr,
            loc=loc,
        )
        new_op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        op_golden_function = get_golden_function(ttir_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            shape_attr,
            start_attr,
            end_attr,
            step_attr,
            arange_dimension_attr,
            mesh_shape_attr,
            mlir_output_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        return new_op_result

    @parse(ttir.ArangeOp)
    def arange_parser(
        self,
        old_op: ttir.ArangeOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.arange_parser)
        result = old_op.result.type
        start_attr = old_op.start
        end_attr = old_op.end
        step_attr = old_op.step
        arange_dimension_attr = old_op.arange_dimension

        new_op = ttir_op(
            result,
            start_attr,
            end_attr,
            step_attr,
            arange_dimension_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        op_golden_function = get_golden_function(ttir_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            old_op.result.type.shape,
            start_attr,
            end_attr,
            step_attr,
            arange_dimension_attr,
            mesh_shape_attr,
            old_op.result.type.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.ArangeOp)
    def arange_split(
        self,
        old_op: ttir.ArangeOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.arange_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            arange_module = Module.create()
            arange_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types: List[Type] = []

            with InsertionPoint(arange_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="arange_module")
                def decorated_func():
                    start_attr = old_op.start
                    end_attr = old_op.end
                    step_attr = old_op.step
                    arange_dimension_attr = old_op.arange_dimension
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        start_attr,
                        end_attr,
                        step_attr,
                        arange_dimension_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    arange_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                arange_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return arange_module, arange_builder

    ############### ttir.CumSumOp ###############

    @tag(ttir.CumSumOp)
    def cumsum(
        self,
        in0: Operand,
        dim: int,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.cumsum)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        dim_attr = IntegerAttr.get(IntegerType.get_signless(64), dim)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, dim_attr, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(result, in0, dim_attr, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.CumSumOp)
    def cumsum_parser(
        self,
        old_op: ttir.CumSumOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.cumsum_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type
        dim_attr = old_op.dim

        new_op = ttir_op(
            result,
            in0,
            dim_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, dim_attr, old_op.result.type.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.CumSumOp)
    def cumsum_split(
        self,
        old_op: ttir.CumSumOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.cumsum_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            cumsum_module = Module.create()
            cumsum_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(cumsum_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="cumsum_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, old_op.dim, loc=old_op.location)
                    new_op_result = new_op.result

                    cumsum_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    cumsum_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                cumsum_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return cumsum_module, cumsum_builder

    ############### ttir.GatherOp ###############

    @tag(ttir.GatherOp)
    def gather(
        self,
        input: Operand,
        start_indices: Operand,
        offset_dims: List[int],
        collapsed_slice_dims: List[int],
        operand_batching_dims: List[int],
        start_indices_batching_dims: List[int],
        start_index_map: List[int],
        index_vector_dim: int,
        slice_sizes: List[int],
        indices_are_sorted: bool = False,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.gather)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        offset_dims_attr = DenseI64ArrayAttr.get(offset_dims, self._ctx)
        collapsed_slice_dims_attr = DenseI64ArrayAttr.get(
            collapsed_slice_dims, self._ctx
        )
        operand_batching_dims_attr = DenseI64ArrayAttr.get(
            operand_batching_dims, self._ctx
        )
        start_indices_batching_dims_attr = DenseI64ArrayAttr.get(
            start_indices_batching_dims, self._ctx
        )
        start_index_map_attr = DenseI64ArrayAttr.get(start_index_map, self._ctx)
        slice_sizes_attr = DenseI64ArrayAttr.get(slice_sizes, self._ctx)
        indices_are_sorted_attr = BoolAttr.get(indices_are_sorted, self._ctx)
        index_vector_dim_attr = IntegerAttr.get(
            IntegerType.get_signed(64), index_vector_dim
        )

        input_golden = self._get_golden_tensor(input)
        indices_golden = self._get_golden_tensor(start_indices)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input_golden,
            indices_golden,
            offset_dims_attr,
            collapsed_slice_dims_attr,
            operand_batching_dims_attr,
            start_indices_batching_dims_attr,
            start_index_map_attr,
            index_vector_dim_attr,
            slice_sizes_attr,
            indices_are_sorted_attr,
            mlir_output_type,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            input,
            start_indices,
            offset_dims_attr,
            collapsed_slice_dims_attr,
            operand_batching_dims_attr,
            start_indices_batching_dims_attr,
            start_index_map_attr,
            index_vector_dim_attr,
            slice_sizes_attr,
            indices_are_sorted_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.GatherOp)
    def gather_parser(
        self,
        old_op: ttir.GatherOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.gather_parser)
        in0 = global_dict[old_op.input]
        in1 = global_dict[old_op.start_indices]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            in1,
            old_op.offset_dims,
            old_op.collapsed_slice_dims,
            old_op.operand_batching_dims,
            old_op.start_indices_batching_dims,
            old_op.start_index_map,
            old_op.index_vector_dim,
            old_op.slice_sizes,
            old_op.indices_are_sorted,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            input1,
            old_op.offset_dims,
            old_op.collapsed_slice_dims,
            old_op.operand_batching_dims,
            old_op.start_indices_batching_dims,
            old_op.start_index_map,
            old_op.index_vector_dim,
            old_op.slice_sizes,
            old_op.indices_are_sorted,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.GatherOp)
    def gather_split(
        self,
        old_op: ttir.GatherOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.gather_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            gather_module = Module.create()
            gather_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type, old_op.start_indices.type]

            with InsertionPoint(gather_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="gather_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        in1,
                        old_op.offset_dims,
                        old_op.collapsed_slice_dims,
                        old_op.operand_batching_dims,
                        old_op.start_indices_batching_dims,
                        old_op.start_index_map,
                        old_op.index_vector_dim,
                        old_op.slice_sizes,
                        old_op.indices_are_sorted,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    input1 = self._get_golden_tensor(old_op.start_indices)
                    gather_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    gather_builder._set_golden_tensor(in0, input0)
                    gather_builder._set_golden_tensor(in1, input1)
                    ordered_inputs.extend([in0, in1])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                gather_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return gather_module, gather_builder

    ############### ttir.OnesOp ###############

    @tag(ttir.OnesOp)
    def ones(
        self,
        shape: List[int],
        dtype: torch.dtype,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.ones)
        mlir_output_type = self._get_type_from_torch_dtype(dtype)
        shape_attr = DenseI32ArrayAttr.get(shape)
        result = self._create_ranked_tensor_type(shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            shape_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        op_golden_function = get_golden_function(ttir_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            shape_attr, mesh_shape_attr, mlir_output_type
        )
        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.OnesOp)
    def ones_parser(
        self,
        old_op: ttir.OnesOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.ones_parser)
        result = old_op.result.type
        shape_attr = old_op.shape

        new_op = ttir_op(
            result,
            shape_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        op_golden_function = get_golden_function(ttir_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            shape_attr, mesh_shape_attr, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.OnesOp)
    def ones_split(
        self,
        old_op: ttir.OnesOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.ones_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            ones_module = Module.create()
            ones_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types: List[Type] = []

            with InsertionPoint(ones_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="ones_module")
                def decorated_func():
                    result = old_op.result.type

                    new_op = ttir_op(result, old_op.shape, loc=old_op.location)
                    new_op_result = new_op.result

                    ones_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                ones_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return ones_module, ones_builder

    ############### ttir.ZerosOp ###############

    @tag(ttir.ZerosOp)
    def zeros(
        self,
        shape: List[int],
        dtype: torch.dtype,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.zeros)
        mlir_output_type = self._get_type_from_torch_dtype(dtype)
        shape_attr = DenseI32ArrayAttr.get(shape)
        result = self._create_ranked_tensor_type(shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            shape_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        op_golden_function = get_golden_function(ttir_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            shape_attr, mesh_shape_attr, result.element_type
        )
        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.ZerosOp)
    def zeros_parser(
        self,
        old_op: ttir.ZerosOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.zeros_parser)
        result = old_op.result.type
        shape_attr = old_op.shape

        new_op = ttir_op(
            result,
            shape_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        op_golden_function = get_golden_function(ttir_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            shape_attr, mesh_shape_attr, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.ZerosOp)
    def zeros_split(
        self,
        old_op: ttir.ZerosOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.zeros_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            zeros_module = Module.create()
            zeros_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types: List[Type] = []

            with InsertionPoint(zeros_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="zeros_module")
                def decorated_func():
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        old_op.shape,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    zeros_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                zeros_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return zeros_module, zeros_builder

    ############### ttir.RandOp ###############

    @tag(ttir.RandOp)
    def rand(
        self,
        size: List[int],
        dtype: torch.dtype,
        low: float = 0.0,
        high: float = 1.0,
        seed: int = 0,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        ttir_op = self.get_opview_from_method(TTIRBuilder.rand)
        mlir_output_type = self._get_type_from_torch_dtype(dtype)
        size_attr = ArrayAttr.get(
            [IntegerAttr.get(IntegerType.get_signless(32), dim) for dim in size]
        )
        low_attr = FloatAttr.get_f32(low)
        high_attr = FloatAttr.get_f32(high)
        seed_attr = IntegerAttr.get(IntegerType.get_unsigned(32), seed)
        result = self._create_ranked_tensor_type(size, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            size_attr,
            mlir_output_type,
            low=low_attr,
            high=high_attr,
            seed=seed_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        op_golden_function = get_golden_function(ttir_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            size_attr,
            low_attr,
            high_attr,
            seed_attr,
            mesh_shape_attr,
            mlir_output_type,
        )
        self._set_golden_tensor(op_result, golden_output)

        self.bypass(op_result)
        return op_result

    @parse(ttir.RandOp)
    def rand_parser(
        self,
        old_op: ttir.RandOp,
        global_dict: Dict[Operand, Operand],
    ) -> Operation:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.rand_parser)

        result = old_op.result.type
        size_attr = old_op.size
        dtype_attr = old_op.dtype
        low_attr = old_op.low
        high_attr = old_op.high
        seed_attr = old_op.seed

        new_op = ttir_op(
            result,
            size_attr,
            dtype_attr,
            low=low_attr,
            high=high_attr,
            seed=seed_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        op_golden_function = get_golden_function(ttir_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            size_attr,
            low_attr,
            high_attr,
            seed_attr,
            mesh_shape_attr,
            dtype_attr,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        self.bypass(new_op_result)
        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.RandOp)
    def rand_split(
        self,
        old_op: ttir.RandOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.rand_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            rand_module = Module.create()
            rand_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types: List[Type] = []

            with InsertionPoint(rand_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="rand_module")
                def decorated_func():
                    result = old_op.result.type
                    size_attr = old_op.size
                    dtype_attr = old_op.dtype
                    low_attr = old_op.low
                    high_attr = old_op.high
                    seed_attr = old_op.seed

                    new_op = ttir_op(
                        result,
                        size_attr,
                        dtype_attr,
                        low=low_attr,
                        high=high_attr,
                        seed=seed_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    rand_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    ordered_outputs.append(new_op_result)

                    rand_builder.bypass(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                rand_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return rand_module, rand_builder

    ############### ttir.DropoutOp ###############

    @tag(ttir.DropoutOp)
    def dropout(
        self,
        input: Operand,
        prob: float = 0.0,
        scale: float = 1.0,
        seed: int = 0,
        use_per_device_seed: bool = True,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        ttir_op = self.get_opview_from_method(TTIRBuilder.dropout)

        # Dropout preserves input type
        mlir_output_type = self.get_type(input)
        result = input.type

        prob_attr = FloatAttr.get_f32(prob)
        scale_attr = FloatAttr.get_f32(scale)
        seed_attr = IntegerAttr.get(IntegerType.get_unsigned(32), seed)
        use_per_device_seed_attr = BoolAttr.get(use_per_device_seed)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            input,
            prob=prob_attr,
            scale=scale_attr,
            seed=seed_attr,
            use_per_device_seed=use_per_device_seed_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        input0 = self._get_golden_tensor(input)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            prob_attr,
            scale_attr,
            seed_attr,
            use_per_device_seed_attr,
            mlir_output_type,
        )
        self._set_golden_tensor(op_result, golden_output)

        self.bypass(op_result)
        return op_result

    @parse(ttir.DropoutOp)
    def dropout_parser(
        self,
        old_op: ttir.DropoutOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.dropout_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            prob=old_op.prob,
            scale=old_op.scale,
            seed=old_op.seed,
            use_per_device_seed=old_op.use_per_device_seed,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            old_op.prob,
            old_op.scale,
            old_op.seed,
            old_op.use_per_device_seed,
            old_op.result.type.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.DropoutOp)
    def dropout_split(
        self,
        old_op: ttir.DropoutOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.dropout_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            dropout_module = Module.create()
            dropout_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(dropout_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="dropout_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        prob=old_op.prob,
                        scale=old_op.scale,
                        seed=old_op.seed,
                        use_per_device_seed=old_op.use_per_device_seed,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    dropout_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    dropout_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    dropout_builder.bypass(new_op_result)
                    return new_op

                new_func_op = decorated_func.func_op
                dropout_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return dropout_module, dropout_builder

    ############### ttir.CosOp ###############

    @tag(ttir.CosOp)
    def cos(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.cos)
        input0 = self._get_golden_tensor(in0)
        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(result, in0, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.CosOp)
    def cos_parser(
        self,
        old_op: ttir.CosOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.cos_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.CosOp)
    def cos_split(
        self,
        old_op: ttir.CosOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.cos_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            cos_module = Module.create()
            cos_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(cos_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="cos_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    cos_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    cos_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                cos_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return cos_module, cos_builder

    ############### ttir.SinOp ###############

    @tag(ttir.SinOp)
    def sin(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.sin)
        input0 = self._get_golden_tensor(in0)
        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(result, in0, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.SinOp)
    def sin_parser(
        self,
        old_op: ttir.SinOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.sin_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.SinOp)
    def sin_split(
        self,
        old_op: ttir.SinOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.sin_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            sin_module = Module.create()
            sin_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(sin_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sin_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    sin_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    sin_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                sin_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sin_module, sin_builder

    ############### ttir.SqrtOp ###############

    @tag(ttir.SqrtOp)
    def sqrt(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.sqrt)
        input0 = self._get_golden_tensor(in0)
        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(result, in0, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.SqrtOp)
    def sqrt_parser(
        self,
        old_op: ttir.SqrtOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.sqrt_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.SqrtOp)
    def sqrt_split(
        self,
        old_op: ttir.SqrtOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.sqrt_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            sqrt_module = Module.create()
            sqrt_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(sqrt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sqrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    sqrt_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    sqrt_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                sqrt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sqrt_module, sqrt_builder

    ############### ttir.GreaterEqualOp ###############

    @tag(ttir.GreaterEqualOp)
    def ge(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.ge)
        lhs = self._get_golden_tensor(in0)
        rhs = self._get_golden_tensor(in1)
        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(lhs, rhs, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(result, in0, in1, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.GreaterEqualOp)
    def ge_parser(
        self,
        old_op: ttir.GreaterEqualOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.ge_parser)
        in0 = global_dict[old_op.lhs]
        in1 = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            in1,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.GreaterEqualOp)
    def ge_split(
        self,
        old_op: ttir.GreaterEqualOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.ge_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            ge_module = Module.create()
            ge_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(ge_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="ge_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    ge_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    ge_builder._set_golden_tensor(in0, input0)
                    ge_builder._set_golden_tensor(in1, input1)
                    ordered_inputs.extend([in0, in1])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                ge_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return ge_module, ge_builder

    ############### ttir.LessThanOp ###############

    @tag(ttir.LessThanOp)
    def lt(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.lt)
        lhs = self._get_golden_tensor(in0)
        rhs = self._get_golden_tensor(in1)
        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(lhs, rhs, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(result, in0, in1, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.LessThanOp)
    def lt_parser(
        self,
        old_op: ttir.LessThanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.lt_parser)
        in0 = global_dict[old_op.lhs]
        in1 = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            in1,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.LessThanOp)
    def lt_split(
        self,
        old_op: ttir.LessThanOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.lt_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            lt_module = Module.create()
            lt_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(lt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="lt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    lt_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    lt_builder._set_golden_tensor(in0, input0)
                    lt_builder._set_golden_tensor(in1, input1)
                    ordered_inputs.extend([in0, in1])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                lt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return lt_module, lt_builder

    ############### ttir.LessEqualOp ###############

    @tag(ttir.LessEqualOp)
    def le(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.le)
        lhs = self._get_golden_tensor(in0)
        rhs = self._get_golden_tensor(in1)
        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(lhs, rhs, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(result, in0, in1, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.LessEqualOp)
    def le_parser(
        self,
        old_op: ttir.LessEqualOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.le_parser)
        in0 = global_dict[old_op.lhs]
        in1 = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            in1,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.LessEqualOp)
    def le_split(
        self,
        old_op: ttir.LessEqualOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.le_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            le_module = Module.create()
            le_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(le_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="le_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    le_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    le_builder._set_golden_tensor(in0, input0)
                    le_builder._set_golden_tensor(in1, input1)
                    ordered_inputs.extend([in0, in1])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                le_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return le_module, le_builder

    ############### ttir.BitwiseAndOp ###############

    @tag(ttir.BitwiseAndOp)
    def bitwise_and(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.bitwise_and)
        lhs = self._get_golden_tensor(in0)
        rhs = self._get_golden_tensor(in1)
        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(lhs, rhs, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(result, in0, in1, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.BitwiseAndOp)
    def bitwise_and_parser(
        self,
        old_op: ttir.BitwiseAndOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.bitwise_and_parser)
        in0 = global_dict[old_op.lhs]
        in1 = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            in1,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.BitwiseAndOp)
    def bitwise_and_split(
        self,
        old_op: ttir.BitwiseAndOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.bitwise_and_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            bitwise_and_module = Module.create()
            bitwise_and_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(bitwise_and_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="bitwise_and_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    bitwise_and_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    bitwise_and_builder._set_golden_tensor(in0, input0)
                    bitwise_and_builder._set_golden_tensor(in1, input1)
                    ordered_inputs.extend([in0, in1])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                bitwise_and_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return bitwise_and_module, bitwise_and_builder

    ############### ttir.PowOp ###############

    @tag(ttir.PowOp)
    def pow(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.pow)
        lhs = self._get_golden_tensor(in0)
        rhs = self._get_golden_tensor(in1)
        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(lhs, rhs, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(result, in0, in1, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.PowOp)
    def pow_parser(
        self,
        old_op: ttir.PowOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.pow_parser)
        in0 = global_dict[old_op.lhs]
        in1 = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            in1,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.PowOp)
    def pow_split(
        self,
        old_op: ttir.PowOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.pow_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            pow_module = Module.create()
            pow_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(pow_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="pow_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    pow_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    pow_builder._set_golden_tensor(in0, input0)
                    pow_builder._set_golden_tensor(in1, input1)
                    ordered_inputs.extend([in0, in1])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                pow_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return pow_module, pow_builder

    ############### ttir.MinimumOp ###############

    @tag(ttir.MinimumOp)
    def minimum(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.minimum)
        lhs = self._get_golden_tensor(in0)
        rhs = self._get_golden_tensor(in1)
        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(lhs, rhs, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(result, in0, in1, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.MinimumOp)
    def minimum_parser(
        self,
        old_op: ttir.MinimumOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.minimum_parser)
        in0 = global_dict[old_op.lhs]
        in1 = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            in1,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.MinimumOp)
    def minimum_split(
        self,
        old_op: ttir.MinimumOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.minimum_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            min_module = Module.create()
            min_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(min_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="minimum_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    min_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    min_builder._set_golden_tensor(in0, input0)
                    min_builder._set_golden_tensor(in1, input1)
                    ordered_inputs.extend([in0, in1])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                min_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return min_module, min_builder

    ############### ttir.LogicalRightShiftOp ###############

    @tag(ttir.LogicalRightShiftOp)
    def logical_right_shift(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.logical_right_shift)
        lhs = self._get_golden_tensor(in0)
        rhs = self._get_golden_tensor(in1)
        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(lhs, rhs, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(result, in0, in1, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.LogicalRightShiftOp)
    def logical_right_shift_parser(
        self,
        old_op: ttir.LogicalRightShiftOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.logical_right_shift_parser)
        in0 = global_dict[old_op.lhs]
        in1 = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            in1,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.LogicalRightShiftOp)
    def logical_right_shift_split(
        self,
        old_op: ttir.LogicalRightShiftOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.logical_right_shift_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            lrs_module = Module.create()
            lrs_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(lrs_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_right_shift_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    lrs_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    lrs_builder._set_golden_tensor(in0, input0)
                    lrs_builder._set_golden_tensor(in1, input1)
                    ordered_inputs.extend([in0, in1])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                lrs_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return lrs_module, lrs_builder

    ############### ttir.LogicalAndOp ###############

    @tag(ttir.LogicalAndOp)
    def logical_and(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.logical_and)
        lhs = self._get_golden_tensor(in0)
        rhs = self._get_golden_tensor(in1)
        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(lhs, rhs, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(result, in0, in1, loc=loc)
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.LogicalAndOp)
    def logical_and_parser(
        self,
        old_op: ttir.LogicalAndOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.logical_and_parser)
        in0 = global_dict[old_op.lhs]
        in1 = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            in1,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.LogicalAndOp)
    def logical_and_split(
        self,
        old_op: ttir.LogicalRightShiftOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.logical_and_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            logical_and_module = Module.create()
            logical_and_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(logical_and_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_and_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    logical_and_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    logical_and_builder._set_golden_tensor(in0, input0)
                    logical_and_builder._set_golden_tensor(in1, input1)
                    ordered_inputs.extend([in0, in1])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                logical_and_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return logical_and_module, logical_and_builder

    ############### ttir.SortOp ###############

    @tag(ttir.SortOp)
    def sort(
        self,
        in0: Operand,
        dim: int = -1,
        descending: bool = False,
        stable: bool = False,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> Tuple[OpResult, OpResult]:
        ttir_op = self.get_opview_from_method(TTIRBuilder.sort)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        dim_attr = IntegerAttr.get(IntegerType.get_signed(32), dim)
        descending_attr = BoolAttr.get(descending)
        stable_attr = BoolAttr.get(stable)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_values, golden_indices = op_golden_function(
            input0, dim_attr, descending_attr, stable_attr, mlir_output_type
        )
        values = self._create_ranked_tensor_type(golden_values.shape, mlir_output_type)
        indices = self._create_ranked_tensor_type(
            golden_indices.shape, self._get_type_from_torch_dtype(torch.int64)
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            values,
            indices,
            in0,
            dim=dim_attr,
            descending=descending_attr,
            stable=stable_attr,
            loc=loc,
        )
        op_values = op.values
        op_indices = op.indices

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_values, golden_values)
        self._set_golden_tensor(op_indices, golden_indices)

        return op_values, op_indices

    ############### ttir.ReverseOp ###############

    @tag(ttir.ReverseOp)
    def reverse(
        self,
        in0: Operand,
        dimensions: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.reverse)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        dimensions_attr = DenseI64ArrayAttr.get(dimensions)
        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, dimensions_attr, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            dimensions_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.ReverseOp)
    def reverse_parser(
        self,
        old_op: ttir.ReverseOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.reverse_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        dimensions_attr = old_op.dimensions

        new_op = ttir_op(
            result,
            in0,
            dimensions_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, dimensions_attr, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.ReverseOp)
    def reverse_split(
        self,
        old_op: ttir.ReverseOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.reverse_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            reverse_module = Module.create()
            reverse_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(reverse_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="reverse_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    dimensions_attr = old_op.dimensions

                    new_op = ttir_op(
                        result,
                        in0,
                        dimensions_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    reverse_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
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

    ############### ttir.ScatterOp ###############

    @tag(ttir.ScatterOp)
    def scatter(
        self,
        in0: Operand,
        index: Operand,
        source: Operand,
        dim: int,
        scatter_reduce_type: ReduceType = ReduceType.Invalid,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.scatter)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        dim_attr = IntegerAttr.get(IntegerType.get_signless(32), dim)
        scatter_reduce_type_attr = ttcore.ir.ReduceTypeAttr.get(
            self._ctx, scatter_reduce_type.value
        )
        input0 = self._get_golden_tensor(in0)
        input_index = self._get_golden_tensor(index)
        input_source = self._get_golden_tensor(source)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            input_index,
            input_source,
            dim_attr,
            scatter_reduce_type_attr,
            mlir_output_type,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            index,
            source,
            dim_attr,
            scatter_reduce_type_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.ScatterOp)
    def scatter_parser(
        self,
        old_op: ttir.ScatterOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.scatter_parser)

        in0 = global_dict[old_op.input]
        index = global_dict[old_op.index]
        source = global_dict[old_op.source]
        result = old_op.result.type
        dim_attr = old_op.dim
        scatter_reduce_type_attr = old_op.scatter_reduce_type

        new_op = ttir_op(
            result,
            in0,
            index,
            source,
            dim_attr,
            scatter_reduce_type_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input_index = self._get_golden_tensor(index)
        input_source = self._get_golden_tensor(source)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            input_index,
            input_source,
            dim_attr,
            scatter_reduce_type_attr,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.ScatterOp)
    def scatter_split(
        self,
        old_op: ttir.ScatterOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.scatter_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            scatter_module = Module.create()
            scatter_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type, old_op.index.type, old_op.source.type]

            with InsertionPoint(scatter_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="scatter_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    index = inputs[1]
                    source = inputs[2]
                    result = old_op.result.type
                    dim_attr = old_op.dim
                    scatter_reduce_type_attr = old_op.scatter_reduce_type

                    new_op = ttir_op(
                        result,
                        in0,
                        index,
                        source,
                        dim_attr,
                        scatter_reduce_type_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    input_index = self._get_golden_tensor(old_op.index)
                    input_source = self._get_golden_tensor(old_op.source)
                    scatter_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    scatter_builder._set_golden_tensor(in0, input0)
                    scatter_builder._set_golden_tensor(index, input_index)
                    scatter_builder._set_golden_tensor(source, input_source)
                    ordered_inputs.extend([in0, index, source])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                scatter_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return scatter_module, scatter_builder

    ############### ttir.MaxPool2dOp ###############

    @tag(ttir.MaxPool2dOp)
    def max_pool2d(
        self,
        in0: Operand,
        kernel: Union[int, List[int]],
        stride: Union[int, List[int]],
        dilation: Union[int, List[int]],
        padding: Union[int, List[int]],
        ceil_mode: bool,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.max_pool2d)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        if isinstance(kernel, int):
            kernel_attr = IntegerAttr.get(IntegerType.get_signless(32), kernel)
        else:
            kernel_attr = DenseI32ArrayAttr.get(kernel)

        if isinstance(stride, int):
            stride_attr = IntegerAttr.get(IntegerType.get_signless(32), stride)
        else:
            stride_attr = DenseI32ArrayAttr.get(stride)

        if isinstance(padding, int):
            padding_attr = IntegerAttr.get(IntegerType.get_signless(32), padding)
        else:
            padding_attr = DenseI32ArrayAttr.get(padding)

        if isinstance(dilation, int):
            dilation_attr = IntegerAttr.get(IntegerType.get_signless(32), dilation)
        else:
            dilation_attr = DenseI32ArrayAttr.get(dilation)

        ceil_mode_attr = BoolAttr.get(ceil_mode)
        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            kernel_attr,
            stride_attr,
            padding_attr,
            dilation_attr,
            ceil_mode_attr,
            mlir_output_type,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            kernel_attr,
            stride_attr,
            dilation_attr,
            padding_attr,
            ceil_mode_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.MaxPool2dOp)
    def max_pool2d_parser(
        self,
        old_op: ttir.MaxPool2dOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.max_pool2d_parser)

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        kernel_attr = old_op.kernel
        stride_attr = old_op.stride
        dilation_attr = old_op.dilation
        padding_attr = old_op.padding
        ceil_mode_attr = old_op.ceil_mode

        new_op = ttir_op(
            result,
            in0,
            kernel_attr,
            stride_attr,
            dilation_attr,
            padding_attr,
            ceil_mode_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            kernel_attr,
            stride_attr,
            padding_attr,
            dilation_attr,
            ceil_mode_attr,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.MaxPool2dOp)
    def max_pool2d_split(
        self,
        old_op: ttir.MaxPool2dOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.max_pool2d_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            max_pool2d_module = Module.create()
            max_pool2d_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(max_pool2d_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="max_pool2d_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    kernel_attr = old_op.kernel
                    stride_attr = old_op.stride
                    dilation_attr = old_op.dilation
                    padding_attr = old_op.padding
                    ceil_mode_attr = old_op.ceil_mode

                    new_op = ttir_op(
                        result,
                        in0,
                        kernel_attr,
                        stride_attr,
                        dilation_attr,
                        padding_attr,
                        ceil_mode_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    max_pool2d_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    max_pool2d_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                max_pool2d_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return max_pool2d_module, max_pool2d_builder

    ############### ttir.MaxPool2dWithIndicesOp ###############

    @tag(ttir.MaxPool2dWithIndicesOp)
    def max_pool2d_with_indices(
        self,
        in0: Operand,
        kernel: Union[int, List[int]],
        stride: Union[int, List[int]],
        dilation: Union[int, List[int]],
        padding: Union[int, List[int]],
        ceil_mode: bool,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> Tuple[OpResult, OpResult]:
        ttir_op = self.get_opview_from_method(TTIRBuilder.max_pool2d_with_indices)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        if isinstance(kernel, int):
            kernel_attr = IntegerAttr.get(IntegerType.get_signless(32), kernel)
        else:
            kernel_attr = DenseI32ArrayAttr.get(kernel)

        if isinstance(stride, int):
            stride_attr = IntegerAttr.get(IntegerType.get_signless(32), stride)
        else:
            stride_attr = DenseI32ArrayAttr.get(stride)

        if isinstance(padding, int):
            padding_attr = IntegerAttr.get(IntegerType.get_signless(32), padding)
        else:
            padding_attr = DenseI32ArrayAttr.get(padding)

        if isinstance(dilation, int):
            dilation_attr = IntegerAttr.get(IntegerType.get_signless(32), dilation)
        else:
            dilation_attr = DenseI32ArrayAttr.get(dilation)

        ceil_mode_attr = BoolAttr.get(ceil_mode)
        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_outputs = op_golden_function(
            input0,
            kernel_attr,
            stride_attr,
            padding_attr,
            dilation_attr,
            ceil_mode_attr,
            mlir_output_type,
        )
        result = self._create_ranked_tensor_type(
            golden_outputs[0].shape, mlir_output_type
        )
        result_indices = self._create_ranked_tensor_type(
            golden_outputs[1].shape, self._get_type_from_torch_dtype(torch.int64)
        )
        output = self._get_empty_op(result)
        output_indices = self._get_empty_op(result_indices)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            result_indices,
            in0,
            [output, output_indices],
            kernel_attr,
            stride_attr,
            dilation_attr,
            padding_attr,
            ceil_mode_attr,
            loc=loc,
        )
        op_result = op.result
        op_result_indices = op.result_indices

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_outputs[0])
        self._set_golden_tensor(op_result_indices, golden_outputs[1])

        return op_result, op_result_indices

    @parse(ttir.MaxPool2dWithIndicesOp)
    def max_pool2d_with_indices_parser(
        self,
        old_op: ttir.MaxPool2dWithIndicesOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[Operand, GoldenMapTensor]]:
        ttir_op = self.get_opview_from_parser(
            TTIRBuilder.max_pool2d_with_indices_parser
        )

        in0 = global_dict[old_op.input]
        result = old_op.result.type
        result_indices = old_op.result_indices.type
        output = self._get_empty_op(result)
        output_indices = self._get_empty_op(result_indices)
        kernel_attr = old_op.kernel
        stride_attr = old_op.stride
        dilation_attr = old_op.dilation
        padding_attr = old_op.padding
        ceil_mode_attr = old_op.ceil_mode

        new_op = ttir_op(
            result,
            result_indices,
            in0,
            [output, output_indices],
            kernel_attr,
            stride_attr,
            dilation_attr,
            padding_attr,
            ceil_mode_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result
        new_op_result_indices = new_op.result_indices

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_outputs = op_golden_function(
            input0,
            kernel_attr,
            stride_attr,
            padding_attr,
            dilation_attr,
            ceil_mode_attr,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_outputs[0])
        self._set_golden_tensor(new_op_result_indices, golden_outputs[1])

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        op_map_dictionary[old_op.result_indices] = new_op_result_indices
        return new_op, op_map_dictionary

    @split(ttir.MaxPool2dWithIndicesOp)
    def max_pool2d_with_indices_split(
        self,
        old_op: ttir.MaxPool2dWithIndicesOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.max_pool2d_with_indices_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            max_pool2d_with_indices_module = Module.create()
            max_pool2d_with_indices_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(max_pool2d_with_indices_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="max_pool2d_with_indices_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    result_indices = old_op.result_indices.type
                    output = max_pool2d_with_indices_builder._get_empty_op(result)
                    output_indices = max_pool2d_with_indices_builder._get_empty_op(
                        result_indices
                    )
                    kernel_attr = old_op.kernel
                    stride_attr = old_op.stride
                    dilation_attr = old_op.dilation
                    padding_attr = old_op.padding
                    ceil_mode_attr = old_op.ceil_mode

                    new_op = ttir_op(
                        result,
                        result_indices,
                        in0,
                        [output, output_indices],
                        kernel_attr,
                        stride_attr,
                        dilation_attr,
                        padding_attr,
                        ceil_mode_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result
                    new_op_result_indices = new_op.result_indices

                    input0 = self._get_golden_tensor(old_op.input)
                    op_golden_function = get_golden_function(ttir_op)
                    golden_outputs = op_golden_function(
                        input0,
                        kernel_attr,
                        stride_attr,
                        padding_attr,
                        dilation_attr,
                        ceil_mode_attr,
                        result.element_type,
                    )
                    max_pool2d_with_indices_builder._set_golden_tensor(
                        new_op_result, golden_outputs[0]
                    )
                    max_pool2d_with_indices_builder._set_golden_tensor(
                        new_op_result_indices, golden_outputs[1]
                    )
                    max_pool2d_with_indices_builder._set_golden_tensor(
                        old_op.input, input0
                    )
                    ordered_inputs.append(in0)
                    ordered_outputs.extend([new_op_result, new_op_result_indices])

                    return new_op

                new_func_op = decorated_func.func_op
                max_pool2d_with_indices_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return max_pool2d_with_indices_module, max_pool2d_with_indices_builder

    ############### ttir.Log1pOp ###############

    @tag(ttir.Log1pOp)
    def log1p(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.log1p)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.Log1pOp)
    def log1p_parser(
        self,
        old_op: ttir.Log1pOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.log1p_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.Log1pOp)
    def log1p_split(
        self,
        old_op: ttir.Log1pOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.log1p_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            log1p_module = Module.create()
            log1p_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(log1p_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="log1p_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    log1p_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
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

    ############### ttir.ConcatOp ###############

    @tag(ttir.ConcatOp)
    def concat(
        self,
        ins: List[Operand],
        dim: int = 0,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.concat)
        dim_attr = IntegerAttr.get(IntegerType.get_signed(32), dim)

        if output_type is None:
            mlir_output_type = self.get_type(ins[0])
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input_tensors = tuple([self._get_golden_tensor(i) for i in ins])
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input_tensors, dim_attr, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.ConcatOp)
    def concat_parser(
        self,
        old_op: ttir.ConcatOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.concat_parser)
        inputs = [global_dict[inp] for inp in old_op.inputs]
        result = old_op.result.type
        dim_attr = old_op.dim

        new_op = ttir_op(
            result,
            inputs,
            dim=dim_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input_tensors = tuple([self._get_golden_tensor(inp) for inp in inputs])
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input_tensors, dim_attr, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.ConcatOp)
    def concat_split(
        self,
        old_op: ttir.ConcatOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.concat_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            concat_module = Module.create()
            concat_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [inp.type for inp in old_op.inputs]

            with InsertionPoint(concat_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="concat_module")
                def decorated_func(*inputs):
                    result = old_op.result.type
                    dim_attr = old_op.dim

                    new_op = ttir_op(result, inputs, dim=dim_attr, loc=old_op.location)
                    new_op_result = new_op.result

                    input_tensors = tuple(
                        [self._get_golden_tensor(inp) for inp in old_op.inputs]
                    )
                    concat_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    for input_operand, input_golden_tensor in zip(
                        inputs, input_tensors
                    ):
                        concat_builder._set_golden_tensor(
                            input_operand, input_golden_tensor
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

    ############### ttir.FullOp ###############

    @tag(ttir.FullOp)
    def full(
        self,
        output_shape: List[int],
        output_type: torch.dtype,
        fill_value: Union[int, float],
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.full)
        mlir_output_type = self._get_type_from_torch_dtype(output_type)

        if isinstance(fill_value, int):
            fill_value_attr = IntegerAttr.get(IntegerType.get_signless(32), fill_value)
        else:
            fill_value_attr = FloatAttr.get_f32(fill_value)

        output_shape_attr = DenseI32ArrayAttr.get(output_shape)
        result = self._create_ranked_tensor_type(output_shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            output_shape_attr,
            fill_value_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        op_golden_function = get_golden_function(ttir_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            output_shape_attr, fill_value_attr, mesh_shape_attr, mlir_output_type
        )
        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.FullOp)
    def full_parser(
        self,
        old_op: ttir.FullOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.full_parser)

        result = old_op.result.type
        output_shape_attr = old_op.shape
        fill_value_attr = old_op.fill_value

        new_op = ttir_op(
            result,
            output_shape_attr,
            fill_value_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        op_golden_function = get_golden_function(ttir_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(
            output_shape_attr, fill_value_attr, mesh_shape_attr, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.FullOp)
    def full_split(
        self,
        old_op: ttir.FullOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.full_split)
        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            full_module = Module.create()
            full_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = []

            with InsertionPoint(full_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="full_module")
                def decorated_func():
                    result = old_op.result.type
                    shape_attr = old_op.shape
                    fill_value_attr = old_op.fill_value

                    new_op = ttir_op(
                        result,
                        shape_attr,
                        fill_value_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    full_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                full_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return full_module, full_builder

    ############### ttir.ClampTensorOp ###############

    @tag(ttir.ClampTensorOp)
    def clamp_tensor(
        self,
        in0: Operand,
        min_tensor: Operand,
        max_tensor: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.clamp_tensor)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        min_tensor_golden = self._get_golden_tensor(min_tensor)
        max_tensor_golden = self._get_golden_tensor(max_tensor)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, min_tensor_golden, max_tensor_golden, mlir_output_type
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.ClampTensorOp)
    def clamp_tensor_parser(
        self,
        old_op: ttir.ClampTensorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.clamp_tensor_parser)
        in0 = global_dict[old_op.input]
        min_tensor = global_dict[old_op.min]
        max_tensor = global_dict[old_op.max]
        result = old_op.result.type

        new_op = ttir_op(
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
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, min_tensor_golden, max_tensor_golden, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.ClampTensorOp)
    def clamp_tensor_split(
        self,
        old_op: ttir.ClampTensorOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.clamp_tensor_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            clamp_tensor_module = Module.create()
            clamp_tensor_builder = TTIRBuilder(old_ctx, old_loc)
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

                    new_op = ttir_op(
                        result, in0, min_tensor, max_tensor, loc=old_op.location
                    )
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    min_tensor_golden = self._get_golden_tensor(old_op.min)
                    max_tensor_golden = self._get_golden_tensor(old_op.max)
                    op_golden_function = get_golden_function(ttir_op)
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

    ############### ttir.ReduceOrOp ###############

    @tag(ttir.ReduceOrOp)
    def reduce_or(
        self,
        in0: Operand,
        dim_arg: List[int] = None,
        keep_dim: bool = True,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.reduce_or)

        if dim_arg is None:
            dim_arg = list(range(len(self.get_shape(in0))))
        dim_arg_attr = ArrayAttr.get(
            [IntegerAttr.get(IntegerType.get_signless(32), d) for d in dim_arg]
        )
        keep_dim_attr = BoolAttr.get(keep_dim)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, dim_arg_attr, keep_dim_attr, mlir_output_type
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            keep_dim_attr,
            dim_arg=dim_arg_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.ReduceOrOp)
    def reduce_or_parser(
        self,
        old_op: ttir.ReduceOrOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.reduce_or_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type
        keep_dim_attr = old_op.keep_dim
        dim_arg_attr = old_op.dim_arg

        new_op = ttir_op(
            result,
            in0,
            keep_dim_attr,
            dim_arg=dim_arg_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, dim_arg_attr, keep_dim_attr, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.ReduceOrOp)
    def reduce_or_split(
        self,
        old_op: ttir.ReduceOrOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.reduce_or_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            reduce_or_module = Module.create()
            reduce_or_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(reduce_or_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="reduce_or_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        old_op.keep_dim,
                        dim_arg=old_op.dim_arg,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    reduce_or_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    reduce_or_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                reduce_or_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return reduce_or_module, reduce_or_builder

    ############### ttir.MaxOp ###############

    @tag(ttir.MaxOp)
    def max(
        self,
        in0: Operand,
        dim_arg: List[int] = None,
        keep_dim: bool = True,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.max)

        if dim_arg is None:
            dim_arg = list(range(len(self.get_shape(in0))))
        dim_arg_attr = ArrayAttr.get(
            [IntegerAttr.get(IntegerType.get_signless(32), d) for d in dim_arg]
        )
        keep_dim_attr = BoolAttr.get(keep_dim)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, dim_arg_attr, keep_dim_attr, mlir_output_type
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            keep_dim_attr,
            dim_arg=dim_arg_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.MaxOp)
    def max_parser(
        self,
        old_op: ttir.MaxOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.max_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type
        keep_dim_attr = old_op.keep_dim
        dim_arg_attr = old_op.dim_arg

        new_op = ttir_op(
            result,
            in0,
            keep_dim_attr,
            dim_arg=dim_arg_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, dim_arg_attr, keep_dim_attr, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.MaxOp)
    def max_split(
        self,
        old_op: ttir.MaxOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.max_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            max_module = Module.create()
            max_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(max_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="max_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        old_op.keep_dim,
                        dim_arg=old_op.dim_arg,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    max_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    max_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                max_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return max_module, max_builder

    ############### ttir.LogicalNotOp ###############

    @tag(ttir.LogicalNotOp)
    def logical_not(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.logical_not)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.LogicalNotOp)
    def logical_not_parser(
        self,
        old_op: ttir.LogicalNotOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.logical_not_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.LogicalNotOp)
    def logical_not_split(
        self,
        old_op: ttir.LogicalNotOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.logical_not_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            logical_not_module = Module.create()
            logical_not_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(logical_not_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_not_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    logical_not_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    logical_not_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                logical_not_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return logical_not_module, logical_not_builder

    ############### ttir.LogOp ###############

    @tag(ttir.LogOp)
    def log(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.log)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.LogOp)
    def log_parser(
        self,
        old_op: ttir.LogOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.log_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.LogOp)
    def log_split(
        self,
        old_op: ttir.LogOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.log_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            log_module = Module.create()
            log_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(log_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="log_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    log_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
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

    ############### ttir.GreaterThanOp ###############

    @tag(ttir.GreaterThanOp)
    def gt(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.gt)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.GreaterThanOp)
    def gt_parser(
        self,
        old_op: ttir.GreaterThanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.gt_parser)
        in0 = global_dict[old_op.lhs]
        in1 = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            in1,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.GreaterThanOp)
    def gt_split(
        self,
        old_op: ttir.GreaterThanOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.gt_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            gt_module = Module.create()
            gt_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(gt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="gt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, in1, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    gt_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    gt_builder._set_golden_tensor(in0, input0)
                    gt_builder._set_golden_tensor(in1, input1)
                    ordered_inputs.extend([in0, in1])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                gt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return gt_module, gt_builder

    ############### ttir.BatchNormInferenceOp ###############

    @tag(ttir.BatchNormInferenceOp)
    def batch_norm_inference(
        self,
        in0: Operand,
        scale: Operand,
        offset: Operand,
        mean: Operand,
        variance: Operand,
        epsilon: float = 1e-5,
        dimension: int = 1,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.batch_norm_inference)
        epsilon_attr = FloatAttr.get_f32(epsilon)
        dimension_attr = IntegerAttr.get(IntegerType.get_signless(32), dimension)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        scale0 = self._get_golden_tensor(scale)
        offset0 = self._get_golden_tensor(offset)
        mean0 = self._get_golden_tensor(mean)
        variance0 = self._get_golden_tensor(variance)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            scale0,
            offset0,
            mean0,
            variance0,
            epsilon_attr,
            dimension_attr,
            mlir_output_type,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            scale,
            offset,
            mean,
            variance,
            epsilon_attr,
            dimension_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.BatchNormInferenceOp)
    def batch_norm_inference_parser(
        self,
        old_op: ttir.BatchNormInferenceOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.batch_norm_inference_parser)
        in0 = global_dict[old_op.operand]
        scale = global_dict[old_op.scale]
        offset = global_dict[old_op.offset]
        mean = global_dict[old_op.mean]
        variance = global_dict[old_op.variance]
        epsilon_attr = old_op.epsilon
        dimension_attr = old_op.dimension
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            scale,
            offset,
            mean,
            variance,
            epsilon_attr,
            dimension_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        scale0 = self._get_golden_tensor(scale)
        offset0 = self._get_golden_tensor(offset)
        mean0 = self._get_golden_tensor(mean)
        variance0 = self._get_golden_tensor(variance)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            scale0,
            offset0,
            mean0,
            variance0,
            epsilon_attr,
            dimension_attr,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.BatchNormInferenceOp)
    def batch_norm_inference_split(
        self,
        old_op: ttir.BatchNormInferenceOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.batch_norm_inference_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            batch_norm_inference_module = Module.create()
            batch_norm_inference_builder = TTIRBuilder(old_ctx, old_loc)
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
                    in0 = inputs[0]
                    scale = inputs[1]
                    offset = inputs[2]
                    mean = inputs[3]
                    variance = inputs[4]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        scale,
                        offset,
                        mean,
                        variance,
                        old_op.epsilon,
                        old_op.dimension,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.operand)
                    scale0 = self._get_golden_tensor(old_op.scale)
                    offset0 = self._get_golden_tensor(old_op.offset)
                    mean0 = self._get_golden_tensor(old_op.mean)
                    variance0 = self._get_golden_tensor(old_op.variance)

                    op_golden_function = get_golden_function(ttir_op)
                    golden_output = op_golden_function(
                        input0,
                        scale0,
                        offset0,
                        mean0,
                        variance0,
                        old_op.epsilon,
                        old_op.dimension,
                        result.element_type,
                    )
                    batch_norm_inference_builder._set_golden_tensor(
                        new_op_result, golden_output
                    )
                    batch_norm_inference_builder._set_golden_tensor(in0, input0)
                    batch_norm_inference_builder._set_golden_tensor(scale, scale0)
                    batch_norm_inference_builder._set_golden_tensor(offset, offset0)
                    batch_norm_inference_builder._set_golden_tensor(mean, mean0)
                    batch_norm_inference_builder._set_golden_tensor(variance, variance0)
                    ordered_inputs.extend([in0, scale, offset, mean, variance])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                batch_norm_inference_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return batch_norm_inference_module, batch_norm_inference_builder

    ############### ttir.BatchNormTrainingOp ###############

    @tag(ttir.BatchNormTrainingOp)
    def batch_norm_training(
        self,
        in0: Operand,
        scale: Operand,
        offset: Operand,
        running_mean: Operand,
        running_variance: Operand,
        epsilon: float = 1e-5,
        dimension: int = 1,
        momentum: float = 0.1,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> Tuple[OpResult, OpResult, OpResult]:
        ttir_op = self.get_opview_from_method(TTIRBuilder.batch_norm_training)
        epsilon_attr = FloatAttr.get_f32(epsilon)
        dimension_attr = IntegerAttr.get(IntegerType.get_signless(32), dimension)
        momentum_attr = FloatAttr.get_f32(momentum)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        scale0 = self._get_golden_tensor(scale)
        offset0 = self._get_golden_tensor(offset)
        running_mean0 = self._get_golden_tensor(running_mean)
        running_variance0 = self._get_golden_tensor(running_variance)
        op_golden_function = get_golden_function(ttir_op)
        golden_output, golden_batch_mean, golden_batch_variance = op_golden_function(
            input0,
            scale0,
            offset0,
            running_mean0,
            running_variance0,
            epsilon_attr,
            dimension_attr,
            momentum_attr,
            mlir_output_type,
            mlir_output_type,
            mlir_output_type,
        )

        result_type = self._create_ranked_tensor_type(
            golden_output.shape, mlir_output_type
        )
        batch_mean_type = self._create_ranked_tensor_type(
            golden_batch_mean.shape, mlir_output_type
        )
        batch_variance_type = self._create_ranked_tensor_type(
            golden_batch_variance.shape, mlir_output_type
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result_type,
            batch_mean_type,
            batch_variance_type,
            in0,
            scale,
            offset,
            running_mean,
            running_variance,
            epsilon_attr,
            dimension_attr,
            momentum_attr,
            loc=loc,
        )
        op_result = op.result
        op_batch_mean = op.batch_mean
        op_batch_variance = op.batch_variance

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)
        self._set_golden_tensor(op_batch_mean, golden_batch_mean)
        self._set_golden_tensor(op_batch_variance, golden_batch_variance)

        return op_result, op_batch_mean, op_batch_variance

    @parse(ttir.BatchNormTrainingOp)
    def batch_norm_training_parser(
        self,
        old_op: ttir.BatchNormTrainingOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.batch_norm_training_parser)
        in0 = global_dict[old_op.operand]
        scale = global_dict[old_op.scale]
        offset = global_dict[old_op.offset]
        running_mean = global_dict[old_op.running_mean]
        running_variance = global_dict[old_op.running_variance]
        epsilon_attr = old_op.epsilon
        dimension_attr = old_op.dimension
        momentum_attr = old_op.momentum
        result_type = old_op.result.type
        batch_mean_type = old_op.batch_mean.type
        batch_variance_type = old_op.batch_variance.type

        new_op = ttir_op(
            result_type,
            batch_mean_type,
            batch_variance_type,
            in0,
            scale,
            offset,
            running_mean,
            running_variance,
            epsilon_attr,
            dimension_attr,
            momentum_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result
        new_op_batch_mean = new_op.batch_mean
        new_op_batch_variance = new_op.batch_variance

        input0 = self._get_golden_tensor(in0)
        scale0 = self._get_golden_tensor(scale)
        offset0 = self._get_golden_tensor(offset)
        running_mean0 = self._get_golden_tensor(running_mean)
        running_variance0 = self._get_golden_tensor(running_variance)
        op_golden_function = get_golden_function(ttir_op)
        (golden_output, golden_batch_mean, golden_batch_variance,) = op_golden_function(
            input0,
            scale0,
            offset0,
            running_mean0,
            running_variance0,
            epsilon_attr,
            dimension_attr,
            momentum_attr,
            result_type.element_type,
            batch_mean_type.element_type,
            batch_variance_type.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)
        self._set_golden_tensor(new_op_batch_mean, golden_batch_mean)
        self._set_golden_tensor(new_op_batch_variance, golden_batch_variance)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        op_map_dictionary[old_op.batch_mean] = new_op_batch_mean
        op_map_dictionary[old_op.batch_variance] = new_op_batch_variance
        return new_op, op_map_dictionary

    @split(ttir.BatchNormTrainingOp)
    def batch_norm_training_split(
        self,
        old_op: ttir.BatchNormTrainingOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.batch_norm_training_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            batch_norm_training_module = Module.create()
            batch_norm_training_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [
                old_op.operand.type,
                old_op.scale.type,
                old_op.offset.type,
                old_op.running_mean.type,
                old_op.running_variance.type,
            ]

            with InsertionPoint(batch_norm_training_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="batch_norm_training_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    scale = inputs[1]
                    offset = inputs[2]
                    running_mean = inputs[3]
                    running_variance = inputs[4]
                    result_type = old_op.result.type
                    batch_mean_type = old_op.batch_mean.type
                    batch_variance_type = old_op.batch_variance.type

                    new_op = ttir_op(
                        result_type,
                        batch_mean_type,
                        batch_variance_type,
                        in0,
                        scale,
                        offset,
                        running_mean,
                        running_variance,
                        old_op.epsilon,
                        old_op.dimension,
                        old_op.momentum,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result
                    new_op_batch_mean = new_op.batch_mean
                    new_op_batch_variance = new_op.batch_variance

                    input0 = self._get_golden_tensor(old_op.operand)
                    scale0 = self._get_golden_tensor(old_op.scale)
                    offset0 = self._get_golden_tensor(old_op.offset)
                    running_mean0 = self._get_golden_tensor(old_op.running_mean)
                    running_variance0 = self._get_golden_tensor(old_op.running_variance)

                    op_golden_function = get_golden_function(ttir_op)
                    (
                        golden_output,
                        golden_batch_mean,
                        golden_batch_variance,
                    ) = op_golden_function(
                        input0,
                        scale0,
                        offset0,
                        running_mean0,
                        running_variance0,
                        old_op.epsilon,
                        old_op.dimension,
                        old_op.momentum,
                        result_type.element_type,
                        batch_mean_type.element_type,
                        batch_variance_type.element_type,
                    )
                    batch_norm_training_builder._set_golden_tensor(
                        new_op_result, golden_output
                    )
                    batch_norm_training_builder._set_golden_tensor(
                        new_op_batch_mean, golden_batch_mean
                    )
                    batch_norm_training_builder._set_golden_tensor(
                        new_op_batch_variance, golden_batch_variance
                    )
                    batch_norm_training_builder._set_golden_tensor(in0, input0)
                    batch_norm_training_builder._set_golden_tensor(scale, scale0)
                    batch_norm_training_builder._set_golden_tensor(offset, offset0)
                    batch_norm_training_builder._set_golden_tensor(
                        running_mean, running_mean0
                    )
                    batch_norm_training_builder._set_golden_tensor(
                        running_variance, running_variance0
                    )
                    ordered_inputs.extend(
                        [in0, scale, offset, running_mean, running_variance]
                    )
                    ordered_outputs.extend(
                        [new_op_result, new_op_batch_mean, new_op_batch_variance]
                    )

                    return new_op

                new_func_op = decorated_func.func_op
                batch_norm_training_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return batch_norm_training_module, batch_norm_training_builder

    ############### ttir.ConstantOp ###############

    @tag(ttir.ConstantOp)
    def constant(
        self,
        tensor: torch.Tensor,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.constant)
        value_attr = DenseElementsAttr.get(tensor.numpy())
        result = self._create_ranked_tensor_type(
            tensor.shape, self._get_type_from_torch_dtype(tensor.dtype)
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            value_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        op_golden_function = get_golden_function(ttir_op)
        mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
        golden_output = op_golden_function(value_attr, mesh_shape_attr)
        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.ConstantOp)
    def constant_parser(
        self,
        old_op: ttir.ConstantOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.constant_parser)

        with old_op.context, old_op.location:
            value_attr = old_op.value
            result = old_op.result.type

            new_op = ttir_op(
                result,
                value_attr,
                loc=old_op.location,
            )
            new_op_result = new_op.result

            op_golden_function = get_golden_function(ttir_op)
            mesh_shape_attr = DenseI32ArrayAttr.get(self._mesh_shape)
            golden_output = op_golden_function(value_attr, mesh_shape_attr)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.ConstantOp)
    def constant_split(
        self,
        old_op: Operation,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.constant_split)

        old_context = old_op.context
        old_location = Location.unknown(old_context)

        with old_context, old_location:
            constant_module = Module.create()
            constant_builder = TTIRBuilder(old_context, old_location)
            op_input_types = []

            with InsertionPoint(constant_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="constant_module")
                def decorated_func(*inputs):
                    value_attr = old_op.value
                    result = old_op.result.type

                    new_op = ttir_op(result, value_attr, loc=old_op.location)
                    new_op_result = new_op.result

                    constant_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                constant_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return constant_module, constant_builder

    ############### ttir.PadOp ###############

    @tag(ttir.PadOp)
    def pad(
        self,
        in0: Operand,
        padding: List[int],
        value: float = 0.0,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.pad)
        padding_attr = DenseI32ArrayAttr.get(padding)
        value_attr = FloatAttr.get(F32Type.get(), value)

        output_shape = []
        for i in range(len(padding_attr) // 2):
            output_shape.append(
                self.get_shape(in0)[i] + padding_attr[2 * i] + padding_attr[2 * i + 1]
            )

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        result = self._create_ranked_tensor_type(output_shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            padding_attr,
            value_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        input = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input, padding_attr, value_attr, mlir_output_type
        )
        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.PadOp)
    def pad_parser(
        self,
        old_op: ttir.PadOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.pad_parser)
        in0 = global_dict[old_op.input]
        padding_attr = old_op.padding
        value_attr = old_op.value
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            padding_attr,
            value_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, padding_attr, value_attr, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.PadOp)
    def pad_split(
        self,
        old_op: ttir.PadOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.pad_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            pad_module = Module.create()
            pad_builder = TTIRBuilder(old_context, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(pad_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="pad_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    padding_attr = old_op.padding
                    value_attr = old_op.value
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        padding_attr,
                        value_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    pad_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    pad_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                pad_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return pad_module, pad_builder

    ############### ttir.DotGeneralOp ###############

    @tag(ttir.DotGeneralOp)
    def dot_general(
        self,
        in0: Operand,
        in1: Operand,
        batch_dims_lhs: List[int],
        contract_dims_lhs: List[int],
        batch_dims_rhs: List[int],
        contract_dims_rhs: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.dot_general)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        batch_dims_lhs_attr = DenseI64ArrayAttr.get(batch_dims_lhs)
        contract_dims_lhs_attr = DenseI64ArrayAttr.get(contract_dims_lhs)
        batch_dims_rhs_attr = DenseI64ArrayAttr.get(batch_dims_rhs)
        contract_dims_rhs_attr = DenseI64ArrayAttr.get(contract_dims_rhs)

        lhs = self._get_golden_tensor(in0)
        rhs = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            lhs,
            rhs,
            batch_dims_lhs_attr,
            contract_dims_lhs_attr,
            batch_dims_rhs_attr,
            contract_dims_rhs_attr,
            mlir_output_type,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            in1,
            batch_dims_lhs_attr,
            contract_dims_lhs_attr,
            batch_dims_rhs_attr,
            contract_dims_rhs_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.DotGeneralOp)
    def dot_general_parser(
        self,
        old_op: ttir.DotGeneralOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.dot_general_parser)
        in0 = global_dict[old_op.lhs]
        in1 = global_dict[old_op.rhs]
        batch_dims_lhs_attr = old_op.batch_dims_lhs
        contract_dims_lhs_attr = old_op.contract_dims_lhs
        batch_dims_rhs_attr = old_op.batch_dims_rhs
        contract_dims_rhs_attr = old_op.contract_dims_rhs
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            in1,
            batch_dims_lhs_attr,
            contract_dims_lhs_attr,
            batch_dims_rhs_attr,
            contract_dims_rhs_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            input1,
            batch_dims_lhs_attr,
            contract_dims_lhs_attr,
            batch_dims_rhs_attr,
            contract_dims_rhs_attr,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.DotGeneralOp)
    def dot_general_split(
        self,
        old_op: ttir.DotGeneralOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.dot_general_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            dot_general_module = Module.create()
            dot_general_builder = TTIRBuilder(old_context, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(dot_general_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="dot_general_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    batch_dims_lhs_attr = old_op.batch_dims_lhs
                    contract_dims_lhs_attr = old_op.contract_dims_lhs
                    batch_dims_rhs_attr = old_op.batch_dims_rhs
                    contract_dims_rhs_attr = old_op.contract_dims_rhs
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        in1,
                        batch_dims_lhs_attr,
                        contract_dims_lhs_attr,
                        batch_dims_rhs_attr,
                        contract_dims_rhs_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    dot_general_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    dot_general_builder._set_golden_tensor(in0, input0)
                    dot_general_builder._set_golden_tensor(in1, input1)
                    ordered_inputs.extend([in0, in1])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                dot_general_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return dot_general_module, dot_general_builder

    ############### ttir.PermuteOp ###############

    @tag(ttir.PermuteOp)
    def permute(
        self,
        in0: Operand,
        permutation: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.permute)
        permutation_attr = DenseI64ArrayAttr.get(permutation)

        output_shape = []
        in0_shape = self.get_shape(in0)
        for i in permutation:
            output_shape.append(in0_shape[i])

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        result = self._create_ranked_tensor_type(output_shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            permutation_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        input = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input, permutation_attr, mlir_output_type)
        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.PermuteOp)
    def permute_parser(
        self,
        old_op: ttir.PermuteOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.permute_parser)
        in0 = global_dict[old_op.input]
        permutation_attr = old_op.permutation
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            permutation_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, permutation_attr, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.PermuteOp)
    def permute_split(
        self,
        old_op: ttir.PermuteOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.permute_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            permute_module = Module.create()
            permute_builder = TTIRBuilder(old_context, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(permute_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="permute_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    permutation_attr = old_op.permutation
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, permutation_attr, loc=old_op.location)
                    new_op_result = new_op.result

                    permute_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    permute_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                permute_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return permute_module, permute_builder

    ############### ttir.BroadcastOp ###############

    @tag(ttir.BroadcastOp)
    def broadcast(
        self,
        in0: Operand,
        broadcast_dimensions: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.broadcast)
        broadcast_dimensions_attr = DenseI64ArrayAttr.get(broadcast_dimensions)

        output_shape = []
        in0_shape = self.get_shape(in0)
        for i in range(len(broadcast_dimensions)):
            if broadcast_dimensions[i] != 1:
                output_shape.append(broadcast_dimensions[i])
            else:
                output_shape.append(in0_shape[i])

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        result = self._create_ranked_tensor_type(output_shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            broadcast_dimensions_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        input = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input, broadcast_dimensions_attr, mlir_output_type
        )
        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.BroadcastOp)
    def broadcast_parser(
        self,
        old_op: ttir.BroadcastOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.broadcast_parser)
        in0 = global_dict[old_op.input]
        broadcast_dimensions_attr = old_op.broadcast_dimensions
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            broadcast_dimensions_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, broadcast_dimensions_attr, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.BroadcastOp)
    def broadcast_split(
        self,
        old_op: ttir.BroadcastOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.broadcast_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            broadcast_module = Module.create()
            broadcast_builder = TTIRBuilder(old_context, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(broadcast_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="broadcast_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    broadcast_dimensions_attr = old_op.broadcast_dimensions
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        broadcast_dimensions_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    broadcast_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    input0 = self._get_golden_tensor(old_op.input)
                    broadcast_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                broadcast_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return broadcast_module, broadcast_builder

    ############### ttir.ReshapeOp ###############

    @tag(ttir.ReshapeOp)
    def reshape(
        self,
        in0: Operand,
        shape: List[int],
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.reshape)
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

        op = ttir_op(
            result,
            in0,
            shape_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        input = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input, shape_attr, mlir_output_type)
        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.ReshapeOp)
    def reshape_parser(
        self,
        old_op: ttir.ReshapeOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.reshape_parser)
        in0 = global_dict[old_op.input]
        shape_attr = old_op.shape
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            shape_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, shape_attr, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.ReshapeOp)
    def reshape_split(
        self,
        old_op: ttir.ReshapeOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.reshape_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            reshape_module = Module.create()
            reshape_builder = TTIRBuilder(old_context, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(reshape_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="reshape_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    shape_attr = old_op.shape
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, shape_attr, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    reshape_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
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

    ############### ttir.ConcatenateHeadsOp ###############

    @tag(ttir.ConcatenateHeadsOp)
    def concatenate_heads(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.concatenate_heads)

        input_shape = self.get_shape(in0)
        # Input: [batch, num_heads, seq_len, head_dim]
        # Output: [batch, seq_len, num_heads * head_dim]
        assert len(input_shape) == 4, f"Expected 4D input, got {len(input_shape)}D"
        batch, num_heads, seq_len, head_dim = input_shape
        output_shape = [batch, seq_len, num_heads * head_dim]

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        result = self._create_ranked_tensor_type(output_shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            loc=loc,
        )
        op_result = op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.ConcatenateHeadsOp)
    def concatenate_heads_parser(
        self,
        old_op: ttir.ConcatenateHeadsOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.concatenate_heads_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result

        return new_op, op_map_dictionary

    @split(ttir.ConcatenateHeadsOp)
    def concatenate_heads_split(
        self,
        old_op: ttir.ConcatenateHeadsOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.concatenate_heads_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            concatenate_heads_module = Module.create()
            concatenate_heads_builder = TTIRBuilder(old_context, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(concatenate_heads_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="concatenate_heads_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    op_golden_function = get_golden_function(ttir_op)
                    input0 = self._get_golden_tensor(old_op.input)
                    golden_output = op_golden_function(input0, result.element_type)
                    concatenate_heads_builder._set_golden_tensor(
                        new_op_result, golden_output
                    )
                    concatenate_heads_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                concatenate_heads_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return concatenate_heads_module, concatenate_heads_builder

    ############### ttir.MaximumOp ###############

    @tag(ttir.MaximumOp)
    def maximum(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.maximum)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.MaximumOp)
    def maximum_parser(
        self,
        old_op: ttir.MaximumOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.maximum_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.MaximumOp)
    def maximum_split(
        self,
        old_op: ttir.MaximumOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.maximum_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            maximum_module = Module.create()
            maximum_builder = TTIRBuilder(old_context, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(maximum_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="maximum_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    maximum_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    maximum_builder._set_golden_tensor(lhs, input0)
                    maximum_builder._set_golden_tensor(rhs, input1)
                    ordered_inputs.extend([lhs, rhs])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                maximum_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return maximum_module, maximum_builder

    ############### ttir.MultiplyOp ###############

    @tag(ttir.MultiplyOp)
    def multiply(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.multiply)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.MultiplyOp)
    def multiply_parser(
        self,
        old_op: ttir.MultiplyOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.multiply_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.MultiplyOp)
    def multiply_split(
        self,
        old_op: ttir.MultiplyOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.multiply_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            multiply_module = Module.create()
            multiply_builder = TTIRBuilder(old_context, old_loc)
            op_input_types = [old_op.lhs.type, old_op.rhs.type]

            with InsertionPoint(multiply_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="multiply_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    multiply_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    multiply_builder._set_golden_tensor(lhs, input0)
                    multiply_builder._set_golden_tensor(rhs, input1)
                    ordered_inputs.extend([lhs, rhs])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                multiply_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return multiply_module, multiply_builder

    # class TTIR_ElementwiseBinaryOp

    ############### ttir.EqualOp ###############

    @tag(ttir.EqualOp)
    def eq(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.eq)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.EqualOp)
    def eq_parser(
        self,
        old_op: ttir.EqualOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.eq_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.EqualOp)
    def eq_split(
        self,
        old_op: ttir.EqualOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.eq_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            eq_module = Module.create()
            eq_builder = TTIRBuilder(old_ctx, old_loc)
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

                    new_op = ttir_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    eq_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
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

    ############### ttir.SumOp ###############

    @tag(ttir.SumOp)
    def sum(
        self,
        in0: Operand,
        dim_arg: List[int] = None,
        keep_dim: bool = True,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.sum)

        if dim_arg is None:
            dim_arg = list(range(len(self.get_shape(in0))))
        dim_arg_attr = ArrayAttr.get(
            [IntegerAttr.get(IntegerType.get_signless(32), d) for d in dim_arg]
        )
        keep_dim_attr = BoolAttr.get(keep_dim)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0, dim_arg_attr, keep_dim_attr, mlir_output_type
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            keep_dim_attr,
            dim_arg=dim_arg_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.SumOp)
    def sum_parser(
        self,
        old_op: ttir.SumOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.sum_parser)
        in0 = global_dict[old_op.input]
        dim_arg_attr = old_op.dim_arg
        keep_dim_attr = old_op.keep_dim
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            keep_dim_attr,
            dim_arg=dim_arg_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            dim_arg_attr,
            keep_dim_attr,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.SumOp)
    def sum_split(
        self,
        old_op: ttir.SumOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.sum_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            sum_module = Module.create()
            sum_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(sum_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sum_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    keep_dim_attr = old_op.keep_dim
                    dim_arg_attr = old_op.dim_arg

                    new_op = ttir_op(
                        result,
                        in0,
                        keep_dim_attr,
                        dim_arg=dim_arg_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    sum_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    sum_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                sum_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sum_module, sum_builder

    ############### ttir.AddOp ###############

    @tag(ttir.AddOp)
    def add(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.add)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.AddOp)
    def add_parser(
        self,
        old_op: ttir.AddOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.add_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.AddOp)
    def add_split(
        self,
        old_op: ttir.AddOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.add_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            add_module = Module.create()
            add_builder = TTIRBuilder(old_context, old_loc)
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

                    new_op = ttir_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    add_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
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

    ############### ttir.SigmoidOp ###############

    @tag(ttir.SigmoidOp)
    def sigmoid(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.sigmoid)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.SigmoidOp)
    def sigmoid_parser(
        self,
        old_op: ttir.SigmoidOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.sigmoid_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.SigmoidOp)
    def sigmoid_split(
        self,
        old_op: ttir.SigmoidOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.sigmoid_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            sigmoid_module = Module.create()
            sigmoid_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(sigmoid_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="sigmoid_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    sigmoid_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    sigmoid_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                sigmoid_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return sigmoid_module, sigmoid_builder

    ################ ttir.HardsigmoidOp ###############

    @tag(ttir.HardsigmoidOp)
    def hardsigmoid(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.hardsigmoid)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.HardsigmoidOp)
    def hardsigmoid_parser(
        self,
        old_op: ttir.HardsigmoidOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.hardsigmoid_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.HardsigmoidOp)
    def hardsigmoid_split(
        self,
        old_op: ttir.HardsigmoidOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.hardsigmoid_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            hardsigmoid_module = Module.create()
            hardsigmoid_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(hardsigmoid_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="hardsigmoid_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    hardsigmoid_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    hardsigmoid_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                hardsigmoid_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return hardsigmoid_module, hardsigmoid_builder

    ################ ttir.SubtractOp ###############

    @tag(ttir.SubtractOp)
    def subtract(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.subtract)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.SubtractOp)
    def subtract_parser(
        self,
        old_op: ttir.SubtractOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.subtract_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.SubtractOp)
    def subtract_split(
        self,
        old_op: ttir.SubtractOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.subtract_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            subtract_module = Module.create()
            subtract_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(subtract_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="subtract_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    subtract_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    subtract_builder._set_golden_tensor(lhs, input0)
                    subtract_builder._set_golden_tensor(rhs, input1)
                    ordered_inputs.extend([lhs, rhs])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                subtract_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return subtract_module, subtract_builder

    ############### ttir.TanhOp ###############

    @tag(ttir.TanhOp)
    def tanh(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.tanh)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.TanhOp)
    def tanh_parser(
        self,
        old_op: ttir.TanhOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.tanh_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.TanhOp)
    def tanh_split(
        self,
        old_op: ttir.TanhOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.tanh_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            tanh_module = Module.create()
            tanh_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(tanh_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="tanh_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    tanh_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
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

    ############### ttir.RsqrtOp ###############

    @tag(ttir.RsqrtOp)
    def rsqrt(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.rsqrt)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.RsqrtOp)
    def rsqrt_parser(
        self,
        old_op: ttir.RsqrtOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.rsqrt_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.RsqrtOp)
    def rsqrt_split(
        self,
        old_op: ttir.RsqrtOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.rsqrt_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            rsqrt_module = Module.create()
            rsqrt_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(rsqrt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="rsqrt_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    rsqrt_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
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

    ############### ttir.NegOp ###############

    @tag(ttir.NegOp)
    def neg(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.neg)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.NegOp)
    def neg_parser(
        self,
        old_op: ttir.NegOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.neg_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.NegOp)
    def neg_split(
        self,
        old_op: ttir.NegOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.neg_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            neg_module = Module.create()
            neg_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(neg_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="neg_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    neg_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
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

    ############### ttir.NotEqualOp ###############

    @tag(ttir.NotEqualOp)
    def ne(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.ne)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.NotEqualOp)
    def ne_parser(
        self,
        old_op: ttir.NotEqualOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.ne_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.NotEqualOp)
    def ne_split(
        self,
        old_op: ttir.NotEqualOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.ne_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            ne_module = Module.create()
            ne_builder = TTIRBuilder(old_ctx, old_loc)
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

                    new_op = ttir_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    ne_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
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

    ############### ttir.WhereOp ###############

    @tag(ttir.WhereOp)
    def where(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.where)

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
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(condition, input1, input2, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.WhereOp)
    def where_parser(
        self,
        old_op: ttir.WhereOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.where_parser)
        first = global_dict[old_op.first]
        second = global_dict[old_op.second]
        third = global_dict[old_op.third]
        result = old_op.result.type

        new_op = ttir_op(
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
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            condition, input1, input2, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.WhereOp)
    def where_split(
        self,
        old_op: ttir.WhereOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.where_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            where_module = Module.create()
            where_builder = TTIRBuilder(old_ctx, old_loc)
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

                    new_op = ttir_op(result, first, second, third, loc=old_op.location)
                    new_op_result = new_op.result

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
                    where_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
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

    ############### ttir.AbsOp ###############

    @tag(ttir.AbsOp)
    def abs(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.abs)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.AbsOp)
    def abs_parser(
        self,
        old_op: ttir.AbsOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.abs_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.AbsOp)
    def abs_split(
        self,
        old_op: ttir.AbsOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.abs_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            abs_module = Module.create()
            abs_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(abs_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="abs_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    abs_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    abs_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                abs_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return abs_module, abs_builder

    ############### ttir.ErfOp ###############

    @tag(ttir.ErfOp)
    def erf(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.erf)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.ErfOp)
    def erf_parser(
        self,
        old_op: ttir.ErfOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.erf_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.ErfOp)
    def erf_split(
        self,
        old_op: ttir.ErfOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.erf_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            erf_module = Module.create()
            erf_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(erf_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="erf_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    erf_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    erf_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                erf_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return erf_module, erf_builder

    ############### ttir.FloorOp ###############

    @tag(ttir.FloorOp)
    def floor(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.floor)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.FloorOp)
    def floor_parser(
        self,
        old_op: ttir.FloorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.floor_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.FloorOp)
    def floor_split(
        self,
        old_op: ttir.FloorOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.floor_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            floor_module = Module.create()
            floor_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(floor_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="floor_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    floor_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    floor_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                floor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return floor_module, floor_builder

    ############### ttir.TypecastOp ###############

    @tag(ttir.TypecastOp)
    def typecast(
        self,
        in0: Operand,
        output_type: torch.dtype,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.typecast)
        output_mlir_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, output_mlir_type)
        result = self._create_ranked_tensor_type(golden_output.shape, output_mlir_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.TypecastOp)
    def typecast_parser(
        self,
        old_op: ttir.TypecastOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.typecast_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.TypecastOp)
    def typecast_split(
        self,
        old_op: ttir.TypecastOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.typecast_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            typecast_module = Module.create()
            typecast_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(typecast_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="typecast_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    output_dtype = self._get_torch_dtype_from_type(result.element_type)
                    typecast_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    typecast_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                typecast_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return typecast_module, typecast_builder

    ############### ttir.ExpOp ###############

    @tag(ttir.ExpOp)
    def exp(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.exp)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.ExpOp)
    def exp_parser(
        self,
        old_op: ttir.ExpOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.exp_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.ExpOp)
    def exp_split(
        self,
        old_op: ttir.ExpOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.exp_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            exp_module = Module.create()
            exp_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(exp_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="exp_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    exp_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    exp_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                exp_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return exp_module, exp_builder

    ############### ttir.DivOp ###############

    @tag(ttir.DivOp)
    def div(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.div)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.DivOp)
    def div_parser(
        self,
        old_op: ttir.DivOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.div_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            lhs,
            rhs,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(lhs)
        input1 = self._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, input1, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.DivOp)
    def div_split(
        self,
        old_op: ttir.DivOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.div_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            div_module = Module.create()
            div_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(div_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="div_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.lhs)
                    input1 = self._get_golden_tensor(old_op.rhs)
                    div_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
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

    ############### ttir.SliceStaticOp ###############

    @tag(ttir.SliceStaticOp)
    def slice(
        self,
        in0: Operand,
        begins: List[int],
        ends: List[int],
        step: List[int] = None,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: List[str] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.slice)

        if step is None:
            step = [1] * len(begins)

        begins_int_attrs = [
            IntegerAttr.get(IntegerType.get_signless(32), b) for b in begins
        ]
        ends_int_attrs = [
            IntegerAttr.get(IntegerType.get_signless(32), e) for e in ends
        ]
        step_int_attrs = [
            IntegerAttr.get(IntegerType.get_signless(32), s) for s in step
        ]

        begins_attr = ArrayAttr.get(begins_int_attrs, self._ctx)
        ends_attr = ArrayAttr.get(ends_int_attrs, self._ctx)
        step_attr = ArrayAttr.get(step_int_attrs, self._ctx)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            begins=begins_attr,
            ends=ends_attr,
            step=step_attr,
            output_type_mlir=mlir_output_type,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            begins_attr,
            ends_attr,
            step_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.SliceStaticOp)
    def slice_parser(
        self,
        old_op: ttir.SliceStaticOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.slice_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type
        begins_attr = old_op.begins
        ends_attr = old_op.ends
        step_attr = old_op.step

        new_op = ttir_op(
            result,
            in0,
            begins_attr,
            ends_attr,
            step_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            begins=begins_attr,
            ends=ends_attr,
            step=step_attr,
            output_type_mlir=result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.SliceStaticOp)
    def slice_split(
        self,
        old_op: ttir.SliceStaticOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.slice_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            slice_module = Module.create()
            slice_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(slice_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="slice_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type
                    begins_attr = old_op.begins
                    ends_attr = old_op.ends
                    step_attr = old_op.step

                    new_op = ttir_op(
                        result,
                        in0,
                        begins_attr,
                        ends_attr,
                        step_attr,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    slice_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
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

    ############### ttir.EmbeddingBackwardOp ###############

    @tag(ttir.EmbeddingBackwardOp)
    def embedding_backward(
        self,
        input: Operand,
        weight: Operand,
        in_gradient: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.embedding_backward)

        if output_type is None:
            mlir_output_type = self.get_type(weight)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input_tensor = self._get_golden_tensor(input)
        weight_tensor = self._get_golden_tensor(weight)
        in_gradient_tensor = self._get_golden_tensor(in_gradient)

        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input_tensor, weight_tensor, in_gradient_tensor, mlir_output_type
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            input,
            weight,
            in_gradient,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.EmbeddingBackwardOp)
    def embedding_backward_parser(
        self,
        old_op: ttir.EmbeddingBackwardOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.embedding_backward_parser)
        input = global_dict[old_op.input]
        weight = global_dict[old_op.weight]
        in_gradient = global_dict[old_op.in_gradient]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            input,
            weight,
            in_gradient,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input_tensor = self._get_golden_tensor(input)
        weight_tensor = self._get_golden_tensor(weight)
        in_gradient_tensor = self._get_golden_tensor(in_gradient)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input_tensor, weight_tensor, in_gradient_tensor, result.element_type
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.EmbeddingBackwardOp)
    def embedding_backward_split(
        self,
        old_op: ttir.EmbeddingBackwardOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.embedding_backward_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            embedding_backward_module = Module.create()
            embedding_backward_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [
                old_op.input.type,
                old_op.weight.type,
                old_op.in_gradient.type,
            ]

            with InsertionPoint(embedding_backward_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="embedding_backward_module")
                def decorated_func(*inputs):
                    input = inputs[0]
                    weight = inputs[1]
                    in_gradient = inputs[2]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result, input, weight, in_gradient, loc=old_op.location
                    )
                    new_op_result = new_op.result

                    input_tensor = self._get_golden_tensor(old_op.input)
                    weight_tensor = self._get_golden_tensor(old_op.weight)
                    in_gradient_tensor = self._get_golden_tensor(old_op.in_gradient)
                    embedding_backward_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    embedding_backward_builder._set_golden_tensor(input, input_tensor)
                    embedding_backward_builder._set_golden_tensor(weight, weight_tensor)
                    embedding_backward_builder._set_golden_tensor(
                        in_gradient, in_gradient_tensor
                    )
                    ordered_inputs.extend([input, weight, in_gradient])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                embedding_backward_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return embedding_backward_module, embedding_backward_builder

    def get_dimension_size(
        self, in0: Operand, dimension: int = 0, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.get_dimension_size``.

        *Dimension size query operation.*

        Produces the size of the given `dimension` of the `operand`.

        .. code-block:: mlir

            %operand: [[3, 2, 7], [1, 4, 4]]
            "ttir.get_dimension_size"(%operand, value = dense<0>, %out) -> %out: [[3]]

        Parameters
        ----------
        in0 : Operand
            Input tensor operand to get dimension size from
        dimension : int, optional
            The dimension index to get size of (default: 0)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.GetDimensionSizeOp,
            [in0],
            ttir_kwargs={"dimension": dimension},
            organize_ttir_args=lambda i, o: (o, i[0]),
            output_type=self._get_type_from_torch_dtype(torch.int32),
            unit_attrs=unit_attrs,
        )

    def cbrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.cbrt``.

        *Elementwise cubic root operation.*

        Computes the cubic root (∛) of each element in the input tensor.
        For each element, returns the real-valued number that, when cubed, equals the input value.
        Unlike square root, cubic root is defined for negative numbers as well as positive numbers.

        .. code-block:: mlir

            // Compute cubic root of all elements
            %result = ttir.cbrt(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [8.0, 27.0, -8.0, 1.0]
            // Output tensor:
            // [2.0, 3.0, -2.0, 1.0]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the cubic root of each element in the input tensor
        """
        return self._op_proxy(ttir.CbrtOp, [in0], unit_attrs)

    def ceil(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.ceil``.

        *Elementwise ceiling operation.*

        Computes the ceiling of each element in the input tensor, rounding up to the nearest integer.
        This operation is idempotent.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with ceiling values
        """
        return self._op_proxy(ttir.CeilOp, [in0], unit_attrs)

    def erfc(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.erfc``.

        *Elementwise complementary error function operation.*

        Computes the complementary error function (erfc) of each element in the input tensor.
        The complementary error function is defined as erfc(x) = 1 - erf(x),
        where erf(x) is the error function. It is commonly used in statistics and probability.

        Mathematical definition: erfc(x) = 1 - (2/sqrt(π)) * ∫[0 to x] e^(-t^2) dt

        .. code-block:: mlir

            // Compute complementary error function of all elements
            %result = ttir.erfc(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [0.0, 0.5, 1.0, -1.0]
            // Output tensor:
            // [1.0, 0.4795, 0.1573, 1.8427]
        """
        return self._op_proxy(ttir.ErfcOp, [in0], unit_attrs)

    def gelu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.gelu``.

        *Elementwise GELU operation.*

        Computes the GELU (Gaussian Error Linear Unit) of each element in the input tensor.
        GELU is a smooth, non-monotonic activation function that approximates the cumulative
        distribution function of a standard normal distribution.

        Mathematical definition: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

        .. code-block:: mlir

            // Compute GELU of all elements
            %result = ttir.gelu(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [1.0, -0.5, 2.0, -2.0]
            // Output tensor:
            // [0.841, -0.154, 1.954, -0.046]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the GELU values of each element in the input tensor
        """
        return self._op_proxy(ttir.GeluOp, [in0], unit_attrs)

    def gelu_backward(
        self,
        grad: Operand,
        input: Operand,
        approximate: str = "none",
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.gelu_backward``.

        *GELU backward operation.*

        Computes the gradient of the GELU activation function during backpropagation.

        Parameters
        ----------
        grad : Operand
            Gradient tensor from the next layer
        input : Operand
            Input tensor from the forward pass
        approximate : str, optional
            Approximation mode: "none" (default) or "tanh"
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Gradient tensor with respect to the input
        """
        return self._op_proxy(
            ttir.GeluBackwardOp,
            [grad, input],
            unit_attrs,
            ttir_kwargs={"approximate": approximate},
        )

    ############### ttir.IsFiniteOp ###############

    @tag(ttir.IsFiniteOp)
    def is_finite(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.is_finite)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.IsFiniteOp)
    def is_finite_parser(
        self,
        old_op: ttir.IsFiniteOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.is_finite_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(input0, result.element_type)
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.IsFiniteOp)
    def is_finite_split(
        self,
        old_op: ttir.IsFiniteOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.is_finite_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            is_finite_module = Module.create()
            is_finite_builder = TTIRBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(is_finite_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="is_finite_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    is_finite_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    is_finite_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.extend([in0])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                is_finite_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return is_finite_module, is_finite_builder

    def bitwise_not(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.bitwise_not``.

        *Elementwise bitwise NOT operation.*

        Computes the bitwise NOT (one's complement) of each element in the input tensor.
        For each element, flips all the bits in the binary representation of the value.

        This operation is typically used with integer data types and has the involution property,
        meaning that applying it twice returns the original value: bitwise_not(bitwise_not(x)) = x.

        .. code-block:: mlir

            // Bitwise NOT with integer tensors
            %result = ttir.bitwise_not(%input, %output) : tensor<2x2xi32>, tensor<2x2xi32> -> tensor<2x2xi32>
            // Input tensor:
            // [[1, 2],
            //  [3, 4]]
            // Output tensor:
            // [[-2, -3],
            //  [-4, -5]]

            // Example with 8-bit integers
            %result = ttir.bitwise_not(%input, %output) : tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
            // Input: [0, 5, 255] (binary: [00000000, 00000101, 11111111])
            // Output: [255, 250, 0] (binary: [11111111, 11111010, 00000000])

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.BitwiseNotOp,
            [in0],
            unit_attrs,
        )

    def tan(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.tan``.

        *Elementwise tangent operation.*

        Computes the tangent of each element in the input tensor.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with tangent values
        """
        return self._op_proxy(ttir.TanOp, [in0], unit_attrs)

    def atan(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.atan``.

        *Elementwise arctangent operation.*

        Computes the inverse tangent (arctangent) of each element in the input tensor.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with arctangent values
        """
        return self._op_proxy(ttir.AtanOp, [in0], unit_attrs)

    def reciprocal(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.reciprocal``.

        *Elementwise reciprocal operation.*

        Computes the reciprocal (1/x) of each element in the input tensor.
        This operation is involutive (applying it twice returns to the original value).

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with reciprocal values
        """
        return self._op_proxy(
            ttir.ReciprocalOp,
            [in0],
            unit_attrs,
        )

    def relu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.relu``.

        *Elementwise ReLU activation operation.*

        Computes the Rectified Linear Unit function for each element in the input tensor.
        This operation is idempotent (applying it multiple times has the same effect as applying it once).

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with ReLU activation values
        """
        return self._op_proxy(ttir.ReluOp, [in0], unit_attrs)

    def relu6(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.relu6``.

        *Elementwise ReLU6 activation operation.*

        Computes the ReLU6 function for each element in the input tensor.
        ReLU6 is defined as: min(max(0, x), 6)
        This activation function clips values between 0 and 6, making it useful
        for quantized neural networks and mobile applications.

        .. code-block:: mlir

            // Compute ReLU6 of all elements
            %result = ttir.relu6(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [-2.0, 3.0, 8.0, 1.5]
            // Output tensor:
            // [0.0, 3.0, 6.0, 1.5]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with ReLU6 activation values
        """
        return self._op_proxy(ttir.Relu6Op, [in0], unit_attrs)

    def silu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.silu``.

        *Elementwise SiLU (Swish) activation operation.*

        Computes the SiLU (Sigmoid Linear Unit) activation function for each element in the input tensor.
        SiLU is also known as Swish activation and is defined as: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

        This activation function is smooth, non-monotonic, and has been shown to work well
        in deep neural networks, particularly in transformer architectures.

        .. code-block:: mlir

            // Compute SiLU activation of all elements
            %result = ttir.silu(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [1.0, -0.5, 2.0, -2.0]
            // Output tensor:
            // [0.731, -0.193, 1.762, -0.238]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the SiLU activation values of each element in the input tensor
        """
        return self._op_proxy(ttir.SiluOp, [in0], unit_attrs)

    def mish(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.mish``.

        *Elementwise Mish activation operation.*

        Applies the Mish activation function element-wise to the input tensor.
        Mish is a smooth, self-regularized, non-monotonic activation function defined as:
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))

        This activation function has been shown to improve performance in deep learning
        applications, particularly in computer vision tasks, by providing smooth gradients
        and better information flow.

        .. code-block:: mlir

            // Apply Mish activation to all elements
            %result = ttir.mish(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [-2.0, -1.0, 0.0, 1.0]
            // Output tensor:
            // [-0.252, -0.303, 0.0, 0.865]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the Mish activation values of each element in the input tensor
        """
        return self._op_proxy(ttir.MishOp, [in0], unit_attrs)

    def relu6(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.relu6``.

        *Elementwise ReLU6 activation operation.*

        Computes the ReLU6 function for each element in the input tensor.
        ReLU6 is defined as: min(max(0, x), 6)
        This activation function clips values between 0 and 6, making it useful
        for quantized neural networks and mobile applications.

        .. code-block:: mlir

            // Compute ReLU6 of all elements
            %result = ttir.relu6(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [-2.0, 3.0, 8.0, 1.5]
            // Output tensor:
            // [0.0, 3.0, 6.0, 1.5]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with ReLU6 activation values
        """
        return self._op_proxy(ttir.Relu6Op, [in0], unit_attrs)

    def sign(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.sign``.

        *Elementwise sign operation.*

        Returns the sign (-1, 0, or 1) of each element in the input tensor.
        This operation is idempotent.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with sign values
        """
        return self._op_proxy(ttir.SignOp, [in0], unit_attrs)

    def silu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.silu``.

        *Elementwise SiLU (Swish) activation operation.*

        Computes the SiLU (Sigmoid Linear Unit) activation function for each element in the input tensor.
        SiLU is also known as Swish activation and is defined as: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

        This activation function is smooth, non-monotonic, and has been shown to work well
        in deep neural networks, particularly in transformer architectures.

        .. code-block:: mlir

            // Compute SiLU activation of all elements
            %result = ttir.silu(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [1.0, -0.5, 2.0, -2.0]
            // Output tensor:
            // [0.731, -0.193, 1.762, -0.238]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the SiLU activation values of each element in the input tensor
        """
        return self._op_proxy(ttir.SiluOp, [in0], unit_attrs)

    def expm1(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.expm1``.

        *Elementwise exponential minus one operation.*

        Computes e^x - 1 for each element in the input tensor, where e is Euler's number.
        This operation provides better numerical precision than computing exp(x) - 1 directly,
        especially for small values of x.

        .. code-block:: mlir

            // Compute exp(x) - 1 for all elements
            %result = ttir.expm1(%input, %output) : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensor:
            // [0.0, 0.1, -0.1]
            // Output tensor:
            // [0.0, 0.10517, -0.09516]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing exp(x) - 1 for each element x in the input tensor
        """
        return self._op_proxy(
            ttir.Expm1Op,
            [in0],
            unit_attrs,
        )

    # class TTIR_ElementwiseUnaryWithFloatParameterOp

    def leaky_relu(
        self,
        in0: Operand,
        parameter: float = 0.01,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.leaky_relu``.

        *Elementwise leaky ReLU activation operation.*

        Computes a leaky version of the Rectified Linear Unit (ReLU) activation function.
        For each element x in the input tensor:
        - If x > 0: returns x
        - If x ≤ 0: returns parameter * x

        The parameter controls the slope for negative values, allowing a small gradient
        when the unit is not active.

        .. code-block:: mlir

            // Compute leaky ReLU with slope 0.01 for negative values
            %result = ttir.leaky_relu(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [2.0, -1.0, 0.0, -3.0]
            // Output tensor:
            // [2.0, -0.01, 0.0, -0.03]

        Parameters
        ----------
        in0 : Operand
            Input tensor to be activated
        parameter : float, optional
            Slope for negative values (default: 0.01)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the leaky ReLU activation values
        """
        ttir_kwargs = {"parameter": parameter}
        return self._op_proxy(
            ttir.LeakyReluOp,
            [in0],
            ttir_kwargs=ttir_kwargs,
            unit_attrs=unit_attrs,
        )

    def logical_left_shift(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.logical_shift_left``.

        *Elementwise logical shift left operation.*

        Performs elementwise logical shift left operation between two tensors.
        For each pair of corresponding elements, shifts the bits of the first element to the left
        by the number of positions specified by the second element.

        .. code-block:: mlir

            // Logical shift left operation
            %result = ttir.logical_shift_left(%lhs, %rhs, %output) : tensor<3xi8>, tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
            // Input tensors:
            // lhs: [2, 4, 8]  (binary: [00000010, 00000100, 00001000])
            // rhs: [1, 2, 3]  (shift amounts)
            // Output tensor:
            // [4, 16, 64]    (binary: [00000100, 00010000, 01000000])

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor (shift amounts)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.LogicalLeftShiftOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def logical_or(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.logical_or``.

        *Elementwise logical OR operation.*

        Performs elementwise logical OR operation between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if at least one element is 1 (true)
        - 0 (false) if both elements are 0 (false)

        This operation is idempotent, meaning logical_or(x, x) = x.

        Mathematical definition: logical_or(x, y) = x || y

        .. code-block:: mlir

            // Logical OR operation
            %result = ttir.logical_or(%lhs, %rhs, %output) : tensor<4xi1>, tensor<4xi1>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1, 0, 1, 0]
            // rhs: [1, 1, 0, 1]
            // Output tensor:
            // [1, 1, 1, 1]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.LogicalOrOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def logical_xor(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.logical_xor``.

        *Elementwise logical XOR operation.*

        Performs elementwise logical XOR (exclusive OR) operation between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if exactly one element is 1 (true)
        - 0 (false) if both elements are the same (both 0 or both 1)

        Mathematical definition: logical_xor(x, y) = (x || y) && !(x && y)

        .. code-block:: mlir

            // Logical XOR operation
            %result = ttir.logical_xor(%lhs, %rhs, %output) : tensor<4xi1>, tensor<4xi1>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1, 0, 1, 0]
            // rhs: [1, 1, 0, 1]
            // Output tensor:
            // [0, 1, 1, 1]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.LogicalXorOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def bitwise_or(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.bitwise_or``.

        *Elementwise bitwise OR operation.*

        Performs elementwise bitwise OR operation between two tensors.
        For each pair of corresponding elements, performs a bitwise OR on their binary representations.

        This operation is typically used with integer data types and has the following properties:
        - Commutative: bitwise_or(x, y) = bitwise_or(y, x)
        - Associative: bitwise_or(x, bitwise_or(y, z)) = bitwise_or(bitwise_or(x, y), z)
        - Identity: bitwise_or(x, 0) = x
        - One: bitwise_or(x, -1) = -1

        .. code-block:: mlir

            // Bitwise OR with integer tensors
            %result = ttir.bitwise_or(%lhs, %rhs, %output) : tensor<3xi8>, tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
            // Input tensors:
            // lhs: [5, 3, 255]  (binary: [00000101, 00000011, 11111111])
            // rhs: [3, 6, 129]   (binary: [00000011, 00000110, 10000001])
            // Output tensor:
            // [7, 7, 255]    (binary: [00000111, 00000111, 11111111])

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.BitwiseOrOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def bitwise_xor(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.bitwise_xor``.

        *Elementwise bitwise XOR operation.*

        Performs elementwise bitwise XOR (exclusive OR) operation between two tensors.
        For each pair of corresponding elements, performs a bitwise XOR on their binary representations.

        .. code-block:: mlir

            // Bitwise XOR with integer tensors
            %result = ttir.bitwise_xor(%input1, %input2, %output) : tensor<2x2xi32>, tensor<2x2xi32> -> tensor<2x2xi32>
            // Input1 tensor:
            // [[1, 3],  // binary: [[0001, 0011],
            //  [5, 7]]  //         [0101, 0111]]
            // Input2 tensor:
            // [[2, 3],  // binary: [[0010, 0011],
            //  [6, 7]]  //         [0110, 0111]]
            // Output tensor:
            // [[3, 0],  // binary: [[0011, 0000],
            //  [3, 0]]  //         [0011, 0000]]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the bitwise XOR of corresponding elements
        """
        return self._op_proxy(
            ttir.BitwiseXorOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def remainder(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.remainder``.

        *Elementwise remainder operation.*

        Computes the element-wise remainder of division (modulo operation).

        Parameters
        ----------
        in0 : Operand
            First input tensor (dividend)
        in1 : Operand
            Second input tensor (divisor)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with remainder values
        """
        return self._op_proxy(
            ttir.RemainderOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    # class TTIR_ReductionOp

    def argmax(
        self,
        in0: Operand,
        dim_arg: List[int] = None,
        keep_dim: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.argmax``.

        *Argmax reduction operation.*

        Returns the indices of the maximum values along the specified dimensions.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim_arg : List[int]
            Dimensions to reduce over
        keep_dim : bool, optional
            If True, retains reduced dimensions with length 1 (default: False)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor containing the indices of maximum values
        """
        kwargs = {"dim_arg": dim_arg, "keep_dim": keep_dim}
        return self._op_proxy(
            ttir.ArgMaxOp,
            [in0],
            ttir_kwargs=kwargs,
            output_type=IntegerType.get_signless(32, self._ctx),
            unit_attrs=unit_attrs,
        )

    def mean(
        self,
        in0: Operand,
        dim_arg: List[int] = None,
        keep_dim: bool = True,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.mean``.

        *Mean reduction operation.*

        Computes the mean of elements along specified dimensions of the input tensor.
        If `dim_arg` is not provided, the mean is computed over all dimensions.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim_arg : List[int], optional
            Dimensions to reduce over (default: [0])
        keep_dim : bool, optional
            If True, retains reduced dimensions with length 1 (default: True)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with mean values
        """
        return self._op_proxy(
            ttir.MeanOp,
            [in0],
            ttir_kwargs={"dim_arg": dim_arg, "keep_dim": keep_dim},
            unit_attrs=unit_attrs,
        )

    def min(
        self,
        in0: Operand,
        dim_arg: List[int] = None,
        keep_dim: bool = True,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.min``.

        *Minimum reduction operation.*

        Returns the minimum values along the specified dimension.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim_arg : int, optional
            Dimension to reduce over (default: None, reduces over all dimensions)
        keep_dim : bool, optional
            If True, retains reduced dimensions with length 1 (default: True)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with minimum values
        """
        kwargs = {"dim_arg": dim_arg, "keep_dim": keep_dim}
        return self._op_proxy(
            ttir.MinOp,
            [in0],
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def prod(
        self,
        in0: Operand,
        dim_arg: List[int] = None,
        keep_dim: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.prod``.

        *Product reduction operation.*

        Computes the product of elements along specified dimensions.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim_arg : List[int]
            Dimensions to reduce over
        keep_dim : bool, optional
            If True, retains reduced dimensions with length 1 (default: False)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with product values
        """
        return self._op_proxy(
            ttir.ProdOp,
            [in0],
            ttir_kwargs={"keep_dim": keep_dim, "dim_arg": dim_arg},
            unit_attrs=unit_attrs,
        )

    def embedding(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.embedding``.

        *Embedding lookup operation.*

        Performs a lookup in an embedding table (in1) using indices (in0).
        Returns a tensor containing the embeddings for the given indices.

        .. code-block:: mlir

            // Lookup embeddings for indices
            %result = ttir.embedding(%indices, %weights, %output) : tensor<2xi32>, tensor<4x3xf32> -> tensor<2x3xf32>
            // Indices tensor:
            // [1, 3]  // Looking up embeddings at indices 1 and 3
            // Weights tensor (embedding table):
            // [[0.1, 0.2, 0.3],  // embedding 0
            //  [0.4, 0.5, 0.6],  // embedding 1
            //  [0.7, 0.8, 0.9],  // embedding 2
            //  [1.0, 1.1, 1.2]]  // embedding 3
            // Output tensor:
            // [[0.4, 0.5, 0.6],  // embedding for index 1
            //  [1.0, 1.1, 1.2]]  // embedding for index 3

        Parameters
        ----------
        in0 : Operand
            Input tensor containing indices
        in1 : Operand
            Weight tensor containing embeddings
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the embeddings for the input indices
        """
        return self._op_proxy(
            ttir.EmbeddingOp,
            [in0, in1],
            organize_golden_args=lambda i: (
                self._get_golden_tensor(i[0]),
                self._get_golden_tensor(i[1]),
            ),
            unit_attrs=unit_attrs,
        )

    def softmax(
        self,
        in0: Operand,
        dimension: int = 1,
        numeric_stable: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.softmax``.

        *Softmax operation.*

        Applies the Softmax function to an n-dimensional input tensor rescaling them
        so that the elements lie in the range [0,1] and sum to 1.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dimension : int, optional
            Dimension along which Softmax will be computed (default: 1)
        numeric_stable : bool, optional
            Whether to use numerically stable softmax computation (default: False)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Output tensor after softmax
        """
        # kwargs handled thru organize_ttir_args
        return self._op_proxy(
            ttir.SoftmaxOp,
            [in0],
            golden_kwargs={"dim": dimension},
            ttir_kwargs={
                "dimension": dimension,
                "numericStable": numeric_stable,
            },
            organize_ttir_args=lambda i, o: (
                o,
                i[0],
            ),
            unit_attrs=unit_attrs,
        )

    def transpose(
        self,
        in0: Operand,
        dim0: int = 0,
        dim1: int = 1,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.transpose``.

        *Tensor transpose operation.*

        Swaps two dimensions of a tensor.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim0 : int, optional
            First dimension to swap (default: 0)
        dim1 : int, optional
            Second dimension to swap (default: 1)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with swapped dimensions
        """
        kwargs = {"dim0": dim0, "dim1": dim1}
        return self._op_proxy(
            ttir.TransposeOp,
            [in0],
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def repeat_interleave(
        self,
        in0: Operand,
        repeats: int,
        dim: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.repeat_interleave``.

        *Tensor repeat interleave operation.*

        Repeats elements of a tensor along a dimension by interleaving the repeated elements.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        repeats : int
            Number of repetitions for each element
        dim : int
            Dimension along which to repeat
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with interleaved repeated elements
        """
        return self._op_proxy(
            ttir.RepeatInterleaveOp,
            [in0],
            ttir_kwargs={"repeats": repeats, "dim": dim},
            organize_ttir_args=lambda i, o: (o, i[0]),
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            output_type=self._get_type_from_torch_dtype(
                self._get_golden_tensor(in0).dtype
            ),
            unit_attrs=unit_attrs,
        )

    def fill_cache(
        self,
        in0: Operand,
        in1: Operand,
        batch_offset: int = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.fill_cache``.

        *Cache fill operation.*

        Fills a cache tensor with new values starting at a specified batch offset.
        This operation is typically used in sequence models to initialize or update
        cached states.

        .. code-block:: mlir

            // Fill cache with new values at batch offset 1
            %result = ttir.fill_cache(%new_values, %cache, batch_offset = 1) : tensor<2x3xf32>, tensor<4x3xf32> -> tensor<4x3xf32>
            // New values tensor:
            // [[1.0, 2.0, 3.0],
            //  [4.0, 5.0, 6.0]]
            // Cache tensor before:
            // [[0.1, 0.2, 0.3],
            //  [0.4, 0.5, 0.6],
            //  [0.7, 0.8, 0.9],
            //  [1.0, 1.1, 1.2]]
            // Cache tensor after:
            // [[0.1, 0.2, 0.3],
            //  [1.0, 2.0, 3.0],
            //  [4.0, 5.0, 6.0],
            //  [1.0, 1.1, 1.2]]

        Parameters
        ----------
        in0 : Operand
            New values to fill into cache
        in1 : Operand
            Cache tensor to be filled
        batch_offset : int, optional
            Starting position in batch dimension (default: 0)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The updated cache tensor
        """
        return self._op_proxy(
            ttir.FillCacheOp,
            [in0, in1],
            ttir_kwargs={"batch_offset": batch_offset},
            organize_ttir_args=lambda i, o: (o, i[0], i[1]),
            organize_golden_args=lambda i: (
                self._get_golden_tensor(i[0]),
                self._get_golden_tensor(i[1]),
            ),
            unit_attrs=unit_attrs,
        )

    def update_cache(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        batch_offset: int = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.update_cache``.

        *Cache update operation.*

        Updates a cache tensor by combining new values with existing cache values,
        starting at a specified batch offset. This operation is typically used in
        sequence models to maintain and update cached states.

        .. code-block:: mlir

            // Update cache with new values at batch offset 1
            %result = ttir.update_cache(%new_values, %old_cache, %mask, batch_offset = 1) \
                : tensor<2x3xf32>, tensor<4x3xf32>, tensor<2xi1> -> tensor<4x3xf32>
            // New values tensor:
            // [[1.0, 2.0, 3.0],
            //  [4.0, 5.0, 6.0]]
            // Old cache tensor:
            // [[0.1, 0.2, 0.3],
            //  [0.4, 0.5, 0.6],
            //  [0.7, 0.8, 0.9],
            //  [1.0, 1.1, 1.2]]
            // Mask tensor:
            // [true, false]  // Only update first new value
            // Output tensor:
            // [[0.1, 0.2, 0.3],
            //  [1.0, 2.0, 3.0],  // Updated with first new value
            //  [0.7, 0.8, 0.9],  // Kept old value due to mask
            //  [1.0, 1.1, 1.2]]

        Parameters
        ----------
        in0 : Operand
            New values to update cache with
        in1 : Operand
            Cache tensor to be updated
        in2 : Operand
            Mask tensor indicating which values to update
        batch_offset : int, optional
            Starting position in batch dimension (default: 0)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The updated cache tensor
        """
        return self._op_proxy(
            ttir.UpdateCacheOp,
            [in0, in1, in2],
            ttir_kwargs={"batch_offset": batch_offset},
            organize_ttir_args=lambda i, o: (o, i[0], i[1], i[2]),
            organize_golden_args=lambda i: (
                self._get_golden_tensor(i[0]),
                self._get_golden_tensor(i[1]),
                self._get_golden_tensor(i[2]),
            ),
            unit_attrs=unit_attrs,
        )

    @tag(ttir.Conv2dOp)
    def conv2d(
        self,
        in0: Operand,
        weight: Operand,
        bias: Optional[Operand],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        """
        Creates ``ttir.conv2d``.

        *Conv2d operation.*

        Applies a 2D convolution over an input image composed of several input planes.
        This operation performs a 2D convolution on the input tensor using the provided weight tensor
        and optional bias. It supports configurable stride, padding, dilation, and grouping parameters.

        .. code-block:: mlir

            // Basic 2D convolution
            %input = ... : tensor<1x28x28x3xf32>    // Batch size 1, 28x28 image, 3 channels
            %weight = ... : tensor<16x3x3x3xf32>    // 16 output channels, 3 input channels, 3x3 kernel
            %bias = ... : tensor<1x1x1x16xf32>      // Bias for 16 output channels
            %output = ttir.empty() : tensor<1x26x26x16xf32>  // Output shape with no padding
            %result = ttir.conv2d(%input, %weight, %bias, %output) {
                stride = [1, 1],
                padding = [0, 0, 0, 0],
                dilation = [1, 1],
                groups = 1
            }

        Parameters
        ----------
        in0 : Operand
            Input tensor in (N, H_in, W_in, C) format
        weight : Operand
            Weight tensor in (O, C/G, K_H, K_W) format
        bias : *Optional[Operand]*
            Optional bias tensor in (1, 1, 1, O) format
        stride : *Union[int, List[int]]*, optional
            Stride for height and width dimensions (default: 1)
        padding : *Union[int, List[int]]*, optional
            Padding for all sides or [top, left, bottom, right] (default: 0)
        dilation : *Union[int, List[int]]*, optional
            Spacing between kernel elements (default: 1)
        groups : int, optional
            Number of blocked connections from input to output channels (default: 1)
        output_type : *Optional[torch.dtype]*, optional
            Optional output data type (default: None, uses input type)
        loc : *Optional[str]*, optional
            Optional location string for debugging
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpResult*)
            Output tensor after convolution
        """
        ttir_op = self.get_opview_from_method(TTIRBuilder.conv2d)

        if not bias:
            bias = None

        stride_attr = (
            IntegerAttr.get(IntegerType.get_signless(32), stride)
            if isinstance(stride, int)
            else DenseI32ArrayAttr.get(stride)
        )
        padding_attr = (
            IntegerAttr.get(IntegerType.get_signless(32), padding)
            if isinstance(padding, int)
            else DenseI32ArrayAttr.get(padding)
        )
        dilation_attr = (
            IntegerAttr.get(IntegerType.get_signless(32), dilation)
            if isinstance(dilation, int)
            else DenseI32ArrayAttr.get(dilation)
        )

        groups_attr = IntegerAttr.get(IntegerType.get_signless(32), groups)

        # Default dimension attributes (NHWC layout)
        batch_dim_attr = IntegerAttr.get(IntegerType.get_signless(32), 0)
        height_dim_attr = IntegerAttr.get(IntegerType.get_signless(32), 1)
        width_dim_attr = IntegerAttr.get(IntegerType.get_signless(32), 2)
        channel_dim_attr = IntegerAttr.get(IntegerType.get_signless(32), 3)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        weight0 = self._get_golden_tensor(weight)
        bias0 = self._get_golden_tensor(bias) if bias is not None else None
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            weight0,
            bias0,
            stride_attr,
            padding_attr,
            dilation_attr,
            groups_attr,
            batch_dim_attr,
            height_dim_attr,
            width_dim_attr,
            channel_dim_attr,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            weight,
            stride_attr,
            padding_attr,
            dilation_attr,
            groups_attr,
            bias=bias,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.Conv2dOp)
    def conv2d_parser(
        self,
        old_op: ttir.Conv2dOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.conv2d_parser)

        in0 = global_dict[old_op.input]
        weight = global_dict[old_op.weight]
        bias = global_dict[old_op.bias] if old_op.bias is not None else None
        result = old_op.result.type

        stride_attr = old_op.stride
        padding_attr = old_op.padding
        dilation_attr = old_op.dilation
        groups_attr = old_op.groups

        # Access optional flattened_compat_info attribute safely
        flattened_compat_info = None
        if "flattened_compat_info" in old_op.operation.attributes:
            flattened_compat_info = old_op.operation.attributes["flattened_compat_info"]

        new_op = ttir_op(
            result,
            in0,
            weight,
            stride_attr,
            padding_attr,
            dilation_attr,
            groups_attr,
            bias=bias,
            batch_dim=old_op.batch_dim,
            height_dim=old_op.height_dim,
            width_dim=old_op.width_dim,
            channel_dim=old_op.channel_dim,
            flattened_compat_info=flattened_compat_info,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input_weight = self._get_golden_tensor(weight)
        input_bias = self._get_golden_tensor(bias) if bias is not None else None
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            input_weight,
            input_bias,
            stride_attr,
            padding_attr,
            dilation_attr,
            groups_attr,
            old_op.batch_dim,
            old_op.height_dim,
            old_op.width_dim,
            old_op.channel_dim,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.Conv2dOp)
    def conv2d_split(
        self,
        old_op: ttir.Conv2dOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.conv2d_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            conv2d_module = Module.create()
            conv2d_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type, old_op.weight.type]
            if old_op.bias is not None:
                op_input_types.append(old_op.bias.type)

            with InsertionPoint(conv2d_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="conv2d_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    weight = inputs[1]
                    bias = inputs[2] if len(inputs) > 2 else None
                    result = old_op.result.type

                    stride_attr = old_op.stride
                    padding_attr = old_op.padding
                    dilation_attr = old_op.dilation
                    groups_attr = old_op.groups

                    # Access optional flattened_compat_info attribute safely
                    flattened_compat_info = None
                    if "flattened_compat_info" in old_op.operation.attributes:
                        flattened_compat_info = old_op.operation.attributes[
                            "flattened_compat_info"
                        ]

                    new_op = ttir_op(
                        result,
                        in0,
                        weight,
                        stride_attr,
                        padding_attr,
                        dilation_attr,
                        groups_attr,
                        bias=bias,
                        batch_dim=old_op.batch_dim,
                        height_dim=old_op.height_dim,
                        width_dim=old_op.width_dim,
                        channel_dim=old_op.channel_dim,
                        flattened_compat_info=flattened_compat_info,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    input_weight = self._get_golden_tensor(old_op.weight)
                    input_bias = (
                        self._get_golden_tensor(old_op.bias)
                        if old_op.bias is not None
                        else None
                    )
                    conv2d_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    conv2d_builder._set_golden_tensor(in0, input0)
                    conv2d_builder._set_golden_tensor(weight, input_weight)
                    ordered_inputs.extend([in0, weight])
                    if bias is not None:
                        conv2d_builder._set_golden_tensor(bias, input_bias)
                        ordered_inputs.append(bias)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                conv2d_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return conv2d_module, conv2d_builder

    @tag(ttir.ConvTranspose2dOp)
    def conv_transpose2d(
        self,
        in0: Operand,
        weight: Operand,
        bias: Optional[Operand],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        output_padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        """
        Creates ``ttir.conv_transpose2d``.

        *2D transposed convolution operation.*

        Applies a 2D transposed convolution over an input image. This operation
        can be seen as the gradient of Conv2d with respect to its input.
        Also known as a deconvolution or fractionally strided convolution.

        .. code-block:: mlir

            // Apply 2D transposed convolution
            %result = ttir.conv_transpose2d(%input, %weight, %bias, %output,
                                          stride = [2, 2], padding = [1, 1],
                                          output_padding = [1, 1], dilation = [1, 1],
                                          groups = 1) :
                tensor<1x1x4x4xf32>, tensor<1x1x3x3xf32>, tensor<1xf32>, tensor<1x1x9x9xf32> -> tensor<1x1x9x9xf32>
            // Input tensor: 4x4 feature map
            // Weight tensor: 3x3 kernel
            // Output tensor: 9x9 upsampled feature map

        Parameters
        ----------
        in0 : Operand
            Input tensor of shape (batch, in_channels, height, width)
        weight : Operand
            Weight tensor of shape (in_channels, out_channels/groups, kernel_height, kernel_width)
        bias : Optional[Operand]
            Optional bias tensor of shape (out_channels)
        stride : *Union[int, List[int]]*
            Stride of the convolution
        padding : *Union[int, List[int]]*
            Padding added to input
        output_padding : *Union[int, List[int]]*
            Additional size added to output shape
        dilation : *Union[int, List[int]]*
            Dilation of the kernel
        groups : int
            Number of blocked connections from input to output channels
        output_type : *Optional[torch.dtype]*, optional
            Optional output data type (default: None, uses input type)
        loc : *Optional[str]*, optional
            Optional location string for debugging
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpResult*)
            The output tensor after transposed convolution
        """
        ttir_op = self.get_opview_from_method(TTIRBuilder.conv_transpose2d)

        if not bias:
            bias = None

        stride_attr = (
            IntegerAttr.get(IntegerType.get_signless(32), stride)
            if isinstance(stride, int)
            else DenseI32ArrayAttr.get(stride)
        )
        padding_attr = (
            IntegerAttr.get(IntegerType.get_signless(32), padding)
            if isinstance(padding, int)
            else DenseI32ArrayAttr.get(padding)
        )
        output_padding_attr = (
            IntegerAttr.get(IntegerType.get_signless(32), output_padding)
            if isinstance(output_padding, int)
            else DenseI32ArrayAttr.get(output_padding)
        )
        dilation_attr = (
            IntegerAttr.get(IntegerType.get_signless(32), dilation)
            if isinstance(dilation, int)
            else DenseI32ArrayAttr.get(dilation)
        )
        groups_attr = IntegerAttr.get(IntegerType.get_signless(32), groups)

        # Default dimension attributes (NHWC layout)
        batch_dim_attr = IntegerAttr.get(IntegerType.get_signless(32), 0)
        height_dim_attr = IntegerAttr.get(IntegerType.get_signless(32), 1)
        width_dim_attr = IntegerAttr.get(IntegerType.get_signless(32), 2)
        channel_dim_attr = IntegerAttr.get(IntegerType.get_signless(32), 3)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        weight0 = self._get_golden_tensor(weight)
        bias0 = self._get_golden_tensor(bias) if bias is not None else None
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            weight0,
            bias0,
            stride_attr,
            padding_attr,
            output_padding_attr,
            dilation_attr,
            groups_attr,
            batch_dim_attr,
            height_dim_attr,
            width_dim_attr,
            channel_dim_attr,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            weight,
            stride_attr,
            padding_attr,
            output_padding_attr,
            dilation_attr,
            groups_attr,
            bias=bias,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    def avg_pool2d(
        self,
        in0: Operand,
        kernel: Union[int, List[int]],
        stride: Union[int, List[int]],
        dilation: Union[int, List[int]],
        padding: Union[int, List[int]],
        ceil_mode: bool,
        count_include_pad: bool = True,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.max_pool2d``.

        *Max pooling operation.*

        Applies a 2D max pooling over an input signal composed of several input planes.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        kernel_size : *Union[int, List[int]]*
            Size of the pooling window
        stride : *Optional[Union[int, List[int]]]*
            Stride of the pooling window (default: None, same as kernel_size)
        padding : *Union[int, List[int]]*, optional
            Padding added to all sides of input (default: 0)
        dilation : *Union[int, List[int]]*, optional
            Controls spacing between kernel elements (default: 1)
        ceil_mode : bool, optional
            When True, use ceil instead of floor for output shape (default: False)
        count_include_pad : bool, optional
            When True , include padding in the average calculation (default: True)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Output tensor after max pooling
        """
        return self._op_proxy(
            ttir.AvgPool2dOp,
            [in0],
            ttir_kwargs={
                "kernel": (
                    IntegerAttr.get(IntegerType.get_signed(32), kernel)
                    if isinstance(kernel, int)
                    else DenseI32ArrayAttr.get(kernel)
                ),
                "stride": (
                    IntegerAttr.get(IntegerType.get_signed(32), stride)
                    if isinstance(stride, int)
                    else DenseI32ArrayAttr.get(stride)
                ),
                "dilation": (
                    IntegerAttr.get(IntegerType.get_signed(32), dilation)
                    if isinstance(dilation, int)
                    else DenseI32ArrayAttr.get(dilation)
                ),
                "padding": (
                    IntegerAttr.get(IntegerType.get_signed(32), padding)
                    if isinstance(padding, int)
                    else DenseI32ArrayAttr.get(padding)
                ),
                "ceil_mode": ceil_mode,
                "count_include_pad": count_include_pad,
            },
            unit_attrs=unit_attrs,
        )

    ############### ttir.GlobalAvgPool2d ###############

    @tag(ttir.GlobalAvgPool2dOp)
    def global_avg_pool2d(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        """
        Creates ``ttir.global_avg_pool2d``.
        *Global average pooling operation.*

        Applies a global average pooling over an input signal composed of several input planes.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Output tensor after global average pooling
        """
        ttir_op = self.get_opview_from_method(TTIRBuilder.global_avg_pool2d)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            output_type_mlir=mlir_output_type,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
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

    @parse(ttir.GlobalAvgPool2dOp)
    def global_avg_pool2d_parser(
        self,
        old_op: ttir.GlobalAvgPool2dOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.global_avg_pool2d_parser)
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {old_op.result: new_op_result}
        return new_op, op_map_dictionary

    @split(ttir.GlobalAvgPool2dOp)
    def global_avg_pool2d_split(
        self,
        old_op: ttir.GlobalAvgPool2dOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.global_avg_pool2d_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            global_avg_pool_module = Module.create()
            global_avg_pool_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(global_avg_pool_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="global_avg_pool2d_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    op_golden_function = get_golden_function(ttir_op)
                    golden_output = op_golden_function(
                        input0,
                        result.element_type,
                    )
                    global_avg_pool_builder._set_golden_tensor(
                        new_op_result, golden_output
                    )
                    global_avg_pool_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                global_avg_pool_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return global_avg_pool_module, global_avg_pool_builder

    def select(
        self,
        in0: Operand,
        dim: int = 0,
        begin: int = 0,
        length: int = 2,
        stride: int = 2,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.select``.

        *Tensor selection operation.*

        Selects a slice of the input tensor along the specified dimension with given stride.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim : int, optional
            Dimension to select from (default: 0)
        begin : int, optional
            Starting index (default: 0)
        length : int, optional
            Length of the slice (default: 2)
        stride : int, optional
            Stride between elements (default: 2)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The selected slice of the tensor
        """
        return self._op_proxy(
            ttir.IndexSelectOp,
            [in0],
            ttir_kwargs={
                "dim": dim,
                "begin": begin,
                "length": length,
                "stride": stride,
            },
            unit_attrs=unit_attrs,
        )

    def index(
        self,
        in0: Operand,
        dim: int,
        begin: int,
        end: int,
        step: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.index``.

        *Tensor indexing operation.*

        Indexes into the input tensor along the specified dimension using a range of indices.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim : int
            Dimension to index into
        begin : int
            Starting index
        end : int
            Ending index (exclusive)
        step : int
            Step size between indices
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The indexed tensor
        """
        return self._op_proxy(
            ttir.IndexOp,
            [in0],
            ttir_kwargs={"dim": dim, "begin": begin, "end": end, "step": step},
            unit_attrs=unit_attrs,
        )

    def squeeze(
        self,
        in0: Operand,
        dim: Optional[int] = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.squeeze``.

        *Tensor squeeze operation.*

        Removes dimensions of size 1 from the shape of a tensor.
        If dim is specified, only squeezes the dimension if it has size 1.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim : Optional[int], optional
            Dimension to squeeze (default: 0)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with specified dimensions of size 1 removed
        """
        kwargs = {"dim": dim}
        return self._op_proxy(
            ttir.SqueezeOp,
            [in0],
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def unsqueeze(
        self,
        in0: Operand,
        dim: Optional[int] = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.unsqueeze``.

        *Tensor unsqueeze operation.*

        Adds a dimension of size 1 at the specified position.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim : Optional[int], optional
            Position to insert the new dimension (default: 0)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with a new dimension of size 1 inserted
        """
        kwargs = {"dim": dim}
        return self._op_proxy(
            ttir.UnsqueezeOp,
            [in0],
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def clamp_scalar(
        self,
        in0: Operand,
        min_arg: Optional[float] = None,
        max_arg: Optional[float] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        def _to_attr(value):
            if isinstance(value, int):
                return IntegerAttr.get(IntegerType.get_signless(32), value)
            return FloatAttr.get_f32(value)

        golden_kwargs = {"min": min_arg, "max": max_arg}
        ttir_kwargs = {"min": _to_attr(min_arg), "max": _to_attr(max_arg)}
        return self._op_proxy(
            ttir.ClampScalarOp,
            [in0],
            ttir_kwargs=ttir_kwargs,
            golden_kwargs=golden_kwargs,
            unit_attrs=unit_attrs,
        )

    def linear(
        self,
        in0: Operand,
        in1: Operand,
        bias: Optional[Operand] = None,
        transpose_a: bool = False,
        transpose_b: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.linear``.

        *Linear transformation operation.*

        Applies a linear transformation to the incoming data: y = xA^T + b

        Parameters
        ----------
        in0 : Operand
            Input tensor
        weight : Operand
            Weight matrix
        bias : *Optional[Operand]*
            Bias vector (default: None)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Output tensor after linear transformation
        """
        inputs = [in0, in1]
        if bias is not None:
            inputs.append(bias)

        # Convert bias operand to tensor for golden function
        golden_bias = self._get_golden_tensor(bias) if bias is not None else None

        return self._op_proxy(
            ttir.LinearOp,
            [in0, in1],
            golden_kwargs={
                "transpose_a": transpose_a,
                "transpose_b": transpose_b,
                "bias": golden_bias,
            },
            ttir_kwargs={
                "transpose_a": transpose_a,
                "transpose_b": transpose_b,
                "bias": bias,
            },
            unit_attrs=unit_attrs,
        )

    @tag(ttir.MatmulOp)
    def matmul(
        self,
        in0: Operand,
        in1: Operand,
        transpose_a: bool = False,
        transpose_b: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        kwargs = {
            "transpose_a": transpose_a,
            "transpose_b": transpose_b,
        }
        return self._op_proxy(
            ttir.MatmulOp,
            [in0, in1],
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    @parse(ttir.MatmulOp)
    def matmul_parser(
        self,
        old_op: ttir.MatmulOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.matmul_parser)
        in0 = global_dict[old_op.a]
        in1 = global_dict[old_op.b]
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            in1,
            loc=old_op.location,
            transpose_a=old_op.transpose_a,
            transpose_b=old_op.transpose_b,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttir_op)
        transpose_a = unpack_mlir_attr(old_op.transpose_a)
        transpose_b = unpack_mlir_attr(old_op.transpose_b)
        golden_output = op_golden_function(
            input0,
            input1,
            transpose_a,
            transpose_b,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.MatmulOp)
    def matmul_split(
        self,
        old_op: ttir.MatmulOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.matmul_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            matmul_module = Module.create()
            matmul_builder = TTIRBuilder(old_context, old_loc)
            op_input_types = [old_op.a.type, old_op.b.type]

            with InsertionPoint(matmul_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="matmul_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    in1 = inputs[1]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        in1,
                        loc=old_op.location,
                        transpose_a=old_op.transpose_a,
                        transpose_b=old_op.transpose_b,
                    )
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.a)
                    input1 = self._get_golden_tensor(old_op.b)
                    transpose_a = unpack_mlir_attr(old_op.transpose_a)
                    transpose_b = unpack_mlir_attr(old_op.transpose_b)
                    matmul_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    matmul_builder._set_golden_tensor(in0, input0)
                    matmul_builder._set_golden_tensor(in1, input1)
                    ordered_inputs.extend([in0, in1])
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                matmul_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return matmul_module, matmul_builder

    def upsample2d(
        self,
        in0: Operand,
        in1: Operand,
        scale_factor: Union[int, List[int]],
        mode: str = "nearest",
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        output_shape = self._get_golden_tensor(in1).shape
        kwargs = {
            "scale_factor": (
                IntegerAttr.get(IntegerType.get_signed(32), scale_factor)
                if isinstance(scale_factor, int)
                else DenseI32ArrayAttr.get(scale_factor)
            ),
            "mode": mode,
        }
        return self._op_proxy(
            ttir.Upsample2dOp,
            [in0, in1],
            ttir_kwargs=kwargs,
            organize_ttir_args=lambda i, _: (self._get_type(i[1]), i[0]),
            output_shape=output_shape,
            unit_attrs=unit_attrs,
        )

    # class TTIR_GenericElementwiseBinaryOp

    def atan2(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.atan2``.

        *Elementwise arctangent operation.*

        Computes the elementwise arctangent of the quotient of its arguments.
        For each pair of corresponding elements (y, x), returns atan2(y, x).

        Mathematical definition: atan2(y, x) = arctan(y / x)

        .. code-block:: mlir

            // Compute arctangent of corresponding elements
            %result = ttir.atan2(%y, %x, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // y: [1.0, 0.0, -1.0]
            // x: [1.0, -1.0, -1.0]
            // Output tensor:
            // [0.7854, 3.1416, -2.3562]
        Parameters
        ----------
        in0 : Operand
            First input tensor (y)
        in1 : Operand
            Second input tensor (x)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes
        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise arctangent of the inputs
        """
        return self._op_proxy(
            ttir.Atan2Op,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def quantize(
        self,
        in0: Operand,
        scale: float,
        zero_point: int,
        dtype: torch.dtype,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.quantize``.

        *Quantize floating-point tensor to integer tensor.*

        Converts a floating-point tensor into a quantized integer tensor using the specified
        scale and zero_point parameters. For each element in the input tensor, computes:
        output[i] = (input[i] / scale) + zero_point

        .. code-block:: mlir

            // Quantize float32 tensor to int8
            %result = ttir.quantize(%input, %output) {scale = 0.1 : f32, zero_point = 128 : i32} : tensor<2x2xf32>, tensor<2x2xi8> -> tensor<2x2xi8>
            // Input tensor:
            // [[1.5, -0.2],
            //  [0.0, 3.7]]
            // Output tensor:
            // [[143, 126],
            //  [128, 165]]

        Parameters
        ----------
        in0 : Operand
            Input floating-point tensor to be quantized
        scale : float
            Scale factor for quantization (each integer step represents this value)
        zero_point : int
            Integer value that represents 0.0 in the quantized space
        dtype : torch.dtype
            Target integer data type for quantization (e.g., torch.int8)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The quantized integer tensor
        """

        # kwargs passed thru output type
        return self._op_proxy(
            ttir.QuantizeOp,
            [in0],
            golden_kwargs={"scale": scale, "zero_point": zero_point, "dtype": dtype},
            output_type=self._get_type_from_torch_dtype(
                dtype=dtype, scale=scale, zero_point=zero_point
            ),
            unit_attrs=unit_attrs,
        )

    def dequantize(
        self,
        in0: Operand,
        scale: float,
        zero_point: int,
        dtype: torch.dtype,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.dequantize``.

        *Dequantize integer tensor to floating-point tensor.*

        Converts a quantized integer tensor back into a floating-point tensor using the
        specified scale and zero_point parameters. For each element in the input tensor,
        computes: output[i] = (input[i] - zero_point) * scale

        .. code-block:: mlir

            // Dequantize int8 tensor to float32
            %result = ttir.dequantize(%input, %output) {scale = 0.1 : f32, zero_point = 128 : i32} : tensor<2x2xi8>, tensor<2x2xf32> -> tensor<2x2xf32>
            // Input tensor:
            // [[143, 126],
            //  [128, 165]]
            // Output tensor:
            // [[1.5, -0.2],
            //  [0.0, 3.7]]

        Parameters
        ----------
        in0 : Operand
            Input quantized integer tensor to be dequantized
        scale : float
            Scale factor used in the original quantization
        zero_point : int
            Integer value that represents 0.0 in the quantized space
        dtype : torch.dtype
            Target floating-point data type (e.g., torch.float32)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The dequantized floating-point tensor
        """
        return self._op_proxy(
            ttir.DequantizeOp,
            [in0],
            output_type=self._get_type_from_torch_dtype(dtype=dtype),
            unit_attrs=unit_attrs,
        )

    def requantize(
        self,
        in0: Operand,
        scale: float,
        zero_point: int,
        dtype: torch.dtype,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.requantize``.

        *Requantize integer tensor to new scale and zero-point.*

        Converts a quantized integer tensor from one quantization scheme to another using
        new scale and zero-point parameters. For each element in the input tensor, computes:
        output[i] = round((input[i] - input_zero_point) * (input_scale / output_scale)) + output_zero_point

        .. code-block:: mlir

            // Requantize int8 tensor to new scale and zero-point
            %result = ttir.requantize(%input, %output) {scale = 0.2 : f32, zero_point = 100 : i32} : tensor<2x2xi8>, tensor<2x2xi8> -> tensor<2x2xi8>
            // Input tensor (scale=0.1, zero_point=128):
            // [[143, 126],
            //  [128, 165]]
            // Output tensor (scale=0.2, zero_point=100):
            // [[107, 98],
            //  [100, 119]]

        Parameters
        ----------
        in0 : Operand
            Input quantized integer tensor to be requantized
        scale : float
            New scale factor for requantization
        zero_point : int
            New integer value that represents 0.0 in the quantized space
        dtype : torch.dtype
            Target integer data type (e.g., torch.int8)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The requantized integer tensor with new scale and zero-point
        """
        # kwargs passed thru output type
        return self._op_proxy(
            ttir.RequantizeOp,
            [in0],
            golden_kwargs={"scale": scale, "zero_point": zero_point, "dtype": dtype},
            output_type=self._get_type_from_torch_dtype(
                dtype=dtype, scale=scale, zero_point=zero_point
            ),
            unit_attrs=unit_attrs,
        )

    def tilize(
        self,
        in0: Operand,
        output_type: RankedTensorType,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.tilize``.

        *Convert tensor to tiled layout.*

        Transforms a tensor into a tiled layout format, where data is organized into
        regular blocks or tiles. This can improve memory access patterns and cache
        utilization for certain operations.

        .. code-block:: mlir

            // Convert tensor to tiled layout
            %result = ttir.tilize(%input) : tensor<128x128xf32> -> tensor<128x128xf32, #tiled<32x32>>
            // Input tensor (standard layout):
            // [[1.5, 2.0, ...],
            //  [3.0, 4.0, ...]]
            // Output tensor (tiled 32x32 layout):
            // Same values but organized in 32x32 tiles

        Parameters
        ----------
        in0 : Operand
            Input tensor to be tiled
        output_type : RankedTensorType
            Target type specifying the desired tiled layout
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The tensor with tiled layout
        """
        return self._op_proxy(
            ttir.ToLayoutOp,
            [in0],
            output_type=output_type,
            output_create_fn=self._create_empty_from_tensor_type,
            organize_ttir_args=lambda i, o: (
                [self._get_type(o)],
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
            golden_kwargs={"tilize": True},
        )

    def untilize(
        self,
        in0: Operand,
        output_type: RankedTensorType,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.untilize``.

        *Convert tensor from tiled to standard layout.*

        Transforms a tensor from a tiled layout back to a standard row-major or
        column-major layout. This is the inverse operation of tilize.

        .. code-block:: mlir

            // Convert tensor from tiled to standard layout
            %result = ttir.untilize(%input) : tensor<128x128xf32, #tiled<32x32>> -> tensor<128x128xf32>
            // Input tensor (tiled 32x32 layout):
            // Data organized in 32x32 tiles
            // Output tensor (standard layout):
            // [[1.5, 2.0, ...],
            //  [3.0, 4.0, ...]]

        Parameters
        ----------
        in0 : Operand
            Input tensor with tiled layout
        output_type : RankedTensorType
            Target type specifying the desired standard layout
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The tensor with standard layout
        """
        return self._op_proxy(
            ttir.ToLayoutOp,
            [in0],
            output_type=output_type,
            output_create_fn=self._create_empty_from_tensor_type,
            organize_ttir_args=lambda i, o: (
                [self._get_type(o)],
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
            golden_kwargs={"tilize": False},
        )

    ############### ttir.RMSNormOp ###############

    @tag(ttir.RMSNormOp)
    def rms_norm(
        self,
        in0: Operand,
        normalized_shape: List[int],
        weight: Optional[Operand] = None,
        bias: Optional[Operand] = None,
        epsilon: float = 1e-5,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.rms_norm)
        normalized_shape_attr = DenseI64ArrayAttr.get(normalized_shape)
        epsilon_attr = FloatAttr.get_f32(epsilon)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        weight0 = self._get_golden_tensor(weight) if weight is not None else None
        bias0 = self._get_golden_tensor(bias) if bias is not None else None
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            weight=weight0,
            bias=bias0,
            normalized_shape=normalized_shape_attr,
            epsilon=epsilon_attr,
            output_type_mlir=mlir_output_type,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            normalized_shape_attr,
            weight=weight,
            bias=bias,
            epsilon=epsilon_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.RMSNormOp)
    def rms_norm_parser(
        self,
        old_op: ttir.RMSNormOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.rms_norm_parser)
        in0 = global_dict[old_op.input]
        weight = global_dict[old_op.weight] if old_op.weight else None
        bias = global_dict[old_op.bias] if old_op.bias else None
        normalized_shape_attr = old_op.normalized_shape
        epsilon_attr = old_op.epsilon
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            normalized_shape_attr,
            weight=weight,
            bias=bias,
            epsilon=epsilon_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        weight0 = self._get_golden_tensor(weight) if weight is not None else None
        bias0 = self._get_golden_tensor(bias) if bias is not None else None
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            weight0,
            bias0,
            normalized_shape_attr,
            epsilon_attr,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {old_op.result: new_op_result}
        return new_op, op_map_dictionary

    @split(ttir.RMSNormOp)
    def rms_norm_split(
        self,
        old_op: ttir.RMSNormOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.rms_norm_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            rms_norm_module = Module.create()
            rms_norm_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]
            if old_op.weight is not None:
                op_input_types.append(old_op.weight.type)
            if old_op.bias is not None:
                op_input_types.append(old_op.bias.type)

            with InsertionPoint(rms_norm_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="rms_norm_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    idx = 1
                    weight = None
                    bias = None
                    if old_op.weight is not None:
                        weight = inputs[idx]
                        idx += 1
                    if old_op.bias is not None:
                        bias = inputs[idx]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        old_op.normalized_shape,
                        weight=weight,
                        bias=bias,
                        epsilon=old_op.epsilon,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    weight0 = (
                        self._get_golden_tensor(old_op.weight)
                        if old_op.weight is not None
                        else None
                    )
                    bias0 = (
                        self._get_golden_tensor(old_op.bias)
                        if old_op.bias is not None
                        else None
                    )

                    rms_norm_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    rms_norm_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    if weight is not None:
                        rms_norm_builder._set_golden_tensor(weight, weight0)
                        ordered_inputs.append(weight)
                    if bias is not None:
                        rms_norm_builder._set_golden_tensor(bias, bias0)
                        ordered_inputs.append(bias)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                rms_norm_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return rms_norm_module, rms_norm_builder

    ############### ttir.SplitQueryKeyValueAndSplitHeadsOp ###############

    @tag(ttir.SplitQueryKeyValueAndSplitHeadsOp)
    def split_query_key_value_and_split_heads(
        self,
        input_tensor: Operand,
        num_heads: int,
        transpose_key: bool = False,
        kv_input_tensor: Optional[Operand] = None,
        num_kv_heads: Optional[int] = None,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> Tuple[OpResult, OpResult, OpResult]:
        ttir_op = self.get_opview_from_method(
            TTIRBuilder.split_query_key_value_and_split_heads
        )
        num_heads_attr = IntegerAttr.get(IntegerType.get_unsigned(32), num_heads)
        num_kv_heads_attr = (
            IntegerAttr.get(IntegerType.get_unsigned(32), num_kv_heads)
            if num_kv_heads is not None
            else None
        )
        transpose_key_attr = BoolAttr.get(transpose_key)

        if output_type is None:
            mlir_output_type = self.get_type(input_tensor)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(input_tensor)
        kv_input0 = (
            self._get_golden_tensor(kv_input_tensor)
            if kv_input_tensor is not None
            else None
        )
        op_golden_function = get_golden_function(ttir_op)
        golden_query, golden_key, golden_value = op_golden_function(
            input0,
            kv_input0,
            num_heads_attr,
            num_kv_heads_attr,
            transpose_key_attr,
            mlir_output_type,
            mlir_output_type,
            mlir_output_type,
        )

        query_type = self._create_ranked_tensor_type(
            golden_query.shape, mlir_output_type
        )
        key_type = self._create_ranked_tensor_type(golden_key.shape, mlir_output_type)
        value_type = self._create_ranked_tensor_type(
            golden_value.shape, mlir_output_type
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        # Build kwargs dict for the op
        # Note: kv_input_tensor must be passed as keyword argument per the MLIR binding signature
        op_kwargs = {
            "num_heads": num_heads_attr,
            "transpose_key": transpose_key_attr,
            "loc": loc,
        }
        if num_kv_heads_attr is not None:
            op_kwargs["num_kv_heads"] = num_kv_heads_attr
        if kv_input_tensor is not None:
            op_kwargs["kv_input_tensor"] = kv_input_tensor

        op = ttir_op(
            query_type,
            key_type,
            value_type,
            input_tensor,
            **op_kwargs,
        )

        op_query = op.query
        op_key = op.key
        op_value = op.value

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_query, golden_query)
        self._set_golden_tensor(op_key, golden_key)
        self._set_golden_tensor(op_value, golden_value)

        return op_query, op_key, op_value

    @parse(ttir.SplitQueryKeyValueAndSplitHeadsOp)
    def split_query_key_value_and_split_heads_parser(
        self,
        old_op: ttir.SplitQueryKeyValueAndSplitHeadsOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(
            TTIRBuilder.split_query_key_value_and_split_heads_parser
        )

        input_tensor = global_dict[old_op.input_tensor]
        kv_input_tensor = (
            global_dict[old_op.kv_input_tensor]
            if old_op.kv_input_tensor is not None
            else None
        )
        num_heads_attr = old_op.num_heads
        num_kv_heads_attr = old_op.num_kv_heads if old_op.num_kv_heads else None
        transpose_key_attr = old_op.transpose_key
        query_type = old_op.query.type
        key_type = old_op.key.type
        value_type = old_op.value.type

        op_kwargs = {
            "num_heads": num_heads_attr,
            "transpose_key": transpose_key_attr,
            "loc": old_op.location,
        }
        if num_kv_heads_attr is not None:
            op_kwargs["num_kv_heads"] = num_kv_heads_attr
        if kv_input_tensor is not None:
            op_kwargs["kv_input_tensor"] = kv_input_tensor

        new_op = ttir_op(
            query_type,
            key_type,
            value_type,
            input_tensor,
            **op_kwargs,
        )

        new_op_query = new_op.query
        new_op_key = new_op.key
        new_op_value = new_op.value

        input0 = self._get_golden_tensor(input_tensor)
        kv_input0 = (
            self._get_golden_tensor(kv_input_tensor)
            if kv_input_tensor is not None
            else None
        )
        op_golden_function = get_golden_function(ttir_op)
        golden_query, golden_key, golden_value = op_golden_function(
            input0,
            kv_input0,
            num_heads_attr,
            num_kv_heads_attr,
            transpose_key_attr,
            query_type.element_type,
            key_type.element_type,
            value_type.element_type,
        )
        self._set_golden_tensor(new_op_query, golden_query)
        self._set_golden_tensor(new_op_key, golden_key)
        self._set_golden_tensor(new_op_value, golden_value)

        op_map_dictionary = {}
        op_map_dictionary[old_op.query] = new_op_query
        op_map_dictionary[old_op.key] = new_op_key
        op_map_dictionary[old_op.value] = new_op_value
        return new_op, op_map_dictionary

    @split(ttir.SplitQueryKeyValueAndSplitHeadsOp)
    def split_query_key_value_and_split_heads_split(
        self,
        old_op: ttir.SplitQueryKeyValueAndSplitHeadsOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(
            TTIRBuilder.split_query_key_value_and_split_heads_split
        )

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            split_qkv_module = Module.create()
            split_qkv_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input_tensor.type]
            if old_op.kv_input_tensor is not None:
                op_input_types.append(old_op.kv_input_tensor.type)

            with InsertionPoint(split_qkv_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="split_qkv_module")
                def decorated_func(*inputs):
                    input_tensor = inputs[0]
                    kv_input_tensor = inputs[1] if len(inputs) > 1 else None
                    query_type = old_op.query.type
                    key_type = old_op.key.type
                    value_type = old_op.value.type

                    op_kwargs = {
                        "num_heads": old_op.num_heads,
                        "transpose_key": old_op.transpose_key,
                        "loc": old_op.location,
                    }
                    if old_op.num_kv_heads:
                        op_kwargs["num_kv_heads"] = old_op.num_kv_heads
                    if kv_input_tensor is not None:
                        op_kwargs["kv_input_tensor"] = kv_input_tensor

                    new_op = ttir_op(
                        query_type,
                        key_type,
                        value_type,
                        input_tensor,
                        **op_kwargs,
                    )

                    new_op_query = new_op.query
                    new_op_key = new_op.key
                    new_op_value = new_op.value

                    input0 = self._get_golden_tensor(old_op.input_tensor)
                    kv_input0 = (
                        self._get_golden_tensor(old_op.kv_input_tensor)
                        if old_op.kv_input_tensor is not None
                        else None
                    )
                    op_golden_function = get_golden_function(ttir_op)
                    golden_query, golden_key, golden_value = op_golden_function(
                        input0,
                        kv_input0,
                        old_op.num_heads,
                        old_op.num_kv_heads if old_op.num_kv_heads else None,
                        old_op.transpose_key,
                        query_type.element_type,
                        key_type.element_type,
                        value_type.element_type,
                    )
                    split_qkv_builder._set_golden_tensor(new_op_query, golden_query)
                    split_qkv_builder._set_golden_tensor(new_op_key, golden_key)
                    split_qkv_builder._set_golden_tensor(new_op_value, golden_value)
                    split_qkv_builder._set_golden_tensor(input_tensor, input0)
                    ordered_inputs.append(input_tensor)
                    if kv_input_tensor is not None:
                        split_qkv_builder._set_golden_tensor(kv_input_tensor, kv_input0)
                        ordered_inputs.append(kv_input_tensor)
                    ordered_outputs.extend([new_op_query, new_op_key, new_op_value])

                    return new_op

                new_func_op = decorated_func.func_op
                split_qkv_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return split_qkv_module, split_qkv_builder

    ############### ttir.DistributedRMSNormOp ###############

    @tag(ttir.DistributedRMSNormOp)
    def distributed_rms_norm(
        self,
        input: Operand,
        cluster_axis: int,
        weight: Optional[Operand] = None,
        residual: Optional[Operand] = None,
        epsilon: float = 1e-5,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        """
        Creates ``ttir.distributed_rms_norm``.

        *Distributed RMS normalization with all-gather operation.*

        Performs a fused distributed RMS normalization followed by an all-gather
        collective operation across mesh devices. This operation combines:
        1. Optional residual addition (input + residual)
        2. RMS normalization: output = input * rsqrt(mean(input^2) + epsilon) * weight
        3. All-gather to collect results across cluster_axis

        This is a multi-device operation that requires the tensor to be sharded
        across a device mesh.

        Parameters
        ----------
        input : Operand
            Input tensor to be normalized (must be width-sharded)
        cluster_axis : int
            Mesh dimension to all-gather across (0 or 1)
        weight : Optional[Operand], optional
            Scale parameter (gamma) tensor
        residual : Optional[Operand], optional
            Optional residual tensor for fused add
        epsilon : float, optional
            Small constant for numerical stability (default: 1e-5)
        output_type : Optional[torch.dtype], optional
            Output data type
        loc : Optional[str], optional
            Location string for debugging
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        (*OpResult*)
            The normalized and gathered tensor
        """
        ttir_op = self.get_opview_from_method(TTIRBuilder.distributed_rms_norm)

        if output_type is None:
            mlir_output_type = self.get_type(input)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        cluster_axis_attr = IntegerAttr.get(IntegerType.get_unsigned(32), cluster_axis)
        epsilon_attr = FloatAttr.get_f32(epsilon)

        input0 = self._get_golden_tensor(input)
        weight0 = self._get_golden_tensor(weight) if weight is not None else None
        residual0 = self._get_golden_tensor(residual) if residual is not None else None
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            weight=weight0,
            residual=residual0,
            cluster_axis_attr=cluster_axis_attr,
            epsilon_attr=epsilon_attr,
            output_type_mlir=mlir_output_type,
        )

        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            input,
            cluster_axis_attr,
            weight=weight,
            residual=residual,
            epsilon=epsilon_attr,
            loc=loc,
        )
        new_op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(new_op_result, golden_output)

        return new_op_result

    @parse(ttir.DistributedRMSNormOp)
    def distributed_rms_norm_parser(
        self,
        old_op: ttir.DistributedRMSNormOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.distributed_rms_norm_parser)

        in0 = global_dict[old_op.input]
        weight = global_dict[old_op.weight] if old_op.weight else None
        residual = global_dict[old_op.residual] if old_op.residual else None
        result = old_op.result.type
        cluster_axis_attr = old_op.cluster_axis
        epsilon_attr = old_op.epsilon

        new_op = ttir_op(
            result,
            in0,
            cluster_axis_attr,
            weight=weight,
            residual=residual,
            epsilon=epsilon_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        weight0 = self._get_golden_tensor(weight) if weight is not None else None
        residual0 = self._get_golden_tensor(residual) if residual is not None else None

        # Compute golden
        if residual0 is not None:
            normalized_input = input0 + residual0
        else:
            normalized_input = input0

        epsilon = epsilon_attr.value
        rms = torch.sqrt(
            torch.mean(normalized_input**2, dim=-1, keepdim=True) + epsilon
        )
        golden_output = normalized_input / rms
        if weight0 is not None:
            golden_output = golden_output * weight0

        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {old_op.result: new_op_result}
        return new_op, op_map_dictionary

    ############### ttir.LayerNormOp ###############

    @tag(ttir.LayerNormOp)
    def layer_norm(
        self,
        in0: Operand,
        normalized_shape: List[int],
        weight: Optional[Operand] = None,
        bias: Optional[Operand] = None,
        epsilon: float = 1e-5,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttir_op = self.get_opview_from_method(TTIRBuilder.layer_norm)
        normalized_shape_attr = DenseI64ArrayAttr.get(normalized_shape)
        epsilon_attr = FloatAttr.get_f32(epsilon)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        weight0 = self._get_golden_tensor(weight) if weight is not None else None
        bias0 = self._get_golden_tensor(bias) if bias is not None else None
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            weight=weight0,
            bias=bias0,
            normalized_shape=normalized_shape_attr,
            epsilon=epsilon_attr,
            output_type_mlir=mlir_output_type,
        )
        result = self._create_ranked_tensor_type(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            result,
            in0,
            normalized_shape_attr,
            weight=weight,
            bias=bias,
            epsilon=epsilon_attr,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttir.LayerNormOp)
    def layer_norm_parser(
        self,
        old_op: ttir.LayerNormOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.layer_norm_parser)
        in0 = global_dict[old_op.input]
        weight = global_dict[old_op.weight] if old_op.weight else None
        bias = global_dict[old_op.bias] if old_op.bias else None
        normalized_shape_attr = old_op.normalized_shape
        epsilon_attr = old_op.epsilon
        result = old_op.result.type

        new_op = ttir_op(
            result,
            in0,
            normalized_shape_attr,
            weight=weight,
            bias=bias,
            epsilon=epsilon_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        input0 = self._get_golden_tensor(in0)
        weight0 = self._get_golden_tensor(weight) if weight is not None else None
        bias0 = self._get_golden_tensor(bias) if bias is not None else None
        op_golden_function = get_golden_function(ttir_op)
        golden_output = op_golden_function(
            input0,
            weight0,
            bias0,
            normalized_shape_attr,
            epsilon_attr,
            result.element_type,
        )
        self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {old_op.result: new_op_result}
        return new_op, op_map_dictionary

    @split(ttir.LayerNormOp)
    def layer_norm_split(
        self,
        old_op: ttir.LayerNormOp,
    ) -> Tuple[Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.layer_norm_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            layer_norm_module = Module.create()
            layer_norm_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]
            if old_op.weight is not None:
                op_input_types.append(old_op.weight.type)
            if old_op.bias is not None:
                op_input_types.append(old_op.bias.type)

            with InsertionPoint(layer_norm_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="layer_norm_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    idx = 1
                    weight = None
                    bias = None
                    if old_op.weight is not None:
                        weight = inputs[idx]
                        idx += 1
                    if old_op.bias is not None:
                        bias = inputs[idx]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        old_op.normalized_shape,
                        weight=weight,
                        bias=bias,
                        epsilon=old_op.epsilon,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    input0 = self._get_golden_tensor(old_op.input)
                    weight0 = (
                        self._get_golden_tensor(old_op.weight)
                        if old_op.weight is not None
                        else None
                    )
                    bias0 = (
                        self._get_golden_tensor(old_op.bias)
                        if old_op.bias is not None
                        else None
                    )

                    layer_norm_builder._set_golden_tensor(
                        new_op_result, self._goldens[old_op.result]
                    )
                    layer_norm_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    if weight is not None:
                        layer_norm_builder._set_golden_tensor(weight, weight0)
                        ordered_inputs.append(weight)
                    if bias is not None:
                        layer_norm_builder._set_golden_tensor(bias, bias0)
                        ordered_inputs.append(bias)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                layer_norm_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return layer_norm_module, layer_norm_builder

    # ----- Parse ttir module ----

    @staticmethod
    def from_module(
        ctx: Context,
        mlir_text: str,
        golden_inputs: Dict[str, List[torch.tensor]] = None,
    ) -> Tuple(Module, TTIRBuilder):
        if golden_inputs is None:
            golden_inputs = {}

        root_module = Module.parse(mlir_text, ctx)
        loc = Location.unknown(ctx)
        with ctx, loc:
            mesh_name = "mesh"
            mesh_shape = OrderedDict([("x", 1), ("y", 1)])

            for named_attr in root_module.operation.attributes:
                if named_attr.name != "ttcore.meshes":
                    continue

                meshes = ttcore.ir.MeshesAttr.maybe_downcast(named_attr.attr)
                mesh = meshes.meshes[0]
                mesh_name = mesh.name
                shape = mesh.shape
                mesh_shape = OrderedDict(
                    x=1 if len(shape) == 1 else shape[0],
                    y=shape[0] if len(shape) == 1 else shape[1],
                )
                break

            ttir_builder = TTIRBuilder(ctx, loc, mesh_name, mesh_shape)
            new_module = ttir_builder.parse_root_module(root_module, golden_inputs)

        return new_module, ttir_builder

    # ----- Split ttir module ----

    def split_op(
        self,
        parsed_op: Operation,
    ) -> Tuple[Module, TTIRBuilder]:
        split_function = self.get_split_from_opview(type(parsed_op))
        return split_function(self, parsed_op)

    @staticmethod
    def split_module(
        module: Module,
        builder: TTIRBuilder,
    ) -> List[Tuple[Module, TTIRBuilder]]:
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
                                        if isinstance(op, func.ReturnOp) or isinstance(
                                            op,
                                            ttir.EmptyOp,
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
                        for op in block.operations:
                            if isinstance(op, func.ReturnOp) or isinstance(
                                op,
                                ttir.EmptyOp,
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

    ############### ttir.TopKOp ###############

    @tag(ttir.TopKOp)
    def topk(
        self,
        in0: Operand,
        k: int,
        dim: int = -1,
        largest: bool = True,
        sorted: bool = True,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> Tuple[OpResult, OpResult]:
        ttir_op = self.get_opview_from_method(TTIRBuilder.topk)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        k_attr = IntegerAttr.get(IntegerType.get_signless(32), k)
        dim_attr = IntegerAttr.get(IntegerType.get_signless(32), dim)
        largest_attr = BoolAttr.get(largest)
        sorted_attr = BoolAttr.get(sorted)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttir_op)
        golden_values, golden_indices = op_golden_function(
            input0, k_attr, dim_attr, largest_attr, sorted_attr, mlir_output_type
        )
        values = self._create_ranked_tensor_type(golden_values.shape, mlir_output_type)
        indices = self._create_ranked_tensor_type(
            golden_indices.shape, golden_indices.dtype
        )

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttir_op(
            values,
            indices,
            in0,
            k=k_attr,
            dim=dim_attr,
            largest=largest_attr,
            sorted=sorted_attr,
            loc=loc,
        )
        op_values = op.values

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_values, golden_values)

        return op_values

    @parse(ttir.TopKOp)
    def topk_parser(
        self,
        old_op: ttir.TopKOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttir_op = self.get_opview_from_parser(TTIRBuilder.topk_parser)
        input_tensor = global_dict[old_op.input_tensor]
        k_attr = old_op.k
        dim_attr = old_op.dim
        largest_attr = old_op.largest
        sorted_attr = old_op.sorted
        result = old_op.result.type

        new_op = ttir_op(
            result,
            input_tensor,
            k=k_attr,
            dim=dim_attr,
            largest=largest_attr,
            sorted=sorted_attr,
            loc=old_op.location,
        )
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input = self._get_golden_tensor(input_tensor)
            op_golden_function = get_golden_function(ttir_op)
            golden_output = op_golden_function(
                input, k_attr, dim_attr, largest_attr, sorted_attr, result.element_type
            )
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttir.TopKOp)
    def topk_split(
        self,
        old_op: ttir.TopKOp,
    ) -> Tuple[torch.Module, TTIRBuilder]:
        ttir_op = self.get_opview_from_split(TTIRBuilder.topk_split)

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:

            topk_module = Module.create()
            topk_builder = TTIRBuilder(old_ctx, old_loc)
            op_input_types = [old_op.input.type]

            with InsertionPoint(topk_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="topk_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = ttir_op(
                        result,
                        in0,
                        old_op.k,
                        old_op.dim,
                        old_op.largest,
                        old_op.sorted,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttir_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(
                            input0,
                            old_op.k,
                            old_op.dim,
                            old_op.largest,
                            old_op.sorted,
                            result.element_type,
                        )
                        topk_builder._set_golden_tensor(new_op_result, golden_output)
                        topk_builder._set_golden_tensor(in0, input0)
                        ordered_inputs.append(in0)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                topk_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return topk_module, topk_builder
