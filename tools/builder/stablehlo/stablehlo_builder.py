# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Callable, Dict, Any
import torch
from enum import Enum, auto
import re

from ttmlir.ir import *
from ttmlir.dialects import stablehlo, sdy

from builder.base.builder import *


class StableHLOBuilder(Builder):
    # ----- Methods -----

    def __init__(self, ctx: Context, location: Location):
        super().__init__(ctx, location)

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

    def _op_proxy(
        self,
        op_golden_function: Callable,
        op_stablehlo_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
        organize_stablehlo_args: Optional[Callable] = None,
        organize_golden_args: Optional[Callable] = None,
        output_shape: Optional[Shape] = None,
        output_type: Optional[Type] = None,
        output_create_fn: Optional[Callable] = None,
        golden_kwargs: dict = {},
        stablehlo_kwargs: dict = {},
        loc: Optional[Union[str, Location]] = None,
    ) -> Any:
        stack = inspect.stack()
        cur_filename = stack[0].filename

        while len(stack) > 0 and stack[0].filename == cur_filename:
            stack = stack[1:]

        if len(stack) == 0:
            raise RuntimeError(
                "Top of callstack to builder funcs must be outside this file"
            )

        if organize_golden_args is None:
            organize_golden_args = self._organize_eltwise_golden

        with self._ctx, self._loc:
            if (
                not isinstance(organize_golden_args(inputs), torch.Tensor)
                and organize_golden_args(inputs) == 0
            ):
                golden_output = op_golden_function(**golden_kwargs)
            else:
                golden_output = op_golden_function(
                    *(organize_golden_args(inputs)),
                    **golden_kwargs,
                )

            golden = (
                Golden(golden_output[0])
                if not isinstance(golden_output, torch.Tensor)
                else Golden(golden_output)
            )
            output_shape = golden.tensor.shape if not output_shape else output_shape
            if not output_type and inputs:
                output_type = self._get_type_from_torch_dtype(
                    self._get_golden_tensor(inputs[0]).dtype
                )
            elif not output_type:
                output_type = self._get_default_dtype()

            id = self._get_next_global_id()
            loc = (
                self._get_loc_from_str(loc)
                if loc is not None
                else self._get_loc_of_extra_file_callee(id=id)
            )
            op = op_stablehlo_function(
                *inputs,
                loc=loc,
                **stablehlo_kwargs,
            )

            if unit_attrs is not None:
                from ttmlir.ir import UnitAttr

                for attr_name in unit_attrs:
                    op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)
            self._id_golden_map[str(loc)] = golden
            self._store_golden(op, golden)
            return op

    def _eltwise_proxy(
        self,
        op_golden_function: Callable,
        op_stablehlo_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        return self._op_proxy(
            op_golden_function, op_stablehlo_function, inputs, unit_attrs
        )

    # ----- Public StableHLO Op Generators ----

    def add(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        return self._eltwise_proxy(
            torch.add,
            stablehlo.AddOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

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

    # ----- Public Shardy Op Generators ----

    def mesh(self, mesh_name: str, mesh_attr: sdy.MeshAttr) -> sdy.MeshOp:
        return sdy.MeshOp(sym_name=mesh_name, mesh=mesh_attr)

    def sharding_constraint(
        self,
        in0: Operand,
        tensor_sharding_attr: sdy.TensorShardingAttr,
    ) -> sdy.ShardingConstraintOp:
        return sdy.ShardingConstraintOp(in0, tensor_sharding_attr)
