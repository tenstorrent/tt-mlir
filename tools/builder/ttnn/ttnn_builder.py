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
from ttmlir.dialects import ttnn, sdy

from builder.base.builder import *


class TTNNBuilder(Builder):
    # ----- Methods -----

    def __init__(self, ctx: Context, location: Location):
        super().__init__(ctx, location)

    # ----- Private Methods ----

    def _op_proxy(
        self,
        op_golden_function: Callable,
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
            op = op_ttnn_function(
                *inputs,
                loc=loc,
                **ttnn_kwargs,
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
        op_ttnn_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        return self._op_proxy(
            op_golden_function, op_ttnn_function, inputs, unit_attrs
        )

    # ----- Public TTNN Op Generators ----

    def add(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.add``.

        *Elementwise addition operation.*

        Performs elementwise addition between two tensors.
        For each pair of corresponding elements, adds the element in the second
        tensor to the element in the first tensor.

        Mathematical definition: add(x, y) = x + y

        .. code-block:: mlir

            // Add corresponding elements
            %result = ttnn.add(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // lhs: [3.5, 0.0, -1.2]
            // rhs: [1.5, 2.0, -3.2]
            // Output tensor:
            // [5.0, 2.0, -4.4]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise sum of the inputs
        """

        return self._eltwise_proxy(
            torch.add,
            ttnn.AddOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )
