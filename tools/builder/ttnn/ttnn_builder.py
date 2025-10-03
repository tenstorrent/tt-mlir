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
from ttmlir import util
from ttmlir.dialects import ttnn, ttcore

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
        golden_kwargs: dict = {},
        ttnn_kwargs: dict = {},
        loc: Optional[Union[str, Location]] = None,
    ) -> Any:
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

            id = self._get_next_global_id()
            loc = (
                self._get_loc_from_str(loc)
                if loc is not None
                else self._get_loc_of_extra_file_callee(id=id)
            )

            op = op_ttnn_function(
                inputs[0].type,
                *inputs,
                loc=loc,
                **ttnn_kwargs,
            )

            if not self._disable_golden_check:
                self._set_golden_tensor(op, golden_output)

            return op

    def _eltwise_proxy(
        self,
        op_golden_function: Callable,
        op_ttnn_function: Callable,
        inputs: List[Operand],
        ttnn_kwargs: dict = {},
    ) -> OpView:
        return self._op_proxy(
            op_golden_function, op_ttnn_function, inputs, ttnn_kwargs=ttnn_kwargs
        )

    # ----- Public Helper Methods ----

    def create_ttnn_tensor(self, shape: Shape, element_type: Type) -> RankedTensorType:
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
            return RankedTensorType.get(shape, element_type, ttnn_layout_attr)

    # ----- Public TTNN Op Generators ----

    def add(self, in0: Operand, in1: Operand) -> OpView:
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

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise sum of the inputs
        """
        dtype = ttnn.ir.TTNNLayoutAttr.maybe_downcast(
            in0.type.encoding
        ).data_type_as_int
        with self._ctx, self._loc:
            dtype_attr = ttcore.ir.DataTypeAttr.get(self._ctx, dtype)
        ttnn_kwargs = {
            "dtype": dtype_attr,
        }
        return self._eltwise_proxy(
            torch.add,
            ttnn.AddOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
        )
