# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Callable, Any
import torch

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
        output_type: RankedTensorType,
        ttnn_kwargs: dict = {},
    ) -> Any:
        with self._ctx, self._loc:
            if len(inputs) == 0:
                golden_output = op_golden_function()
            else:
                golden_output = op_golden_function(
                    *self._organize_eltwise_golden(inputs)
                )

            id = self._get_next_global_id()
            loc = self._get_loc_of_extra_file_callee(id=id)

            op = op_ttnn_function(
                output_type,
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
        # Eltwise operations require dtype attribute to be set
        # so we extract it from the input operand
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(inputs[0]),
        }
        output_type = self.create_ttnn_tensor(
            shape=inputs[0].type.shape,
            element_type=inputs[0].type.element_type,
        )
        return self._op_proxy(
            op_golden_function,
            op_ttnn_function,
            inputs,
            output_type,
            ttnn_kwargs=ttnn_kwargs,
        )

    def _get_data_type_attribute(self, operand: Operand) -> ttcore.ir.DataTypeAttr:
        with self._ctx, self._loc:
            dtype = ttnn.ir.TTNNLayoutAttr.maybe_downcast(
                operand.type.encoding
            ).data_type_as_int
            return ttcore.ir.DataTypeAttr.get(self._ctx, dtype)

    # ----- Public Helper Methods ----

    def create_ttnn_tensor(self, shape: Shape, element_type: Type) -> RankedTensorType:
        """
        TTNN tensors require that encoding information is present.
        This method creates a TTNN tensor with encoding information.
        For simplicity we will always create DRAM/Interlaved tiled tensor.
        """
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

    def multiply(self, in0: Operand, in1: Operand) -> OpView:
        """
        Creates ``ttnn.multiply``.
        """
        return self._eltwise_proxy(
            torch.multiply,
            ttnn.MultiplyOp,
            [in0, in1],
        )
