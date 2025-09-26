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
from collections import OrderedDict

from ttmlir.ir import *
from ttmlir.dialects import d2m, ttcore, tensor, quant
from ttmlir.passes import GoldenTensor, DataType

from builder.base.builder import *
from builder.base import builder_golden


class D2MBuilder(Builder):
    """
    Builder class for creating D2M (Direct-to-Metal) operations.

    This builder creates D2M operations directly, bypassing the TTIR dialect.
    It's designed for use cases where operations need to be generated for
    pipelines that work directly with D2M operations.
    """

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

    # ----- Private methods -----

    def _op_proxy(
        self,
        op_d2m_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
        organize_d2m_args: Optional[Callable] = None,
        organize_golden_args: Optional[Callable] = None,
        output_shape: Optional[Shape] = None,
        output_type: Optional[Type] = None,
        output_create_fn: Optional[Callable] = None,
        golden_kwargs: dict = {},
        d2m_kwargs: dict = {},
        loc: Optional[Union[str, Location]] = None,
        skip_golden: bool = False,
    ) -> Any:
        """
        Proxy method for creating D2M operations with golden tensor support.
        Similar to TTIRBuilder._op_proxy but for D2M operations.
        """
        if not golden_kwargs:
            golden_kwargs = d2m_kwargs

        if organize_d2m_args is None:
            organize_d2m_args = self._organize_eltwise_d2m

        if organize_golden_args is None:
            organize_golden_args = self._organize_eltwise_golden

        if output_create_fn is None:
            output_create_fn = self._create_empty_from_tensor_type

        if output_type is None:
            output_type = self._get_type(inputs[0])

        if output_shape is None:
            output_shape = self._get_type(inputs[0]).shape

        with self._ctx, self._loc:
            output = output_create_fn(output_shape, output_type)

            if loc is not None:
                loc = Location.name(loc, context=self._ctx)

            op = op_d2m_function(
                *organize_d2m_args(inputs, output, output_shape),
                loc=loc,
                **d2m_kwargs,
            )

            # Set unit attributes if provided.
            if unit_attrs is not None:
                for attr_name in unit_attrs:
                    op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

            if not skip_golden and not self._disable_golden_check:
                op_golden_function = builder_golden.get_golden_function(
                    op_d2m_function, **golden_kwargs
                )
                if op_golden_function is not None:
                    if len(inputs) == 0:
                        golden_output = op_golden_function(**golden_kwargs)
                    else:
                        golden_output = op_golden_function(
                            *(organize_golden_args(inputs)), **golden_kwargs
                        )
                    self._set_golden_tensor(op, golden_output)

            return op

    # ----- Public methods -----

    # ----- D2M Operation Creation Methods -----

    def _get_empty_op(self, tensor_type: RankedTensorType) -> OpView:
        """Get D2M-specific empty operation."""
        return d2m.EmptyOp(tensor_type)

    # ----- D2M Layout Operations -----

    def to_layout(
        self,
        input: Operand,
        output_type: Type,
        unit_attrs: Optional[List[str]] = None,
        loc: Optional[Union[str, Location]] = None,
    ) -> OpView:
        """
        Create a D2M to_layout operation.

        Args:
            input: Input operand
            output_type: Desired output type with layout
            unit_attrs: Optional unit attributes
            loc: Optional location string

        Returns:
            OpView of the D2M to_layout operation
        """

        def organize_to_layout_args(inputs, output, output_shape):
            # D2M ToLayoutOp expects: (results_, input, output)
            return ([output_type], inputs[0], output)

        return self._op_proxy(
            d2m.ToLayoutOp,
            [input],
            unit_attrs=unit_attrs,
            organize_d2m_args=organize_to_layout_args,
            output_type=output_type,
            output_shape=output_type.shape,
            loc=loc,
        )

    def view_layout(
        self,
        input: Operand,
        output_type: Type,
        reinterpret_layout: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Create a D2M view_layout operation."""

        return self._op_proxy(
            d2m.ViewLayoutOp,
            [input],
            d2m_kwargs={"reinterpretLayout": reinterpret_layout},
            output_type=output_type,
            output_shape=output_type.shape,
            output_create_fn=self._create_empty_from_tensor_type,
            organize_d2m_args=lambda i, o, _: (self._get_type(o), i[0]),
            unit_attrs=unit_attrs,
        )

    def stream_layout(
        self,
        input: Operand,
        storage: Operand,
    ) -> OpView:
        """Create a D2M stream_layout operation."""
        with self._ctx, self._loc:
            # Determine result type based on input type
            result_type = input.type
            return d2m.StreamLayoutOp(input, storage, result_type)

    # ----- D2M Layout Convenience Methods -----

    def tilize(
        self,
        input: Operand,
        output_type: Type,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Create a D2M tilize operation (specialized to_layout with tiled output).

        This is a convenience method that creates a to_layout operation
        with a tiled layout output type.
        """
        return self._op_proxy(
            d2m.ToLayoutOp,
            [input],
            output_type=output_type,
            output_create_fn=self._create_empty_from_tensor_type,
            organize_d2m_args=lambda i, o, _: (
                [self._get_type(o)],
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
            golden_kwargs={"tilize": True},
        )

    def untilize(
        self,
        input: Operand,
        output_type: Type,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Create a D2M untilize operation (specialized to_layout with row-major output).

        This is a convenience method that creates a to_layout operation
        with a row-major (untiled) layout output type.
        """
        return self._op_proxy(
            d2m.ToLayoutOp,
            [input],
            output_type=output_type,
            output_create_fn=self._create_empty_from_tensor_type,
            organize_d2m_args=lambda i, o, _: (
                [self._get_type(o)],
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
            golden_kwargs={"tilize": False},
        )

    # NOTE: GenericRegionOperations are not implemented here; currently, we have no testcases which require generating a generic op from these bindings as opposed to going through ttir named ops.
