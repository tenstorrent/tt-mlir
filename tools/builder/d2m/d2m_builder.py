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
from ttmlir._mlir_libs import _ttmlir

from builder.base.builder import *
from golden import *


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
                op_golden_function = get_golden_function(
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

    def empty(
        self,
        output_type: Type,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Create a D2M empty operation.

        Args:
            output_type: The type of the empty tensor to create
            unit_attrs: Optional unit attributes

        Returns:
            OpView of the empty operation result
        """
        with self._ctx, self._loc:
            result = d2m.EmptyOp(output_type)

            # Set unit attributes if provided
            if unit_attrs:
                for attr_name in unit_attrs:
                    result.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        return result

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

    def generic_identity(
        self,
        input: Operand,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Create a D2M generic operation that performs identity (forwarding).

        This creates a generic op that simply yields its input unchanged.
        Useful for materializing views created by ViewLayoutOp.

        Args:
            input: Input operand
            unit_attrs: Optional unit attributes

        Returns:
            OpView of the generic operation result
        """
        # Handle both OpView and Value inputs
        if hasattr(input, "result"):
            input_val = input.result
        else:
            input_val = input

        input_type = input_val.type

        # Create output operand with same type as input
        output = self.empty(output_type=input_type, unit_attrs=unit_attrs)
        output_val = output.result

        # Get tensor shape to determine rank
        shaped_type = ShapedType(input_type)
        rank = shaped_type.rank

        with self._ctx, self._loc:
            # Create identity affine map and iterator types manually
            # Create two separate identity maps to avoid potential reuse issues
            identity_map1 = AffineMap.get_identity(rank, context=self._ctx)
            identity_map2 = AffineMap.get_identity(rank, context=self._ctx)
            indexing_maps = ArrayAttr.get(
                [
                    AffineMapAttr.get(identity_map1),
                    AffineMapAttr.get(identity_map2),
                ]
            )

            # Create parallel iterator types
            # Parse each one separately to avoid sharing
            iterator_types = ArrayAttr.get(
                [
                    Attribute.parse("#ttcore.iterator_type<parallel>")
                    for _ in range(rank)
                ]
            )

            # Create grid and threads
            grid_attr = Attribute.parse("#ttcore.grid<1x1>")
            threads_attr = ArrayAttr.get([Attribute.parse("#d2m.thread<compute>")])

            # Create block_factors as ArrayAttr with i64 type
            i64_type = IntegerType.get_signless(64)
            block_factors_list = []
            for _ in range(rank):
                block_factors_list.append(IntegerAttr.get(i64_type, 1))
            block_factors_attr = ArrayAttr.get(block_factors_list)

            # Create operandSegmentSizes: [num_inputs, num_outputs]
            operand_segment_sizes = DenseI32ArrayAttr.get([1, 1])

            # Create the generic op
            generic_op = Operation.create(
                "d2m.generic",
                results=[input_type],
                operands=[input_val, output_val],
                attributes={
                    "indexing_maps": indexing_maps,
                    "iterator_types": iterator_types,
                    "grid": grid_attr,
                    "block_factors": block_factors_attr,
                    "threads": threads_attr,
                    "operandSegmentSizes": operand_segment_sizes,
                },
                regions=1,
            )

            # Populate the region with a block that yields the input
            region = generic_op.regions[0]

            # Block arguments should be circular buffer types for inputs and outputs
            cb_input_type = Type.parse(f"!d2m.cb<{input_type}>", context=self._ctx)
            cb_output_type = Type.parse(f"!d2m.cb<{input_type}>", context=self._ctx)

            block = Block.create_at_start(region, [cb_input_type, cb_output_type])
            with InsertionPoint(block):
                # Identity operation: wait on input, reserve output, yield it
                input_cb = block.arguments[0]
                output_cb = block.arguments[1]

                input_tensor = d2m.WaitOp(input_type, input_cb).result
                output_tensor = d2m.ReserveOp(input_type, output_cb).result

                # For identity, we just copy input to output (element-wise copy via linalg/tensor ops)
                # But for now, just yield the input directly
                d2m.YieldOp([input_tensor])

            result = OpView(generic_op)

            # Set unit attributes if provided
            if unit_attrs:
                for attr_name in unit_attrs:
                    result.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

            # Track golden tensor (pass through from input)
            if hasattr(input, "__hash__") and input in self._goldens:
                self._set_golden_tensor(result, self._goldens[input])

            return result

    # NOTE: GenericRegionOperations are not implemented here; currently, we have no testcases which require generating a generic op from these bindings as opposed to going through ttir named ops.
