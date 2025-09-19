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
            output_type = inputs[0].type

        if output_shape is None:
            output_shape = inputs[0].type.shape

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
                else:
                    # For D2M operations without golden functions, create a placeholder
                    # This maintains parity with the test infrastructure while we port golden functions
                    import torch

                    placeholder_tensor = torch.zeros(output_shape, dtype=torch.float32)
                    # Create a BuilderGoldenTensor with a single shard (device 0)
                    shard_map = {0: placeholder_tensor}
                    mesh_shape = (1, 1)  # Single device mesh
                    placeholder_golden = builder_golden.BuilderGoldenTensor(
                        shard_map, mesh_shape
                    )
                    self._set_golden_tensor(op, placeholder_golden)

            return op

    def _organize_eltwise_d2m(self, inputs, output, output_shape):
        """Organize arguments for D2M elementwise operations."""
        return (self._get_type(output), *inputs, output)

    def _organize_eltwise_golden(self, inputs):
        """Organize arguments for golden tensor computation."""
        return [self._get_golden_tensor(inp) for inp in inputs]

    # ----- Public methods -----

    def get_metal_tensor_layout(
        self,
        logical_shape: Shape,
        tiled=False,
        oobVal=ttcore.OOBVal.Undef,
        memorySpace=ttcore.MemorySpace.DeviceL1,
        grid: Optional[Tuple[int, int]] = None,
        index_map: Optional[AffineMap] = None,
    ):
        """
        Create a metal tensor layout for D2M operations.
        This is similar to TTIRBuilder but creates layouts suitable for D2M.
        """
        ctx = self._ctx

        # Create grid shape by 1s filling logical rank.
        if grid is None:
            original_rank = len(logical_shape)
            grid_shape = [1] * original_rank
        else:
            grid_shape = list(grid)

        worker_grid = [8, 8]

        # Create layout with original logical shape.
        if index_map is None:
            layout = ttcore.ir.MetalLayoutAttr.get(
                ctx, logical_shape, worker_grid, oobVal, memorySpace
            )
        else:
            layout = ttcore.ir.MetalLayoutAttr.get(
                ctx, logical_shape, worker_grid, oobVal, memorySpace, index_map
            )

        shard_shape = []
        for l, g in zip(logical_shape, grid_shape):
            assert l % g == 0, f"Logical shape {l} must be divisible by grid shape {g}"
            shard_shape.append(l // g)

        # Get sharded shape w/ proper collapse & alignment logic.
        typed_layout = ttcore.ir.MetalLayoutAttr.maybe_downcast(layout)
        if typed_layout is None:
            raise RuntimeError("Failed to downcast MetalLayoutAttr")
        device_shape = typed_layout.getDeviceShape(
            grid_shape, [32, 32] if tiled else [1, 1]
        )

        elemType = F32Type.get(ctx)

        # For tiled layouts, ensure the device shape accounts for tiles.
        if tiled:
            elemType = ttcore.ir.TileType.get(ctx, 32, 32, ttcore.DataType.Float32)
            if grid is None or grid == (1, 1):
                # For default 1x1 grid, use exact tile count.
                tile_count_h = (logical_shape[-2] + 31) // 32
                tile_count_w = (logical_shape[-1] + 31) // 32
                device_shape[-2] = tile_count_h
                device_shape[-1] = tile_count_w
            else:
                # For explicit grids, calculate proper sharded tile count.
                shard_h, shard_w = shard_shape[-2], shard_shape[-1]
                tiles_per_shard_h = (shard_h + 31) // 32
                tiles_per_shard_w = (shard_w + 31) // 32
                device_shape[-2] = tiles_per_shard_h
                device_shape[-1] = tiles_per_shard_w

        return RankedTensorType.get(
            device_shape, elemType, layout, Location.unknown(ctx)
        )

    # ----- D2M Operation Creation Methods -----

    def _empty(self, shape: Shape, data_type: Optional[Type] = None) -> OpView:
        """Create a D2M empty operation."""
        dtype = data_type if data_type is not None else F32Type.get(self._ctx)
        return self._create_empty_from_tensor_type(
            shape, self._create_ranked_tensor_type(shape, dtype)
        )

    def _create_empty_from_tensor_type(
        self, shape: Shape, tensor_type: RankedTensorType
    ) -> OpView:
        """Create D2M empty operation from tensor type."""
        with self._ctx, self._loc:
            op = d2m.EmptyOp(tensor_type)
            return op

    # NOTE: We don't implement elementwise operations here because they would
    # still go through TTIR and get converted via ttir-to-d2m-generic pass.
    # D2MBuilder is specifically for operations that need to be D2M from the start,
    # like layout operations that bypass the full pipeline.

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

        def organize_view_layout_args(inputs, output, output_shape):
            # D2M ViewLayoutOp expects: (result_type, input) + reinterpret_layout as kwarg
            return (output_type, inputs[0])

        return self._op_proxy(
            d2m.ViewLayoutOp,
            [input],
            unit_attrs=unit_attrs,
            organize_d2m_args=organize_view_layout_args,
            d2m_kwargs={"reinterpretLayout": reinterpret_layout},
            output_type=output_type,
            output_shape=output_type.shape,
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
        return self.to_layout(input, output_type, unit_attrs)

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
        return self.to_layout(input, output_type, unit_attrs)

    # NOTE: Elementwise operations (add, multiply, exp, etc.) are NOT implemented here.
    # Those should continue to use TTIRBuilder and get converted via ttir-to-d2m-generic.
    # This builder is specifically for top-level D2M operations that need to bypass
    # the TTIR dialect entirely, primarily layout operations.
