# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import inspect
import functools
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
from builder.base.builder_utils import *

from golden import *


class D2MBuilder(Builder):

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
        self.goldens_set = False

    def set_goldens(self, *args, **kwargs):
        super().set_goldens(*args, **kwargs)
        self.goldens_set = True

    def _get_golden_tensor(self, *args, **kwargs):
        if not self.goldens_set:
            raise RuntimeError(
                "You must manually use builder.set_goldens(...) API for D2MBuilder"
            )
        return super()._get_golden_tensor(*args, **kwargs)

    # ----- Private methods -----

    def _op_proxy(
        self,
        op_d2m_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
        organize_d2m_args: Optional[Callable] = None,
        output_shape: Optional[Shape] = None,
        output_type: Optional[Type] = None,
        output_create_fn: Optional[Callable] = None,
        d2m_kwargs: dict = {},
        loc: Optional[Union[str, Location]] = None,
    ) -> Any:
        """
        Proxy method for creating D2M operations with golden tensor support.
        Similar to TTIRBuilder._op_proxy but for D2M operations.
        """

        if organize_d2m_args is None:
            organize_d2m_args = self._organize_eltwise_d2m

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

            return op.result

    def create_tensor_encoding(
        self, shape: Shape, element_type: Union[torch.dtype, TypeInfo]
    ) -> ttnn.ir.TTNNLayoutAttr:
        return None

    # ----- Public methods -----

    # ----- D2M Operation Creation Methods -----

    def _get_empty_op(self, tensor_type: RankedTensorType) -> OpView:
        """Get D2M-specific empty operation."""
        return d2m.EmptyOp(tensor_type).result

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
        remapping: Optional[Union[AffineMap, AffineMapAttr]] = None,
        reinterpret_layout: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """Create a D2M view_layout operation."""

        if remapping is None:
            remapping = AffineMap.get_identity(len(output_type.shape), self._ctx)
        remapping_attr = (
            remapping
            if isinstance(remapping, AffineMapAttr)
            else AffineMapAttr.get(remapping)
        )

        result = self._op_proxy(
            d2m.ViewLayoutOp,
            [input],
            d2m_kwargs={
                "remapping": remapping_attr,
                "reinterpretLayout": reinterpret_layout,
            },
            output_type=output_type,
            output_shape=output_type.shape,
            output_create_fn=self._create_empty_from_tensor_type,
            organize_d2m_args=lambda i, o, _: (self._get_type(o), i[0]),
            unit_attrs=unit_attrs,
        )

        return result

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
        )

    def reblock(
        self,
        input: Operand,
        new_grid: List[int],
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        assert (
            len(input.type.shape) % 2 == 0
        ), f"Input shape must be multiple of 2 {input.type.shape}"
        grid_rank = len(input.type.shape) // 2
        old_grid = list(input.type.shape)[:grid_rank]
        old_shard = list(input.type.shape)[grid_rank:]
        assert (
            len(new_grid) == grid_rank
        ), f"Mismatched input/output grid rank for in {new_grid} and out {old_grid}"
        canonical_shape = [gd * sd for gd, sd in zip(old_grid, old_shard)]
        output_shape = list(new_grid)
        for i, d in enumerate(canonical_shape):
            assert (
                d % new_grid[i] == 0
            ), f"Illegal dims for new grid that don't divide canonical shape at dim[{i}] {d} % {new_grid[i]} != 0"
            output_shape.append(d // new_grid[i])
        layout = input.type.encoding
        output_type = RankedTensorType.get(
            output_shape, input.type.element_type, layout
        )
        remapping = d2m.ir.calculate_reblock_map(
            input.type.shape, output_shape, output_type.context
        )
        return self.view_layout(
            input,
            output_type,
            remapping,
            unit_attrs=unit_attrs,
        )

    # ----- D2M Generic + Region ops -----

    def _create_generic(
        self,
        operands,
        grid,
        block_factors,
        indexing_maps,
        iterator_types,
    ):
        if (
            isinstance(block_factors, list)
            and len(block_factors) > 0
            and isinstance(block_factors[0], tuple)
        ):
            assert isinstance(block_factors, list)
            assert isinstance(block_factors[0], tuple)
            block_factors = [b for bs in block_factors for b in bs]

        inputs = operands[:-1]
        outputs = operands[-1:]
        assert len(outputs) == 1
        ret_type = outputs[0].type
        ctx = ret_type.context
        threads = ArrayAttr.get([d2m.ir.ThreadAttr.get(ctx, "unified")])
        return d2m.GenericOp(
            [ret_type],
            inputs,
            outputs,
            [],  # additional_args
            ttcore.ir.GridAttr.get(ctx, grid),
            block_factors,
            list(map(affine_map_from_lambda, indexing_maps)),
            ArrayAttr.get(
                list(
                    ttcore.ir.IteratorTypeAttr.get(
                        ctx, ttcore.IteratorType[i.title()].value
                    )
                    for i in iterator_types
                )
            ),
            threads,
            len(threads),
        )

    def generic(
        self,
        grid=None,
        block_factors=None,
        indexing_maps=None,
        iterator_types=None,
        skip_grid_selection=False,
    ):
        assert (
            not skip_grid_selection or grid is not None
        ), "grid must be specified if skip_grid_selection is set"
        implicit_blocked_form = block_factors is not None
        if implicit_blocked_form:
            assert (
                indexing_maps is not None
            ), "indexing_maps must be set for generic in implicit blocked form"
            assert (
                iterator_types is not None
            ), "iterator_types must be set for generic in implicit blocked form"
            for indexing_map in indexing_maps:
                num_dims = len(inspect.signature(indexing_map).parameters)
                if iterator_types is not None:
                    assert num_dims == len(iterator_types)
                assert len(block_factors) == num_dims
                num_results = len(indexing_map(*tuple(range(num_dims))))
                if grid is None:
                    grid = [1] * num_results
                assert num_results == len(grid)
        else:
            assert (
                indexing_maps is None
            ), "indexing_maps must not be set for generic in explicit blocked form"
            assert (
                iterator_types is None
            ), "iterator_types must not be set for generic in explicit blocked form"
            assert (
                grid is not None
            ), "grid must be set for generic in explicit blocked form"
            indexing_maps = []
            iterator_types = []

        def _decorator(f):
            @functools.wraps(f)
            def _wrapper(*args, **kwargs):
                nonlocal self
                nonlocal grid
                nonlocal block_factors
                nonlocal indexing_maps
                nonlocal iterator_types
                nonlocal skip_grid_selection

                generic = self._create_generic(
                    args,
                    grid,
                    block_factors,
                    indexing_maps,
                    iterator_types,
                )
                assert len(generic.regions[0].blocks) == 0
                generic.regions[0].blocks.append()
                block = generic.regions[0].blocks[0]
                ctx = generic.context
                loc = generic.location
                if skip_grid_selection:
                    generic.attributes["d2m.skip_grid_selection"] = UnitAttr.get(ctx)
                with InsertionPoint(block):
                    f(*args, **kwargs)

                return generic.result

            return _wrapper

        return _decorator

    def spatial(
        self,
        inputs: Sequence[Value],
        outputs: Sequence[Value],
        grid_ranges: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
        region_builders: Sequence[Callable[[], None]],
        result_types: Optional[Sequence[Type]] = None,
        unit_attrs: Optional[List[str]] = None,
        loc: Optional[Union[str, Location]] = None,
    ) -> Union[Value, OpResultList]:
        """
        Create a d2m.spatial op with one region per grid_ranges entry.

        Each callable in region_builders runs with InsertionPoint at that
        region's entry block. Each region must contain exactly one d2m.generic
        (see SpatialOp::verify). d2m.spatial has NoTerminator. If a region needs
        an explicit terminator, use d2m.spatial_yield / d2m.SpatialYieldOp like
        d2m.yield_ inside d2m.generic.

        Args:
            inputs: Variadic ins operands.
            outputs: Variadic outs (DPS inits).
            grid_ranges: One ((y, x), (y, x)) pair per region: inclusive start
                and inclusive end, same as ttcore.core_range<start, end> today.
            region_builders: One no-argument callable per core range / region.
            result_types: Result tensor types.
            unit_attrs: Optional unit attributes on the spatial op.
            loc: Optional location string.

        Returns:
            The sole OpResult if there is one result, otherwise op.results.
        """
        num_regions = len(region_builders)
        assert num_regions == len(
            grid_ranges
        ), "len(region_builders) must match len(grid_ranges)"

        with self._ctx, self._loc:
            range_attrs: List[Attribute] = []
            for (sy, sx), (ey, ex) in grid_ranges:
                text = f"#ttcore.core_range<({sy}, {sx}), ({ey}, {ex})>"
                range_attrs.append(Attribute.parse(text, self._ctx))
            grid_attr = ArrayAttr.get(range_attrs)
            if loc is not None:
                loc = Location.name(loc, context=self._ctx)
            spatial_op = d2m.SpatialOp(
                list(result_types),
                list(inputs),
                list(outputs),
                grid_attr,
                num_regions,
                loc=loc,
            )
            if unit_attrs is not None:
                for attr_name in unit_attrs:
                    spatial_op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

            for region_idx, build_region in enumerate(region_builders):
                region = spatial_op.regions[region_idx]
                assert len(region.blocks) == 0
                region.blocks.append()
                block = region.blocks[0]
                with InsertionPoint(block):
                    build_region()

            nres = len(spatial_op.results)
            if nres == 1:
                return spatial_op.result
            return spatial_op.results

    def remote_load(
        self, src, indices, mcast_start_index=None, mcast_shape=None, mcast_dims=None
    ):
        dst = tensor.empty(src.type.shape[len(indices) :], src.type.element_type)
        return d2m.remote_load(
            RankedTensorType.get(dst.type.shape, dst.type.element_type),
            src,
            indices,
            mcast_start_index=mcast_start_index,
            mcast_shape=mcast_shape,
            mcast_dims=mcast_dims,
            local_buffer=dst,
        )
