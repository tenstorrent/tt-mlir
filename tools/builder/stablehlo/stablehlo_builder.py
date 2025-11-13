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
from collections import OrderedDict

from ttmlir.ir import *
from ttmlir.dialects import stablehlo, sdy, mpmd

from builder.base.builder import *
from golden import *


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

        self._arg_attrs: Dict[Operand, Dict[str, Attribute]] = {}

    # ----- Public methods -----

    @property
    def arg_attrs(self) -> Dict[Operand, Dict[str, Attribute]]:
        return self._arg_attrs

    def get_arg_attrs(self, func_op: FuncOp) -> ArrayAttr:
        attrs = []
        for i, operand in enumerate(self._ordered_inputs):
            if operand in self._arg_attrs:
                attrs.append(DictAttr.get(self._arg_attrs[operand]))
            else:
                attrs.append(func_op.arg_attrs[i])

        return ArrayAttr.get(attrs)

    def create_sharding_attr_from_tuples(
        self,
        mesh_name: str,
        shardings: List[Tuple[str, bool]],
        replicated_axes: List[sdy.AxisRefAttr] = [],
        unreduced_axes: List[sdy.AxisRefAttr] = [],
    ) -> sdy.TensorShardingPerValueAttr:
        """
        Creates a tensor sharding per value attribute from a list of tuples.
        Each tuple contains a mesh name and a boolean indicating whether the sharding is closed.

        Parameters
        ----------
        mesh_name : str
            The name of the mesh to which the tensor sharding applies
        shardings : List[Tuple[str, bool]]
            A list of tuples, each containing a mesh name and a boolean indicating whether the sharding is closed

        Returns
        -------
        (*sdy.TensorShardingPerValueAttr*)
            A tensor sharding per value attribute that describes how tensors are distributed across the mesh
        """
        dimension_shardings = []
        for sharding in shardings:
            axis_ref_name, is_closed = sharding
            axes = []
            if axis_ref_name != "":
                axes = [self.axis_ref_attr(name=axis_ref_name)]
            dimension_sharding = self.dimension_sharding_attr(
                axes=axes, is_closed=is_closed
            )
            dimension_shardings.append(dimension_sharding)

        tensor_sharding = self.tensor_sharding_attr(
            mesh_name, dimension_shardings, replicated_axes, unreduced_axes
        )
        return self.tensor_sharding_per_value_attr([tensor_sharding])

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
        op_golden_function = builder_golden.get_golden_function(
            op_stablehlo_function, **golden_kwargs
        )
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
                    self._set_golden_tensor(op, golden_output)

            return op

    def _eltwise_proxy(
        self,
        op_stablehlo_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        return self._op_proxy(op_stablehlo_function, inputs, unit_attrs, sharding_attr)

    # ----- Public StableHLO Op Generators ----

    def add(
        self,
        in0: Operand,
        in1: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.add``.

        *Elementwise addition operation.*

        Performs elementwise addition between two tensors.
        For each pair of corresponding elements, adds the element in the second
        tensor to the element in the first tensor.

        Mathematical definition: add(x, y) = x + y

        .. code-block:: mlir

            // Add corresponding elements
            %result = stablehlo.add(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
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
            stablehlo.AddOp,
            [in0, in1],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def clamp(
        self,
        min: Operand,
        operand: Operand,
        max: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.clamp``.

        *Elementwise clamp operation.*

        Clamps each element of the operand tensor between a minimum and maximum value.
        For each element, returns min if element < min, max if element > max, otherwise element.

        Mathematical definition: clamp(min, x, max) = min(max(x, min), max)

        .. code-block:: mlir

            // Clamp elements between min and max
            %result = stablehlo.clamp(%min, %operand, %max) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // min: [5, 10, 15]
            // operand: [3, 13, 23]
            // max: [10, 15, 20]
            // Output tensor:
            // [5, 13, 20]

        Parameters
        ----------
        min : Operand
            Minimum value tensor (can be scalar or tensor)
        operand : Operand
            Input tensor to be clamped
        max : Operand
            Maximum value tensor (can be scalar or tensor)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes
        sharding_attr : *Optional[sdy.TensorShardingPerValueAttr]*
            Optional sharding attribute

        Returns
        -------
        (*OpView*)
            A tensor containing the clamped values
        """
        return self._eltwise_proxy(
            stablehlo.ClampOp,
            [min, operand, max],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    # ----- Elementwise Unary Operations -----

    def abs(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.abs``.

        *Elementwise absolute value operation.*

        Computes the element-wise absolute value of the input tensor.

        Mathematical definition: abs(x) = |x|

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise absolute values of the input
        """
        return self._eltwise_proxy(
            stablehlo.AbsOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def ceil(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.ceil``.

        *Elementwise ceiling operation.*

        Computes the element-wise ceiling of the input tensor.

        Mathematical definition: ceil(x) = ⌈x⌉

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise ceiling values of the input
        """
        return self._eltwise_proxy(
            stablehlo.CeilOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def cosine(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.cosine``.

        *Elementwise cosine operation.*

        Computes the element-wise cosine of the input tensor.

        Mathematical definition: cosine(x) = cos(x)

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise cosine values of the input
        """
        return self._eltwise_proxy(
            stablehlo.CosineOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def exp(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.exponential``.

        *Elementwise exponential operation.*

        Computes the element-wise exponential of the input tensor.

        Mathematical definition: exp(x) = e^x

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise exponential values of the input
        """
        return self._eltwise_proxy(
            stablehlo.ExpOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def floor(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.floor``.

        *Elementwise floor operation.*

        Computes the element-wise floor of the input tensor.

        Mathematical definition: floor(x) = ⌊x⌋

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise floor values of the input
        """
        return self._eltwise_proxy(
            stablehlo.FloorOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def neg(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.negate``.

        *Elementwise negation operation.*

        Computes the element-wise negation of the input tensor.

        Mathematical definition: neg(x) = -x

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise negated values of the input
        """
        return self._eltwise_proxy(
            stablehlo.NegOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def rsqrt(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.rsqrt``.

        *Elementwise reciprocal square root operation.*

        Computes the element-wise reciprocal square root of the input tensor.

        Mathematical definition: rsqrt(x) = 1/√x

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise reciprocal square root values of the input
        """
        return self._eltwise_proxy(
            stablehlo.RsqrtOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def sine(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.sine``.

        *Elementwise sine operation.*

        Computes the element-wise sine of the input tensor.

        Mathematical definition: sine(x) = sin(x)

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise sine values of the input
        """
        return self._eltwise_proxy(
            stablehlo.SineOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def sqrt(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.sqrt``.

        *Elementwise square root operation.*

        Computes the element-wise square root of the input tensor.

        Mathematical definition: sqrt(x) = √x

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise square root values of the input
        """
        return self._eltwise_proxy(
            stablehlo.SqrtOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def logistic(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.logistic``.

        *Elementwise logistic (sigmoid) operation.*

        Computes the element-wise logistic function of the input tensor.

        Mathematical definition: logistic(x) = 1 / (1 + exp(-x))

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise logistic values of the input
        """
        return self._eltwise_proxy(
            stablehlo.LogisticOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def tan(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.tan``.

        *Elementwise tangent operation.*

        Computes the element-wise tangent of the input tensor.

        Mathematical definition: tan(x) = sin(x) / cos(x)

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise tangent values of the input
        """
        return self._eltwise_proxy(
            stablehlo.TanOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
        )

    def log(
        self,
        in0: Operand,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.log``.

        *Elementwise natural logarithm operation.*

        Computes the element-wise natural logarithm of the input tensor.

        Mathematical definition: log(x) = ln(x)

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise natural logarithm values of the input
        """
        return self._eltwise_proxy(
            stablehlo.LogOp,
            [in0],
            unit_attrs=unit_attrs,
        )

    def slice(
        self,
        in0: Operand,
        start_indices: List[int],
        limit_indices: List[int],
        strides: Optional[List[int]] = None,
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.slice``.

        *Slice operation.*

        Extracts a slice from the operand using statically-computed starting indices
        and produces a result tensor. start_indices contain the starting indices of
        the slice for each dimension, limit_indices contain the ending indices
        (exclusive) for the slice for each dimension, and strides contain the
        strides for each dimension.

        More formally: result[result_index] = operand[operand_index] where
        operand_index = start_indices + result_index * strides.

        .. code-block:: mlir

            // %operand: [
            //            [0, 0, 0, 0],
            //            [0, 0, 1, 1],
            //            [0, 0, 1, 1]
            //           ]
            %result = "stablehlo.slice"(%operand) {
              start_indices = array<i64: 1, 2>,
              limit_indices = array<i64: 3, 4>,
              strides = array<i64: 1, 1>
            } : (tensor<3x4xi64>) -> tensor<2x2xi64>
            // %result: [
            //            [1, 1],
            //            [1, 1]
            //           ]

        Parameters
        ----------
        in0 : Operand
            Input tensor to slice
        start_indices : List[int]
            Starting indices of the slice for each dimension
        limit_indices : List[int]
            Ending indices (exclusive) of the slice for each dimension
        strides : *Optional[List[int]]*
            Strides for each dimension (default: [1, 1, ...])
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the extracted slice
        """
        if strides is None:
            strides = [1] * len(start_indices)

        if not (len(start_indices) == len(limit_indices) == len(strides)):
            raise ValueError(
                "start_indices, limit_indices, and strides must have the same length"
            )

        start_indices_attr = DenseI64ArrayAttr.get(start_indices, context=self._ctx)
        limit_indices_attr = DenseI64ArrayAttr.get(limit_indices, context=self._ctx)
        strides_attr = DenseI64ArrayAttr.get(strides, context=self._ctx)

        return self._op_proxy(
            stablehlo.SliceOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
            stablehlo_kwargs={
                "start_indices": start_indices_attr,
                "limit_indices": limit_indices_attr,
                "strides": strides_attr,
            },
            golden_kwargs={
                "start_indices": start_indices,
                "limit_indices": limit_indices,
                "strides": strides,
            },
        )

    # ----- Tensor Manipulation Operations -----

    def concatenate(
        self,
        inputs: List[Operand],
        dim: int = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.concatenate``.

        *Tensor concatenation operation.*

        Concatenates a variadic number of tensors in `inputs` along `dim`
        dimension in the same order as the given arguments. All input tensors
        must have the same shape except in the concatenating dimension.

        .. code-block:: mlir

            // Concatenate two tensors along dimension 0
            %result = stablehlo.concatenate %input0, %input1, dim = 0 : (tensor<2x3xf32>, tensor<1x3xf32>) -> tensor<3x3xf32>
            // Input tensors:
            // input0: [[1.0, 2.0, 3.0],
            //          [4.0, 5.0, 6.0]]
            // input1: [[7.0, 8.0, 9.0]]
            // Output tensor:
            // [[1.0, 2.0, 3.0],
            //  [4.0, 5.0, 6.0],
            //  [7.0, 8.0, 9.0]]

        Parameters
        ----------
        inputs : List[Operand]
            List of input tensors to concatenate. All tensors must have the same
            rank and matching dimensions except along the concatenation dimension.
        dim : int, optional
            Dimension along which to concatenate. Must be in range [0, rank).
            Default is 0.
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing all input tensors concatenated along the specified dimension
        """
        return self._op_proxy(
            stablehlo.ConcatenateOp,
            inputs,
            organize_stablehlo_args=lambda i, o, k: (i,),
            stablehlo_kwargs={"dimension": dim},
            organize_golden_args=lambda i: (
                tuple([self._get_golden_tensor(inp) for inp in i]),
            ),
            golden_kwargs={"dim": dim},
        )

    def transpose(
        self,
        in0: Operand,
        permutation: List[int],
        unit_attrs: Optional[List[str]] = None,
        sharding_attr: Optional[sdy.TensorShardingPerValueAttr] = None,
    ) -> OpView:
        """
        Creates ``stablehlo.transpose``.

        *Tensor transpose operation.*

        Permutes the dimensions of the input tensor according to the given permutation.
        This operation rearranges the axes of the tensor without changing the data.

        Mathematical definition: For a tensor with dimensions [d0, d1, ..., dn-1] and
        permutation [p0, p1, ..., pn-1], the output tensor has dimensions
        [d_p0, d_p1, ..., d_pn-1].

        .. code-block:: mlir
            // Transpose a 2x3 tensor by swapping dimensions 0 and 1
            %result = stablehlo.transpose(%input) {permutation = array<i64: 1, 0>} :
                tensor<2x3xf32> -> tensor<3x2xf32>
            // Input tensor:
            // [[1.0, 2.0, 3.0],
            //  [4.0, 5.0, 6.0]]
            // Output tensor:
            // [[1.0, 4.0],
            //  [2.0, 5.0],
            //  [3.0, 6.0]]

        Parameters
        ----------
        in0 : Operand
            Input tensor to transpose
        permutation : List[int]
            The desired ordering of dimensions (0-indexed)
        unit_attrs : Optional[List[str]]
            Optional list of unit attributes
        sharding_attr : Optional[sdy.TensorShardingPerValueAttr]
            Optional tensor sharding attribute for distributed execution

        Returns
        -------
        (*OpView*)
            A tensor with permuted dimensions according to the permutation
        """
        return self._op_proxy(
            stablehlo.TransposeOp,
            [in0],
            unit_attrs=unit_attrs,
            sharding_attr=sharding_attr,
            stablehlo_kwargs={"permutation": permutation},
        )

    # ----- Public Shardy Attribute Generators ----

    def mesh_axis_attr(
        self,
        name: str,
        size: int,
    ) -> sdy.MeshAxisAttr:
        """
        Creates a mesh axis attribute.
        This attribute represents a single axis in a mesh, defined by its name and size.

        Parameters
        ----------
        name : str
            The name of the mesh axis
        size : int
            The size of the mesh axis, indicating how many elements are along this axis

        Returns
        -------
        (*sdy.MeshAxisAttr*)
            A mesh axis attribute representing the specified axis with its name and size
        """
        return sdy.MeshAxisAttr.get(name, size)

    def mesh_attr(
        self,
        axes: List[sdy.MeshAxisAttr],
    ) -> MeshAttr:
        """
        Creates a mesh attribute from a list of mesh axis attributes.
        This attribute represents a mesh, which is a collection of axes that can be used
        to define the layout of tensors across multiple devices or processing units.

        Parameters
        ----------
        axes : List[sdy.MeshAxisAttr]
            A list of mesh axis attributes that define the axes of the mesh

        Returns
        -------
        (*sdy.MeshAttr*)
            A mesh attribute representing the collection of axes in the mesh
        """
        return sdy.MeshAttr.get(axes)

    def axis_ref_attr(
        self,
        name: str,
        sub_axis_info_attr: Optional[sdy.AxisRefAttr] = None,
    ) -> sdy.AxisRefAttr:
        """
        Creates an axis reference attribute.
        This attribute is used to reference a specific axis in a mesh, optionally with additional
        sub-axis information.

        Parameters
        ----------
        name : str
            The name of the axis reference
        sub_axis_info_attr : *Optional[sdy.AxisRefAttr]*
            An optional sub-axis reference attribute that provides additional information about the axis

        Returns
        -------
        (*sdy.AxisRefAttr*)
            An axis reference attribute that can be used to refer to a specific axis in a mesh
        """
        return sdy.AxisRefAttr.get(name, sub_axis_info_attr)

    def dimension_sharding_attr(
        self,
        axes: List[sdy.AxisRefAttr],
        is_closed: bool,
        priority: Optional[int] = None,
    ) -> sdy.DimensionShardingAttr:
        """
        Creates a dimension sharding attribute.
        This attribute defines how a tensor is sharded across multiple devices or processing units
        based on the specified axes. It can also indicate whether the sharding is closed and an optional priority for the sharding.

        Parameters
        ----------
        axes : List[sdy.AxisRefAttr]
            A list of axis reference attributes that define how the tensor is sharded across the mesh
        is_closed : bool
            A boolean indicating whether the sharding is closed
        priority : *Optional[int]*
            An optional integer that specifies the priority of the sharding. If not provided, defaults to None.

        Returns
        -------
        (*sdy.DimensionShardingAttr*)
            A dimension sharding attribute that describes how a tensor is distributed across the mesh
        """
        return sdy.DimensionShardingAttr.get(axes, is_closed, priority)

    def tensor_sharding_attr(
        self,
        mesh_name: str,
        dimension_shardings: List[sdy.DimensionShardingAttr],
        replicated_axes: List[sdy.AxisRefAttr] = [],
        unreduced_axes: List[sdy.AxisRefAttr] = [],
    ) -> sdy.TensorShardingAttr:
        """
        Creates a tensor sharding attribute.
        This attribute describes how a tensor is sharded across a mesh, including the mesh name,
        the dimension shardings, and any replicated or unreduced axes.

        Parameters
        ----------
        mesh_name : str
            The name of the mesh to which the tensor sharding applies
        dimension_shardings : List[sdy.DimensionShardingAttr]
            A list of dimension sharding attributes that define how the tensor is sharded across the mesh
        replicated_axes : List[sdy.AxisRefAttr]
            A list of axis reference attributes that are replicated across the mesh. Defaults to an empty list
        unreduced_axes : List[sdy.AxisRefAttr]
            A list of axis reference attributes that are not reduced in the sharding. Defaults to an empty list

        Returns
        -------
        (*sdy.TensorShardingAttr*)
            A tensor sharding attribute that describes how a tensor is distributed across the mesh
        """
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
        """
        Creates a tensor sharding per value attribute from a list of tensor sharding attributes.
        This attribute allows for specifying different sharding strategies for different tensors.

        Parameters
        ----------
        shardings : List[sdy.TensorShardingAttr]
            A list of tensor sharding attributes, each defining a sharding strategy for a tensor

        Returns
        -------
        (*sdy.TensorShardingPerValueAttr*)
            A tensor sharding per value attribute that describes how multiple tensors are distributed across the mesh
        """
        return sdy.TensorShardingPerValueAttr.get(
            shardings,
        )

    # ----- Public Shardy Op Generators ----

    def mesh(self, mesh_name: str, mesh_attr: sdy.MeshAttr) -> sdy.MeshOp:
        """
        Creates a mesh operation.
        This operation defines a mesh in the system, which can be used to distribute tensors
        across multiple devices or processing units. The mesh is identified by its name and
        defined by the provided mesh attribute.

        Parameters
        ----------
        mesh_name : str
            The name of the mesh to be created
        mesh_attr : sdy.MeshAttr
            The mesh attribute that defines the axes and properties of the mesh

        Returns
        -------
        (*sdy.MeshOp*)
            A mesh operation that represents the defined mesh in the system
        """
        return sdy.MeshOp(sym_name=mesh_name, mesh=mesh_attr)

    def sharding_constraint(
        self,
        in0: Operand,
        tensor_sharding_attr: sdy.TensorShardingAttr,
    ) -> sdy.ShardingConstraintOp:
        """
        Creates a sharding constraint operation.
        This operation applies a sharding constraint to a tensor, specifying how the tensor should be distributed
        across a mesh based on the provided tensor sharding attribute.

        Parameters
        ----------
        in0 : Operand
            The input tensor to which the sharding constraint will be applied
        tensor_sharding_attr : sdy.TensorShardingAttr
            The tensor sharding attribute that defines how the tensor should be sharded across the mesh

        Returns
        -------
        (*sdy.ShardingConstraintOp*)
            A sharding constraint operation that applies the specified sharding to the input tensor
        """
        return sdy.ShardingConstraintOp(in0, tensor_sharding_attr)

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
