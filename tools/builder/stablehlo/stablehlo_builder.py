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
from builder.base import builder_golden


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

    def _op_proxy(
        self,
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

            op = op_stablehlo_function(
                *inputs,
                loc=loc,
                **stablehlo_kwargs,
            )

            if unit_attrs is not None:
                for attr_name in unit_attrs:
                    op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

            if not skip_golden and not self._disable_golden_check:
                op_golden_function = builder_golden.get_golden_function(
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
    ) -> OpView:
        return self._op_proxy(op_stablehlo_function, inputs, unit_attrs)

    # ----- Public StableHLO Op Generators ----

    def add(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
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
        )

    def div(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``stablehlo.divide``.

        *Elementwise division operation.*

        Performs elementwise division between two tensors.
        For each pair of corresponding elements, divides the element in the first
        tensor by the element in the second tensor.

        Mathematical definition: div(x, y) = x / y

        Parameters
        ----------
        in0 : Operand
            First input tensor (dividend)
        in1 : Operand
            Second input tensor (divisor)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise division of the inputs
        """
        return self._eltwise_proxy(
            stablehlo.DivOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def max(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``stablehlo.maximum``.

        *Elementwise maximum operation.*

        Performs elementwise maximum between two tensors.
        For each pair of corresponding elements, returns the maximum of the two elements.

        Mathematical definition: max(x, y) = max(x, y)

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
            A tensor containing the elementwise maximum of the inputs
        """
        return self._eltwise_proxy(
            stablehlo.MaxOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def min(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``stablehlo.minimum``.

        *Elementwise minimum operation.*

        Performs elementwise minimum between two tensors.
        For each pair of corresponding elements, returns the minimum of the two elements.

        Mathematical definition: min(x, y) = min(x, y)

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
            A tensor containing the elementwise minimum of the inputs
        """
        return self._eltwise_proxy(
            stablehlo.MinOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def multiply(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``stablehlo.multiply``.

        *Elementwise multiplication operation.*

        Performs elementwise multiplication between two tensors.
        For each pair of corresponding elements, multiplies the element in the first
        tensor by the element in the second tensor.

        Mathematical definition: multiply(x, y) = x * y

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
            A tensor containing the elementwise product of the inputs
        """
        return self._eltwise_proxy(
            stablehlo.MulOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def subtract(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``stablehlo.subtract``.

        *Elementwise subtraction operation.*

        Performs elementwise subtraction between two tensors.
        For each pair of corresponding elements, subtracts the element in the second
        tensor from the element in the first tensor.

        Mathematical definition: subtract(x, y) = x - y

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
            A tensor containing the elementwise difference of the inputs
        """
        return self._eltwise_proxy(
            stablehlo.SubtractOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def remainder(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``stablehlo.remainder``.

        *Elementwise remainder operation.*

        Performs elementwise remainder between two tensors.
        For each pair of corresponding elements, computes the remainder of the division
        of the first element by the second element.

        Mathematical definition: remainder(x, y) = x % y

        Parameters
        ----------
        in0 : Operand
            First input tensor (dividend)
        in1 : Operand
            Second input tensor (divisor)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise remainder of the inputs
        """
        return self._eltwise_proxy(
            stablehlo.RemOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def pow(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``stablehlo.power``.

        *Elementwise power operation.*

        Performs elementwise power between two tensors.
        For each pair of corresponding elements, raises the element in the first
        tensor to the power of the element in the second tensor.

        Mathematical definition: pow(x, y) = x^y

        Parameters
        ----------
        in0 : Operand
            First input tensor (base)
        in1 : Operand
            Second input tensor (exponent)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise power of the inputs
        """
        return self._eltwise_proxy(
            stablehlo.PowOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def atan2(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``stablehlo.atan2``.

        *Elementwise arctangent2 operation.*

        Performs elementwise arctangent2 between two tensors.
        For each pair of corresponding elements, computes the arctangent of y/x
        in radians, where y is the first tensor and x is the second tensor.

        Mathematical definition: atan2(y, x) = arctan(y/x)

        Parameters
        ----------
        in0 : Operand
            First input tensor (y coordinates)
        in1 : Operand
            Second input tensor (x coordinates)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise atan2 of the inputs
        """
        return self._eltwise_proxy(
            stablehlo.Atan2Op,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def shift_right_logical(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``stablehlo.shift_right_logical``.

        *Elementwise logical right shift operation.*

        Performs elementwise logical right shift between two tensors.
        For each pair of corresponding elements, shifts the bits of the first
        element to the right by the number of positions specified by the second element.

        Mathematical definition: shift_right_logical(x, y) = x >> y

        Parameters
        ----------
        in0 : Operand
            First input tensor (value to shift)
        in1 : Operand
            Second input tensor (shift amount)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise logical right shift of the inputs
        """
        return self._eltwise_proxy(
            stablehlo.ShiftRightLogicalOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def shift_left(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``stablehlo.shift_left``.

        *Elementwise left shift operation.*

        Performs elementwise left shift between two tensors.
        For each pair of corresponding elements, shifts the bits of the first
        element to the left by the number of positions specified by the second element.

        Mathematical definition: shift_left(x, y) = x << y

        Parameters
        ----------
        in0 : Operand
            First input tensor (value to shift)
        in1 : Operand
            Second input tensor (shift amount)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise left shift of the inputs
        """
        return self._eltwise_proxy(
            stablehlo.ShiftLeftOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def select(
        self, condition: Operand, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``stablehlo.select``.

        *Elementwise select operation.*

        Performs elementwise selection between two tensors based on a condition.
        For each element, selects the corresponding element from the first tensor
        if the condition is true, otherwise selects from the second tensor.

        Mathematical definition: select(cond, x, y) = cond ? x : y

        Parameters
        ----------
        condition : Operand
            Boolean condition tensor
        in0 : Operand
            First input tensor (selected when condition is true)
        in1 : Operand
            Second input tensor (selected when condition is false)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise selection based on the condition
        """
        return self._eltwise_proxy(
            stablehlo.SelectOp,
            [condition, in0, in1],
            unit_attrs=unit_attrs,
        )

    # ----- Elementwise Unary Operations -----

    def abs(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
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
        )

    def ceil(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
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
        )

    def cosine(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
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
        )

    def exp(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
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
        )

    def floor(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
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
        )

    def neg(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
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
        )

    def rsqrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
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
        )

    def sine(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
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
        )

    def sqrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
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
        )

    def logistic(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
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
        )

    def tan(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
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
        )

    def log(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
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
