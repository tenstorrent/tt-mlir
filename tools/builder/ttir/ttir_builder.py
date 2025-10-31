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

from ttmlir.ir import *
from ttmlir.dialects import ttir, ttcore, tensor, quant
from ttmlir.passes import GoldenTensor, DataType

from builder.base.builder import *
from builder.base import builder_golden


class TTIRBuilder(Builder):
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

    # ----- Public methods -----

    # ----- Private methods ----

    def _get_output_shape_and_type(
        self,
        organize_golden_args: Callable,
        inputs: List[Operand],
        op_ttir_function: Callable,
        golden_kwargs: dict = {},
    ):
        op_golden_function = builder_golden.get_golden_function(
            op_ttir_function, **golden_kwargs
        )
        if op_golden_function is None:
            return

        # If the op has no input, just call golden function with kwargs (eg ttir.zeros).
        if len(inputs) == 0:
            golden_output = op_golden_function(**golden_kwargs)
        else:
            golden_output = op_golden_function(
                *(organize_golden_args(inputs)), **golden_kwargs
            )

        return golden_output.shape, golden_output.dtype

    def _get_empty_op(self, tensor_type: RankedTensorType) -> OpView:
        """Get TTIR-specific empty operation."""
        return ttir.EmptyOp(tensor_type)

    def _organize_eltwise_ttir(
        self, inputs: List[Operand], output: OpView, _: Optional[Shape]
    ):
        return (self._get_type(output), *inputs, output)

    def _op_proxy(
        self,
        op_ttir_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
        organize_ttir_args: Optional[Callable] = None,
        organize_golden_args: Optional[Callable] = None,
        output_shape: Optional[Shape] = None,
        output_type: Optional[Type] = None,
        output_create_fn: Optional[Callable] = None,
        golden_kwargs: dict = {},
        ttir_kwargs: dict = {},
        loc: Optional[Union[str, Location]] = None,
        skip_golden: bool = False,
    ) -> Any:
        if not golden_kwargs:
            golden_kwargs = ttir_kwargs

        if organize_ttir_args is None:
            organize_ttir_args = self._organize_eltwise_ttir

        if organize_golden_args is None:
            organize_golden_args = self._organize_eltwise_golden

        with self._ctx, self._loc:
            # If output shape or type is not provided, calculate it using golden function.
            # This is needed because TTIR ops do not have shape or type MLIR inference trait.
            output_shape_and_type = self._get_output_shape_and_type(
                organize_golden_args, inputs, op_ttir_function, golden_kwargs
            )
            if not output_shape_and_type:
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

            # Use provided output shape/type if available, otherwise use calculated ones.
            output_shape = calculated_output_shape if not output_shape else output_shape
            output_type = (
                self._get_type_from_torch_dtype(calculated_output_type)
                if not output_type
                else output_type
            )

            # Create output tensor using provided function or create empty tensor.
            if output_create_fn:
                output = output_create_fn(output_shape, output_type)
            else:
                output = self._empty(output_shape, output_type)

            # Prepare location for the op.
            id = self._get_next_global_id()
            loc = (
                self._get_loc_from_str(loc)
                if loc is not None
                else self._get_loc_of_extra_file_callee(id=id)
            )

            # Organize arguments and create the TTIR op.
            if organize_ttir_args(inputs, output, output_shape) == 0:
                op = op_ttir_function(
                    loc=loc,
                    **ttir_kwargs,
                )
            else:
                op = op_ttir_function(
                    *organize_ttir_args(inputs, output, output_shape),
                    loc=loc,
                    **ttir_kwargs,
                )

            # Set unit attributes if provided.
            if unit_attrs is not None:
                for attr_name in unit_attrs:
                    op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

            if not skip_golden and not self._disable_golden_check:
                op_golden_function = builder_golden.get_golden_function(
                    op_ttir_function, **golden_kwargs
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

    # ----- Public Op Generators ----

    def get_dimension_size(
        self, in0: Operand, dimension: int = 0, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.get_dimension_size``.

        *Dimension size query operation.*

        Produces the size of the given `dimension` of the `operand`.

        .. code-block:: mlir

            %operand: [[3, 2, 7], [1, 4, 4]]
            "ttir.get_dimension_size"(%operand, value = dense<0>, %out) -> %out: [[3]]

        Parameters
        ----------
        in0 : Operand
            Input tensor operand to get dimension size from
        dimension : int, optional
            The dimension index to get size of (default: 0)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.GetDimensionSizeOp,
            [in0],
            ttir_kwargs={"dimension": dimension},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0]),
            output_type=self._get_type_from_torch_dtype(torch.int32),
            unit_attrs=unit_attrs,
        )

    def dot_general(
        self,
        in0: Operand,
        in1: Operand,
        batch_dims_lhs: List[int],
        contract_dims_lhs: List[int],
        batch_dims_rhs: List[int],
        contract_dims_rhs: List[int],
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.dot_general``.

        *Generalized dot product operation.*

        Flexible tensor operation that generalizes matrix multiplication by allowing user to specify which
        dimensions of two tensors to contract. Matrix multiplication is a special case of this operation,
        where the contraction happens along the last axis of the first tensor and the second-to-last axis of the second tensor.
        From StableHLO DotGeneral Op https://openxla.org/stablehlo/spec#dot_general

        Parameters
        ----------
        in0 : Operand
            Left-hand side input tensor
        in1 : Operand
            Right-hand side input tensor
        batch_dims_lhs : *List[int]*
            Batch dimensions for the left-hand side tensor
        contract_dims_lhs : *List[int]*
            Contracting dimensions for the left-hand side tensor
        batch_dims_rhs : *List[int]*
            Batch dimensions for the right-hand side tensor
        contract_dims_rhs : *List[int]*
            Contracting dimensions for the right-hand side tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.DotGeneralOp,
            [in0, in1],
            ttir_kwargs={
                "batch_dims_lhs": batch_dims_lhs,
                "contract_dims_lhs": contract_dims_lhs,
                "batch_dims_rhs": batch_dims_rhs,
                "contract_dims_rhs": contract_dims_rhs,
            },
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1]),
            output_type=self._get_type_from_torch_dtype(
                self._get_golden_tensor(in0).dtype
            ),
            unit_attrs=unit_attrs,
        )

    def where(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.where``.

        *Elementwise conditional selection operation.*

        For each element position, selects between two values based on a boolean condition:
        - If the condition is true (non-zero), selects from the first value tensor
        - If the condition is false (zero), selects from the second value tensor

        Supports broadcasting according to standard broadcasting rules.

        .. code-block:: mlir

            // Basic selection between two tensors
            %result = ttir.where(%cond, %true_vals, %false_vals) :
                tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
            // Input tensors:
            // %cond: [[1, 0], [0, 1]]
            // %true_vals: [[1.0, 2.0], [3.0, 4.0]]
            // %false_vals: [[5.0, 6.0], [7.0, 8.0]]
            // Output tensor:
            // [[1.0, 6.0], [7.0, 4.0]]

            // With broadcasting (scalar condition)
            %result = ttir.where(%scalar_cond, %true_vals, %false_vals) :
                tensor<i1>, tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>

        Parameters
        ----------
        in0 : Operand
            Condition tensor (predicate)
        in1 : Operand
            Tensor containing values to select when condition is true
        in2 : Operand
            Tensor containing values to select when condition is false
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        # Handle golden condition tensor
        in0_tensor = self._get_golden_tensor(in0)
        condition = in0_tensor.apply_shardwise(
            lambda shard: torch.where(
                shard > 0,
                torch.tensor(True, device=shard.device),
                torch.tensor(False, device=shard.device),
            )
        )
        return self._op_proxy(
            ttir.WhereOp,
            [in0, in1, in2],
            organize_golden_args=lambda i: (
                condition,
                self._get_golden_tensor(i[1]),
                self._get_golden_tensor(i[2]),
            ),
            unit_attrs=unit_attrs,
        )

    # class TTIR_ElementwiseUnaryOp

    def abs(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.abs``.

        *Elementwise absolute value operation.*

        Computes the absolute value of each element in the input tensor.

        .. code-block:: mlir

            // Compute absolute values of all elements in %input
            %result = ttir.abs(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
            // Input tensor:
            // [[-2.5,  3.7,  0.0,  1.2], ... ]
            // Output tensor:
            // [[2.5, 3.7, 0.0, 1.2], ... ]

            // Example with integer tensor
            %result = ttir.abs(%int_input, %int_output) : tensor<10xi32>, tensor<10xi32> -> tensor<10xi32>
            // Input tensor:
            // [-5, 0, 3, -2, ...]
            // Output tensor:
            // [5, 0, 3, 2, ...]

        Parameters
        ----------
        in0 : Operand
            Input tensor to compute absolute value of
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """

        return self._op_proxy(ttir.AbsOp, [in0], unit_attrs)

    def cbrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.cbrt``.

        *Elementwise cubic root operation.*

        Computes the cubic root (∛) of each element in the input tensor.
        For each element, returns the real-valued number that, when cubed, equals the input value.
        Unlike square root, cubic root is defined for negative numbers as well as positive numbers.

        .. code-block:: mlir

            // Compute cubic root of all elements
            %result = ttir.cbrt(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [8.0, 27.0, -8.0, 1.0]
            // Output tensor:
            // [2.0, 3.0, -2.0, 1.0]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the cubic root of each element in the input tensor
        """
        return self._op_proxy(ttir.CbrtOp, [in0], unit_attrs)

    def ceil(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.ceil``.

        *Elementwise ceiling operation.*

        Computes the ceiling of each element in the input tensor, rounding up to the nearest integer.
        This operation is idempotent.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with ceiling values
        """
        return self._op_proxy(ttir.CeilOp, [in0], unit_attrs)

    def cos(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.cos``.

        *Elementwise cosine operation.*

        Computes the cosine of each element in the input tensor.
        Input values are expected to be in radians.

        .. code-block:: mlir

            // Compute cosine of all elements
            %result = ttir.cos(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor (in radians):
            // [0.0, 3.14159, 1.5708, -1.5708]
            // Output tensor:
            // [1.0, -1.0, 0.0, 0.0]

        Parameters
        ----------
        in0 : Operand
            Input tensor (values in radians)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the cosine of each element in the input tensor
        """
        return self._op_proxy(ttir.CosOp, [in0], unit_attrs)

    def erf(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.erf``.

        *Elementwise error function operation.*

        Computes the error function (erf) of each element in the input tensor.
        The error function is a mathematical function used in probability, statistics,
        and partial differential equations related to the normal distribution.

        Mathematical definition: erf(x) = (2/sqrt(π)) * ∫[0 to x] e^(-t^2) dt

        .. code-block:: mlir

            // Compute error function of all elements
            %result = ttir.erf(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [0.0, 0.5, 1.0, -1.0]
            // Output tensor:
            // [0.0, 0.5205, 0.8427, -0.8427]
        """
        return self._op_proxy(ttir.ErfOp, [in0], unit_attrs)

    def erfc(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.erfc``.

        *Elementwise complementary error function operation.*

        Computes the complementary error function (erfc) of each element in the input tensor.
        The complementary error function is defined as erfc(x) = 1 - erf(x),
        where erf(x) is the error function. It is commonly used in statistics and probability.

        Mathematical definition: erfc(x) = 1 - (2/sqrt(π)) * ∫[0 to x] e^(-t^2) dt

        .. code-block:: mlir

            // Compute complementary error function of all elements
            %result = ttir.erfc(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [0.0, 0.5, 1.0, -1.0]
            // Output tensor:
            // [1.0, 0.4795, 0.1573, 1.8427]
        """
        return self._op_proxy(ttir.ErfcOp, [in0], unit_attrs)

    def floor(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.floor``.

        *Elementwise floor operation.*

        Computes the floor of each element in the input tensor, rounding down to the nearest integer.
        This operation is idempotent.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with floor values
        """
        return self._op_proxy(
            ttir.FloorOp,
            [in0],
            unit_attrs,
        )

    def gelu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.gelu``.

        *Elementwise GELU operation.*

        Computes the GELU (Gaussian Error Linear Unit) of each element in the input tensor.
        GELU is a smooth, non-monotonic activation function that approximates the cumulative
        distribution function of a standard normal distribution.

        Mathematical definition: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

        .. code-block:: mlir

            // Compute GELU of all elements
            %result = ttir.gelu(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [1.0, -0.5, 2.0, -2.0]
            // Output tensor:
            // [0.841, -0.154, 1.954, -0.046]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the GELU values of each element in the input tensor
        """
        return self._op_proxy(ttir.GeluOp, [in0], unit_attrs)

    def is_finite(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.is_finite``.

        *Elementwise finite check operation.*

        Checks if each element in the input tensor is finite (neither infinite nor NaN).
        For each element, returns a boolean value indicating whether the element is finite.

        Mathematical definition: isfinite(x) = x ∈ ℝ

        .. code-block:: mlir

            // Check if elements are finite
            %result = ttir.is_finite(%input, %output) : tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
            // Input tensor:
            // [1.0, inf, -inf, nan]
            // Output tensor:
            // [true, false, false, false]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.IsFiniteOp,
            [in0],
            unit_attrs,
            output_type=F32Type.get(self._ctx),
        )

    def logical_not(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.logical_not``.

        *Elementwise logical NOT operation.*

        Computes the logical NOT of each element in the input tensor.
        For each element x, returns True if x is False, and False if x is True.

        .. code-block:: mlir

            // Compute logical NOT of all elements
            %result = ttir.logical_not(%input, %output) : tensor<3xi1>, tensor<3xi1> -> tensor<3xi1>
            // Input tensor:
            // [true, false, true]
            // Output tensor:
            // [false, true, false]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the logical NOT of each element in the input tensor
        """
        return self._op_proxy(
            ttir.LogicalNotOp,
            [in0],
            unit_attrs=unit_attrs,
        )

    def bitwise_not(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.bitwise_not``.

        *Elementwise bitwise NOT operation.*

        Computes the bitwise NOT (one's complement) of each element in the input tensor.
        For each element, flips all the bits in the binary representation of the value.

        This operation is typically used with integer data types and has the involution property,
        meaning that applying it twice returns the original value: bitwise_not(bitwise_not(x)) = x.

        .. code-block:: mlir

            // Bitwise NOT with integer tensors
            %result = ttir.bitwise_not(%input, %output) : tensor<2x2xi32>, tensor<2x2xi32> -> tensor<2x2xi32>
            // Input tensor:
            // [[1, 2],
            //  [3, 4]]
            // Output tensor:
            // [[-2, -3],
            //  [-4, -5]]

            // Example with 8-bit integers
            %result = ttir.bitwise_not(%input, %output) : tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
            // Input: [0, 5, 255] (binary: [00000000, 00000101, 11111111])
            // Output: [255, 250, 0] (binary: [11111111, 11111010, 00000000])

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.BitwiseNotOp,
            [in0],
            unit_attrs,
        )

    def neg(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.neg``.

        *Elementwise negate operation.*

        Computes the negation of each element in the input tensor.
        For each element, returns the negation of the value.

        Mathematical definition: neg(x) = -x

        .. code-block:: mlir

            // Compute negation of all elements
            %result = ttir.neg(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [1.7, 2.0, -0.3, 4.5]
            // Output tensor:
            // [-1.7, -2.0, 0.3, -4.5]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the negation of each input element
        """
        return self._op_proxy(ttir.NegOp, [in0], unit_attrs)

    def tan(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.tan``.

        *Elementwise tangent operation.*

        Computes the tangent of each element in the input tensor.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with tangent values
        """
        return self._op_proxy(ttir.TanOp, [in0], unit_attrs)

    def atan(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.atan``.

        *Elementwise arctangent operation.*

        Computes the inverse tangent (arctangent) of each element in the input tensor.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with arctangent values
        """
        return self._op_proxy(ttir.AtanOp, [in0], unit_attrs)

    def tanh(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.tanh``.

        *Elementwise hyperbolic tangent operation.*

        Computes the hyperbolic tangent of each element in the input tensor.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with hyperbolic tangent values
        """
        return self._op_proxy(ttir.TanhOp, [in0], unit_attrs)

    def reciprocal(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.reciprocal``.

        *Elementwise reciprocal operation.*

        Computes the reciprocal (1/x) of each element in the input tensor.
        This operation is involutive (applying it twice returns to the original value).

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with reciprocal values
        """
        return self._op_proxy(
            ttir.ReciprocalOp,
            [in0],
            unit_attrs,
        )

    def relu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.relu``.

        *Elementwise ReLU activation operation.*

        Computes the Rectified Linear Unit function for each element in the input tensor.
        This operation is idempotent (applying it multiple times has the same effect as applying it once).

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with ReLU activation values
        """
        return self._op_proxy(ttir.ReluOp, [in0], unit_attrs)

    def relu6(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.relu6``.

        *Elementwise ReLU6 activation operation.*

        Computes the ReLU6 function for each element in the input tensor.
        ReLU6 is defined as: min(max(0, x), 6)
        This activation function clips values between 0 and 6, making it useful
        for quantized neural networks and mobile applications.

        .. code-block:: mlir

            // Compute ReLU6 of all elements
            %result = ttir.relu6(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [-2.0, 3.0, 8.0, 1.5]
            // Output tensor:
            // [0.0, 3.0, 6.0, 1.5]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with ReLU6 activation values
        """
        return self._op_proxy(ttir.Relu6Op, [in0], unit_attrs)

    def silu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.silu``.

        *Elementwise SiLU (Swish) activation operation.*

        Computes the SiLU (Sigmoid Linear Unit) activation function for each element in the input tensor.
        SiLU is also known as Swish activation and is defined as: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

        This activation function is smooth, non-monotonic, and has been shown to work well
        in deep neural networks, particularly in transformer architectures.

        .. code-block:: mlir

            // Compute SiLU activation of all elements
            %result = ttir.silu(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [1.0, -0.5, 2.0, -2.0]
            // Output tensor:
            // [0.731, -0.193, 1.762, -0.238]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the SiLU activation values of each element in the input tensor
        """
        return self._op_proxy(ttir.SiluOp, [in0], unit_attrs)

    def relu6(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.relu6``.

        *Elementwise ReLU6 activation operation.*

        Computes the ReLU6 function for each element in the input tensor.
        ReLU6 is defined as: min(max(0, x), 6)
        This activation function clips values between 0 and 6, making it useful
        for quantized neural networks and mobile applications.

        .. code-block:: mlir

            // Compute ReLU6 of all elements
            %result = ttir.relu6(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [-2.0, 3.0, 8.0, 1.5]
            // Output tensor:
            // [0.0, 3.0, 6.0, 1.5]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with ReLU6 activation values
        """
        return self._op_proxy(ttir.Relu6Op, [in0], unit_attrs)

    def rsqrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.rsqrt``.

        *Elementwise reciprocal square root operation.*

        Computes the reciprocal of the square root (1/√x) of each element in the input tensor.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with reciprocal square root values
        """
        return self._op_proxy(
            ttir.RsqrtOp,
            [in0],
            unit_attrs,
        )

    def sigmoid(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.sigmoid``.

        *Elementwise sigmoid activation operation.*

        Computes the sigmoid function (1/(1 + e^-x)) for each element in the input tensor.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with sigmoid activation values
        """
        return self._op_proxy(
            ttir.SigmoidOp,
            [in0],
            unit_attrs,
        )

    def sign(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.sign``.

        *Elementwise sign operation.*

        Returns the sign (-1, 0, or 1) of each element in the input tensor.
        This operation is idempotent.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with sign values
        """
        return self._op_proxy(ttir.SignOp, [in0], unit_attrs)

    def silu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.silu``.

        *Elementwise SiLU (Swish) activation operation.*

        Computes the SiLU (Sigmoid Linear Unit) activation function for each element in the input tensor.
        SiLU is also known as Swish activation and is defined as: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

        This activation function is smooth, non-monotonic, and has been shown to work well
        in deep neural networks, particularly in transformer architectures.

        .. code-block:: mlir

            // Compute SiLU activation of all elements
            %result = ttir.silu(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [1.0, -0.5, 2.0, -2.0]
            // Output tensor:
            // [0.731, -0.193, 1.762, -0.238]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the SiLU activation values of each element in the input tensor
        """
        return self._op_proxy(ttir.SiluOp, [in0], unit_attrs)

    def sin(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.sin``.

        *Elementwise sine operation.*

        Computes the sine of each element in the input tensor.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with sine values
        """
        return self._op_proxy(ttir.SinOp, [in0], unit_attrs)

    def sqrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.sqrt``.

        *Elementwise square root operation.*

        Computes the square root of each element in the input tensor.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with square root values
        """
        return self._op_proxy(ttir.SqrtOp, [in0], unit_attrs)

    def typecast(
        self,
        in0: Operand,
        output_type: torch.dtype,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.typecast``.

        *Elementwise type casting operation.*

        Casts each element in the input tensor to the type of the output tensor.
        The output type can be any supported tensor element type.

        .. code-block:: mlir

            // Cast float32 to int32
            %result = ttir.typecast(%input, %output) : tensor<2x2xf32>, tensor<2x2xi32> -> tensor<2x2xi32>
            // Input tensor:
            // [[1.7, 2.3],
            //  [3.8, 4.1]]
            // Output tensor:
            // [[1, 2],
            //  [3, 4]]

        Parameters
        ----------
        in0 : Operand
            Input tensor to cast
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the input values cast to the output type
        """
        return self._op_proxy(
            ttir.TypecastOp,
            [in0],
            golden_kwargs={"dtype": output_type},
            output_type=self._get_type_from_torch_dtype(output_type),
            unit_attrs=unit_attrs,
        )

    def log(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.log``.

        *Elementwise natural logarithm operation.*

        Computes the natural logarithm of each element in the input tensor.
        For each element x, returns ln(x), where ln is the natural logarithm.

        .. code-block:: mlir

            // Compute natural logarithm of all elements
            %result = ttir.log(%input, %output) : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensor:
            // [1.0, 2.71828, 7.38906]
            // Output tensor:
            // [0.0, 1.0, 2.0]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the natural logarithm of each element in the input tensor
        """
        return self._op_proxy(ttir.LogOp, [in0], unit_attrs)

    def log1p(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise natural logarithm of one plus input operation.

        The `log1p` operation computes the natural logarithm of one plus each element in the
        input tensor. For each element x, it returns ln(1 + x). This operation is more
        accurate than computing log(1 + x) directly for x values close to zero, and it is
        defined for x > -1. For values less than or equal to -1, the behavior depends on
        the implementation (may return NaN or negative infinity).

        .. code-block:: mlir

            // Compute log1p of all elements
            %result = ttir.log1p(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [0.0, -0.999, 7.0, 6.38905621, 15.0]
            // Output tensor:
            // [0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the log1p values of the input tensor
        """
        return self._op_proxy(
            ttir.Log1pOp,
            [in0],
            unit_attrs,
        )

    def expm1(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.expm1``.

        *Elementwise exponential minus one operation.*

        Computes e^x - 1 for each element in the input tensor, where e is Euler's number.
        This operation provides better numerical precision than computing exp(x) - 1 directly,
        especially for small values of x.

        .. code-block:: mlir

            // Compute exp(x) - 1 for all elements
            %result = ttir.expm1(%input, %output) : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensor:
            // [0.0, 0.1, -0.1]
            // Output tensor:
            // [0.0, 0.10517, -0.09516]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing exp(x) - 1 for each element x in the input tensor
        """
        return self._op_proxy(
            ttir.Expm1Op,
            [in0],
            unit_attrs,
        )

    # class TTIR_ElementwiseUnaryWithFloatParameterOp

    def leaky_relu(
        self,
        in0: Operand,
        parameter: float = 0.01,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.leaky_relu``.

        *Elementwise leaky ReLU activation operation.*

        Computes a leaky version of the Rectified Linear Unit (ReLU) activation function.
        For each element x in the input tensor:
        - If x > 0: returns x
        - If x ≤ 0: returns parameter * x

        The parameter controls the slope for negative values, allowing a small gradient
        when the unit is not active.

        .. code-block:: mlir

            // Compute leaky ReLU with slope 0.01 for negative values
            %result = ttir.leaky_relu(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [2.0, -1.0, 0.0, -3.0]
            // Output tensor:
            // [2.0, -0.01, 0.0, -0.03]

        Parameters
        ----------
        in0 : Operand
            Input tensor to be activated
        parameter : float, optional
            Slope for negative values (default: 0.01)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the leaky ReLU activation values
        """
        ttir_kwargs = {"parameter": parameter}
        return self._op_proxy(
            ttir.LeakyReluOp,
            [in0],
            ttir_kwargs=ttir_kwargs,
            unit_attrs=unit_attrs,
        )

    # class TTIR_ElementwiseBinaryOp

    def eq(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.eq``.

        *Elementwise equality comparison operation.*

        Performs an elementwise equality comparison between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if the elements are equal
        - 0 (false) if the elements are not equal

        Note that special handling may be required for floating-point NaN values, as NaN is not
        equal to any value, including itself.

        Mathematical definition: equal(x, y) = x == y

        .. code-block:: mlir

            // Compare elements for equality
            %result = ttir.eq(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xi1> -> tensor<3xi1>
            // Input tensors:
            // lhs: [1.0, 2.0, 3.0]
            // rhs: [1.0, 2.0, 4.0]
            // Output tensor:
            // [1, 1, 0]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.EqualOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def ne(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.ne``.

        *Elementwise inequality comparison operation.*

        Performs elementwise inequality comparison between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if the elements are not equal
        - 0 (false) if the elements are equal

        Note: Special handling may be required for floating-point NaN values, as NaN is not
        equal to any value, including itself. This means ne(NaN, NaN) should return true.

        Mathematical definition: not_equal(x, y) = x != y

        .. code-block:: mlir

            // Compare elements for inequality
            %result = ttir.ne(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1.0, 2.0, 3.0, 2.0]
            // rhs: [1.0, 2.0, 4.0, 5.0]
            // Output tensor:
            // [0, 0, 1, 1]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.NotEqualOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def ge(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.ge``.

        *Elementwise greater than or equal to comparison operation.*

        Performs elementwise greater than or equal to comparison between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if the left element is greater than or equal to the right element
        - 0 (false) if the left element is less than the right element

        Mathematical definition: greater_equal(x, y) = x >= y

        .. code-block:: mlir

            // Compare elements for greater than or equal to
            %result = ttir.ge(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1.0, 2.0, 3.0, 2.0]
            // rhs: [1.0, 2.0, 4.0, 5.0]
            // Output tensor:
            // [1, 1, 0, 0]

        Parameters
        ----------
        in0 : Operand
            First input tensor (left-hand side)
        in1 : Operand
            Second input tensor (right-hand side)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.GreaterEqualOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def gt(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.gt``.

        *Elementwise greater than comparison operation.*

        Performs elementwise greater than comparison between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if the left element is greater than the right element
        - 0 (false) if the left element is less than or equal to the right element

        Mathematical definition: greater(x, y) = x > y

        .. code-block:: mlir

            // Compare elements for greater than
            %result = ttir.gt(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1.0, 2.0, 3.0, 2.0]
            // rhs: [1.0, 1.0, 4.0, 5.0]
            // Output tensor:
            // [0, 1, 0, 0]

        Parameters
        ----------
        in0 : Operand
            First input tensor (left-hand side)
        in1 : Operand
            Second input tensor (right-hand side)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.GreaterThanOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def le(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.le``.

        *Elementwise less than or equal to comparison operation.*

        Performs elementwise less than or equal to comparison between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if the left element is less than or equal to the right element
        - 0 (false) if the left element is greater than the right element

        Mathematical definition: less_equal(x, y) = x <= y

        .. code-block:: mlir

            // Compare elements for less than or equal to
            %result = ttir.le(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1.0, 2.0, 3.0, 2.0]
            // rhs: [1.0, 2.0, 4.0, 5.0]
            // Output tensor:
            // [1, 1, 1, 1]

        Parameters
        ----------
        in0 : Operand
            First input tensor (left-hand side)
        in1 : Operand
            Second input tensor (right-hand side)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.LessEqualOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def lt(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.lt``.

        *Elementwise less than comparison operation.*

        The `lt` operation performs an elementwise less than comparison between two tensors.
        For each pair of corresponding elements, it returns:
        - 1 (true) if the left element is less than the right element
        - 0 (false) if the left element is greater than or equal to the right element

        Mathematical definition: less(x, y) = x < y

        .. code-block:: mlir

            // Compare elements for less than
            %result = ttir.lt(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensors:
            // lhs: [1.0, 2.0, 3.0, 2.0]
            // rhs: [1.0, 2.0, 4.0, 5.0]
            // Output tensor: [0, 0, 1, 1]  # 1 where less, 0 where greater or equal

        Parameters
        ----------
        in0 : Operand
            First input tensor (left-hand side)
        in1 : Operand
            Second input tensor (right-hand side)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A boolean tensor with 1s where left < right and 0s otherwise
        """
        return self._op_proxy(
            ttir.LessThanOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def logical_and(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.logical_and``.

        *Elementwise logical AND operation.*

        Performs elementwise logical AND operation between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if both elements are 1 (true)
        - 0 (false) if at least one element is 0 (false)

        This operation is idempotent, meaning logical_and(x, x) = x.

        .. code-block:: mlir

            // Logical AND operation
            %result = ttir.logical_and(%lhs, %rhs, %output) : tensor<4xi1>, tensor<4xi1>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1, 0, 1, 0]
            // rhs: [1, 1, 0, 1]
            // Output tensor:
            // [1, 0, 0, 0]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.LogicalAndOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def logical_left_shift(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.logical_shift_left``.

        *Elementwise logical shift left operation.*

        Performs elementwise logical shift left operation between two tensors.
        For each pair of corresponding elements, shifts the bits of the first element to the left
        by the number of positions specified by the second element.

        .. code-block:: mlir

            // Logical shift left operation
            %result = ttir.logical_shift_left(%lhs, %rhs, %output) : tensor<3xi8>, tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
            // Input tensors:
            // lhs: [2, 4, 8]  (binary: [00000010, 00000100, 00001000])
            // rhs: [1, 2, 3]  (shift amounts)
            // Output tensor:
            // [4, 16, 64]    (binary: [00000100, 00010000, 01000000])

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor (shift amounts)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.LogicalLeftShiftOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def logical_or(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.logical_or``.

        *Elementwise logical OR operation.*

        Performs elementwise logical OR operation between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if at least one element is 1 (true)
        - 0 (false) if both elements are 0 (false)

        This operation is idempotent, meaning logical_or(x, x) = x.

        Mathematical definition: logical_or(x, y) = x || y

        .. code-block:: mlir

            // Logical OR operation
            %result = ttir.logical_or(%lhs, %rhs, %output) : tensor<4xi1>, tensor<4xi1>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1, 0, 1, 0]
            // rhs: [1, 1, 0, 1]
            // Output tensor:
            // [1, 1, 1, 1]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.LogicalOrOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def logical_right_shift(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.logical_right_shift``.

        *Elementwise logical right shift operation.*

        Performs elementwise logical right shift operation between two tensors.
        For each pair of corresponding elements, shifts the bits of the first element to the right
        by the number of positions specified by the second element.

        .. code-block:: mlir

            // Logical right shift operation
            %result = ttir.logical_right_shift(%lhs, %rhs, %output) : tensor<3xi8>, tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
            // Input tensors:
            // lhs: [8, 16, 32]  (binary: [00001000, 00010000, 00100000])
            // rhs: [1, 2, 3]    (shift amounts)
            // Output tensor:
            // [4, 4, 4]        (binary: [00000100, 00000100, 00000100])
        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor (shift amounts)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.LogicalRightShiftOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def logical_xor(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.logical_xor``.

        *Elementwise logical XOR operation.*

        Performs elementwise logical XOR (exclusive OR) operation between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if exactly one element is 1 (true)
        - 0 (false) if both elements are the same (both 0 or both 1)

        Mathematical definition: logical_xor(x, y) = (x || y) && !(x && y)

        .. code-block:: mlir

            // Logical XOR operation
            %result = ttir.logical_xor(%lhs, %rhs, %output) : tensor<4xi1>, tensor<4xi1>, tensor<4xi1> -> tensor<4xi1>
            // Input tensors:
            // lhs: [1, 0, 1, 0]
            // rhs: [1, 1, 0, 1]
            // Output tensor:
            // [0, 1, 1, 1]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.LogicalXorOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def bitwise_and(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.bitwise_and``.

        *Elementwise bitwise AND operation.*

        Performs elementwise bitwise AND operation between two tensors.
        For each pair of corresponding elements, performs a bitwise AND on their binary representations.

        This operation is typically used with integer data types and has the following properties:
        - Commutative: bitwise_and(x, y) = bitwise_and(y, x)
        - Associative: bitwise_and(x, bitwise_and(y, z)) = bitwise_and(bitwise_and(x, y), z)
        - Identity: bitwise_and(x, -1) = x
        - Zero: bitwise_and(x, 0) = 0

        .. code-block:: mlir

            // Bitwise AND with integer tensors
            %result = ttir.bitwise_and(%lhs, %rhs, %output) : tensor<3xi8>, tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
            // Input tensors:
            // lhs: [5, 3, 255]  (binary: [00000101, 00000011, 11111111])
            // rhs: [3, 6, 129]   (binary: [00000011, 00000110, 10000001])
            // Output tensor:
            // [1, 2, 129]    (binary: [00000001, 00000010, 10000001])

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.BitwiseAndOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def bitwise_or(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.bitwise_or``.

        *Elementwise bitwise OR operation.*

        Performs elementwise bitwise OR operation between two tensors.
        For each pair of corresponding elements, performs a bitwise OR on their binary representations.

        This operation is typically used with integer data types and has the following properties:
        - Commutative: bitwise_or(x, y) = bitwise_or(y, x)
        - Associative: bitwise_or(x, bitwise_or(y, z)) = bitwise_or(bitwise_or(x, y), z)
        - Identity: bitwise_or(x, 0) = x
        - One: bitwise_or(x, -1) = -1

        .. code-block:: mlir

            // Bitwise OR with integer tensors
            %result = ttir.bitwise_or(%lhs, %rhs, %output) : tensor<3xi8>, tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
            // Input tensors:
            // lhs: [5, 3, 255]  (binary: [00000101, 00000011, 11111111])
            // rhs: [3, 6, 129]   (binary: [00000011, 00000110, 10000001])
            // Output tensor:
            // [7, 7, 255]    (binary: [00000111, 00000111, 11111111])

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        return self._op_proxy(
            ttir.BitwiseOrOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def bitwise_xor(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.bitwise_xor``.

        *Elementwise bitwise XOR operation.*

        Performs elementwise bitwise XOR (exclusive OR) operation between two tensors.
        For each pair of corresponding elements, performs a bitwise XOR on their binary representations.

        .. code-block:: mlir

            // Bitwise XOR with integer tensors
            %result = ttir.bitwise_xor(%input1, %input2, %output) : tensor<2x2xi32>, tensor<2x2xi32> -> tensor<2x2xi32>
            // Input1 tensor:
            // [[1, 3],  // binary: [[0001, 0011],
            //  [5, 7]]  //         [0101, 0111]]
            // Input2 tensor:
            // [[2, 3],  // binary: [[0010, 0011],
            //  [6, 7]]  //         [0110, 0111]]
            // Output tensor:
            // [[3, 0],  // binary: [[0011, 0000],
            //  [3, 0]]  //         [0011, 0000]]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the bitwise XOR of corresponding elements
        """
        return self._op_proxy(
            ttir.BitwiseXorOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def minimum(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.minimum``.

        *Elementwise minimum operation.*

        Returns the element-wise minimum of two tensors.
        This operation is idempotent and partially broadcastable.

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with minimum values
        """
        return self._op_proxy(
            ttir.MinimumOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def subtract(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.subtract``.

        *Elementwise subtraction operation.*

        Performs elementwise subtraction between two tensors.
        For each pair of corresponding elements, subtracts the element in the second
        tensor from the element in the first tensor.

        Mathematical definition: subtract(x, y) = x - y

        .. code-block:: mlir

            // Subtract corresponding elements
            %result = ttir.subtract(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // lhs: [3.5, 0.0, -1.2]
            // rhs: [1.5, 2.0, -3.2]
            // Output tensor:
            // [2.0, -2.0, 2.0]

        Parameters
        ----------
        in0 : Operand
            First input tensor (minuend)
        in1 : Operand
            Second input tensor (subtrahend)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise difference of the inputs
        """
        return self._op_proxy(
            ttir.SubtractOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def remainder(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.remainder``.

        *Elementwise remainder operation.*

        Computes the element-wise remainder of division (modulo operation).

        Parameters
        ----------
        in0 : Operand
            First input tensor (dividend)
        in1 : Operand
            Second input tensor (divisor)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with remainder values
        """
        return self._op_proxy(
            ttir.RemainderOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def pow(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.pow``.

        *Elementwise power operation.*

        Takes the first tensor to the power of the second tensor element-wise.

        Parameters
        ----------
        in0 : Operand
            First input tensor (base)
        in1 : Operand
            Second input tensor (exponent)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with power values
        """
        return self._op_proxy(
            ttir.PowOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    # class TTIR_ReductionOp

    def argmax(
        self,
        in0: Operand,
        dim_arg: List[int],
        keep_dim: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.argmax``.

        *Argmax reduction operation.*

        Returns the indices of the maximum values along the specified dimensions.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim_arg : List[int]
            Dimensions to reduce over
        keep_dim : bool, optional
            If True, retains reduced dimensions with length 1 (default: False)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor containing the indices of maximum values
        """
        kwargs = {"dim_arg": dim_arg, "keep_dim": keep_dim}
        return self._op_proxy(
            ttir.ArgMaxOp,
            [in0],
            ttir_kwargs=kwargs,
            output_type=IntegerType.get_signless(32, self._ctx),
            unit_attrs=unit_attrs,
        )

    def sum(
        self,
        in0: Operand,
        dim_arg: List[int] = [0],
        keep_dim: bool = True,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.sum``.

        *Sum reduction operation.*

        The `sum` operation computes the sum of elements along specified dimensions of the input tensor.
        If `dim_arg` is not provided, the sum is computed over all dimensions. If `keep_dim` is True,
        the reduced dimensions are retained with a size of 1.

        .. code-block:: mlir

            // Sum along dimension 1
            %input = ... : tensor<2x3xf32>
            %output = ttir.empty() : tensor<2xf32>
            %result = ttir.sum(%input, %output) {keep_dim = false, dim_arg = [1: i32]} : tensor<2x3xf32>, tensor<2xf32> -> tensor<2xf32>
            // Input: [[1.0, 2.0, 3.0],
            //         [4.0, 5.0, 6.0]]
            // Output: [6.0, 15.0]  // Sum of each row

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim_arg : List[int], optional
            Dimensions to reduce over (default: [0])
        keep_dim : bool, optional
            If True, retains reduced dimensions with length 1 (default: True)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with summed values
        """
        return self._op_proxy(
            ttir.SumOp,
            [in0],
            ttir_kwargs={"dim_arg": dim_arg, "keep_dim": keep_dim},
            unit_attrs=unit_attrs,
        )

    def mean(
        self,
        in0: Operand,
        dim_arg: List[int] = [0],
        keep_dim: bool = True,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.mean``.

        *Mean reduction operation.*

        Computes the mean of elements along specified dimensions of the input tensor.
        If `dim_arg` is not provided, the mean is computed over all dimensions.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim_arg : List[int], optional
            Dimensions to reduce over (default: [0])
        keep_dim : bool, optional
            If True, retains reduced dimensions with length 1 (default: True)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with mean values
        """
        return self._op_proxy(
            ttir.MeanOp,
            [in0],
            ttir_kwargs={"dim_arg": dim_arg, "keep_dim": keep_dim},
            unit_attrs=unit_attrs,
        )

    def max(
        self,
        in0: Operand,
        dim_arg: int = None,
        keep_dim: bool = True,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.max``.

        *Maximum reduction operation.*

        Returns the maximum values along the specified dimension.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim_arg : int, optional
            Dimension to reduce over (default: None, reduces over all dimensions)
        keep_dim : bool, optional
            If True, retains reduced dimensions with length 1 (default: True)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with maximum values
        """
        # Handle ttir and golden function arguments for edge cases
        ttir_kwargs = {"keep_dim": keep_dim}
        input_shape = list(self.get_shape(in0))
        ndim = len(input_shape)
        if dim_arg is not None:
            golden_kwargs = {"dim_arg": dim_arg, "keep_dim": keep_dim}
            ttir_kwargs["dim_arg"] = [dim_arg]
            output_shape = input_shape.copy()
            output_shape[dim_arg] = 1
        else:
            golden_kwargs = {"dim_arg": None, "keep_dim": keep_dim}
            output_shape = [1] * ndim

        return self._op_proxy(
            ttir.MaxOp,
            [in0],
            golden_kwargs=golden_kwargs,
            ttir_kwargs=ttir_kwargs,
            output_shape=output_shape,
            unit_attrs=unit_attrs,
        )

    def min(
        self,
        in0: Operand,
        dim_arg: int = None,
        keep_dim: bool = True,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.min``.

        *Minimum reduction operation.*

        Returns the minimum values along the specified dimension.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim_arg : int, optional
            Dimension to reduce over (default: None, reduces over all dimensions)
        keep_dim : bool, optional
            If True, retains reduced dimensions with length 1 (default: True)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with minimum values
        """
        # Handle ttir and golden function arguments for edge cases
        ttir_kwargs = {"keep_dim": keep_dim}
        output_shape = [1] * len(self.get_shape(in0))
        if dim_arg:
            ttir_kwargs["dim_arg"] = [dim_arg]
        if not keep_dim:
            output_shape = torch.Size([1])

        return self._op_proxy(
            ttir.MinOp,
            [in0],
            ttir_kwargs=ttir_kwargs,
            output_shape=output_shape,
            unit_attrs=unit_attrs,
        )

    # NOTE: Not useable. Boolean tensors are not supported by the runtime. Issue #1775
    def reduce_and(
        self,
        in0: Operand,
        keep_dim: bool = True,
        dim_args: Optional[List] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.reduce_and``.

        *Logical AND reduction operation.*

        Computes the logical AND of elements along specified dimensions.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        keep_dim : bool, optional
            If True, retains reduced dimensions with length 1 (default: True)
        dim_args : Optional[List], optional
            Dimensions to reduce over (default: None, reduces over all dimensions)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with logical AND values
        """
        return self._op_proxy(
            ttir.ReduceAndOp,
            [in0],
            ttir_kwargs={"dim_arg": dim_args, "keep_dim": keep_dim},
            unit_attrs=unit_attrs,
        )

    # NOTE: Not useable. Boolean tensors are not supported by the runtime. Issue #1775
    def reduce_or(
        self,
        in0: Operand,
        keep_dim: bool = True,
        dim_args: Optional[List] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.reduce_or``.

        *Logical OR reduction operation.*

        Computes the logical OR of elements along specified dimensions.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        keep_dim : bool, optional
            If True, retains reduced dimensions with length 1 (default: True)
        dim_args : Optional[List], optional
            Dimensions to reduce over (default: None, reduces over all dimensions)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with logical OR values
        """
        return self._op_proxy(
            ttir.ReduceOrOp,
            [in0],
            ttir_kwargs={"dim_arg": dim_args, "keep_dim": keep_dim},
            unit_attrs=unit_attrs,
            output_type=F32Type.get(self._ctx),
        )

    def prod(
        self,
        in0: Operand,
        dim_arg: List[int],
        keep_dim: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.prod``.

        *Product reduction operation.*

        Computes the product of elements along specified dimensions.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim_arg : List[int]
            Dimensions to reduce over
        keep_dim : bool, optional
            If True, retains reduced dimensions with length 1 (default: False)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with product values
        """
        return self._op_proxy(
            ttir.ProdOp,
            [in0],
            ttir_kwargs={"keep_dim": keep_dim, "dim_arg": dim_arg},
            unit_attrs=unit_attrs,
        )

    def embedding(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.embedding``.

        *Embedding lookup operation.*

        Performs a lookup in an embedding table (in1) using indices (in0).
        Returns a tensor containing the embeddings for the given indices.

        .. code-block:: mlir

            // Lookup embeddings for indices
            %result = ttir.embedding(%indices, %weights, %output) : tensor<2xi32>, tensor<4x3xf32> -> tensor<2x3xf32>
            // Indices tensor:
            // [1, 3]  // Looking up embeddings at indices 1 and 3
            // Weights tensor (embedding table):
            // [[0.1, 0.2, 0.3],  // embedding 0
            //  [0.4, 0.5, 0.6],  // embedding 1
            //  [0.7, 0.8, 0.9],  // embedding 2
            //  [1.0, 1.1, 1.2]]  // embedding 3
            // Output tensor:
            // [[0.4, 0.5, 0.6],  // embedding for index 1
            //  [1.0, 1.1, 1.2]]  // embedding for index 3

        Parameters
        ----------
        in0 : Operand
            Input tensor containing indices
        in1 : Operand
            Weight tensor containing embeddings
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the embeddings for the input indices
        """
        return self._op_proxy(
            ttir.EmbeddingOp,
            [in0, in1],
            organize_golden_args=lambda i: (
                self._get_golden_tensor(i[0]),
                self._get_golden_tensor(i[1]),
            ),
            unit_attrs=unit_attrs,
        )

    def cumsum(
        self,
        in0: Operand,
        dim: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.cumsum``.

        *Cumulative sum operation.*

        Computes the cumulative sum of elements along a specified dimension.
        For each element at index i in the dimension, computes the sum of all elements
        with indices ≤ i in that dimension.

        .. code-block:: mlir

            // Compute cumulative sum along dimension 1
            %result = ttir.cumsum(%input, %output, dim = 1) : tensor<2x3xf32>, tensor<2x3xf32> -> tensor<2x3xf32>
            // Input tensor:
            // [[1.0, 2.0, 3.0],
            //  [4.0, 5.0, 6.0]]
            // Output tensor:
            // [[1.0, 3.0, 6.0],
            //  [4.0, 9.0, 15.0]]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim : int
            Dimension along which to compute cumulative sum
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the cumulative sums along the specified dimension
        """
        return self._op_proxy(
            ttir.CumSumOp,
            [in0],
            ttir_kwargs={"dim": dim},
            unit_attrs=unit_attrs,
        )

    def softmax(
        self,
        in0: Operand,
        dimension: int = 1,
        numeric_stable: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.softmax``.

        *Softmax operation.*

        Applies the Softmax function to an n-dimensional input tensor rescaling them
        so that the elements lie in the range [0,1] and sum to 1.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dimension : int, optional
            Dimension along which Softmax will be computed (default: 1)
        numeric_stable : bool, optional
            Whether to use numerically stable softmax computation (default: False)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Output tensor after softmax
        """
        # kwargs handled thru organize_ttir_args
        return self._op_proxy(
            ttir.SoftmaxOp,
            [in0],
            golden_kwargs={"dim": dimension},
            ttir_kwargs={
                "dimension": dimension,
                "numericStable": numeric_stable,
            },
            organize_ttir_args=lambda i, o, _: (
                self._get_type(o),
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
        )

    def transpose(
        self,
        in0: Operand,
        dim0: int = 0,
        dim1: int = 1,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.transpose``.

        *Tensor transpose operation.*

        Swaps two dimensions of a tensor.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim0 : int, optional
            First dimension to swap (default: 0)
        dim1 : int, optional
            Second dimension to swap (default: 1)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with swapped dimensions
        """
        kwargs = {"dim0": dim0, "dim1": dim1}
        return self._op_proxy(
            ttir.TransposeOp,
            [in0],
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def concat(
        self, ins: List[Operand], dim: int = 0, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.concat``.

        *Tensor concatenation operation.*

        Concatenates the given sequence of tensors in the given dimension.
        All tensors must have the same shape, except in the concatenating dimension.

        Parameters
        ----------
        ins : *List[Operand]*
            List of input tensors to concatenate
        dim : int, optional
            Dimension along which to concatenate (default: 0)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Concatenated tensor
        """
        kwargs = {"dim": dim}
        return self._op_proxy(
            ttir.ConcatOp,
            ins,
            ttir_kwargs=kwargs,
            # special handling is needed here to get around arg expansion; `torch.concat` takes a tuple of tensors on input
            organize_golden_args=lambda i: (
                tuple([self._get_golden_tensor(i_i) for i_i in i]),
            ),
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i, o),
            unit_attrs=unit_attrs,
        )

    def repeat(
        self, in0: Operand, dims: List[int], unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.repeat``.

        *Tensor repeat operation.*

        Repeats the tensor along each dimension the number of times given by dims.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dims : *List[int]*
            Number of repetitions for each dimension
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with repeated elements
        """
        return self._op_proxy(
            ttir.RepeatOp,
            [in0],
            ttir_kwargs={"repeat_dimensions": dims},
            unit_attrs=unit_attrs,
        )

    def repeat_interleave(
        self,
        in0: Operand,
        in1: Operand,
        repeats: int,
        dim: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.repeat_interleave``.

        *Tensor repeat interleave operation.*

        Repeats elements of a tensor along a dimension by interleaving the repeated elements.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        repeats : int
            Number of repetitions for each element
        dim : int
            Dimension along which to repeat
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with interleaved repeated elements
        """
        return self._op_proxy(
            ttir.RepeatInterleaveOp,
            [in0, in1],
            ttir_kwargs={"repeats": repeats, "dim": dim},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1]),
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            output_type=self._get_type_from_torch_dtype(
                self._get_golden_tensor(in1).dtype
            ),
            unit_attrs=unit_attrs,
        )

    def fill_cache(
        self,
        in0: Operand,
        in1: Operand,
        batch_offset: int = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.fill_cache``.

        *Cache fill operation.*

        Fills a cache tensor with new values starting at a specified batch offset.
        This operation is typically used in sequence models to initialize or update
        cached states.

        .. code-block:: mlir

            // Fill cache with new values at batch offset 1
            %result = ttir.fill_cache(%new_values, %cache, batch_offset = 1) : tensor<2x3xf32>, tensor<4x3xf32> -> tensor<4x3xf32>
            // New values tensor:
            // [[1.0, 2.0, 3.0],
            //  [4.0, 5.0, 6.0]]
            // Cache tensor before:
            // [[0.1, 0.2, 0.3],
            //  [0.4, 0.5, 0.6],
            //  [0.7, 0.8, 0.9],
            //  [1.0, 1.1, 1.2]]
            // Cache tensor after:
            // [[0.1, 0.2, 0.3],
            //  [1.0, 2.0, 3.0],
            //  [4.0, 5.0, 6.0],
            //  [1.0, 1.1, 1.2]]

        Parameters
        ----------
        in0 : Operand
            New values to fill into cache
        in1 : Operand
            Cache tensor to be filled
        batch_offset : int, optional
            Starting position in batch dimension (default: 0)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The updated cache tensor
        """
        return self._op_proxy(
            ttir.FillCacheOp,
            [in0, in1],
            ttir_kwargs={"batch_offset": batch_offset},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1]),
            organize_golden_args=lambda i: (
                self._get_golden_tensor(i[0]),
                self._get_golden_tensor(i[1]),
            ),
            unit_attrs=unit_attrs,
        )

    def update_cache(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        batch_offset: int = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.update_cache``.

        *Cache update operation.*

        Updates a cache tensor by combining new values with existing cache values,
        starting at a specified batch offset. This operation is typically used in
        sequence models to maintain and update cached states.

        .. code-block:: mlir

            // Update cache with new values at batch offset 1
            %result = ttir.update_cache(%new_values, %old_cache, %mask, batch_offset = 1) \
                : tensor<2x3xf32>, tensor<4x3xf32>, tensor<2xi1> -> tensor<4x3xf32>
            // New values tensor:
            // [[1.0, 2.0, 3.0],
            //  [4.0, 5.0, 6.0]]
            // Old cache tensor:
            // [[0.1, 0.2, 0.3],
            //  [0.4, 0.5, 0.6],
            //  [0.7, 0.8, 0.9],
            //  [1.0, 1.1, 1.2]]
            // Mask tensor:
            // [true, false]  // Only update first new value
            // Output tensor:
            // [[0.1, 0.2, 0.3],
            //  [1.0, 2.0, 3.0],  // Updated with first new value
            //  [0.7, 0.8, 0.9],  // Kept old value due to mask
            //  [1.0, 1.1, 1.2]]

        Parameters
        ----------
        in0 : Operand
            New values to update cache with
        in1 : Operand
            Cache tensor to be updated
        in2 : Operand
            Mask tensor indicating which values to update
        batch_offset : int, optional
            Starting position in batch dimension (default: 0)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The updated cache tensor
        """
        return self._op_proxy(
            ttir.UpdateCacheOp,
            [in0, in1, in2],
            ttir_kwargs={"batch_offset": batch_offset},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1], i[2]),
            organize_golden_args=lambda i: (
                self._get_golden_tensor(i[0]),
                self._get_golden_tensor(i[1]),
                self._get_golden_tensor(i[2]),
            ),
            unit_attrs=unit_attrs,
        )

    def broadcast(
        self,
        in0: Operand,
        broadcast_dimensions: List[int],
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.broadcast``.

        *Tensor broadcast operation.*

        Broadcasts a tensor to a new shape by replicating its values along specified dimensions.
        The broadcast_dimensions parameter specifies how dimensions of the input map to
        dimensions of the output.

        .. code-block:: mlir

            // Broadcast a 1D tensor to 2D
            %result = ttir.broadcast(%input, %output, broadcast_dimensions = [1]) : tensor<3xf32>, tensor<2x3xf32> -> tensor<2x3xf32>
            // Input tensor:
            // [1.0, 2.0, 3.0]
            // Output tensor:
            // [[1.0, 2.0, 3.0],
            //  [1.0, 2.0, 3.0]]

        Parameters
        ----------
        in0 : Operand
            Input tensor to broadcast
        broadcast_dimensions : *List[int]*
            List of dimension mappings from input to output
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The broadcasted tensor
        """
        output_shape = []
        for i in range(len(broadcast_dimensions)):
            if broadcast_dimensions[i] != 1:
                output_shape.append(broadcast_dimensions[i])
            else:
                output_shape.append(self.get_shape(in0)[i])
        return self._op_proxy(
            ttir.BroadcastOp,
            [in0],
            golden_kwargs={"size": output_shape},
            ttir_kwargs={"broadcast_dimensions": broadcast_dimensions},
            output_shape=output_shape,
            unit_attrs=unit_attrs,
        )

    def conv2d(
        self,
        in0: Operand,
        weight: Operand,
        bias: Optional[Operand],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.conv2d``.

        *Conv2d operation.*

        Applies a 2D convolution over an input image composed of several input planes.
        This operation performs a 2D convolution on the input tensor using the provided weight tensor
        and optional bias. It supports configurable stride, padding, dilation, and grouping parameters.

        .. code-block:: mlir

            // Basic 2D convolution
            %input = ... : tensor<1x28x28x3xf32>    // Batch size 1, 28x28 image, 3 channels
            %weight = ... : tensor<16x3x3x3xf32>    // 16 output channels, 3 input channels, 3x3 kernel
            %bias = ... : tensor<1x1x1x16xf32>      // Bias for 16 output channels
            %output = ttir.empty() : tensor<1x26x26x16xf32>  // Output shape with no padding
            %result = ttir.conv2d(%input, %weight, %bias, %output) {
                stride = [1, 1],
                padding = [0, 0, 0, 0],
                dilation = [1, 1],
                groups = 1
            }

        Parameters
        ----------
        in0 : Operand
            Input tensor in (N, H_in, W_in, C) format
        weight : Operand
            Weight tensor in (O, C/G, K_H, K_W) format
        bias : *Optional[Operand]*
            Optional bias tensor in (1, 1, 1, O) format
        stride : *Union[int, List[int]]*, optional
            Stride for height and width dimensions (default: 1)
        padding : *Union[int, List[int]]*, optional
            Padding for all sides or [top, left, bottom, right] (default: 0)
        dilation : *Union[int, List[int]]*, optional
            Spacing between kernel elements (default: 1)
        groups : int, optional
            Number of blocked connections from input to output channels (default: 1)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Output tensor after convolution
        """
        if not bias:
            bias = None
        golden_bias = self._get_golden_tensor(bias) if bias is not None else None
        return self._op_proxy(
            ttir.Conv2dOp,
            [in0, weight],
            ttir_kwargs={
                "stride": (
                    IntegerAttr.get(IntegerType.get_signed(32), stride)
                    if isinstance(stride, int)
                    else DenseI32ArrayAttr.get(stride)
                ),
                "padding": (
                    IntegerAttr.get(IntegerType.get_signed(32), padding)
                    if isinstance(padding, int)
                    else DenseI32ArrayAttr.get(padding)
                ),
                "dilation": (
                    IntegerAttr.get(IntegerType.get_signed(32), dilation)
                    if isinstance(dilation, int)
                    else DenseI32ArrayAttr.get(dilation)
                ),
                "groups": groups,
                "bias": bias,
            },
            golden_kwargs={
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
                "bias": golden_bias,
            },
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1], o),
            unit_attrs=unit_attrs,
        )

    def conv_transpose2d(
        self,
        in0: Operand,
        weight: Operand,
        bias: Optional[Operand],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        output_padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.conv_transpose2d``.

        *2D transposed convolution operation.*

        Applies a 2D transposed convolution over an input image. This operation
        can be seen as the gradient of Conv2d with respect to its input.
        Also known as a deconvolution or fractionally strided convolution.

        .. code-block:: mlir

            // Apply 2D transposed convolution
            %result = ttir.conv_transpose2d(%input, %weight, %bias, %output,
                                          stride = [2, 2], padding = [1, 1],
                                          output_padding = [1, 1], dilation = [1, 1],
                                          groups = 1) :
                tensor<1x1x4x4xf32>, tensor<1x1x3x3xf32>, tensor<1xf32>, tensor<1x1x9x9xf32> -> tensor<1x1x9x9xf32>
            // Input tensor: 4x4 feature map
            // Weight tensor: 3x3 kernel
            // Output tensor: 9x9 upsampled feature map

        Parameters
        ----------
        in0 : Operand
            Input tensor of shape (batch, in_channels, height, width)
        weight : Operand
            Weight tensor of shape (in_channels, out_channels/groups, kernel_height, kernel_width)
        bias : Optional[Operand]
            Optional bias tensor of shape (out_channels)
        stride : *Union[int, List[int]]*
            Stride of the convolution
        padding : *Union[int, List[int]]*
            Padding added to input
        output_padding : *Union[int, List[int]]*
            Additional size added to output shape
        dilation : *Union[int, List[int]]*
            Dilation of the kernel
        groups : int
            Number of blocked connections from input to output channels
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The output tensor after transposed convolution
        """
        if not bias:
            bias = None
        return self._op_proxy(
            ttir.ConvTranspose2dOp,
            [in0, weight],
            ttir_kwargs={
                "stride": (
                    IntegerAttr.get(IntegerType.get_signless(32), stride)
                    if isinstance(stride, int)
                    else DenseI32ArrayAttr.get(stride)
                ),
                "padding": (
                    IntegerAttr.get(IntegerType.get_signless(32), padding)
                    if isinstance(padding, int)
                    else DenseI32ArrayAttr.get(padding)
                ),
                "output_padding": (
                    IntegerAttr.get(IntegerType.get_signless(32), output_padding)
                    if isinstance(output_padding, int)
                    else DenseI32ArrayAttr.get(output_padding)
                ),
                "dilation": (
                    IntegerAttr.get(IntegerType.get_signless(32), dilation)
                    if isinstance(dilation, int)
                    else DenseI32ArrayAttr.get(dilation)
                ),
                "groups": (
                    IntegerAttr.get(IntegerType.get_signless(32), groups)
                    if isinstance(groups, int)
                    else DenseI32ArrayAttr.get(groups)
                ),
                "bias": bias,
            },
            unit_attrs=unit_attrs,
        )

    def max_pool2d(
        self,
        in0: Operand,
        kernel: Union[int, List[int]],
        stride: Union[int, List[int]],
        dilation: Union[int, List[int]],
        padding: Union[int, List[int]],
        ceil_mode: bool,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.max_pool2d``.

        *Max pooling operation.*

        Applies a 2D max pooling over an input signal composed of several input planes.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        kernel_size : *Union[int, List[int]]*
            Size of the pooling window
        stride : *Optional[Union[int, List[int]]]*
            Stride of the pooling window (default: None, same as kernel_size)
        padding : *Union[int, List[int]]*, optional
            Padding added to all sides of input (default: 0)
        dilation : *Union[int, List[int]]*, optional
            Controls spacing between kernel elements (default: 1)
        ceil_mode : bool, optional
            When True, use ceil instead of floor for output shape (default: False)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Output tensor after max pooling
        """

        return self._op_proxy(
            ttir.MaxPool2dOp,
            [in0],
            ttir_kwargs={
                "kernel": (
                    IntegerAttr.get(IntegerType.get_signed(32), kernel)
                    if isinstance(kernel, int)
                    else DenseI32ArrayAttr.get(kernel)
                ),
                "stride": (
                    IntegerAttr.get(IntegerType.get_signed(32), stride)
                    if isinstance(stride, int)
                    else DenseI32ArrayAttr.get(stride)
                ),
                "dilation": (
                    IntegerAttr.get(IntegerType.get_signed(32), dilation)
                    if isinstance(dilation, int)
                    else DenseI32ArrayAttr.get(dilation)
                ),
                "padding": (
                    IntegerAttr.get(IntegerType.get_signed(32), padding)
                    if isinstance(padding, int)
                    else DenseI32ArrayAttr.get(padding)
                ),
                "ceil_mode": ceil_mode,
            },
            unit_attrs=unit_attrs,
        )

    def avg_pool2d(
        self,
        in0: Operand,
        kernel: Union[int, List[int]],
        stride: Union[int, List[int]],
        dilation: Union[int, List[int]],
        padding: Union[int, List[int]],
        ceil_mode: bool,
        count_include_pad: bool = True,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.max_pool2d``.

        *Max pooling operation.*

        Applies a 2D max pooling over an input signal composed of several input planes.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        kernel_size : *Union[int, List[int]]*
            Size of the pooling window
        stride : *Optional[Union[int, List[int]]]*
            Stride of the pooling window (default: None, same as kernel_size)
        padding : *Union[int, List[int]]*, optional
            Padding added to all sides of input (default: 0)
        dilation : *Union[int, List[int]]*, optional
            Controls spacing between kernel elements (default: 1)
        ceil_mode : bool, optional
            When True, use ceil instead of floor for output shape (default: False)
        count_include_pad : bool, optional
            When True , include padding in the average calculation (default: True)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Output tensor after max pooling
        """

        return self._op_proxy(
            ttir.AvgPool2dOp,
            [in0],
            ttir_kwargs={
                "kernel": (
                    IntegerAttr.get(IntegerType.get_signed(32), kernel)
                    if isinstance(kernel, int)
                    else DenseI32ArrayAttr.get(kernel)
                ),
                "stride": (
                    IntegerAttr.get(IntegerType.get_signed(32), stride)
                    if isinstance(stride, int)
                    else DenseI32ArrayAttr.get(stride)
                ),
                "dilation": (
                    IntegerAttr.get(IntegerType.get_signed(32), dilation)
                    if isinstance(dilation, int)
                    else DenseI32ArrayAttr.get(dilation)
                ),
                "padding": (
                    IntegerAttr.get(IntegerType.get_signed(32), padding)
                    if isinstance(padding, int)
                    else DenseI32ArrayAttr.get(padding)
                ),
                "ceil_mode": ceil_mode,
                "count_include_pad": count_include_pad,
            },
            unit_attrs=unit_attrs,
        )

    def batch_norm(
        self,
        in0: Operand,
        scale: Operand,
        offset: Operand,
        mean: Operand,
        variance: Operand,
        epsilon: float = 1e-5,
        dimension: int = 1,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.batch_norm``.

        *Batch normalization operation.*

        Applies batch normalization to the input tensor using the provided scale, offset,
        mean, and variance. This operation normalizes the input tensor across the specified dimension:
        batch_norm(x, scale, offset, mean, variance, epsilon, dimension) =
         (x - mean) / sqrt(variance + epsilon) * scale + offset

        Parameters
        ----------
        in0 : Operand
            Input tensor to normalize
        scale : Operand
            Scale tensor for normalization
        offset : Operand
            Offset tensor for normalization
        mean : Operand
            Mean tensor for normalization
        variance : Operand
            Variance tensor for normalization
        epsilon : float, optional
            Small value added to variance for numerical stability (default: 1e-5)
        dimension : int, optional
            Dimension along which to normalize (default: 1)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Output tensor after batch normalization
        """
        return self._op_proxy(
            ttir.BatchNormInferenceOp,
            [in0, scale, offset, mean, variance],
            golden_kwargs={
                "epsilon": epsilon,
                "dim": dimension,
            },
            ttir_kwargs={
                "epsilon": FloatAttr.get_f32(epsilon),
                "dimension": IntegerAttr.get(IntegerType.get_signless(32), dimension),
            },
            # organize_ttir_args=lambda i, o, _: (self._get_type(o), *i, o),
            unit_attrs=unit_attrs,
        )

    def reshape(
        self, in0: Operand, shape: Shape, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.reshape``.

        *Tensor reshape operation.*

        The `reshape` operation changes the shape of a tensor without changing the data or number of elements.
        The total number of elements in the tensor must remain the same after reshaping. This operation is
        commonly used in neural networks to change the dimensionality of tensors between layers.

        .. code-block:: mlir

            // Reshape a 2x3 tensor to a 1x6 tensor
            %input = ... : tensor<2x3xf32>  // Input tensor with shape [2,3]
            %output = ttir.empty() : tensor<1x6xf32>  // Output tensor with shape [1,6]
            %result = ttir.reshape(%input, %output) {shape = [1, 6]} :
                tensor<2x3xf32>, tensor<1x6xf32> -> tensor<1x6xf32>

        Parameters
        ----------
        in0 : Operand
            Input tensor to reshape
        shape : Shape
            The new shape for the tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The reshaped tensor
        """
        kwargs = {"shape": shape}
        return self._op_proxy(
            ttir.ReshapeOp,
            [in0],
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def pad(
        self,
        in0: Operand,
        padding: List[int],
        value: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.pad``.

        *Tensor padding operation.*

        Pads a tensor with a constant value. The padding amount is specified for each dimension
        and can be asymmetric (different padding at the start and end of each dimension).

        Parameters
        ----------
        in0 : Operand
            Input tensor to pad
        in1 : Operand
            Output tensor
        padding : *List[int]*
            Amount of padding for each dimension
        value : int
            Value to use for padding
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The padded tensor
        """
        output_shape = []
        for i in range(len(padding) // 2):
            output_shape.append(
                self.get_shape(in0)[i] + padding[2 * i] + padding[2 * i + 1]
            )
        return self._op_proxy(
            ttir.PadOp,
            [in0],
            ttir_kwargs={"padding": padding, "value": value},
            output_shape=output_shape,
            unit_attrs=unit_attrs,
        )

    def select(
        self,
        in0: Operand,
        dim: int = 0,
        begin: int = 0,
        length: int = 2,
        stride: int = 2,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.select``.

        *Tensor selection operation.*

        Selects a slice of the input tensor along the specified dimension with given stride.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim : int, optional
            Dimension to select from (default: 0)
        begin : int, optional
            Starting index (default: 0)
        length : int, optional
            Length of the slice (default: 2)
        stride : int, optional
            Stride between elements (default: 2)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The selected slice of the tensor
        """
        return self._op_proxy(
            ttir.IndexSelectOp,
            [in0],
            ttir_kwargs={
                "dim": dim,
                "begin": begin,
                "length": length,
                "stride": stride,
            },
            unit_attrs=unit_attrs,
        )

    def index(
        self,
        in0: Operand,
        dim: int,
        begin: int,
        end: int,
        step: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.index``.

        *Tensor indexing operation.*

        Indexes into the input tensor along the specified dimension using a range of indices.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim : int
            Dimension to index into
        begin : int
            Starting index
        end : int
            Ending index (exclusive)
        step : int
            Step size between indices
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The indexed tensor
        """
        return self._op_proxy(
            ttir.IndexOp,
            [in0],
            ttir_kwargs={"dim": dim, "begin": begin, "end": end, "step": step},
            unit_attrs=unit_attrs,
        )

    def squeeze(
        self,
        in0: Operand,
        dim: Optional[int] = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.squeeze``.

        *Tensor squeeze operation.*

        Removes dimensions of size 1 from the shape of a tensor.
        If dim is specified, only squeezes the dimension if it has size 1.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim : Optional[int], optional
            Dimension to squeeze (default: 0)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with specified dimensions of size 1 removed
        """
        kwargs = {"dim": dim}
        return self._op_proxy(
            ttir.SqueezeOp,
            [in0],
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def unsqueeze(
        self,
        in0: Operand,
        dim: Optional[int] = 0,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.unsqueeze``.

        *Tensor unsqueeze operation.*

        Adds a dimension of size 1 at the specified position.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dim : Optional[int], optional
            Position to insert the new dimension (default: 0)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with a new dimension of size 1 inserted
        """
        kwargs = {"dim": dim}
        return self._op_proxy(
            ttir.UnsqueezeOp,
            [in0],
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def clamp_scalar(
        self,
        in0: Operand,
        min_arg: Optional[float] = None,
        max_arg: Optional[float] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        kwargs = {"min": min_arg, "max": max_arg}
        return self._op_proxy(
            ttir.ClampScalarOp,
            [in0],
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def clamp_tensor(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        return self._op_proxy(
            ttir.ClampTensorOp,
            [in0, in1, in2],
            organize_golden_args=lambda i: [
                self._get_golden_tensor(in0),
                self._get_golden_tensor(in1),
                self._get_golden_tensor(in2),
            ],
            unit_attrs=unit_attrs,
        )

    def zeros(
        self,
        shape: Shape,
        data_type: Optional[Type] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.zeros``.

        *Creates a tensor filled with zeros.*

        Returns a tensor of given shape filled with zeros.

        Parameters
        ----------
        shape : Shape
            Shape of the output tensor
        data_type : *Optional[Type]*, optional
            Optional data type of the output tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor of zeros with specified shape
        """
        dtype = self._get_type_from_torch_dtype(data_type)
        output = self._create_ranked_tensor_type(shape, dtype)
        return self._op_proxy(
            ttir.ZerosOp,
            [],
            ttir_kwargs={"result": output, "shape": shape},
            organize_ttir_args=lambda i, o, shape: 0,
            output_type=dtype,
            unit_attrs=unit_attrs,
        )

    def ones(self, shape: Shape, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.ones``.

        *Creates a tensor filled with ones.*

        Returns a tensor of given shape filled with ones.

        Parameters
        ----------
        shape : Shape
            Shape of the output tensor
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor of ones with specified shape
        """
        output = self._create_ranked_tensor_type(shape)
        return self._op_proxy(
            ttir.OnesOp,
            [],
            ttir_kwargs={"result": output, "shape": shape},
            organize_ttir_args=lambda i, o, shape: 0,
            unit_attrs=unit_attrs,
        )

    # Note: TTRT runtime supports float32, bfloat16, uint8, uint16, uint32, int32
    # For bfloat16, use precision-friendly values [-1.0 to 1.0] for best results
    def constant(
        self,
        tensor: torch.Tensor,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.constant``.

        *Creates a tensor with the specified values.*

        Returns a tensor of given shape with the specified values.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor containing the constant values
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with the specified values
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("constant expects a torch.Tensor input")

        if tensor.numel() == 0:
            raise ValueError("Cannot create constant with empty tensor")

        mlir_dtype = self._get_type_from_torch_dtype(tensor.dtype)

        shape = list(tensor.shape)
        tensor_type = RankedTensorType.get(shape, mlir_dtype)

        flat_values = tensor.flatten().tolist()

        if tensor.numel() == 1 or len(set(flat_values)) == 1:
            # Splat case
            if isinstance(mlir_dtype, IntegerType):
                attr = IntegerAttr.get(mlir_dtype, int(flat_values[0]))
            elif isinstance(mlir_dtype, FloatType):
                attr = FloatAttr.get(mlir_dtype, float(flat_values[0]))
            else:
                raise NotImplementedError(f"Unsupported MLIR type: {mlir_dtype}")

            value_attr = DenseElementsAttr.get_splat(tensor_type, attr)
        else:
            # Non-splat case
            if isinstance(mlir_dtype, IntegerType):
                elems = [IntegerAttr.get(mlir_dtype, int(v)) for v in flat_values]
            elif isinstance(mlir_dtype, FloatType):
                elems = [FloatAttr.get(mlir_dtype, float(v)) for v in flat_values]
            else:
                raise NotImplementedError(f"Unsupported MLIR type: {mlir_dtype}")

            value_attr = DenseElementsAttr.get(elems, tensor_type)

        return self._op_proxy(
            ttir.ConstantOp,
            [],
            golden_kwargs={"value": tensor},
            ttir_kwargs={"result": tensor_type, "value": value_attr},
            organize_ttir_args=lambda i, o, _: 0,
            unit_attrs=unit_attrs,
        )

    def reverse(
        self, in0: Operand, dims: List[int], unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.reverse``.

        *Tensor reverse operation.*

        Reverses the order of elements along specified dimensions.
        The input and output shapes must match.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        dims : *List[int]*
            Dimensions to reverse
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with reversed elements
        """
        return self._op_proxy(
            ttir.ReverseOp,
            [in0],
            ttir_kwargs={"dimensions": dims},
            unit_attrs=unit_attrs,
        )

    def linear(
        self,
        in0: Operand,
        in1: Operand,
        bias: Optional[Operand] = None,
        transpose_a: bool = False,
        transpose_b: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.linear``.

        *Linear transformation operation.*

        Applies a linear transformation to the incoming data: y = xA^T + b

        Parameters
        ----------
        in0 : Operand
            Input tensor
        weight : Operand
            Weight matrix
        bias : *Optional[Operand]*
            Bias vector (default: None)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Output tensor after linear transformation
        """
        inputs = [in0, in1]
        if bias is not None:
            inputs.append(bias)

        # Convert bias operand to tensor for golden function
        golden_bias = self._get_golden_tensor(bias) if bias is not None else None

        return self._op_proxy(
            ttir.LinearOp,
            [in0, in1],
            golden_kwargs={
                "transpose_a": transpose_a,
                "transpose_b": transpose_b,
                "bias": golden_bias,
            },
            ttir_kwargs={
                "transpose_a": transpose_a,
                "transpose_b": transpose_b,
                "bias": bias,
            },
            unit_attrs=unit_attrs,
        )

    def matmul(
        self,
        in0: Operand,
        in1: Operand,
        bias: Optional[Operand] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        inputs = [in0, in1]
        if bias:
            inputs.append(bias)
        return self._op_proxy(
            ttir.MatmulOp,
            inputs,
            unit_attrs=unit_attrs,
        )

    def permute(
        self,
        in0: Operand,
        permutation: List[int],
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.permute``.

        *Tensor permutation operation.*

        Permutes the dimensions of the input tensor according to the given permutation.

        Parameters
        ----------
        in0 : Operand
            Input tensor
        permutation : *List[int]*
            The desired ordering of dimensions
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor with permuted dimensions
        """
        return self._op_proxy(
            ttir.PermuteOp,
            [in0],
            ttir_kwargs={"permutation": DenseI64ArrayAttr.get(permutation)},
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            unit_attrs=unit_attrs,
        )

    def upsample2d(
        self,
        in0: Operand,
        in1: Operand,
        scale_factor: Union[int, List[int]],
        mode: str = "nearest",
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        output_shape = self._get_golden_tensor(in1).shape
        kwargs = {
            "scale_factor": (
                IntegerAttr.get(IntegerType.get_signed(32), scale_factor)
                if isinstance(scale_factor, int)
                else DenseI32ArrayAttr.get(scale_factor)
            ),
            "mode": mode,
        }
        return self._op_proxy(
            ttir.Upsample2dOp,
            [in0, in1],
            ttir_kwargs=kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(i[1]), i[0], o),
            output_shape=output_shape,
            unit_attrs=unit_attrs,
        )

    def arange(
        self,
        result: Operand,
        start: int,
        end: int,
        step: int,
        arange_dimension: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.arange``.

        *Creates a 1-D tensor of sequential values.*

        Returns a 1-D tensor of size (end - start) / step with values from start to end taken with common difference step.

        Parameters
        ----------
        start : int
            Starting value
        end : int
            Ending value (exclusive)
        step : int, optional
            Step size between values (default: 1)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            1-D tensor with sequential values
        """
        result_tensor = self._get_golden_tensor(result)

        # Build single-dim tensor
        single_dim_tensor = torch.arange(
            start=start, end=end, step=step, dtype=result_tensor.dtype
        )

        # Compute repeat dimensions
        shape = self.get_shape(result)
        repeat_dims = []
        for i in range(len(shape)):
            if i == arange_dimension:
                repeat_dims.append(int(shape[i] / ((end - start) / step)))
            else:
                repeat_dims.append(shape[i])

        # Apply repeat shard-wise
        single_dim_tensor_bt = BuilderGoldenTensor(
            {
                k: shard.repeat(*repeat_dims)
                for k, shard in result_tensor.shard_map.items()
            },
            result_tensor.mesh_shape,
        )

        return self._op_proxy(
            ttir.ArangeOp,
            [result, single_dim_tensor_bt],
            golden_kwargs={
                "start": start,
                "end": end,
                "step": step,
                "arange_dimension": arange_dimension,
            },
            ttir_kwargs={
                "start": start,
                "end": end,
                "step": step,
                "arange_dimension": arange_dimension,
            },
            organize_ttir_args=lambda i, o, _: (self._get_type(o),),
            organize_golden_args=lambda i: [i[1]],
            output_shape=shape,
            output_type=self._get_type_from_torch_dtype(
                self._get_golden_tensor(result).dtype
            ),
            unit_attrs=unit_attrs,
        )

    def exp(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttir.exp``.

        *Elementwise exponential operation.*

        Computes the exponential of each element in the input tensor.
        For each element x, returns e^x, where e is Euler's number (approximately 2.71828).

        .. code-block:: mlir

            // Compute exponential of all elements
            %result = ttir.exp(%input, %output) : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensor:
            // [0.0, 1.0, 2.0]
            // Output tensor:
            // [1.0, 2.71828, 7.38906]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the exponential of each element in the input tensor
        """
        return self._op_proxy(
            ttir.ExpOp,
            [in0],
            unit_attrs=unit_attrs,
        )

    # class TTIR_GenericElementwiseBinaryOp

    def add(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.add``.

        *Elementwise addition operation.*

        Performs elementwise addition between two tensors.
        For each pair of corresponding elements, adds the element in the second
        tensor to the element in the first tensor.

        Mathematical definition: add(x, y) = x + y

        .. code-block:: mlir

            // Add corresponding elements
            %result = ttir.add(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
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
        return self._op_proxy(
            ttir.AddOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def atan2(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.atan2``.

        *Elementwise arctangent operation.*

        Computes the elementwise arctangent of the quotient of its arguments.
        For each pair of corresponding elements (y, x), returns atan2(y, x).

        Mathematical definition: atan2(y, x) = arctan(y / x)

        .. code-block:: mlir

            // Compute arctangent of corresponding elements
            %result = ttir.atan2(%y, %x, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // y: [1.0, 0.0, -1.0]
            // x: [1.0, -1.0, -1.0]
            // Output tensor:
            // [0.7854, 3.1416, -2.3562]
        Parameters
        ----------
        in0 : Operand
            First input tensor (y)
        in1 : Operand
            Second input tensor (x)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes
        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise arctangent of the inputs
        """
        return self._op_proxy(
            ttir.Atan2Op,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def multiply(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.multiply``.

        *Elementwise multiplication operation.*

        Performs elementwise multiplication between two tensors.
        For each pair of corresponding elements, multiplies the element in the first
        tensor by the element in the second tensor.

        Mathematical definition: multiply(x, y) = x * y

        .. code-block:: mlir

            // Multiply corresponding elements
            %result = ttir.multiply(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // lhs: [3.5, 0.0, -1.2]
            // rhs: [1.5, 2.0, -3.2]
            // Output tensor:
            // [5.25, 0.0, 3.84]

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
        return self._op_proxy(
            ttir.MultiplyOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def div(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.div``.

        *Elementwise division operation.*

        Performs elementwise division between two tensors.
        For each pair of corresponding elements, divides the element in the first
        tensor by the element in the second tensor.

        Note: Division by zero behavior depends on the implementation and data type.

        Mathematical definition: div(x, y) = x / y

        .. code-block:: mlir

            // Divide corresponding elements
            %result = ttir.div(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensors:
            // lhs: [3.5, 0.0, -1.2]
            // rhs: [1.5, 2.0, -3.2]
            // Output tensor:
            // [2.333, 0.0, 0.375]

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
            A tensor containing the elementwise quotient of the inputs
        """
        return self._op_proxy(
            ttir.DivOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def maximum(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttir.maximum``.

        *Elementwise maximum operation.*

        Returns the element-wise maximum of two tensors.

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
            Tensor with maximum values
        """
        return self._op_proxy(
            ttir.MaximumOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def quantize(
        self,
        in0: Operand,
        scale: float,
        zero_point: int,
        dtype: torch.dtype,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.quantize``.

        *Quantize floating-point tensor to integer tensor.*

        Converts a floating-point tensor into a quantized integer tensor using the specified
        scale and zero_point parameters. For each element in the input tensor, computes:
        output[i] = (input[i] / scale) + zero_point

        .. code-block:: mlir

            // Quantize float32 tensor to int8
            %result = ttir.quantize(%input, %output) {scale = 0.1 : f32, zero_point = 128 : i32} : tensor<2x2xf32>, tensor<2x2xi8> -> tensor<2x2xi8>
            // Input tensor:
            // [[1.5, -0.2],
            //  [0.0, 3.7]]
            // Output tensor:
            // [[143, 126],
            //  [128, 165]]

        Parameters
        ----------
        in0 : Operand
            Input floating-point tensor to be quantized
        scale : float
            Scale factor for quantization (each integer step represents this value)
        zero_point : int
            Integer value that represents 0.0 in the quantized space
        dtype : torch.dtype
            Target integer data type for quantization (e.g., torch.int8)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The quantized integer tensor
        """

        # kwargs passed thru output type
        return self._op_proxy(
            ttir.QuantizeOp,
            [in0],
            golden_kwargs={"scale": scale, "zero_point": zero_point, "dtype": dtype},
            output_type=self._get_type_from_torch_dtype(
                dtype=dtype, scale=scale, zero_point=zero_point
            ),
            unit_attrs=unit_attrs,
        )

    def dequantize(
        self,
        in0: Operand,
        scale: float,
        zero_point: int,
        dtype: torch.dtype,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.dequantize``.

        *Dequantize integer tensor to floating-point tensor.*

        Converts a quantized integer tensor back into a floating-point tensor using the
        specified scale and zero_point parameters. For each element in the input tensor,
        computes: output[i] = (input[i] - zero_point) * scale

        .. code-block:: mlir

            // Dequantize int8 tensor to float32
            %result = ttir.dequantize(%input, %output) {scale = 0.1 : f32, zero_point = 128 : i32} : tensor<2x2xi8>, tensor<2x2xf32> -> tensor<2x2xf32>
            // Input tensor:
            // [[143, 126],
            //  [128, 165]]
            // Output tensor:
            // [[1.5, -0.2],
            //  [0.0, 3.7]]

        Parameters
        ----------
        in0 : Operand
            Input quantized integer tensor to be dequantized
        scale : float
            Scale factor used in the original quantization
        zero_point : int
            Integer value that represents 0.0 in the quantized space
        dtype : torch.dtype
            Target floating-point data type (e.g., torch.float32)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The dequantized floating-point tensor
        """
        return self._op_proxy(
            ttir.DequantizeOp,
            [in0],
            output_type=self._get_type_from_torch_dtype(dtype=dtype),
            unit_attrs=unit_attrs,
        )

    def requantize(
        self,
        in0: Operand,
        scale: float,
        zero_point: int,
        dtype: torch.dtype,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.requantize``.

        *Requantize integer tensor to new scale and zero-point.*

        Converts a quantized integer tensor from one quantization scheme to another using
        new scale and zero-point parameters. For each element in the input tensor, computes:
        output[i] = round((input[i] - input_zero_point) * (input_scale / output_scale)) + output_zero_point

        .. code-block:: mlir

            // Requantize int8 tensor to new scale and zero-point
            %result = ttir.requantize(%input, %output) {scale = 0.2 : f32, zero_point = 100 : i32} : tensor<2x2xi8>, tensor<2x2xi8> -> tensor<2x2xi8>
            // Input tensor (scale=0.1, zero_point=128):
            // [[143, 126],
            //  [128, 165]]
            // Output tensor (scale=0.2, zero_point=100):
            // [[107, 98],
            //  [100, 119]]

        Parameters
        ----------
        in0 : Operand
            Input quantized integer tensor to be requantized
        scale : float
            New scale factor for requantization
        zero_point : int
            New integer value that represents 0.0 in the quantized space
        dtype : torch.dtype
            Target integer data type (e.g., torch.int8)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The requantized integer tensor with new scale and zero-point
        """
        # kwargs passed thru output type
        return self._op_proxy(
            ttir.RequantizeOp,
            [in0],
            golden_kwargs={"scale": scale, "zero_point": zero_point, "dtype": dtype},
            output_type=self._get_type_from_torch_dtype(
                dtype=dtype, scale=scale, zero_point=zero_point
            ),
            unit_attrs=unit_attrs,
        )

    def to_layout(
        self,
        in0: Operand,
        output_type: RankedTensorType,
        unit_attrs: Optional[List[str]] = None,
        **kwargs,
    ) -> OpView:
        """
        Creates ``ttir.to_layout``.

        *Layout operation.*

        ToLayout operation, transition tensors from one layout to another. Some examples include:
        - Transitioning between different memory spaces, e.g. DRAM to L1.
        - Transitioning between different data types, e.g. f32 to f16.
        - Transitioning between different tile sizes, e.g. 1x16 to 32x32
        - Transitioning between different tensor sharding
        - Some combination of the above

        .. code-block:: mlir

            #layout = #ttcore.metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #system>>
            #layout1 = #ttcore.metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #l1_>>
            %1 = "ttir.to_layout"(%arg0, %0) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>

        Parameters
        ----------
        in0 : Operand
            Input tensor to be transformed
        output_type : RankedTensorType
            Target type specifying the desired layout
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes
        **kwargs : dict
            Additional keyword arguments for layout transformation

        Returns
        -------
        (*OpView*)
            The tensor with transformed layout
        """
        return self._op_proxy(
            ttir.ToLayoutOp,
            [in0],
            output_type=output_type,
            output_create_fn=self._create_empty_from_tensor_type,
            organize_ttir_args=lambda i, o, _: (
                [self._get_type(o)],
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
            **kwargs,
        )

    def tilize(
        self,
        in0: Operand,
        output_type: RankedTensorType,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.tilize``.

        *Convert tensor to tiled layout.*

        Transforms a tensor into a tiled layout format, where data is organized into
        regular blocks or tiles. This can improve memory access patterns and cache
        utilization for certain operations.

        .. code-block:: mlir

            // Convert tensor to tiled layout
            %result = ttir.tilize(%input) : tensor<128x128xf32> -> tensor<128x128xf32, #tiled<32x32>>
            // Input tensor (standard layout):
            // [[1.5, 2.0, ...],
            //  [3.0, 4.0, ...]]
            // Output tensor (tiled 32x32 layout):
            // Same values but organized in 32x32 tiles

        Parameters
        ----------
        in0 : Operand
            Input tensor to be tiled
        output_type : RankedTensorType
            Target type specifying the desired tiled layout
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The tensor with tiled layout
        """
        return self._op_proxy(
            ttir.ToLayoutOp,
            [in0],
            output_type=output_type,
            output_create_fn=self._create_empty_from_tensor_type,
            organize_ttir_args=lambda i, o, _: (
                [self._get_type(o)],
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
            golden_kwargs={"tilize": True},
        )

    def untilize(
        self,
        in0: Operand,
        output_type: RankedTensorType,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.untilize``.

        *Convert tensor from tiled to standard layout.*

        Transforms a tensor from a tiled layout back to a standard row-major or
        column-major layout. This is the inverse operation of tilize.

        .. code-block:: mlir

            // Convert tensor from tiled to standard layout
            %result = ttir.untilize(%input) : tensor<128x128xf32, #tiled<32x32>> -> tensor<128x128xf32>
            // Input tensor (tiled 32x32 layout):
            // Data organized in 32x32 tiles
            // Output tensor (standard layout):
            // [[1.5, 2.0, ...],
            //  [3.0, 4.0, ...]]

        Parameters
        ----------
        in0 : Operand
            Input tensor with tiled layout
        output_type : RankedTensorType
            Target type specifying the desired standard layout
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The tensor with standard layout
        """
        return self._op_proxy(
            ttir.ToLayoutOp,
            [in0],
            output_type=output_type,
            output_create_fn=self._create_empty_from_tensor_type,
            organize_ttir_args=lambda i, o, _: (
                [self._get_type(o)],
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
            golden_kwargs={"tilize": False},
        )

    def gather(
        self,
        input: Operand,
        start_indices: Operand,
        offset_dims: List[int],
        collapsed_slice_dims: List[int],
        operand_batching_dims: List[int],
        start_indices_batching_dims: List[int],
        start_index_map: List[int],
        index_vector_dim: int,
        slice_sizes: List[int],
        indices_are_sorted: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.gather``.

        *Gather operation.*

        Collects slices from an input tensor at positions specified by start indices.
        This operation is based on the StableHLO Gather operation and allows for flexible
        slicing and indexing of tensors. It can be used to implement operations like array
        indexing, slicing, dynamic indexing, and more complex gathering patterns.

        .. code-block:: mlir

            // Basic gather example: gather elements from a 2D tensor using indices
            %input = ... : tensor<5x3xf32>         // Input tensor with shape [5,3]
            %indices = ... : tensor<2xi64>         // Indices tensor with values [2, 1]
            %output = ttir.empty() : tensor<3xf32> // Output tensor
            %result = ttir.gather(%input, %indices, %output) {
                offset_dims = [0],                 // Output dimensions that are gathered from input
                collapsed_slice_dims = [0],        // Input dimensions that are collapsed
                operand_batching_dims = [],        // Batch dimensions of the input
                start_indices_batching_dims = [],  // Batch dimensions of the indices
                start_index_map = [0],             // Maps indices to input dimensions
                index_vector_dim = 0,              // Which dimension of indices contains the index vector
                slice_sizes = [1, 3],              // Size of the slice to extract from each position
                indices_are_sorted = false         // Whether indices are sorted
            } : tensor<5x3xf32>, tensor<2xi64>, tensor<3xf32> -> tensor<3xf32>

        Parameters
        ----------
        input : Operand
            The tensor from which to gather values
        start_indices : Operand
            Tensor containing the starting indices for slices
        offset_dims : *List[int]*
            Output dimensions that correspond to dimensions of the gathered slice
        collapsed_slice_dims : *List[int]*
            Input dimensions that are collapsed when gathering
        operand_batching_dims : *List[int]*
            Batch dimensions of the input tensor
        start_indices_batching_dims : *List[int]*
            Batch dimensions of the indices tensor
        start_index_map : *List[int]*
            Maps index values to input dimensions
        index_vector_dim : int
            Which dimension of indices contains the index vector
        slice_sizes : *List[int]*
            Size of the slice to extract from each position
        indices_are_sorted : bool, optional
            Whether indices are sorted (for optimization)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The gathered tensor
        """
        # Create the attributes
        from ttmlir.ir import ArrayAttr, IntegerAttr, IntegerType, BoolAttr

        # Create DenseI64ArrayAttr attributes directly from lists
        offset_dims_attr = DenseI64ArrayAttr.get(offset_dims, self._ctx)
        collapsed_slice_dims_attr = DenseI64ArrayAttr.get(
            collapsed_slice_dims, self._ctx
        )
        operand_batching_dims_attr = DenseI64ArrayAttr.get(
            operand_batching_dims, self._ctx
        )
        start_indices_batching_dims_attr = DenseI64ArrayAttr.get(
            start_indices_batching_dims, self._ctx
        )
        start_index_map_attr = DenseI64ArrayAttr.get(start_index_map, self._ctx)
        slice_sizes_attr = DenseI64ArrayAttr.get(slice_sizes, self._ctx)

        # Create boolean attribute
        indices_are_sorted_attr = BoolAttr.get(indices_are_sorted, self._ctx)

        # Create integer attribute for index_vector_dim
        index_vector_dim_attr = IntegerAttr.get(
            IntegerType.get_signed(64), index_vector_dim
        )

        # Calculate output shape based on gather semantics
        input_shape = self.get_shape(input)
        indices_shape = self.get_shape(start_indices)

        # Batch dimensions: all dims from start_indices except index_vector_dim
        batch_dims = [
            indices_shape[i] for i in range(len(indices_shape)) if i != index_vector_dim
        ]
        offset_sizes = [
            slice_sizes[i]
            for i in range(len(slice_sizes))
            if i not in collapsed_slice_dims
        ]
        result_rank = len(batch_dims) + len(offset_sizes)
        assert len(offset_dims) == len(
            offset_sizes
        ), "offset_dims length must match offset_sizes length"
        output_shape = [-1] * result_rank
        for j, pos in enumerate(offset_dims):
            output_shape[pos] = offset_sizes[j]
        b = 0
        for p in range(result_rank):
            if output_shape[p] == -1:
                output_shape[p] = batch_dims[b]
                b += 1

        # Define kwargs for the TTIR operation
        ttir_kwargs = {
            "offset_dims": offset_dims_attr,
            "collapsed_slice_dims": collapsed_slice_dims_attr,
            "operand_batching_dims": operand_batching_dims_attr,
            "start_indices_batching_dims": start_indices_batching_dims_attr,
            "start_index_map": start_index_map_attr,
            "index_vector_dim": index_vector_dim_attr,
            "slice_sizes": slice_sizes_attr,
            "indices_are_sorted": indices_are_sorted_attr,
        }

        # Use op_proxy to create the operation
        return self._op_proxy(
            ttir.GatherOp,
            [input, start_indices],
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1], o),
            output_shape=output_shape,
            unit_attrs=unit_attrs,
            ttir_kwargs=ttir_kwargs,
        )

    def slice(
        self,
        in0: Operand,
        begins: List[int],
        ends: List[int],
        step: List[int] = None,
        unit_attrs: List[str] = None,
    ) -> OpView:
        # If step is not provided, use 1 for each dimension
        if step is None:
            step = [1] * len(begins)

        # Ensure begins, ends, and step have the same length
        if not (len(begins) == len(ends) == len(step)):
            raise ValueError("begins, ends, and step must have the same length")

        # Get the input shape
        input_shape = self.get_shape(in0)

        # Ensure we're not slicing more dimensions than exist
        if len(begins) > len(input_shape):
            raise ValueError("Cannot slice more dimensions than input has")

        # Calculate the output shape
        output_shape = []

        # Process dimensions that are being sliced
        for i, (b, e, s) in enumerate(zip(begins, ends, step)):
            # Handle negative indices
            dim_size = input_shape[i]
            if b < 0:
                b += dim_size
            if e < 0:
                e += dim_size

            # Clamp to valid range
            b = max(0, min(b, dim_size))
            e = max(0, min(e, dim_size))

            # Calculate dimension size using correct formula
            # For positive step: ceil((e - b) / s)
            if s > 0:
                if e > b:
                    size = (e - b + s - 1) // s
                else:
                    size = 0
            else:
                # Negative step not typically supported in MLIR/TOSA
                raise ValueError("Negative step not supported")

            output_shape.append(size)

        # Add remaining dimensions that aren't being sliced
        for i in range(len(begins), len(input_shape)):
            output_shape.append(input_shape[i])

        # Create the attributes
        from ttmlir.ir import ArrayAttr, IntegerAttr, IntegerType

        # Create integer attributes for each value
        begins_int_attrs = [
            IntegerAttr.get(IntegerType.get_signless(32), b) for b in begins
        ]
        ends_int_attrs = [
            IntegerAttr.get(IntegerType.get_signless(32), e) for e in ends
        ]
        step_int_attrs = [
            IntegerAttr.get(IntegerType.get_signless(32), s) for s in step
        ]

        # Create array attributes
        begins_attr = ArrayAttr.get(begins_int_attrs, self._ctx)
        ends_attr = ArrayAttr.get(ends_int_attrs, self._ctx)
        step_attr = ArrayAttr.get(step_int_attrs, self._ctx)

        # Use op_proxy
        return self._op_proxy(
            ttir.SliceStaticOp,
            [in0],
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
            output_shape=output_shape,
            unit_attrs=unit_attrs,
            ttir_kwargs={"begins": begins_attr, "ends": ends_attr, "step": step_attr},
        )

    # CCL ops

    def mesh_shard(
        self,
        input: Operand,
        shard_type: str,
        shard_direction: str,
        shard_shape: Tuple[int, ...],
        shard_dims: Tuple[int, ...],
    ) -> OpView:
        """
        Creates ``ttir.mesh_shard``.

        *Shard a tensor across a device mesh.*

        Distributes a tensor across multiple devices in a mesh according to the specified
        sharding configuration. The sharding can be performed along one or more dimensions
        of the tensor.

        .. code-block:: mlir

            // Shard a tensor across a 2x2 device mesh
            %result = ttir.mesh_shard(%input) {
                shard_type = "block",
                shard_direction = "row",
                shard_shape = [2, 2],
                shard_dims = [0, 1]
            } : tensor<128x128xf32> -> tensor<64x64xf32>
            // Input tensor on single device:
            // [[1.0, 2.0, ...],
            //  [3.0, 4.0, ...]]
            // Output tensor sharded across devices:
            // Device 0: [[1.0, 2.0], [3.0, 4.0]]
            // Device 1: [[1.1, 2.1], [3.1, 4.1]]
            // Device 2: [[1.2, 2.2], [3.2, 4.2]]
            // Device 3: [[1.3, 2.3], [3.3, 4.3]]

        Parameters
        ----------
        input : Operand
            Input tensor to be sharded
        shard_type : str
            Type of sharding (e.g., "block", "cyclic")
        shard_direction : str
            Direction of sharding (e.g., "row", "col")
        shard_shape : Tuple[int, ...]
            Shape of the device mesh
        shard_dims : Tuple[int, ...]
            Tensor dimensions to shard along

        Returns
        -------
        (*OpView*)
        """
        ttir_kwargs = {
            "shard_type": Attribute.parse(shard_type),
            "shard_direction": Attribute.parse(shard_direction),
            "shard_shape": shard_shape,
            "shard_dims": shard_dims,
        }
        golden_kwargs = dict(ttir_kwargs, mesh_shape=self.mesh_shape)
        return self._op_proxy(
            ttir.MeshShardOp,
            [input],
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0]),
            ttir_kwargs=ttir_kwargs,
            golden_kwargs=golden_kwargs,
        )

    def all_gather(
        self,
        input: Operand,
        all_gather_dim: int = None,
        cluster_axis: int = None,
    ) -> OpView:
        """
        Creates ``ttir.all_gather``.

        *Gather tensor data from all devices.*

        Collects tensor data from all devices in the system and concatenates them along
        the specified dimension. The gather operation can be performed along different
        axes of the device mesh.

        For a mesh shape of [2,4] with device IDs:
        [[0, 1, 2, 3],
        [4, 5, 6, 7]]

        - If cluster_axis=0: Gathers along columns (0,4), (1,5), (2,6), (3,7)
        - If cluster_axis=1: Gathers along rows (0,1,2,3), (4,5,6,7)

        .. code-block:: mlir

            // Gather tensor data from all devices along dimension 0
            %result = ttir.all_gather(%input) {all_gather_dim = 0, cluster_axis = 1} : tensor<32x64xf32> -> tensor<128x64xf32>
            // Input tensor on device 0:
            // [[1.0, 2.0],
            //  [3.0, 4.0]]
            // Output tensor after gathering:
            // [[1.0, 2.0],  // from device 0
            //  [5.0, 6.0],  // from device 1
            //  [9.0, 10.0], // from device 2
            //  [13.0, 14.0]] // from device 3

        Parameters
        ----------
        input : Operand
            Input tensor to be gathered
        all_gather_dim : int, optional
            Dimension along which to concatenate gathered tensors
        cluster_axis : int, optional
            Axis of device mesh for gathering (0 or 1)

        Returns
        -------
        (*OpView*)
        """
        kwargs = {"all_gather_dim": all_gather_dim, "cluster_axis": cluster_axis}
        return self._op_proxy(
            ttir.AllGatherOp,
            [input],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
        )

    def all_reduce(
        self,
        input: Operand,
        reduce_type: str,
        cluster_axis: int,
    ) -> OpView:
        """
        Creates ``ttir.all_reduce``.

        *AllReduce operation.*

        AllReduce op.

        Parameters
        ----------
        input : Operand
            Input tensor to be reduced
        reduce_type : str
            Type of reduction operation (e.g., "sum", "max")
        cluster_axis : int
            Axis of device mesh for reduction (0 or 1)

        Returns
        -------
        (*OpView*)
        """
        kwargs = {
            "reduce_type": Attribute.parse(reduce_type),
            "cluster_axis": cluster_axis,
        }
        return self._op_proxy(
            ttir.AllReduceOp,
            [input],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
        )

    def reduce_scatter(
        self,
        input: Operand,
        reduce_type: str,
        scatter_dim: int,
        cluster_axis: int,
    ) -> OpView:
        """
        Creates ``ttir.reduce_scatter``.

        *Reduce scatter operation.*

        Reduce scatter op.

        Parameters
        ----------
        input : Operand
            Input tensor to be reduced and scattered
        reduce_type : str
            Type of reduction operation (e.g., "sum", "max")
        scatter_dim : int
            Dimension along which to scatter the reduced results
        cluster_axis : int
            Axis of device mesh for reduction (0 or 1)

        Returns
        -------
        (*OpView*)
        """
        kwargs = {
            "reduce_type": Attribute.parse(reduce_type),
            "scatter_dim": scatter_dim,
            "cluster_axis": cluster_axis,
        }
        return self._op_proxy(
            ttir.ReduceScatterOp,
            [input],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
        )

    def collective_permute(
        self,
        input: Operand,
        source_target_pairs: List[Tuple[int, int]],
    ) -> OpView:
        """
        Creates ``ttir.collective_permute``.

        *Collective permute operation.*

        Collective permute op. This operation ingests a multi-device tensor spread across multi-devices and will shuffle the data according to source_target_pairs [['src', 'dest']].

        Example:
            For a 1x2 mesh, the following will take the device shard living in device 0 and move it to device 1. The device shard living in device 1 will move to device 0.
        %source_target_pairs: [[0, 1], [1, 0]]

        In the case of missing 'dest', the device shard living on that device will contain values of 0. For example, device shard living in device 0 will contain 0 values.
        %source_target_pairs: [[0, 1]]

        Parameters
        ----------
        input : Operand
            The input tensor to be permuted
        source_target_pairs : *List[Tuple[int, int]]*
            List of pairs of source and target device ids

        Returns
        -------
        (*OpView*)
        """
        kwargs = {
            "source_target_pairs": source_target_pairs,
        }
        return self._op_proxy(
            ttir.CollectivePermuteOp,
            [input],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
        )

    def all_to_all(
        self,
        input: Operand,
        split_dim: int,
        concat_dim: int,
        split_count: int,
        replica_groups: List[List[int]],
    ) -> OpView:
        """
        Creates ``ttir.all_to_all``.

        *all to all operation.*

        The all_to_all operation redistributes slices of a tensor across a cluster of devices. It splits each local tensor along split_dimension, sends the resulting slices to other devices along cluster_axis, and then concatenates the received slices along concat_dimension.

        Example:
            For a 1x2 mesh and a local input of shape [8, 4]:
            - split_dimension = 1
            - concat_dimension = 0
            - split_count = 2
            - cluster_axis = 1

            Each device splits its [8, 4] tensor into two [8, 2] slices. After the exchange, each device concatenates the two received [8, 2] slices into a [16, 2] output tensor.

        Parameters
        ----------
        input : Operand
            Input tensor to be redistributed
        split_dim : int
            Dimension along which to split the input tensor
        concat_dim : int
            Dimension along which to concatenate the reorganized tensors
        split_count : int
            Number of splits to perform
        replica_groups : List[List[int]]
            List of replica group indices

        Returns
        -------
        (*OpView*)
        """
        kwargs = {
            "split_dim": split_dim,
            "concat_dim": concat_dim,
            "split_count": split_count,
            "replica_groups": replica_groups,
        }
        return self._op_proxy(
            ttir.AllToAllOp,
            [input],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
        )

    def collective_broadcast(
        self,
        input: Operand,
        replica_groups: List[Tuple[int, int]],
    ) -> OpView:
        """
        Creates ``ttir.collective_broadcast``.
        *Collective broadcast operation.*
        The collective_broadcast operation distributes a tensor from a single source device to all
        other devices within each replica group. Each replica group defines a subset of devices that
        participate in the broadcast, and the operation is applied independently within each group.
        By convention, the first device listed in each replica group is treated as the broadcast source.
        The value of the `input` tensor on that source device is sent to all other devices in the same
        group. The `input` tensor values on non-source devices are ignored and will be overwritten
        during the operation.
        Parameters
        ----------
        input: The tensor to broadcast. Only the value on the first device of each replica group
              (the source) is used; values on other devices are ignored.
        replica_groups: A list of replica groups. Each group is a list of device IDs, and the first
                        ID in each group is treated as the broadcast source for that group.
        Returns
        -------
        (*OpView*)
        """
        kwargs = {
            "replica_groups": replica_groups,
        }
        return self._op_proxy(
            ttir.CollectiveBroadcastOp,
            [input],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
        )

    def rms_norm(
        self,
        in0: Operand,
        normalized_shape: List[int],
        weight: Optional[Operand] = None,
        bias: Optional[Operand] = None,
        epsilon: float = 1e-5,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.rms_norm``.

        *RMS normalization operation.*

        Performs RMS (Root Mean Square) normalization on the input tensor. This operation
        normalizes the input tensor by computing the root mean square of elements across
        the specified dimensions and dividing by that value, optionally scaling and
        shifting the result.

        Mathematical definition: rms_norm(x, weight, bias, epsilon) =
          (x / sqrt(mean(x^2, dims=normalized_dims) + epsilon)) * weight + bias

        Parameters
        ----------
        in0 : Operand
            Input tensor to be normalized
        normalized_shape : List[int]
            Shape over which to normalize (typically the last few dimensions)
        weight : Optional[Operand], optional
            Scale parameter (gamma) tensor with shape matching normalized_shape
        bias : Optional[Operand], optional
            Shift parameter (beta) tensor with shape matching normalized_shape
        epsilon : float, optional
            Small constant for numerical stability (default: 1e-5)
        unit_attrs : Optional[List[str]], optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
        """
        # Prepare TTIR kwargs:
        ttir_kwargs = {
            "normalized_shape": normalized_shape,
            "epsilon": epsilon,
        }

        golden_kwargs = {
            "normalized_shape": normalized_shape,
            "epsilon": epsilon,
        }

        if weight is not None:
            ttir_kwargs["weight"] = weight
            golden_kwargs["weight"] = self._get_golden_tensor(weight)
        if bias is not None:
            ttir_kwargs["bias"] = bias
            golden_kwargs["bias"] = self._get_golden_tensor(bias)

        return self._op_proxy(
            ttir.RMSNormOp,
            [in0],
            golden_kwargs=golden_kwargs,
            ttir_kwargs=ttir_kwargs,
            organize_ttir_args=lambda i, o, _: (
                self._get_type(o),
                i[0],
                o,
            ),
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            unit_attrs=unit_attrs,
        )
