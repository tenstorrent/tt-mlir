# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Callable, Dict, Any
from ttmlir.ir import *
from ttmlir.dialects import ttir, ttcore, tensor, quant
from ttmlir.passes import GoldenTensor, DataType
import torch
from enum import Enum, auto
import re
from .ccl_golden import *

# Alias for operands of ops which can be either BlockArguments, Values, or other
# ops wrapped in OpView or Operation.
Operand = Union[Value, OpView, Operation]

# Convenience alias for shape
Shape = Union[List[int], Tuple[int, ...]]


def autodoc_skip(func):
    func.__autodoc_skip__ = True
    return func


class TTIRBuilderOps:
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
        golden_data = [self._get_golden_tensor(in0).size(dimension)]
        return self.op_proxy(
            torch.tensor,
            ttir.GetDimensionSizeOp,
            [in0],
            golden_kwargs={"data": golden_data, "dtype": torch.int32},
            ttir_kwargs={"dimension": dimension},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0]),
            organize_golden_args=lambda i: 0,
            output_type=self.get_type_from_torch_dtype(torch.int32),
            unit_attrs=unit_attrs,
        )

    def dot_general(
        self,
        in0: Operand,
        in1: Operand,
        out0: Operand,
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
        out0 : Operand
            Output tensor
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
        kwargs = {
            "batch_dims_lhs": batch_dims_lhs,
            "contract_dims_lhs": contract_dims_lhs,
            "batch_dims_rhs": batch_dims_rhs,
            "contract_dims_rhs": contract_dims_rhs,
        }
        return self.op_proxy(
            self.dot_general_golden_function,
            ttir.DotGeneralOp,
            [in0, in1, out0],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1]),
            output_type=self.get_type_from_torch_dtype(
                self._get_golden_tensor(in0).dtype
            ),
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def dot_general_golden_function(
        self,
        lhs,
        rhs,
        out,
        batch_dims_lhs,
        contract_dims_lhs,
        batch_dims_rhs,
        contract_dims_rhs,
    ):
        non_batch_dims_lhs = [d for d in range(lhs.dim()) if d not in batch_dims_lhs]
        non_batch_dims_rhs = [d for d in range(rhs.dim()) if d not in batch_dims_rhs]
        transposed_lhs = torch.permute(lhs, (batch_dims_lhs + non_batch_dims_lhs))
        transposed_rhs = torch.permute(rhs, (batch_dims_rhs + non_batch_dims_rhs))
        result_batching_dims = list(range(len(batch_dims_lhs)))
        result = torch.empty(*out.shape, dtype=lhs.dtype)

        dim_ranges = []
        for i in range(len(result_batching_dims)):
            dim_ranges.append([j for j in range(list(lhs.shape)[i])])
        import itertools

        batch_indices = list(itertools.product(*dim_ranges))
        for index in batch_indices:
            transposed_lhs_slice = transposed_lhs[index]
            transposed_rhs_slice = transposed_rhs[index]
            dot_dims_lhs = [d - len(index) for d in contract_dims_lhs]
            dot_dims_rhs = [d - len(index) for d in contract_dims_rhs]
            out_index = index
            result[out_index] = torch.tensordot(
                transposed_lhs_slice,
                transposed_rhs_slice,
                dims=(dot_dims_lhs, dot_dims_rhs),
            )
        return result

    # TTIR top level named ops
    # class TTIR_ElementwiseTernaryOp

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
        condition = torch.full(in0_tensor.shape, False)
        condition[in0_tensor > 0] = True
        return self.op_proxy(
            torch.where,
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
        return self.eltwise_proxy(torch.abs, ttir.AbsOp, [in0], unit_attrs)

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
        golden = self._get_golden_tensor(in0)
        golden_sign = torch.sign(golden)
        golden_cbrt = torch.pow(torch.abs(golden), 1 / 3)
        return self.op_proxy(
            torch.mul,
            ttir.CbrtOp,
            [in0],
            golden_kwargs={"input": golden_sign, "other": golden_cbrt},
            organize_golden_args=lambda i: 0,
            unit_attrs=unit_attrs,
        )

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
        return self.eltwise_proxy(torch.ceil, ttir.CeilOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(torch.cos, ttir.CosOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(torch.floor, ttir.FloorOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(
            torch.nn.functional.gelu, ttir.GeluOp, [in0], unit_attrs
        )

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
        return self.eltwise_proxy(torch.isfinite, ttir.IsFiniteOp, [in0], unit_attrs)

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
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.logical_not,
            ttir.LogicalNotOp,
            [in0],
            golden_kwargs={"out": golden_output},
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
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
        return self.eltwise_proxy(
            torch.bitwise_not, ttir.BitwiseNotOp, [in0], unit_attrs
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
        return self.eltwise_proxy(torch.neg, ttir.NegOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(torch.tan, ttir.TanOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(torch.atan, ttir.AtanOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(torch.tanh, ttir.TanhOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(
            torch.reciprocal, ttir.ReciprocalOp, [in0], unit_attrs
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
        return self.eltwise_proxy(torch.relu, ttir.ReluOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(torch.rsqrt, ttir.RsqrtOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(torch.sigmoid, ttir.SigmoidOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(torch.sign, ttir.SignOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(torch.sin, ttir.SinOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(torch.sqrt, ttir.SqrtOp, [in0], unit_attrs)

    def typecast(
        self, in0: Operand, out: Operand, unit_attrs: Optional[List[str]] = None
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
        out : Operand
            Output tensor with desired type
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the input values cast to the output type
        """
        output_type = self.get_type_from_torch_dtype(self._get_golden_tensor(out).dtype)
        return self.op_proxy(
            torch.Tensor.type,
            ttir.TypecastOp,
            [in0],
            golden_kwargs={"dtype": self._get_golden_tensor(out).type()},
            output_type=output_type,
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
        return self.eltwise_proxy(torch.log, ttir.LogOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(torch.log1p, ttir.Log1pOp, [in0], unit_attrs)

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
        return self.eltwise_proxy(torch.expm1, ttir.Expm1Op, [in0], unit_attrs)

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
        # TODO: reconcile this naming mismatch
        ttir_kwargs = {"parameter": parameter}
        golden_kwargs = {"negative_slope": parameter}
        return self.op_proxy(
            torch.nn.functional.leaky_relu,
            ttir.LeakyReluOp,
            [in0],
            golden_kwargs=golden_kwargs,
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
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.eq,
            ttir.EqualOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
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
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.ne,
            ttir.NotEqualOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
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
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.ge,
            ttir.GreaterEqualOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
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
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.gt,
            ttir.GreaterThanOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
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
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.le,
            ttir.LessEqualOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
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
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.lt,
            ttir.LessThanOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
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
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.logical_and,
            ttir.LogicalAndOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
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
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.logical_or,
            ttir.LogicalOrOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
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
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.logical_xor,
            ttir.LogicalXorOp,
            [in0, in1],
            golden_kwargs={"out": golden_output},
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
        return self.eltwise_proxy(
            torch.bitwise_and, ttir.BitwiseAndOp, [in0, in1], unit_attrs=unit_attrs
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
        return self.eltwise_proxy(
            torch.bitwise_or, ttir.BitwiseOrOp, [in0, in1], unit_attrs=unit_attrs
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
        return self.eltwise_proxy(
            torch.bitwise_xor, ttir.BitwiseXorOp, [in0, in1], unit_attrs=unit_attrs
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
        return self.eltwise_proxy(
            torch.minimum, ttir.MinimumOp, [in0, in1], unit_attrs=unit_attrs
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
        return self.eltwise_proxy(
            torch.subtract, ttir.SubtractOp, [in0, in1], unit_attrs=unit_attrs
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
        return self.eltwise_proxy(
            torch.remainder, ttir.RemainderOp, [in0, in1], unit_attrs=unit_attrs
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
        return self.eltwise_proxy(
            torch.pow, ttir.PowOp, [in0, in1], unit_attrs=unit_attrs
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
        return self.op_proxy(
            self.argmax_golden_function,
            ttir.ArgMaxOp,
            [in0],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            output_type=IntegerType.get_signless(32, self._ctx),
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def argmax_golden_function(
        self, in0: Operand, dim_arg: List[int], keep_dim: bool = False
    ) -> OpView:
        in1 = torch.argmax(in0, dim=dim_arg[0], keepdim=keep_dim)
        return in1.to(torch.int32)

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
        return self.op_proxy(
            torch.sum,
            ttir.SumOp,
            [in0],
            golden_kwargs={"dim": dim_arg, "keepdim": keep_dim},
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
        return self.op_proxy(
            torch.mean,
            ttir.MeanOp,
            [in0],
            golden_kwargs={"dim": dim_arg, "keepdim": keep_dim},
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
        golden_kwargs = {}
        ttir_kwargs = {"keep_dim": True}
        input_shape = list(self.get_shape(in0))
        ndim = len(input_shape)
        if dim_arg is not None:
            golden_kwargs = {"dim": dim_arg, "keepdim": True}
            ttir_kwargs["dim_arg"] = [dim_arg]

            output_shape = input_shape.copy()
            output_shape[dim_arg] = 1
            golden_fn = lambda x, *args, **kwargs: torch.max(
                x, dim=kwargs["dim"], keepdim=kwargs["keepdim"]
            )
        else:
            output_shape = [1] * ndim
            golden_fn = lambda x, *args, **kwargs: torch.max(x).reshape(*output_shape)

        return self.op_proxy(
            golden_fn,
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
        golden_kwargs = {}
        ttir_kwargs = {"keep_dim": keep_dim}
        output_shape = [1] * len(self.get_shape(in0))
        if dim_arg:
            golden_kwargs = {"dim": dim_arg, "keepdim": keep_dim}
            ttir_kwargs["dim_arg"] = [dim_arg]
        if not keep_dim:
            output_shape = torch.Size([1])

        return self.op_proxy(
            torch.min,
            ttir.MinOp,
            [in0],
            golden_kwargs=golden_kwargs,
            ttir_kwargs=ttir_kwargs,
            output_shape=output_shape,
            unit_attrs=unit_attrs,
        )

    # NOTE: Not useable. Boolean tensors are not supported by the runtime. Issue #1775
    @autodoc_skip
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
        return self.op_proxy(
            torch.all,
            ttir.ReduceAndOp,
            [in0],
            golden_kwargs={"dim": tuple(dim_args), "keepdim": keep_dim},
            ttir_kwargs={"dim_arg": dim_args, "keep_dim": keep_dim},
            unit_attrs=unit_attrs,
        )

    # NOTE: Not useable. Boolean tensors are not supported by the runtime. Issue #1775
    @autodoc_skip
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
        return self.op_proxy(
            torch.any,
            ttir.ReduceOrOp,
            [in0],
            golden_kwargs={"dim": tuple(dim_args)},
            ttir_kwargs={"dim_arg": dim_args, "keep_dim": keep_dim},
            unit_attrs=unit_attrs,
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
        golden_kwargs = {}
        if len(dim_arg) == 1:
            golden_kwargs["dim"] = dim_arg[0]
            golden_kwargs["keepdim"] = keep_dim
            golden_function = torch.prod
        else:
            golden_function = lambda i: torch.tensor([torch.prod(i[0]).item()])
        return self.op_proxy(
            golden_function,
            ttir.ProdOp,
            [in0],
            golden_kwargs=golden_kwargs,
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
        embedding = torch.nn.Embedding.from_pretrained(self._get_golden_tensor(in1))
        golden_typecast = self._get_golden_tensor(in0).to(torch.int32)
        golden_input = torch.clamp(
            golden_typecast, 0, (self._get_golden_tensor(in1).size()[0] - 1)
        )
        return self.op_proxy(
            embedding,
            ttir.EmbeddingOp,
            [in0, in1],
            organize_golden_args=lambda i: (golden_input,),
            unit_attrs=unit_attrs,
        )

    def cumsum(
        self,
        in0: Operand,
        in1: Operand,
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
        in1 : Operand
            Output tensor
        dim : int
            Dimension along which to compute cumulative sum
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the cumulative sums along the specified dimension
        """
        return self.op_proxy(
            torch.cumsum,
            ttir.CumSumOp,
            [in0, in1],
            golden_kwargs={"dim": dim},
            ttir_kwargs={"dim": dim, "output": in1},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0]),
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            unit_attrs=unit_attrs,
        )

    def softmax(
        self, in0: Operand, dimension: int = 1, unit_attrs: Optional[List[str]] = None
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
        dim : int, optional
            Dimension along which Softmax will be computed (default: -1)
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Output tensor after softmax
        """
        return self.op_proxy(
            torch.nn.functional.softmax,
            ttir.SoftmaxOp,
            [in0],
            golden_kwargs={"dim": dimension},
            organize_ttir_args=lambda i, o, _: (
                self._get_type(o),
                i[0],
                o,
                dimension,
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
        return self.op_proxy(
            torch.transpose,
            ttir.TransposeOp,
            [in0],
            golden_kwargs=kwargs,
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
        return self.op_proxy(
            torch.concat,
            ttir.ConcatOp,
            ins,
            golden_kwargs=kwargs,
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
        return self.op_proxy(
            torch.Tensor.repeat,
            ttir.RepeatOp,
            [in0],
            golden_kwargs={"repeats": dims},
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
        return self.op_proxy(
            torch.repeat_interleave,
            ttir.RepeatInterleaveOp,
            [in0, in1],
            golden_kwargs={"repeats": repeats, "dim": dim},
            ttir_kwargs={"repeats": repeats, "dim": dim},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1]),
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            output_type=self.get_type_from_torch_dtype(
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
        cache_tensor = self._get_golden_tensor(in0)
        input_tensor = self._get_golden_tensor(in1)
        cache_tensor[:, :, : input_tensor.shape[2], :] = input_tensor
        return self.op_proxy(
            torch.clone,
            ttir.FillCacheOp,
            [in0, in1],
            golden_kwargs={"input": cache_tensor},
            ttir_kwargs={"batch_offset": batch_offset},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1]),
            organize_golden_args=lambda i: 0,
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
        cache = self._get_golden_tensor(in0)
        input_tensor = self._get_golden_tensor(in1)
        index = torch.clamp(self._get_golden_tensor(in2), 0, cache.size()[2])
        a = cache[:, :, : index[0], :]
        b = cache[:, :, : (cache.size()[2] - index[0] - 1), :]
        return self.op_proxy(
            torch.cat,
            ttir.UpdateCacheOp,
            [in0, in1, in2],
            golden_kwargs={"tensors": (a, input_tensor, b), "dim": 2},
            ttir_kwargs={"batch_offset": batch_offset},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1], i[2]),
            organize_golden_args=lambda i: 0,
            unit_attrs=unit_attrs,
        )

    def broadcast(
        self,
        in0: Operand,
        in1: Operand,
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
        in1 : Operand
            Output tensor with target shape
        broadcast_dimensions : *List[int]*
            List of dimension mappings from input to output
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            The broadcasted tensor
        """
        return self.op_proxy(
            torch.broadcast_to,
            ttir.BroadcastOp,
            [in0],
            golden_kwargs={"size": self.get_shape(in1)},
            ttir_kwargs={"broadcast_dimensions": broadcast_dimensions},
            unit_attrs=unit_attrs,
        )

    def conv2d(
        self,
        in0: Operand,
        weight: Operand,
        bias: Optional[Operand],
        in1: Operand,
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
        output : Operand
            Output tensor specification
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
        return self.op_proxy(
            self.conv2d_golden_function,
            ttir.Conv2dOp,
            [in0, weight, bias],
            golden_kwargs={
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
            },
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
            },
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1], o),
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def conv2d_golden_function(
        self,
        input_tensor: Operand,
        weight: Operand,
        bias: Optional[Operand],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int,
    ) -> Operand:
        # Reorganize ttir_kwargs into golden_kwargs
        stride = list(stride) if not isinstance(stride, int) else int(stride)
        padding = list(padding) if not isinstance(padding, int) else int(padding)
        dilation = list(dilation) if not isinstance(dilation, int) else int(dilation)

        # ttir can handle a broadcastable bias in the shape [1, 1, 1, C_out], but PyTorch requires the bias is rank 1: [C_out]
        bias = bias.squeeze()  # Removes all dims of size 1

        # Reorganize input and output tensors, golden and ttir functions have different expected tensor shapes
        input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
        result = torch.nn.functional.conv2d(
            input_tensor,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        result = result.transpose(-3, -2).transpose(-2, -1)
        return result

    def conv_transpose2d(
        self,
        in0: Operand,
        weight: Operand,
        bias: Optional[Operand],
        in1: Operand,
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
        in1 : Operand
            Output tensor shape reference
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
        return self.op_proxy(
            self.conv_transpose2d_golden_function,
            ttir.ConvTranspose2dOp,
            [in0, weight],
            golden_kwargs={
                "stride": stride,
                "padding": padding,
                "output_padding": output_padding,
                "dilation": dilation,
                "groups": groups,
            },
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

    @autodoc_skip
    def conv_transpose2d_golden_function(
        self,
        input_tensor: Operand,
        weight: Operand,
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        output_padding: Union[int, List[int]],
        dilation: Union[int, List[int]],
        groups: int,
    ) -> Operand:
        # Reorganize ttir_kwargs into golden_kwargs
        stride = list(stride) if not isinstance(stride, int) else int(stride)
        padding = list(padding) if not isinstance(padding, int) else int(padding)
        output_padding = (
            list(output_padding)
            if not isinstance(output_padding, int)
            else int(output_padding)
        )
        dilation = list(dilation) if not isinstance(dilation, int) else int(dilation)
        golden_bias = torch.rand((weight.size()[0]), dtype=input_tensor.dtype)

        # Reorganize input and output tensors, golden and ttir functions have different expected tensor shapes
        input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
        result = torch.nn.functional.conv_transpose2d(
            input_tensor,
            weight,
            bias=golden_bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )
        result = result.transpose(-3, -2).transpose(-2, -1)
        return result

    def max_pool2d(
        self,
        in0: Operand,
        in1: Operand,
        kernel_height: int,
        kernel_width: int,
        stride_height: int,
        stride_width: int,
        dilation_height: int,
        dilation_width: int,
        ceil_mode: bool,
        padding_left: int,
        padding_right: int,
        padding_top: int,
        padding_bottom: int,
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
        return self.op_proxy(
            self.max_pool2d_golden_function,
            ttir.MaxPool2dOp,
            [in0],
            golden_kwargs={
                "kernel_size": (kernel_height, kernel_width),
                "stride": (stride_height, stride_width),
                "padding": (padding_top, padding_left),
                "dilation": (dilation_height, dilation_width),
                "ceil_mode": ceil_mode,
            },
            ttir_kwargs={
                "kernel_height": kernel_height,
                "kernel_width": kernel_width,
                "stride_height": stride_height,
                "stride_width": stride_width,
                "dilation_height": dilation_height,
                "dilation_width": dilation_width,
                "ceil_mode": ceil_mode,
                "padding_left": padding_left,
                "padding_right": padding_right,
                "padding_top": padding_top,
                "padding_bottom": padding_bottom,
            },
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def tilize_golden(self, input: torch.Tensor) -> torch.Tensor:
        shape = input.shape
        TILE_SIZE = 32
        FACE_SIZE = 16
        Y_TILES = shape[0] // TILE_SIZE
        X_TILES = shape[1] // TILE_SIZE
        FACES_PER_TILE = TILE_SIZE // FACE_SIZE

        tilized = torch.zeros((input.numel(),))

        idx = 0
        for tile_y in range(Y_TILES):
            for tile_x in range(X_TILES):
                for face_y in range(FACES_PER_TILE):
                    for face_x in range(FACES_PER_TILE):
                        for datum_y in range(FACE_SIZE):
                            for datum_x in range(FACE_SIZE):
                                tilized[idx] = input[
                                    datum_y + tile_y * TILE_SIZE + face_y * FACE_SIZE,
                                    datum_x + tile_x * TILE_SIZE + face_x * FACE_SIZE,
                                ]
                                idx += 1

        tilized = tilized.reshape(shape)
        return tilized

    @autodoc_skip
    def untilize_golden(self, input: torch.Tensor) -> torch.Tensor:
        shape = input.shape
        TILE_SIZE = 32
        FACE_SIZE = 16
        Y_TILES = shape[0] // TILE_SIZE
        X_TILES = shape[1] // TILE_SIZE
        FACES_PER_TILE = TILE_SIZE // FACE_SIZE

        untilized = torch.zeros_like(input)
        flattened = input.flatten()

        idx = 0
        for tile_y in range(Y_TILES):
            for tile_x in range(X_TILES):
                for face_y in range(FACES_PER_TILE):
                    for face_x in range(FACES_PER_TILE):
                        for datum_y in range(FACE_SIZE):
                            for datum_x in range(FACE_SIZE):
                                # Calculate the original position
                                orig_y = (
                                    datum_y + tile_y * TILE_SIZE + face_y * FACE_SIZE
                                )
                                orig_x = (
                                    datum_x + tile_x * TILE_SIZE + face_x * FACE_SIZE
                                )

                                # Place the value from the tilized tensor back to its original position
                                untilized[orig_y, orig_x] = flattened[idx]
                                idx += 1

        return untilized

    @autodoc_skip
    def max_pool2d_golden_function(
        self,
        input_tensor: Operand,
        kernel_size: tuple[int],
        stride: tuple[int],
        padding: tuple[int],
        dilation: tuple[int],
        ceil_mode: bool,
    ):
        # TTIR  max_pool2d is channels last. PyTorch max_pool2d is channels first.
        # We need to transpose the input tensor to channels first before applying max_pool2d,
        # and transpose back to channels last afterward to properly calculate the golden tensor.
        # TTIR  max_pool2d is channels last. PyTorch max_pool2d is channels first.
        # We need to transpose the input tensor to channels first before applying max_pool2d,
        # and transpose back to channels last afterward to properly calculate the golden tensor.
        maxpool_object = torch.nn.MaxPool2d(
            kernel_size, stride, padding, dilation, ceil_mode
        )
        input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
        result = maxpool_object(input_tensor)
        result = result.transpose(-3, -2).transpose(-2, -1)
        return result

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
        return self.op_proxy(
            torch.reshape,
            ttir.ReshapeOp,
            [in0],
            ttir_kwargs=kwargs,
            golden_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def pad(
        self,
        in0: Operand,
        in1: Operand,
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
        # Reformatting padding dimensions for golden tensor:
        golden_padding = []
        for i in range(len(padding) // 2):
            golden_padding.append(padding[-((2 * i) + 2)])
            golden_padding.append(padding[-((2 * i) + 1)])
        return self.op_proxy(
            torch.nn.functional.pad,
            ttir.PadOp,
            [in0, in1],
            golden_kwargs={"pad": golden_padding, "mode": "constant", "value": value},
            ttir_kwargs={"padding": padding, "value": value},
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1]),
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
        end = begin + length - 1
        index = torch.tensor([begin, end])
        return self.op_proxy(
            torch.index_select,
            ttir.IndexSelectOp,
            [in0],
            golden_kwargs={"dim": dim, "index": index},
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
        import math

        num_indices = math.ceil((end - begin) / step)
        indices = []
        for i in range(num_indices):
            indices.append((begin + i) * step)
        index = torch.tensor(indices)
        return self.op_proxy(
            torch.index_select,
            ttir.IndexOp,
            [in0],
            golden_kwargs={"dim": dim, "index": index},
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
        return self.op_proxy(
            torch.squeeze,
            ttir.SqueezeOp,
            [in0],
            golden_kwargs=kwargs,
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
        return self.op_proxy(
            torch.unsqueeze,
            ttir.UnsqueezeOp,
            [in0],
            golden_kwargs=kwargs,
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
        return self.op_proxy(
            torch.clamp,
            ttir.ClampScalarOp,
            [in0],
            ttir_kwargs=kwargs,
            golden_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def clamp_tensor(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        in3: Operand,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        return self.op_proxy(
            torch.clamp,
            ttir.ClampTensorOp,
            [in0, in1, in2, in3],
            golden_kwargs={
                "input": self._get_golden_tensor(in0),
                "min": self._get_golden_tensor(in1),
                "max": self._get_golden_tensor(in2),
                "out": self._get_golden_tensor(in3),
            },
            organize_ttir_args=lambda i, o, _: (
                self._get_type(o),
                i[0],
                i[1],
                i[2],
                i[3],
            ),
            organize_golden_args=lambda i: 0,
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
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Tensor of zeros with specified shape
        """
        output = self.ranked_tensor_type(shape)
        dtype = data_type if data_type is not None else self._default_dtype
        return self.op_proxy(
            torch.zeros,
            ttir.ZerosOp,
            [],
            golden_kwargs={"size": shape},
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
        output = self.ranked_tensor_type(shape)
        return self.op_proxy(
            torch.ones,
            ttir.OnesOp,
            [],
            golden_kwargs={"size": shape},
            ttir_kwargs={"result": output, "shape": shape},
            organize_ttir_args=lambda i, o, shape: 0,
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
        return self.op_proxy(
            torch.flip,
            ttir.ReverseOp,
            [in0],
            golden_kwargs={"dims": dims},
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
        kwargs = {"transpose_a": transpose_a, "transpose_b": transpose_b, "bias": bias}
        return self.op_proxy(
            self.linear_golden_function,
            ttir.LinearOp,
            [in0, in1],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def linear_golden_function(
        self,
        a: Operand,
        b: Operand,
        bias: Optional[Operand] = None,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ) -> OpView:
        a = torch.transpose(a, 0, 1) if transpose_a else a
        b = torch.transpose(b, 0, 1) if transpose_a else b
        output = torch.matmul(a, b)
        bias = (
            torch.zeros(list(output.shape))
            if not bias
            else self._get_golden_tensor(bias)
        )
        bias = (
            torch.broadcast_to(bias, list(output.shape))
            if bias.shape != output.shape
            else bias
        )
        return torch.add(output, bias)

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
        return self.op_proxy(
            torch.matmul,
            ttir.MatmulOp,
            inputs,
            unit_attrs=unit_attrs,
        )

    def permute(
        self,
        in0: Operand,
        in1: Operand,
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
        return self.op_proxy(
            torch.permute,
            ttir.PermuteOp,
            [in0, in1],
            golden_kwargs={"dims": tuple(permutation)},
            ttir_kwargs={"permutation": DenseI64ArrayAttr.get(permutation)},
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            organize_ttir_args=lambda i, o, _: (self._get_type(i[1]), i[0], i[1]),
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
        return self.op_proxy(
            self.upsample2d_golden_function,
            ttir.Upsample2dOp,
            [in0, in1],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(i[1]), i[0], o),
            output_shape=output_shape,
            unit_attrs=unit_attrs,
        )

    @autodoc_skip
    def upsample2d_golden_function(
        self,
        in0: Operand,
        in1: Operand,
        scale_factor: Union[SI32Attr, DenseI32ArrayAttr],
        mode: str = "nearest",
    ) -> OpView:
        transposed_golden = torch.transpose(in0, 1, 3)
        golden_output_shape = in1.shape[1:-1]
        output = torch.nn.functional.interpolate(
            transposed_golden, size=golden_output_shape, mode=mode
        )
        return torch.transpose(output, 1, 3)

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
        single_dim_tensor = torch.arange(
            start=start, end=end, step=step, dtype=self._get_golden_tensor(result).dtype
        )
        shape = self.get_shape(result)
        repeat_dims = []
        for i in range(len(shape)):
            if i == arange_dimension:
                repeat_dims.append(int(shape[i] / ((end - start) / step)))
            else:
                repeat_dims.append(shape[i])

        return self.op_proxy(
            torch.Tensor.repeat,
            ttir.ArangeOp,
            [result, single_dim_tensor],
            golden_kwargs={"repeats": tuple(repeat_dims)},
            ttir_kwargs={
                "start": start,
                "end": end,
                "step": step,
                "arange_dimension": arange_dimension,
            },
            organize_ttir_args=lambda i, o, _: (self._get_type(o),),
            organize_golden_args=lambda i: [i[1]],
            output_shape=shape,
            output_type=self.get_type_from_torch_dtype(
                self._get_golden_tensor(result).dtype
            ),
            unit_attrs=unit_attrs,
        )

    # TTIR top level generic ops
    # class TTIR_GenericElementwiseUnaryOp

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
        return self.eltwise_proxy(torch.exp, ttir.ExpOp, [in0], unit_attrs=unit_attrs)

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
        return self.eltwise_proxy(
            torch.add,
            ttir.AddOp,
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
        return self.eltwise_proxy(
            torch.multiply,
            ttir.MultiplyOp,
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
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise difference of the inputs
        """
        return self.eltwise_proxy(
            torch.sub,
            ttir.SubtractOp,
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
        return self.eltwise_proxy(
            torch.div, ttir.DivOp, [in0, in1], unit_attrs=unit_attrs
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
        return self.eltwise_proxy(
            torch.maximum, ttir.MaximumOp, [in0, in1], unit_attrs=unit_attrs
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
        golden_kwargs = {"scale": scale, "zero_point": zero_point, "dtype": dtype}
        return self.op_proxy(
            lambda *args, **kwargs: torch.quantize_per_tensor(
                *args, **kwargs
            ).int_repr(),
            ttir.QuantizeOp,
            [in0],
            golden_kwargs=golden_kwargs,
            output_type=self.get_type_from_torch_dtype(
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
        return self.op_proxy(
            torch.dequantize,
            ttir.DequantizeOp,
            [in0],
            output_type=self.get_type_from_torch_dtype(dtype=dtype),
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
        golden_kwargs = {"scale": scale, "zero_point": zero_point, "dtype": dtype}
        return self.op_proxy(
            lambda *args, **kwargs: torch.quantize_per_tensor(
                torch.dequantize(args[0]), **kwargs
            ),
            ttir.RequantizeOp,
            [in0],
            golden_kwargs=golden_kwargs,
            output_type=self.get_type_from_torch_dtype(
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
        return self.op_proxy(
            lambda *args, **kwargs: args[0],
            ttir.ToLayoutOp,
            [in0],
            output_type=output_type,
            output_create_fn=self.empty_from_tensor_type,
            organize_ttir_args=lambda i, o, _: (
                [self._get_type(o)],
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
            **kwargs,
        )

    def view_layout(
        self,
        in0: Operand,
        output_type: RankedTensorType,
        reinterpret_layout: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttir.view_layout``.

        *Create a new view of tensor with different layout.*

        Creates a new view of the input tensor with a different layout without copying
        or moving data. This is useful for reinterpreting the same data with different
        layout metadata.

        - If reinterpretLayout is true, the layout view change can include a data type cast, but note this does not actually change the format of the data in memory.
        - All ViewLayout ops can trivially be converted to ToLayout ops.

        .. code-block:: mlir

            #layout = #ttcore.metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #system>>
            #layout1 = #ttcore.metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #l1_>>
            %1 = "ttir.view_layout"(%arg0, %0) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>

        Parameters
        ----------
        in0 : Operand
            Input tensor to create new view from
        output_type : RankedTensorType
            Type of output tensor with desired layout
        reinterpret_layout : bool, optional
            If true, allows data type cast in layout view change (default: False)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A new view of the tensor with the specified layout
        """
        return self.op_proxy(
            lambda *args, **kwargs: args[0],
            ttir.ViewLayoutOp,
            [in0],
            ttir_kwargs={"reinterpretLayout": reinterpret_layout},
            output_type=output_type,
            output_create_fn=self.empty_from_tensor_type,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0]),
            unit_attrs=unit_attrs,
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
        return self.op_proxy(
            self.tilize_golden,
            ttir.ToLayoutOp,
            [in0],
            output_type=output_type,
            output_create_fn=self.empty_from_tensor_type,
            organize_ttir_args=lambda i, o, _: (
                [self._get_type(o)],
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
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
        return self.op_proxy(
            self.untilize_golden,
            ttir.ToLayoutOp,
            [in0],
            output_type=output_type,
            output_create_fn=self.empty_from_tensor_type,
            organize_ttir_args=lambda i, o, _: (
                [self._get_type(o)],
                i[0],
                o,
            ),
            unit_attrs=unit_attrs,
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
        kwargs = {
            "shard_type": Attribute.parse(shard_type),
            "shard_direction": Attribute.parse(shard_direction),
            "shard_shape": shard_shape,
            "shard_dims": shard_dims,
        }
        return self.ccl_proxy(
            mesh_shard_golden,
            ttir.MeshShardOp,
            [input],
            kwargs=kwargs,
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
        return self.ccl_proxy(
            all_gather_golden,
            ttir.AllGatherOp,
            [input],
            kwargs=kwargs,
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
        return self.ccl_proxy(
            all_reduce_golden,
            ttir.AllReduceOp,
            [input],
            kwargs=kwargs,
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
        return self.ccl_proxy(
            reduce_scatter_golden,
            ttir.ReduceScatterOp,
            [input],
            kwargs=kwargs,
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
        return self.ccl_proxy(
            collective_permute_golden,
            ttir.CollectivePermuteOp,
            [input],
            kwargs=kwargs,
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
        batch_dims = []
        for i in range(len(indices_shape)):
            if i != index_vector_dim:
                batch_dims.append(indices_shape[i])

        # Offset dimensions: dimensions from slice_sizes that aren't collapsed
        offset_sizes = []
        for i in range(len(slice_sizes)):
            if i not in collapsed_slice_dims:
                offset_sizes.append(slice_sizes[i])

        output_shape = batch_dims + offset_sizes

        # Create a closure that captures all the gather parameters
        def gather_golden_fn(input_tensor, start_indices_tensor):
            import torch
            import numpy as np

            input_np = input_tensor.cpu().numpy()
            indices_np = start_indices_tensor.cpu().numpy()
            device = input_tensor.device

            # For the case where collapsed_slice_dims == start_index_map
            # and we're collapsing dimensions with size > 1, we need to
            # interpret this as indexing into a flattened space

            # Check if this is the embedding-like pattern
            if (
                set(collapsed_slice_dims) == set(start_index_map)
                and len(collapsed_slice_dims) > 0
            ):

                # We're collapsing all indexed dimensions
                # This means we're selecting a single element from the indexed dimensions
                # and keeping full slices from non-indexed dimensions

                output = np.zeros(output_shape, dtype=input_np.dtype)

                # Get batch positions
                batch_shape = [
                    s for i, s in enumerate(indices_shape) if i != index_vector_dim
                ]
                if not batch_shape:
                    batch_positions = [()]
                else:
                    batch_positions = list(np.ndindex(*batch_shape))

                for batch_idx, batch_pos in enumerate(batch_positions):
                    # Extract index vector
                    if batch_pos:
                        full_idx = list(batch_pos)
                        if index_vector_dim < len(indices_shape):
                            full_idx.insert(index_vector_dim, slice(None))
                        index_vec = indices_np[tuple(full_idx)]
                    else:
                        index_vec = indices_np

                    if np.isscalar(index_vec):
                        index_vec = [index_vec]

                    # Build the exact index for collapsed dimensions
                    indices_list = [slice(None)] * len(input_shape)

                    # For each indexed dimension, use the index value directly
                    # (not as a slice start)
                    for i, input_dim in enumerate(start_index_map):
                        if i < len(index_vec):
                            indices_list[input_dim] = int(index_vec[i])
                        else:
                            indices_list[input_dim] = 0

                    # For non-indexed dimensions, take the full slice
                    for dim in range(len(input_shape)):
                        if dim not in start_index_map:
                            indices_list[dim] = slice(0, slice_sizes[dim])

                    # Extract the result
                    result = input_np[tuple(indices_list)]

                    # Place in output
                    if batch_pos:
                        output[batch_pos] = result
                    else:
                        output = result

                return torch.tensor(output, device=device)
            else:
                # General gather case (not used in your tests)
                raise NotImplementedError("General gather not implemented")

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
        return self.op_proxy(
            gather_golden_fn,
            ttir.GatherOp,
            [input, start_indices],
            golden_kwargs={},
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
        assert (
            len(begins) == len(ends) == len(step)
        ), "begins, ends, and step must have the same length"

        # Get the input shape
        input_shape = self.get_shape(in0)

        # Ensure we're not slicing more dimensions than exist
        assert len(begins) <= len(
            input_shape
        ), "Cannot slice more dimensions than input has"

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

        # Golden function for slicing
        def slice_golden_fn(x):
            # Build slice objects for each dimension
            slices = []

            # Add slices for dimensions being sliced
            for i, (b, e, s) in enumerate(zip(begins, ends, step)):
                # Handle negative indices
                dim_size = x.shape[i]
                if b < 0:
                    b += dim_size
                if e < 0:
                    e += dim_size

                # Clamp to valid range
                b = max(0, min(b, dim_size))
                e = max(0, min(e, dim_size))

                # Create slice object
                slices.append(slice(b, e, s))

            # Add full slices for remaining dimensions
            for i in range(len(begins), len(x.shape)):
                slices.append(slice(None))

            # Apply the slice
            return x[tuple(slices)]

        # Use op_proxy
        return self.op_proxy(
            slice_golden_fn,
            ttir.SliceOp,
            [in0],
            golden_kwargs={},  # No kwargs needed - closure captures begins/ends/step
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
            output_shape=output_shape,
            unit_attrs=unit_attrs,
            ttir_kwargs={"begins": begins_attr, "ends": ends_attr, "step": step_attr},
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
        return self.ccl_proxy(
            all_to_all_golden,
            ttir.AllToAllOp,
            [input],
            kwargs=kwargs,
        )


# Remove autodoc_skip from Sphinx documentation
del autodoc_skip
