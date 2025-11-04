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
from golden import *


class TTNNBuilder(Builder):
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

    def _get_output_shape_and_type(
        self,
        organize_golden_args: Callable,
        inputs: List[Operand],
        op_ttnn_function: Callable,
        golden_kwargs: dict = {},
    ):
        op_golden_function = get_golden_function(op_ttnn_function, **golden_kwargs)
        if op_golden_function is None:
            assert len(inputs) > 0, (
                f"Cannot infer output shape for {op_ttnn_function.__name__}: "
                "no golden function available"
            )
            return (
                self._get_type(inputs[0]).shape,
                self._get_type(inputs[0]).element_type,
            )

        # If the op has no input, just call golden function with kwargs (eg ttnn.zeros).
        if len(inputs) == 0:
            golden_output = op_golden_function(**golden_kwargs)
        else:
            golden_output = op_golden_function(
                *(organize_golden_args(inputs)), **golden_kwargs
            )

        return golden_output.shape, self._get_type_from_torch_dtype(golden_output.dtype)

    def _organize_eltwise_ttnn(
        self,
        inputs: List[Operand],
        ttnn_tensor: RankedTensorType,
    ):
        return (ttnn_tensor, *inputs)

    def _op_proxy(
        self,
        op_ttnn_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
        organize_ttnn_args: Optional[Callable] = None,
        organize_golden_args: Optional[Callable] = None,
        output_shape: Optional[Shape] = None,
        output_type: Optional[Type] = None,
        output_create_fn: Optional[Callable] = None,
        golden_kwargs: dict = {},
        ttnn_kwargs: dict = {},
        loc: Optional[Union[str, Location]] = None,
        skip_golden: bool = False,
    ) -> Any:
        if organize_ttnn_args is None:
            organize_ttnn_args = self._organize_eltwise_ttnn

        if organize_golden_args is None:
            organize_golden_args = self._organize_eltwise_golden

        with self._ctx, self._loc:
            # If output shape or type is not provided, calculate it using golden function.
            # This is needed because ttnn ops do not have shape or type MLIR inference trait.

            output_shape_and_type = self._get_output_shape_and_type(
                organize_golden_args, inputs, op_ttnn_function, golden_kwargs
<<<<<<< HEAD
=======
            )

            output_type = self.create_ttnn_tensor(
                shape=output_shape_and_type[0],
                element_type=output_shape_and_type[1],
>>>>>>> c5bfbb6e0 (Add remaining ttnn builder tests)
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
            result_tensor = self.create_ttnn_tensor(output_shape, output_type)

            # Prepare location for the op.
            id = self._get_next_global_id()
            loc = (
                self._get_loc_from_str(loc)
                if loc is not None
                else self._get_loc_of_extra_file_callee(id=id)
            )

            # Organize arguments and create the ttnn op.
            if organize_ttnn_args(inputs, result_tensor) == 0:
                op = op_ttnn_function(
                    loc=loc,
                    **ttnn_kwargs,
                )
            else:
                op = op_ttnn_function(
                    *organize_ttnn_args(inputs, result_tensor),
                    loc=loc,
                    **ttnn_kwargs,
                )

            # Set unit attributes if provided.
            if unit_attrs is not None:
                for attr_name in unit_attrs:
                    op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

            if not skip_golden and not self._disable_golden_check:
                op_golden_function = get_golden_function(
                    op_ttnn_function, **golden_kwargs
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

    def _get_data_type_attribute(self, operand: Operand) -> ttcore.ir.DataTypeAttr:
        with self._ctx, self._loc:
            dtype = ttnn.ir.TTNNLayoutAttr.maybe_downcast(
                self._get_type(operand).encoding
            ).data_type_as_int
            return ttcore.ir.DataTypeAttr.get(self._ctx, dtype)

    # ----- Public Helper Methods ----

    def create_tensor_encoding(
        self, shape: Shape, element_type: Union[torch.dtype, TypeInfo]
    ) -> ttnn.ir.TTNNLayoutAttr:
        """
        TTNN tensors require that encoding information is present.
        This method creates a TTNN tensor with encoding information.
        For simplicity we will always create DRAM/Interlaved tiled tensor.
        """
        if isinstance(element_type, torch.dtype):
            element_type = self._get_type_from_torch_dtype(element_type)
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
            return ttnn_layout_attr

    def create_ttnn_tensor(self, shape: Shape, element_type: Type) -> RankedTensorType:
        """
        TTNN tensors require that encoding information is present.
        This method creates a TTNN tensor with encoding information.
        For simplicity we will always create DRAM/Interlaved tiled tensor.
        """
        with self._ctx, self._loc:
            ttnn_layout_attr = self.create_tensor_encoding(shape, element_type)
            return RankedTensorType.get(shape, element_type, ttnn_layout_attr)

    # ----- Public TTNN Op Generators ----

<<<<<<< HEAD
    def mish(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.mish``.

        *Elementwise Mish activation operation.*

        Applies the Mish activation function element-wise to the input tensor.
        Mish is a smooth, self-regularized, non-monotonic activation function defined as:
        f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))

        Mathematical definition: mish(x) = x * tanh(ln(1 + e^x))

        .. code-block:: mlir

            // Apply Mish activation
            %result = ttnn.mish(%input, %output) : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
            // Input tensor:
            // input: [0.0, 1.0, -1.0]
            // Output tensor:
            // [0.0, 0.865, -0.303]

        Parameters
        ----------
        in0 : Operand
            Input tensor
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise Mish activation of the input
        """
        return self._op_proxy(ttnn.MishOp, [in0], unit_attrs=unit_attrs, ttnn_kwargs={})

    def where(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttnn.where``.

        *Elementwise conditional selection operation.*

        For each element position, selects between two values based on a boolean condition:
        - If the condition is true (non-zero), selects from the first value tensor
        - If the condition is false (zero), selects from the second value tensor

        Supports broadcasting according to standard broadcasting rules.

        .. code-block:: mlir

            // Basic selection between two tensors
            %result = ttnn.where(%cond, %true_vals, %false_vals) :
                tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
            // Input tensors:
            // %cond: [[1, 0], [0, 1]]
            // %true_vals: [[1.0, 2.0], [3.0, 4.0]]
            // %false_vals: [[5.0, 6.0], [7.0, 8.0]]
            // Output tensor:
            // [[1.0, 6.0], [7.0, 4.0]]

            // With broadcasting (scalar condition)
            %result = ttnn.where(%scalar_cond, %true_vals, %false_vals) :
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
            ttnn.WhereOp,
            [in0, in1, in2],
            organize_golden_args=lambda i, o: (
                condition,
                self._get_golden_tensor(i[1]),
                self._get_golden_tensor(i[2]),
            ),
            unit_attrs=unit_attrs,
        )

    # class TTNN_ElementwiseUnaryOp

    def abs(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.abs``.

        *Elementwise absolute value operation.*

        Computes the absolute value of each element in the input tensor.

        .. code-block:: mlir

            // Compute absolute values of all elements in %input
            %result = ttnn.abs(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
            // Input tensor:
            // [[-2.5,  3.7,  0.0,  1.2], ... ]
            // Output tensor:
            // [[2.5, 3.7, 0.0, 1.2], ... ]

            // Example with integer tensor
            %result = ttnn.abs(%int_input, %int_output) : tensor<10xi32>, tensor<10xi32> -> tensor<10xi32>
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

        return self._op_proxy(ttnn.AbsOp, [in0], unit_attrs)

    def cbrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.cbrt``.

        *Elementwise cubic root operation.*

        Computes the cubic root (∛) of each element in the input tensor.
        For each element, returns the real-valued number that, when cubed, equals the input value.
        Unlike square root, cubic root is defined for negative numbers as well as positive numbers.

        .. code-block:: mlir

            // Compute cubic root of all elements
            %result = ttnn.cbrt(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
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
        return self._op_proxy(ttnn.CbrtOp, [in0], unit_attrs)

    def ceil(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.ceil``.

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
        return self._op_proxy(ttnn.CeilOp, [in0], unit_attrs)

    def cos(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.cos``.

        *Elementwise cosine operation.*

        Computes the cosine of each element in the input tensor.
        Input values are expected to be in radians.

        .. code-block:: mlir

            // Compute cosine of all elements
            %result = ttnn.cos(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
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
        return self._op_proxy(ttnn.CosOp, [in0], unit_attrs)

    def erf(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.erf``.

        *Elementwise error function operation.*

        Computes the error function (erf) of each element in the input tensor.
        The error function is a mathematical function used in probability, statistics,
        and partial differential equations related to the normal distribution.

        Mathematical definition: erf(x) = (2/sqrt(π)) * ∫[0 to x] e^(-t^2) dt

        .. code-block:: mlir

            // Compute error function of all elements
            %result = ttnn.erf(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [0.0, 0.5, 1.0, -1.0]
            // Output tensor:
            // [0.0, 0.5205, 0.8427, -0.8427]
        """
        return self._op_proxy(ttnn.ErfOp, [in0], unit_attrs)

    def erfc(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.erfc``.

        *Elementwise complementary error function operation.*

        Computes the complementary error function (erfc) of each element in the input tensor.
        The complementary error function is defined as erfc(x) = 1 - erf(x),
        where erf(x) is the error function. It is commonly used in statistics and probability.

        Mathematical definition: erfc(x) = 1 - (2/sqrt(π)) * ∫[0 to x] e^(-t^2) dt

        .. code-block:: mlir

            // Compute complementary error function of all elements
            %result = ttnn.erfc(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
            // Input tensor:
            // [0.0, 0.5, 1.0, -1.0]
            // Output tensor:
            // [1.0, 0.4795, 0.1573, 1.8427]
        """
        return self._op_proxy(ttnn.ErfcOp, [in0], unit_attrs)

    def exp(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.exp``.

        *Elementwise exponential operation.*

        Computes the exponential of each element in the input tensor.
        For each element x, returns e^x, where e is Euler's number (approximately 2.71828).

        .. code-block:: mlir

            // Compute exponential of all elements
            %result = ttnn.exp(%input, %output) : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
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
            ttnn.ExpOp,
            [in0],
            unit_attrs=unit_attrs,
        )

    def floor(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.floor``.

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
            ttnn.FloorOp,
            [in0],
            unit_attrs,
        )

    def gelu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.gelu``.

        *Elementwise GELU operation.*

        Computes the GELU (Gaussian Error Linear Unit) of each element in the input tensor.
        GELU is a smooth, non-monotonic activation function that approximates the cumulative
        distribution function of a standard normal distribution.

        Mathematical definition: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

        .. code-block:: mlir

            // Compute GELU of all elements
            %result = ttnn.gelu(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
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
        return self._op_proxy(ttnn.GeluOp, [in0], unit_attrs)

    def isfinite(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.isfinite``.

        *Elementwise finite check operation.*

        Checks if each element in the input tensor is finite (neither infinite nor NaN).
        For each element, returns a boolean value indicating whether the element is finite.

        Mathematical definition: isfinite(x) = x ∈ ℝ

        .. code-block:: mlir

            // Check if elements are finite
            %result = ttnn.isfinite(%input, %output) : tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
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
            ttnn.IsFiniteOp,
            [in0],
            output_type=self._get_type_from_torch_dtype(
                self._get_golden_tensor(in0).dtype
            ),
            unit_attrs=unit_attrs,
        )

    def logical_not(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.logical_not``.

        *Elementwise logical NOT operation.*

        Computes the logical NOT of each element in the input tensor.
        For each element x, returns True if x is False, and False if x is True.

        .. code-block:: mlir

            // Compute logical NOT of all elements
            %result = ttnn.logical_not(%input, %output) : tensor<3xi1>, tensor<3xi1> -> tensor<3xi1>
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
            ttnn.LogicalNotOp,
            [in0],
            unit_attrs=unit_attrs,
        )

    def bitwise_not(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.bitwise_not``.

        *Elementwise bitwise NOT operation.*

        Computes the bitwise NOT (one's complement) of each element in the input tensor.
        For each element, flips all the bits in the binary representation of the value.

        This operation is typically used with integer data types and has the involution property,
        meaning that applying it twice returns the original value: bitwise_not(bitwise_not(x)) = x.

        .. code-block:: mlir

            // Bitwise NOT with integer tensors
            %result = ttnn.bitwise_not(%input, %output) : tensor<2x2xi32>, tensor<2x2xi32> -> tensor<2x2xi32>
            // Input tensor:
            // [[1, 2],
            //  [3, 4]]
            // Output tensor:
            // [[-2, -3],
            //  [-4, -5]]

            // Example with 8-bit integers
            %result = ttnn.bitwise_not(%input, %output) : tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
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
            ttnn.BitwiseNotOp,
            [in0],
            unit_attrs,
        )

    def neg(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.neg``.

        *Elementwise negate operation.*

        Computes the negation of each element in the input tensor.
        For each element, returns the negation of the value.

        Mathematical definition: neg(x) = -x

        .. code-block:: mlir

            // Compute negation of all elements
            %result = ttnn.neg(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
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
        return self._op_proxy(ttnn.NegOp, [in0], unit_attrs)

    def tan(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.tan``.

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
        return self._op_proxy(ttnn.TanOp, [in0], unit_attrs)

    def atan(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.atan``.

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
        return self._op_proxy(ttnn.AtanOp, [in0], unit_attrs)

    def tanh(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.tanh``.

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
        return self._op_proxy(ttnn.TanhOp, [in0], unit_attrs)

    def reciprocal(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.reciprocal``.

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
            ttnn.ReciprocalOp,
            [in0],
            unit_attrs,
        )

    def relu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.relu``.

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
        return self._op_proxy(ttnn.ReluOp, [in0], unit_attrs)

    def relu6(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.relu6``.

        *Elementwise ReLU6 activation operation.*

        Computes the ReLU6 function for each element in the input tensor.
        ReLU6 is defined as: min(max(0, x), 6)
        This activation function clips values between 0 and 6, making it useful
        for quantized neural networks and mobile applications.

        .. code-block:: mlir

            // Compute ReLU6 of all elements
            %result = ttnn.relu6(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
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
        return self._op_proxy(ttnn.Relu6Op, [in0], unit_attrs)

    def silu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.silu``.

        *Elementwise SiLU (Swish) activation operation.*

        Computes the SiLU (Sigmoid Linear Unit) activation function for each element in the input tensor.
        SiLU is also known as Swish activation and is defined as: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

        This activation function is smooth, non-monotonic, and has been shown to work well
        in deep neural networks, particularly in transformer architectures.

        .. code-block:: mlir

            // Compute SiLU activation of all elements
            %result = ttnn.silu(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
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
        return self._op_proxy(ttnn.SiluOp, [in0], unit_attrs)

    def relu6(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.relu6``.

        *Elementwise ReLU6 activation operation.*

        Computes the ReLU6 function for each element in the input tensor.
        ReLU6 is defined as: min(max(0, x), 6)
        This activation function clips values between 0 and 6, making it useful
        for quantized neural networks and mobile applications.

        .. code-block:: mlir

            // Compute ReLU6 of all elements
            %result = ttnn.relu6(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
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
        return self._op_proxy(ttnn.Relu6Op, [in0], unit_attrs)

    def rsqrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.rsqrt``.

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
            ttnn.RsqrtOp,
            [in0],
            unit_attrs,
        )

    def sigmoid(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.sigmoid``.

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
            ttnn.SigmoidOp,
            [in0],
            unit_attrs,
        )

    def sign(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.sign``.

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
        return self._op_proxy(ttnn.SignOp, [in0], unit_attrs)

    def silu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.silu``.

        *Elementwise SiLU (Swish) activation operation.*

        Computes the SiLU (Sigmoid Linear Unit) activation function for each element in the input tensor.
        SiLU is also known as Swish activation and is defined as: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

        This activation function is smooth, non-monotonic, and has been shown to work well
        in deep neural networks, particularly in transformer architectures.

        .. code-block:: mlir

            // Compute SiLU activation of all elements
            %result = ttnn.silu(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
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
        return self._op_proxy(ttnn.SiluOp, [in0], unit_attrs)

    def sin(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.sin``.

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
        return self._op_proxy(ttnn.SinOp, [in0], unit_attrs)

    def sqrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.sqrt``.

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
        return self._op_proxy(ttnn.SqrtOp, [in0], unit_attrs)

    def typecast(
        self,
        in0: Operand,
        output_type: torch.dtype,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttnn.typecast``.

        *Elementwise type casting operation.*

        Casts each element in the input tensor to the type of the output tensor.
        The output type can be any supported tensor element type.

        .. code-block:: mlir

            // Cast float32 to int32
            %result = ttnn.typecast(%input, %output) : tensor<2x2xf32>, tensor<2x2xi32> -> tensor<2x2xi32>
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
            ttnn.TypecastOp,
            [in0],
            golden_kwargs={"dtype": output_type},
            output_type=self._get_type_from_torch_dtype(output_type),
            unit_attrs=unit_attrs,
        )

    def log(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.log``.

        *Elementwise natural logarithm operation.*

        Computes the natural logarithm of each element in the input tensor.
        For each element x, returns ln(x), where ln is the natural logarithm.

        .. code-block:: mlir

            // Compute natural logarithm of all elements
            %result = ttnn.log(%input, %output) : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
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
        return self._op_proxy(ttnn.LogOp, [in0], unit_attrs)

    def log1p(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """Elementwise natural logarithm of one plus input operation.

        The `log1p` operation computes the natural logarithm of one plus each element in the
        input tensor. For each element x, it returns ln(1 + x). This operation is more
        accurate than computing log(1 + x) directly for x values close to zero, and it is
        defined for x > -1. For values less than or equal to -1, the behavior depends on
        the implementation (may return NaN or negative infinity).

        .. code-block:: mlir

            // Compute log1p of all elements
            %result = ttnn.log1p(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
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
            ttnn.Log1pOp,
            [in0],
            unit_attrs,
        )

    def expm1(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        """
        Creates ``ttnn.expm1``.

        *Elementwise exponential minus one operation.*

        Computes e^x - 1 for each element in the input tensor, where e is Euler's number.
        This operation provides better numerical precision than computing exp(x) - 1 directly,
        especially for small values of x.

        .. code-block:: mlir

            // Compute exp(x) - 1 for all elements
            %result = ttnn.expm1(%input, %output) : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
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
            ttnn.Expm1Op,
            [in0],
            unit_attrs,
        )

    # class TTNN_ElementwiseUnaryWithFloatParameterOp

    def leaky_relu(
        self,
        in0: Operand,
        parameter: float = 0.01,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttnn.leaky_relu``.

        *Elementwise leaky ReLU activation operation.*

        Computes a leaky version of the Rectified Linear Unit (ReLU) activation function.
        For each element x in the input tensor:
        - If x > 0: returns x
        - If x ≤ 0: returns parameter * x

        The parameter controls the slope for negative values, allowing a small gradient
        when the unit is not active.

        .. code-block:: mlir

            // Compute leaky ReLU with slope 0.01 for negative values
            %result = ttnn.leaky_relu(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
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
        ttnn_kwargs = {"parameter": parameter}
        return self._op_proxy(
            ttnn.LeakyReluOp,
            [in0],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    # class TTNN_ElementwiseBinaryOp

    def eq(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.eq``.

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
            %result = ttnn.eq(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xi1> -> tensor<3xi1>
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
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.EqualOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def ne(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.ne``.

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
            %result = ttnn.ne(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
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
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.NotEqualOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def ge(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.ge``.

        *Elementwise greater than or equal to comparison operation.*

        Performs elementwise greater than or equal to comparison between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if the left element is greater than or equal to the right element
        - 0 (false) if the left element is less than the right element

        Mathematical definition: greater_equal(x, y) = x >= y

        .. code-block:: mlir

            // Compare elements for greater than or equal to
            %result = ttnn.ge(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
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
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.GreaterEqualOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def gt(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.gt``.

        *Elementwise greater than comparison operation.*

        Performs elementwise greater than comparison between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if the left element is greater than the right element
        - 0 (false) if the left element is less than or equal to the right element

        Mathematical definition: greater(x, y) = x > y

        .. code-block:: mlir

            // Compare elements for greater than
            %result = ttnn.gt(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
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
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.GreaterThanOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def le(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.le``.

        *Elementwise less than or equal to comparison operation.*

        Performs elementwise less than or equal to comparison between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if the left element is less than or equal to the right element
        - 0 (false) if the left element is greater than the right element

        Mathematical definition: less_equal(x, y) = x <= y

        .. code-block:: mlir

            // Compare elements for less than or equal to
            %result = ttnn.le(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xi1> -> tensor<4xi1>
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
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.LessEqualOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def lt(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.lt``.

        *Elementwise less than comparison operation.*

        The `lt` operation performs an elementwise less than comparison between two tensors.
        For each pair of corresponding elements, it returns:
        - 1 (true) if the left element is less than the right element
        - 0 (false) if the left element is greater than or equal to the right element

        Mathematical definition: less(x, y) = x < y

        .. code-block:: mlir

            // Compare elements for less than
            %result = ttnn.lt(%lhs, %rhs, %output) : tensor<4xf32>, tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
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
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.LessThanOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def logical_and(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.logical_and``.

        *Elementwise logical AND operation.*

        Performs elementwise logical AND operation between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if both elements are 1 (true)
        - 0 (false) if at least one element is 0 (false)

        This operation is idempotent, meaning logical_and(x, x) = x.

        .. code-block:: mlir

            // Logical AND operation
            %result = ttnn.logical_and(%lhs, %rhs, %output) : tensor<4xi1>, tensor<4xi1>, tensor<4xi1> -> tensor<4xi1>
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
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.LogicalAndOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def logical_left_shift(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.logical_shift_left``.

        *Elementwise logical shift left operation.*

        Performs elementwise logical shift left operation between two tensors.
        For each pair of corresponding elements, shifts the bits of the first element to the left
        by the number of positions specified by the second element.

        .. code-block:: mlir

            // Logical shift left operation
            %result = ttnn.logical_shift_left(%lhs, %rhs, %output) : tensor<3xi8>, tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
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
            ttnn.LogicalLeftShiftOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def logical_or(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.logical_or``.

        *Elementwise logical OR operation.*

        Performs elementwise logical OR operation between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if at least one element is 1 (true)
        - 0 (false) if both elements are 0 (false)

        This operation is idempotent, meaning logical_or(x, x) = x.

        Mathematical definition: logical_or(x, y) = x || y

        .. code-block:: mlir

            // Logical OR operation
            %result = ttnn.logical_or(%lhs, %rhs, %output) : tensor<4xi1>, tensor<4xi1>, tensor<4xi1> -> tensor<4xi1>
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
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.LogicalOrOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def logical_right_shift(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.logical_right_shift``.

        *Elementwise logical right shift operation.*

        Performs elementwise logical right shift operation between two tensors.
        For each pair of corresponding elements, shifts the bits of the first element to the right
        by the number of positions specified by the second element.

        .. code-block:: mlir

            // Logical right shift operation
            %result = ttnn.logical_right_shift(%lhs, %rhs, %output) : tensor<3xi8>, tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
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
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.LogicalRightShiftOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def logical_xor(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.logical_xor``.

        *Elementwise logical XOR operation.*

        Performs elementwise logical XOR (exclusive OR) operation between two tensors.
        For each pair of corresponding elements, returns:
        - 1 (true) if exactly one element is 1 (true)
        - 0 (false) if both elements are the same (both 0 or both 1)

        Mathematical definition: logical_xor(x, y) = (x || y) && !(x && y)

        .. code-block:: mlir

            // Logical XOR operation
            %result = ttnn.logical_xor(%lhs, %rhs, %output) : tensor<4xi1>, tensor<4xi1>, tensor<4xi1> -> tensor<4xi1>
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
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.LogicalXorOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def bitwise_and(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.bitwise_and``.

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
            %result = ttnn.bitwise_and(%lhs, %rhs, %output) : tensor<3xi8>, tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
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
            ttnn.BitwiseAndOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def bitwise_or(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.bitwise_or``.

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
            %result = ttnn.bitwise_or(%lhs, %rhs, %output) : tensor<3xi8>, tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
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
            ttnn.BitwiseOrOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def bitwise_xor(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.bitwise_xor``.

        *Elementwise bitwise XOR operation.*

        Performs elementwise bitwise XOR (exclusive OR) operation between two tensors.
        For each pair of corresponding elements, performs a bitwise XOR on their binary representations.

        .. code-block:: mlir

            // Bitwise XOR with integer tensors
            %result = ttnn.bitwise_xor(%input1, %input2, %output) : tensor<2x2xi32>, tensor<2x2xi32> -> tensor<2x2xi32>
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
            ttnn.BitwiseXorOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def minimum(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.minimum``.

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
            ttnn.MinimumOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def maximum(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.maximum``.

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
            ttnn.MaximumOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def subtract(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.subtract``.

        *Elementwise subtraction operation.*

        Performs elementwise subtraction between two tensors.
        For each pair of corresponding elements, subtracts the element in the second
        tensor from the element in the first tensor.

        Mathematical definition: subtract(x, y) = x - y

        .. code-block:: mlir

            // Subtract corresponding elements
            %result = ttnn.subtract(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
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
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.SubtractOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def remainder(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.remainder``.

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
            ttnn.RemainderOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def pow_tensor(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.pow_tensor``.

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
            ttnn.PowTensorOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    # class TTNN_GenericElementwiseBinaryOp

    def add(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
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
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            A tensor containing the elementwise sum of the inputs
        """
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.AddOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def atan2(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.atan2``.

        *Elementwise arctangent operation.*

        Computes the elementwise arctangent of the quotient of its arguments.
        For each pair of corresponding elements (y, x), returns atan2(y, x).

        Mathematical definition: atan2(y, x) = arctan(y / x)

        .. code-block:: mlir

            // Compute arctangent of corresponding elements
            %result = ttnn.atan2(%y, %x, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
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
            ttnn.Atan2Op,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

=======
>>>>>>> c5bfbb6e0 (Add remaining ttnn builder tests)
    def multiply(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.multiply``.

        *Elementwise multiplication operation.*

        Performs elementwise multiplication between two tensors.
        For each pair of corresponding elements, multiplies the element in the first
        tensor by the element in the second tensor.

        Mathematical definition: multiply(x, y) = x * y

        .. code-block:: mlir

            // Multiply corresponding elements
            %result = ttnn.multiply(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
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
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.MultiplyOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def divide(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.divide``.

        *Elementwise division operation.*

        Performs elementwise division between two tensors.
        For each pair of corresponding elements, divides the element in the first
        tensor by the element in the second tensor.

        Note: Division by zero behavior depends on the implementation and data type.

        Mathematical definition: divide(x, y) = x / y

        .. code-block:: mlir

            // Divide corresponding elements
            %result = ttnn.divide(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
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
        ttnn_kwargs = {
            "dtype": self._get_data_type_attribute(in0),
        }
        return self._op_proxy(
            ttnn.DivideOp,
            [in0, in1],
            ttnn_kwargs=ttnn_kwargs,
            unit_attrs=unit_attrs,
        )

    def matmul(
        self,
        in0: Operand,
        in1: Operand,
        transpose_a: bool = False,
        transpose_b: bool = False,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttnn.matmul``.

        *Matrix multiplication operation.*

        Performs matrix multiplication between two tensors. Supports optional
        transposition of either input tensor before multiplication. For 2D tensors,
        this computes the standard matrix product. For tensors with more dimensions,
        it applies batched matrix multiplication.

        Mathematical definition: matmul(A, B) = A @ B

        .. code-block:: mlir

            // Basic matrix multiplication
            %result = ttnn.matmul(%a, %b) : tensor<3x4xf32>, tensor<4x5xf32> -> tensor<3x5xf32>
            // Input tensors:
            // a: [[1.0, 2.0, 3.0, 4.0],
            //     [5.0, 6.0, 7.0, 8.0],
            //     [9.0, 10.0, 11.0, 12.0]]
            // b: [[1.0, 0.0, 0.0, 0.0, 0.0],
            //     [0.0, 1.0, 0.0, 0.0, 0.0],
            //     [0.0, 0.0, 1.0, 0.0, 0.0],
            //     [0.0, 0.0, 0.0, 1.0, 0.0]]
            // Output tensor:
            // [[1.0, 2.0, 3.0, 4.0, 0.0],
            //  [5.0, 6.0, 7.0, 8.0, 0.0],
            //  [9.0, 10.0, 11.0, 12.0, 0.0]]

        Parameters
        ----------
        in0 : Operand
            First input tensor
        in1 : Operand
            Second input tensor
        transpose_a : bool
            Whether to transpose the first tensor before multiplication (default: False)
        transpose_b : bool
            Whether to transpose the second tensor before multiplication (default: False)
        unit_attrs : *Optional[List[str]]*
            Optional list of unit attributes

        Returns
        -------
        (*OpView*)
            Result of matrix multiplication
        """
        return self._op_proxy(
            ttnn.MatmulOp,
            [in0, in1],
            golden_kwargs={
                "transpose_a": transpose_a,
                "transpose_b": transpose_b,
            },
            ttnn_kwargs={"transpose_a": transpose_a, "transpose_b": transpose_b},
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
        Creates ``ttnn.linear``.

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
            ttnn.LinearOp,
            [in0, in1],
            golden_kwargs={
                "transpose_a": transpose_a,
                "transpose_b": transpose_b,
                "bias": golden_bias,
            },
            ttnn_kwargs={
                "transpose_a": transpose_a,
                "transpose_b": transpose_b,
                "bias": bias,
            },
            unit_attrs=unit_attrs,
        )

    def clamp_scalar(
        self,
        in0: Operand,
        min_arg: Optional[float] = None,
        max_arg: Optional[float] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Clamp tensor values to a specified range using scalar min/max values.

        Args:
            in0: Input tensor to clamp
            min_arg: Minimum scalar value for clamping (optional)
            max_arg: Maximum scalar value for clamping (optional)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: Clamped tensor with values constrained between min_arg and max_arg

        Example:
            If min_arg=2.0, max_arg=5.0, and input=[[0, 1, 2, 3, 4, 5, 6, 7]],
            then output=[[2, 2, 2, 3, 4, 5, 5, 5]]
        """
        kwargs = {"min": min_arg, "max": max_arg}
        return self._op_proxy(
            ttnn.ClampScalarOp,
            [in0],
            ttnn_kwargs=kwargs,
            golden_kwargs=kwargs,
            unit_attrs=unit_attrs,
        )

    def clamp_tensor(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Clamp tensor values to a specified range using tensor min/max values.

        Args:
            in0: Input tensor to clamp
            in1: Minimum tensor values for clamping (element-wise)
            in2: Maximum tensor values for clamping (element-wise)
            unit_attrs: Optional list of unit attributes

        Returns:
            OpView: Clamped tensor with values constrained element-wise between in1 and in2

        Example:
            If min=[[2, 2, 2, 3, 3, 3, 0, 0]], input=[[0, 1, 2, 3, 4, 5, 6, 7]],
            and max=[[5, 5, 5, 9, 9, 9, 6, 6]], then output=[[2, 2, 2, 3, 4, 5, 6, 6]]
        """
        return self._op_proxy(
            ttnn.ClampTensorOp,
            [in0, in1, in2],
            organize_golden_args=lambda i: [
                self._get_golden_tensor(in0),
                self._get_golden_tensor(in1),
                self._get_golden_tensor(in2),
            ],
            unit_attrs=unit_attrs,
        )

    def concat(
        self, ins: List[Operand], dim: int = 0, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.concat``.

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
<<<<<<< HEAD
        kwargs = {"dim": dim}
        return self._op_proxy(
            ttnn.ConcatOp,
            ins,
            ttnn_kwargs=kwargs,
            # special handling is needed here to get around arg expansion; `torch.concat` takes a tuple of tensors on input
            organize_golden_args=lambda i: (
                tuple([self._get_golden_tensor(i_i) for i_i in i]),
            ),
            organize_ttnn_args=lambda i, o: (o, i),
            unit_attrs=unit_attrs,
=======
        return self._op_proxy(
            ttnn.SigmoidOp, [in0], unit_attrs=unit_attrs, ttnn_kwargs={}
>>>>>>> c5bfbb6e0 (Add remaining ttnn builder tests)
        )

    def repeat(
        self, in0: Operand, dims: List[int], unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        """
        Creates ``ttnn.repeat``.

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
            ttnn.RepeatOp,
            [in0],
            ttnn_kwargs={"repeat_dims": ttnn.ir.ShapeAttr.get(self._ctx, dims)},
            golden_kwargs={"repeat_dimensions": dims},
            unit_attrs=unit_attrs,
        )

    def repeat_interleave(
        self,
        in0: Operand,
        repeats: int,
        dim: int,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttnn.repeat_interleave``.

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
            ttnn.RepeatInterleaveOp,
            [in0],
            ttnn_kwargs={"repeats": repeats, "dim": dim},
            organize_ttnn_args=lambda i, o: (o, i[0]),
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            output_type=self._get_type_from_torch_dtype(
                self._get_golden_tensor(in0).dtype
            ),
            unit_attrs=unit_attrs,
        )
