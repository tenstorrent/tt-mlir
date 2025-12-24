# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Callable, Any
import torch

from ttmlir.ir import *
from ttmlir import util
from ttmlir.dialects import ttnn, ttcore, func

from builder.base.builder import *
from builder.base.builder_utils import *

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
                    self._set_golden_tensor(op.result, golden_output)

            return op.result

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

    def create_l1_width_sharded_tiled_encoding(
        self, shape: Shape, element_type: Type
    ) -> ttnn.ir.TTNNLayoutAttr:
        with self._ctx, self._loc:
            data_type = util.element_type_to_data_type(element_type)
            tile_element_type = ttcore.ir.TileType.get(self._ctx, 32, 32, data_type)
            buffer_type = ttnn.BufferType.L1
            grid_attr = ttcore.ir.GridAttr.get(self._ctx, [1, 1])
            return ttnn.ir.TTNNLayoutAttr.get(
                self._ctx,
                shape,
                tile_element_type,
                buffer_type,
                grid_attr,
                ttnn.TensorMemoryLayout.WidthSharded,
            )

    def create_l1_width_sharded_tiled_ttnn_tensor(
        self, shape: Shape, element_type: Type
    ) -> RankedTensorType:
        with self._ctx, self._loc:
            ttnn_layout_attr = self.create_l1_width_sharded_tiled_encoding(
                shape, element_type
            )
            return RankedTensorType.get(shape, element_type, ttnn_layout_attr)

    def _get_location(self) -> Location:
        stack = inspect.stack()
        caller_frame = stack[2]
        filename = caller_frame.filename
        lineno = caller_frame.lineno
        return Location.name(f"{filename}:{lineno}")

    # ----- Public TTNN Op Generators ----

    ############### ttnn.AddOp ###############

    @tag(ttnn.AddOp)
    def add(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.add)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.AddOp)
    def add_parser(
        self,
        old_op: ttnn.AddOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.add_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.AddOp)
    def add_split(
        self,
        old_op: ttnn.AddOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.add_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            add_module = Module.create()
            add_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(add_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="add_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        add_builder._set_golden_tensor(new_op_result, golden_output)
                        add_builder._set_golden_tensor(lhs, input0)
                        add_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                add_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return add_module, add_builder

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

    @tag(ttnn.LogicalNotOp)
    def logical_not(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.logical_not)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LogicalNotOp)
    def logical_not_parser(
        self,
        old_op: ttnn.LogicalNotOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.logical_not_parser)
        inp = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, inp, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(inp)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LogicalNotOp)
    def logical_not_split(
        self,
        old_op: ttnn.LogicalNotOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_not_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            not_module = Module.create()
            not_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(not_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_not_module")
                def decorated_func(*inputs):
                    inp = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, inp, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        not_builder._set_golden_tensor(new_op_result, golden_output)
                        not_builder._set_golden_tensor(inp, input0)
                        ordered_inputs.append(inp)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                not_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return not_module, not_builder

    @tag(ttnn.BitwiseNotOp)
    def bitwise_not(
        self,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.bitwise_not)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.BitwiseNotOp)
    def bitwise_not_parser(
        self,
        old_op: ttnn.BitwiseNotOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.bitwise_not_parser)
        inp = global_dict[old_op.input]
        result = old_op.result.type

        new_op = ttnn_op(result, inp, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(inp)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.BitwiseNotOp)
    def bitwise_not_split(
        self,
        old_op: ttnn.BitwiseNotOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.bitwise_not_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            not_module = Module.create()
            not_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.input.type,
            ]

            with InsertionPoint(not_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="bitwise_not_module")
                def decorated_func(*inputs):
                    inp = inputs[0]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, inp, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.input)
                        golden_output = op_golden_function(input0, result.element_type)
                        not_builder._set_golden_tensor(new_op_result, golden_output)
                        not_builder._set_golden_tensor(inp, input0)
                        ordered_inputs.append(inp)
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                not_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return not_module, not_builder

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

    @tag(ttnn.EqualOp)
    def eq(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.eq)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.EqualOp)
    def eq_parser(
        self,
        old_op: ttnn.EqualOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.eq_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.EqualOp)
    def eq_split(
        self,
        old_op: ttnn.EqualOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.eq_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            eq_module = Module.create()
            eq_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(eq_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="eq_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        eq_builder._set_golden_tensor(new_op_result, golden_output)
                        eq_builder._set_golden_tensor(lhs, input0)
                        eq_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                eq_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return eq_module, eq_builder

    @tag(ttnn.NotEqualOp)
    def ne(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.ne)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.NotEqualOp)
    def ne_parser(
        self,
        old_op: ttnn.NotEqualOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.ne_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.NotEqualOp)
    def ne_split(
        self,
        old_op: ttnn.NotEqualOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.ne_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            ne_module = Module.create()
            ne_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(ne_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="ne_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        ne_builder._set_golden_tensor(new_op_result, golden_output)
                        ne_builder._set_golden_tensor(lhs, input0)
                        ne_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                ne_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return ne_module, ne_builder

    @tag(ttnn.GreaterEqualOp)
    def ge(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.ge)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.GreaterEqualOp)
    def ge_parser(
        self,
        old_op: ttnn.GreaterEqualOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.ge_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.GreaterEqualOp)
    def ge_split(
        self,
        old_op: ttnn.GreaterEqualOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.ge_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            ge_module = Module.create()
            ge_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(ge_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="ge_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        ge_builder._set_golden_tensor(new_op_result, golden_output)
                        ge_builder._set_golden_tensor(lhs, input0)
                        ge_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                ge_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return ge_module, ge_builder

    @tag(ttnn.GreaterThanOp)
    def gt(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.gt)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.GreaterThanOp)
    def gt_parser(
        self,
        old_op: ttnn.GreaterThanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.gt_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.GreaterThanOp)
    def gt_split(
        self,
        old_op: ttnn.GreaterThanOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.gt_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            gt_module = Module.create()
            gt_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(gt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="gt_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        gt_builder._set_golden_tensor(new_op_result, golden_output)
                        gt_builder._set_golden_tensor(lhs, input0)
                        gt_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                gt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return gt_module, gt_builder

    @tag(ttnn.LessEqualOp)
    def le(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.le)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LessEqualOp)
    def le_parser(
        self,
        old_op: ttnn.LessEqualOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.le_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LessEqualOp)
    def le_split(
        self,
        old_op: ttnn.LessEqualOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.le_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            le_module = Module.create()
            le_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(le_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="le_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        le_builder._set_golden_tensor(new_op_result, golden_output)
                        le_builder._set_golden_tensor(lhs, input0)
                        le_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                le_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return le_module, le_builder

    @tag(ttnn.LessThanOp)
    def lt(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.lt)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LessThanOp)
    def lt_parser(
        self,
        old_op: ttnn.LessThanOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.lt_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LessThanOp)
    def lt_split(
        self,
        old_op: ttnn.LessThanOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.lt_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            lt_module = Module.create()
            lt_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(lt_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="lt_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        lt_builder._set_golden_tensor(new_op_result, golden_output)
                        lt_builder._set_golden_tensor(lhs, input0)
                        lt_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                lt_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return lt_module, lt_builder

    @tag(ttnn.LogicalAndOp)
    def logical_and(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.logical_and)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LogicalAndOp)
    def logical_and_parser(
        self,
        old_op: ttnn.LogicalAndOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.logical_and_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LogicalAndOp)
    def logical_and_split(
        self,
        old_op: ttnn.LogicalAndOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_and_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            and_module = Module.create()
            and_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(and_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_and_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        and_builder._set_golden_tensor(new_op_result, golden_output)
                        and_builder._set_golden_tensor(lhs, input0)
                        and_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                and_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return and_module, and_builder

    @tag(ttnn.LogicalLeftShiftOp)
    def logical_left_shift(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.logical_left_shift)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LogicalLeftShiftOp)
    def logical_left_shift_parser(
        self,
        old_op: ttnn.LogicalLeftShiftOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.logical_left_shift_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LogicalLeftShiftOp)
    def logical_left_shift_split(
        self,
        old_op: ttnn.LogicalLeftShiftOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_left_shift_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            lshift_module = Module.create()
            lshift_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(lshift_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_left_shift_module")
                def decorated_func(*inputs):
                    pass

                new_func_op = decorated_func.func_op
                lshift_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return lshift_module, lshift_builder

    @tag(ttnn.LogicalOrOp)
    def logical_or(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.logical_or)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LogicalOrOp)
    def logical_or_parser(
        self,
        old_op: ttnn.LogicalOrOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.logical_or_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LogicalOrOp)
    def logical_or_split(
        self,
        old_op: ttnn.LogicalOrOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_or_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            or_module = Module.create()
            or_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(or_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_or_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        or_builder._set_golden_tensor(new_op_result, golden_output)
                        or_builder._set_golden_tensor(lhs, input0)
                        or_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                or_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return or_module, or_builder

    @tag(ttnn.LogicalRightShiftOp)
    def logical_right_shift(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.logical_right_shift)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LogicalRightShiftOp)
    def logical_right_shift_parser(
        self,
        old_op: ttnn.LogicalRightShiftOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.logical_right_shift_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LogicalRightShiftOp)
    def logical_right_shift_split(
        self,
        old_op: ttnn.LogicalRightShiftOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_right_shift_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            rshift_module = Module.create()
            rshift_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(rshift_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_right_shift_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        rshift_builder._set_golden_tensor(new_op_result, golden_output)
                        rshift_builder._set_golden_tensor(lhs, input0)
                        rshift_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                rshift_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return rshift_module, rshift_builder

    @tag(ttnn.LogicalXorOp)
    def logical_xor(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.logical_xor)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
            dtype=dtype,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.LogicalXorOp)
    def logical_xor_parser(
        self,
        old_op: ttnn.LogicalXorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.logical_xor_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location, dtype=old_op.dtype)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.LogicalXorOp)
    def logical_xor_split(
        self,
        old_op: ttnn.LogicalXorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.logical_xor_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            xor_module = Module.create()
            xor_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(xor_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="logical_xor_module")
                def decorated_func(*inputs):
                    pass

                new_func_op = decorated_func.func_op
                xor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return xor_module, xor_builder

    @tag(ttnn.BitwiseAndOp)
    def bitwise_and(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.bitwise_and)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.BitwiseAndOp)
    def bitwise_and_parser(
        self,
        old_op: ttnn.BitwiseAndOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.bitwise_and_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.BitwiseAndOp)
    def bitwise_and_split(
        self,
        old_op: ttnn.BitwiseAndOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.bitwise_and_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            and_module = Module.create()
            and_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(and_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="bitwise_and_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(
                        result,
                        lhs,
                        rhs,
                        loc=old_op.location,
                    )
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        and_builder._set_golden_tensor(new_op_result, golden_output)
                        and_builder._set_golden_tensor(lhs, input0)
                        and_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                and_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return and_module, and_builder

    @tag(ttnn.BitwiseOrOp)
    def bitwise_or(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.bitwise_or)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.BitwiseOrOp)
    def bitwise_or_parser(
        self,
        old_op: ttnn.BitwiseOrOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.bitwise_or_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.BitwiseOrOp)
    def bitwise_or_split(
        self,
        old_op: ttnn.BitwiseOrOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.bitwise_or_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            or_module = Module.create()
            or_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(or_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="bitwise_or_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        or_builder._set_golden_tensor(new_op_result, golden_output)
                        or_builder._set_golden_tensor(lhs, input0)
                        or_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                or_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return or_module, or_builder

    @tag(ttnn.BitwiseXorOp)
    def bitwise_xor(
        self,
        in0: Operand,
        in1: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        ttnn_op = self.get_opview_from_method(TTNNBuilder.bitwise_xor)
        dtype = self._get_data_type_attribute(in0)

        if output_type is None:
            mlir_output_type = self.get_type(in0)
        else:
            mlir_output_type = self._get_type_from_torch_dtype(output_type)

        input0 = self._get_golden_tensor(in0)
        input1 = self._get_golden_tensor(in1)
        op_golden_function = get_golden_function(ttnn_op)
        golden_output = op_golden_function(input0, input1, mlir_output_type)
        result = self.create_ttnn_tensor(golden_output.shape, mlir_output_type)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = ttnn_op(
            result,
            in0,
            in1,
            loc=loc,
        )
        op_result = op.result

        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

        if not self._disable_golden_check:
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse(ttnn.BitwiseXorOp)
    def bitwise_xor_parser(
        self,
        old_op: ttnn.BitwiseXorOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        ttnn_op = self.get_opview_from_parser(TTNNBuilder.bitwise_xor_parser)
        lhs = global_dict[old_op.lhs]
        rhs = global_dict[old_op.rhs]
        result = old_op.result.type

        new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
        new_op_result = new_op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(lhs)
            input1 = self._get_golden_tensor(rhs)
            op_golden_function = get_golden_function(ttnn_op)
            golden_output = op_golden_function(input0, input1, result.element_type)
            self._set_golden_tensor(new_op_result, golden_output)

        op_map_dictionary = {}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split(ttnn.BitwiseXorOp)
    def bitwise_xor_split(
        self,
        old_op: ttnn.BitwiseXorOp,
    ) -> Tuple[Module, TTNNBuilder]:
        ttnn_op = self.get_opview_from_split(TTNNBuilder.bitwise_xor_split)

        old_context = old_op.context
        old_loc = Location.unknown(old_context)
        with old_context, old_loc:
            xor_module = Module.create()
            xor_builder = TTNNBuilder(old_context, old_loc)
            op_input_types = [
                old_op.lhs.type,
                old_op.rhs.type,
            ]

            with InsertionPoint(xor_module.body):

                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="bitwise_xor_module")
                def decorated_func(*inputs):
                    lhs = inputs[0]
                    rhs = inputs[1]
                    result = old_op.result.type

                    new_op = ttnn_op(result, lhs, rhs, loc=old_op.location)
                    new_op_result = new_op.result

                    if not self._disable_golden_check:
                        op_golden_function = get_golden_function(ttnn_op)
                        input0 = self._get_golden_tensor(old_op.lhs)
                        input1 = self._get_golden_tensor(old_op.rhs)
                        golden_output = op_golden_function(
                            input0, input1, result.element_type
                        )
                        xor_builder._set_golden_tensor(new_op_result, golden_output)
                        xor_builder._set_golden_tensor(lhs, input0)
                        xor_builder._set_golden_tensor(rhs, input1)
                        ordered_inputs.extend([lhs, rhs])
                        ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                xor_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return xor_module, xor_builder

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

    def _op_proxy_l1_sharded_executed_op_with_dram_final_output(
        self,
        op_function: Callable,
        inputs: List[Operand],
        ttnn_kwargs: dict,
        unit_attrs: Optional[List[str]] = None,
        golden_kwargs: Optional[dict] = None,
        skip_golden: bool = False,
    ) -> OpView:
        """
        Helper method to create L1 width-sharded operations with DRAM output conversion.
        """
        with self._ctx, self._loc:
            # Create L1 width-sharded output tensor
            sharded_output_type = self.create_l1_width_sharded_tiled_ttnn_tensor(
                shape=inputs[0].type.shape,
                element_type=inputs[0].type.element_type,
            )

            # Prepare location for the operation
            id = self._get_next_global_id()
            loc = self._get_loc_of_extra_file_callee(id=id)

            # Create the operation with L1 sharded output
            op = op_function(
                sharded_output_type,
                *inputs,
                loc=loc,
                **ttnn_kwargs,
            )

            # Set unit attributes if provided
            if unit_attrs is not None:
                for attr_name in unit_attrs:
                    op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

            # Convert L1 sharded output to DRAM, to match expected final output layout
            final_output_type = self.create_ttnn_tensor(
                shape=op.result.type.shape,
                element_type=op.result.type.element_type,
            )

            tensor_memory_layout_attr = ttnn.ir.TensorMemoryLayoutAttr.get(
                self._ctx, ttnn.TensorMemoryLayout.Interleaved
            )
            buffer_type_attr = ttnn.ir.BufferTypeAttr.get(
                self._ctx, ttnn.BufferType.DRAM
            )
            memoryConfigAttr = ttnn.ir.MemoryConfigAttr.get(
                self._ctx, tensor_memory_layout_attr, buffer_type_attr
            )
            data_type = self._get_data_type_attribute(op.result)

            # Prepare location for the helper ToLayout operation
            id = self._get_next_global_id()
            loc = self._get_loc_of_extra_file_callee(id=id)

            output_to_dram = ttnn.ToLayoutOp(
                final_output_type,
                op.result,
                layout=ttnn.ir.LayoutAttr.get(self._ctx, ttnn.Layout.Tile),
                memory_config=memoryConfigAttr,
                loc=loc,
                dtype=data_type,
            )

            if not skip_golden and not self._disable_golden_check:
                golden_func = get_golden_function(op_function)
                if golden_func:
                    golden_args = self._organize_eltwise_golden(inputs)
                    golden = golden_func(*golden_args, **golden_kwargs or {})
                    self._set_golden_tensor(output_to_dram, golden)

            return output_to_dram

    def rms_norm(
        self,
        input_tensor: Operand,
        weight: Optional[Operand] = None,
        bias: Optional[Operand] = None,
        epsilon: float = 1.0e-5,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates ``ttnn.rms_norm``.

        *RMS Normalization operation.*

        Applies Root Mean Square (RMS) normalization to the input tensor.
        RMS normalization normalizes the input tensor based on the root mean square of its elements.

        Mathematical definition: rms_norm(x) = (x / sqrt(mean(x^2) + epsilon)) * weight + bias

        Parameters
        ----------
        input_tensor : Operand
            Input tensor to be normalized
        weight : Optional[Operand]
            Optional weight tensor for scaling (default: None)
        bias : Optional[Operand]
            Optional bias tensor for shifting (default: None)
        epsilon : float
            Small constant to avoid division by zero (default is 1.0e-5)
        unit_attrs : Optional[List[str]]
            Optional list of unit attributes, here used to specify L1 width sharding

        Returns
        -------
        OpView
            A tensor containing the RMS normalized output
        """
        # Check if L1 width sharding is requested
        l1_width_sharded = unit_attrs is not None and "l1_width_sharded" in unit_attrs
        if l1_width_sharded:
            # Remove the l1_width_sharded attribute for internal processing
            unit_attrs = [attr for attr in unit_attrs if attr != "l1_width_sharded"]

        # Build TTNN kwargs - only include non-None optional parameters
        ttnn_kwargs = {"epsilon": epsilon}
        if weight is not None:
            ttnn_kwargs["weight"] = weight
        if bias is not None:
            ttnn_kwargs["bias"] = bias

        # Determine normalized_shape from weight/bias tensor if available, otherwise use last input dimension
        if weight is not None:
            normalized_shape = list(weight.type.shape)
        elif bias is not None:
            normalized_shape = list(bias.type.shape)
        else:
            normalized_shape = [input_tensor.type.shape[-1]]

        golden_kwargs = {"epsilon": epsilon, "normalized_shape": normalized_shape}
        if weight is not None:
            golden_kwargs["weight"] = self._get_golden_tensor(weight)
        if bias is not None:
            golden_kwargs["bias"] = self._get_golden_tensor(bias)

        if l1_width_sharded:
            return self._op_proxy_l1_sharded_executed_op_with_dram_final_output(
                ttnn.RMSNormOp,
                [input_tensor],
                ttnn_kwargs=ttnn_kwargs,
                unit_attrs=unit_attrs,
                golden_kwargs=golden_kwargs,
            )
        else:
            return self._op_proxy(
                ttnn.RMSNormOp,
                [input_tensor],
                output_shape=input_tensor.type.shape,
                output_type=input_tensor.type.element_type,
                ttnn_kwargs=ttnn_kwargs,
                unit_attrs=unit_attrs,
                golden_kwargs=golden_kwargs,
            )

    # ----- Parse ttnn module ----

    @staticmethod
    def from_module(
        ctx: Context,
        mlir_text: str,
        golden_inputs: Dict[str, List[torch.tensor]] = None,
    ) -> Tuple(Module, TTNNBuilder):
        if golden_inputs is None:
            golden_inputs = {}

        root_module = Module.parse(mlir_text, ctx)
        loc = Location.unknown(ctx)
        with ctx, loc:
            ttnn_builder = TTNNBuilder(ctx, loc)
            new_module = ttnn_builder.parse_root_module(root_module, golden_inputs)

        return new_module, ttnn_builder

    # ----- Split ttnn module ----

    def split_op(
        self,
        parsed_op: Operation,
    ) -> Tuple[Module, TTNNBuilder]:
        split_function = self.get_split_from_opview(type(parsed_op))
        return split_function(self, parsed_op)

    @staticmethod
    def split_module(
        module: Module,
        builder: TTNNBuilder,
    ) -> List[Tuple[Module, TTNNBuilder]]:
        sub_modules_and_builders = []
        old_ctx = module.context
        old_loc = Location.unknown(old_ctx)

        with old_ctx, old_loc:
            for entry in module.body.operations:
                if isinstance(entry, ttcore.DeviceModuleOp):
                    for device_op_module in entry.regions[0].blocks[0].operations:
                        for device_module_op in (
                            device_op_module.regions[0].blocks[0].operations
                        ):
                            if isinstance(device_module_op, func.FuncOp):
                                if device_module_op.name.value in builder._nested_funcs:
                                    continue

                                for block in device_module_op.body:
                                    for op in block.operations:
                                        if isinstance(op, func.ReturnOp):
                                            continue
                                        elif isinstance(op, func.CallOp):
                                            sub_op_module_builder = (
                                                builder.split_call_op(op)
                                            )
                                            if len(sub_op_module_builder) != 0:
                                                sub_modules_and_builders.append(
                                                    sub_op_module_builder
                                                )
                                        else:
                                            sub_op_module_builder = builder.split_op(op)
                                            if len(sub_op_module_builder) != 0:
                                                sub_modules_and_builders.append(
                                                    sub_op_module_builder
                                                )
                elif isinstance(entry, func.FuncOp):
                    if entry.name.value in builder._nested_funcs:
                        continue

                    for block in entry.body:
                        for op in block.operations:
                            if isinstance(op, func.ReturnOp):
                                continue
                            elif isinstance(op, func.CallOp):
                                sub_op_module_builder = builder.split_call_op(op)
                                if len(sub_op_module_builder) != 0:
                                    sub_modules_and_builders.append(
                                        sub_op_module_builder
                                    )
                            else:
                                sub_op_module_builder = builder.split_op(op)
                                if len(sub_op_module_builder) != 0:
                                    sub_modules_and_builders.append(
                                        sub_op_module_builder
                                    )

        return sub_modules_and_builders
