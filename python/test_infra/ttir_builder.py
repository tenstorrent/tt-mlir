# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Callable, Dict, Any
from ttmlir.ir import *
from ttmlir.dialects import ttir, tt, tensor, quant
from ttmlir.passes import GoldenTensor, DataType
import torch
import array

# Alias for operands of ops which can be either BlockArguments, Values, or other
# ops wrapped in OpView or Operation.
Operand = Union[Value, OpView, Operation]

# Convenience alias for shape
Shape = Union[List[int], Tuple[int, ...]]


def get_loc_of_extra_file_callee(id: int = 0) -> Location:
    """When called, this function returns a `Location` referring to first
    callee outside the file of the caller of this function. E.G., if a function
    in `foo.py` called a function in `bar.py` that then called this function,
    the location would be pointing to the call in `foo.py`.

    NOTE: this location is _NOT_ in the form of
    {filename}:{line_number}:{col_number}, but instead in the form:
    {filename}:{line_number}:id({id}), where id is supplied to this function as
    a disambiguator for calls that happen on the same line

    Arguments
    ---------

    id : int
        An optional variable that defaults to 0 to be appended to the location,
        disambiguating calls on the same line.

    Returns
    -------

    A `Location` referring to the first extra file callee of the caller of this function

    """

    stack = inspect.stack()

    # find the innermost frame outside of this file
    caller_filename = stack[1].filename

    while len(stack) > 0 and stack[0].filename == caller_filename:
        stack = stack[1:]

    assert (
        len(stack) > 0
    ), "Top of callstack to builder funcs must be outside the caller's file"

    # FIXME: this should be a `Location.file`, but for some reason it causes
    # strange decomposition inheritance behaviour that breaks using this as
    # a key into the golden map
    return Location.name(f"{stack[0].filename}:{str(stack[0].lineno)}:id({str(id)})")


@dataclass(frozen=True)
class Golden:
    """
    Dataclass used to store information about golden tensor which will be used
    for comparison with TT device output.

    Each TTIR op should have a matching torch op, and for same inputs, they
    should generate same outputs.
    """

    tensor: torch.Tensor

    # `torch.manual_seed` arg with which tensor was generated. Valid (not None)
    # only for randomly generated tensors, for example args of MLIR function
    # wrapped around user-written op graph. Every other tensor is output of some
    # op from graph.
    seed: int = None

    def __repr__(self) -> str:
        s = f"\nRandom seed: {self.seed}" if self.seed is not None else ""
        s += f"\nGolden tensor:\n{self.tensor}"
        return s

    def contiguous(self) -> Golden:
        return Golden(self.tensor.contiguous())


@dataclass
class TypeInfo:
    """Encapsulates type information for quantized tensors.

    Contains both the base data type and quantization parameters (scale and zero point)
    required for quantized operations.

    Attributes:
        dtype: Base PyTorch data type (e.g. torch.float32, torch.qint32).
        scale: Scaling factor for quantization. Required for quantized types.
        zero_point: Zero point offset for quantization. Required for quantized types.
    """

    dtype: torch.dtype
    scale: Optional[float] = None
    zero_point: Optional[int] = None


class TTIRBuilder:
    """Builder class providing API for creating TTIR ops."""

    def __init__(self, ctx: Context, location: Location):
        self._ctx = ctx
        self._loc = location

        tt.register_dialect(self._ctx)
        ttir.register_dialect(self._ctx)

        self._seed = 0
        # Dictionary to store Golden for each Operand we encounter in MLIR
        # graph.
        self._goldens: Dict[Operand, Golden] = {}

        # global ID of operations
        self._global_id = -1

        # id to golden map
        self.id_golden_map = {}

        # mesh_shape for multi-device
        self.mesh_shape = ()

    # ----- Public helpers -----

    @property
    def goldens(self) -> Dict:
        return self._goldens

    def get_next_global_id(self) -> int:
        self._global_id += 1
        return self._global_id

    def print_goldens(self) -> None:
        """
        Prints saved operands and their respective goldens in descriptive form
        which follows SSA ordering from MLIR graph.
        """
        i = 0
        for operand, golden in self._goldens.items():
            operand_name = self._get_name(operand)

            if self._operand_is_mlir_func_arg(operand):
                print(f"Func arg: {operand_name}", golden, "\n")
            else:
                print(f"%{i}: {operand_name}", golden, "\n")
                i += 1

    def get_shape(self, input: Operand) -> Shape:
        """Retrieves shape of operand which is expected to be a shaped type."""
        return self._get_type(input).shape

    def generate_and_store_random_golden(
        self, operand: Operand, dtype: torch.dtype = torch.float32
    ) -> Golden:
        """
        Generates random tensor of `dtype`s of `operand`s shape, assigns it to a golden,
        and maps `operand` to that golden.

        Returns generated golden.
        """
        seed = self._get_seed()
        random_tensor = self._generate_random_tensor(
            self.get_shape(operand), dtype, seed
        )
        golden = Golden(random_tensor, seed)
        self._store_golden(operand, golden)
        return golden

    def generate_input_golden(
        self, operand: Operand, dtype: torch.dtype, index: int
    ) -> None:
        """
        Generates random tensor of `dtype`s of `input`s shape, assigns it to a golden,
        and maps `input` to that golden.
        """
        self.id_golden_map[f"input_{index}"] = self.generate_and_store_random_golden(
            operand, dtype
        )

    def get_golden_map(self) -> Dict:
        golden_info = {}
        for name, golden_tensor in self.id_golden_map.items():
            golden_tensor = golden_tensor.contiguous()
            data_type = self.get_datatype_from_torch_dtype(golden_tensor.tensor.dtype)
            golden_info[name] = GoldenTensor(
                name,
                list(golden_tensor.tensor.shape),
                list(golden_tensor.tensor.stride()),
                data_type if data_type is not None else DataType.Float32,
                golden_tensor.tensor.data_ptr(),
                golden_tensor.tensor.numel() * golden_tensor.tensor.dtype.itemsize,
            )
        return golden_info

    # set mesh_shape for multi-device environment
    def set_mesh_shape(self, mesh_shape: Tuple[int, int]):
        self.mesh_shape = mesh_shape

    def set_graph_input_output(
        self, inputs: List[torch.Tensor], outputs: Optional[List[torch.Tensor]] = None
    ) -> None:
        """
        Records the input and output tensors for the graph.
        """
        for index, tensor in enumerate(inputs):
            input_key = f"input_{index}"
            if input_key in self.id_golden_map:
                assert self.id_golden_map[input_key].tensor.shape == tensor.shape
                assert self.id_golden_map[input_key].tensor.dtype == tensor.dtype
            self.id_golden_map[input_key] = Golden(tensor)

        if outputs is not None:
            for index, tensor in enumerate(outputs):
                output_key = f"output_{index}"
                self.id_golden_map[output_key] = Golden(tensor)

    # ----- Private helpers -----

    @staticmethod
    def _get_name(operand: Operand) -> str:
        """Retrieves descriptive operand name."""
        # Try to call get_name() if it exists, otherwise return operand.name.
        name = getattr(operand, "get_name", lambda: None)() or getattr(
            operand, "name", None
        )
        assert name is not None, (
            f"Couldn't retrieve name for operand {operand}. Check if this "
            f"operand type is properly supported."
        )
        return name

    @staticmethod
    def _operand_is_mlir_func_arg(operand: Operand) -> bool:
        """Checks if operand is an argument of surrounding MLIR function."""
        return isinstance(operand, BlockArgument) and "arg" in TTIRBuilder._get_name(
            operand
        )

    def _get_seed(self) -> int:
        """Monotonically increasing seed for reproducibility."""
        seed = self._seed
        self._seed += 1
        return seed

    @staticmethod
    def _generate_random_tensor(
        shape: Shape, dtype: torch.dtype, seed: int
    ) -> torch.Tensor:
        """
        Generates random tensor of shape `shape`, with type `dtype`, using `seed` to seed torch
        random generator.
        """
        if dtype.is_floating_point:
            return torch.randn(shape, generator=torch.manual_seed(seed), dtype=dtype)
        else:
            min_int = torch.iinfo(dtype).min
            max_int = torch.iinfo(dtype).max
            return torch.randint(
                low=min_int,
                high=max_int,
                size=shape,
                generator=torch.manual_seed(seed),
                dtype=dtype,
            )

    def _get_golden(self, operand: Operand) -> Golden:
        """Retrieves stored golden for `operand`."""
        golden = self._goldens.get(operand)
        assert golden is not None, f"Expected to have a golden stored for {operand}"
        return golden

    def _store_golden(self, operand: Operand, golden: Golden) -> None:
        """Maps `operand` to `golden`."""
        assert (
            self._goldens.get(operand) == None
        ), f"Golden for {operand} already exists."
        self._goldens[operand] = golden

    def _override_golden(self, operand: Operand, golden: Golden) -> None:
        """
        Overrides existing golden for `operand`.

        Used to override randomly generated goldens for empty tensors which are
        used as outputs of TTIR ops with golden for that TIIR op.
        """
        assert (
            self._goldens.get(operand) is not None
        ), f"Expected golden for {operand} to already exist before overriding it."
        self._goldens[operand] = golden

    def _get_golden_tensor(self, operand: Operand) -> torch.Tensor:
        return self._get_golden(operand).tensor

    @property
    def _default_dtype(self) -> Type:
        return F32Type.get(self._ctx)

    def _get_type(self, input: Operand):
        """
        Helper method which retrieves underlying mlir Type of Operand based on
        which concrete type it is.

        We always expect it to be a RankedTensorType.
        """
        if isinstance(input, Value):
            typ = input.type
        elif isinstance(input, OpView):
            typ = input.operation.result.type
        elif isinstance(input, Operation):
            typ = input.result.type
        else:
            raise TypeError(f"Invalid input {type(input)}")

        assert isinstance(typ, RankedTensorType), "Only ranked tensors are supported"

        return typ

    # ----- Utility Conversion ----

    def get_datatype_from_torch_dtype(self, dtype: torch.dtype) -> DataType:
        """
        Returns a MLIR `DataType` obj corresponding to `dtype`.
        """
        match dtype:
            case torch.float16:
                return DataType.Float16
            case torch.bfloat16:
                return DataType.BFloat16
            case torch.float32:
                return DataType.Float32
            case torch.int32:
                return DataType.Int32
            case None:
                return DataType.Float32

    def get_type_from_torch_dtype(self, dtype: Union[torch.dtype, TypeInfo]) -> Type:
        """Converts PyTorch dtype or TypeInfo to corresponding MLIR Type.

        For quantized types (e.g. qint32), scale and zero_point must be provided via TypeInfo.
        For non-quantized types, a plain torch.dtype can be used.

        Args:
            dtype: Either a torch.dtype or TypeInfo containing dtype and quantization params.

        Returns:
            MLIR Type corresponding to the input dtype.

        Raises:
            ValueError: If quantization parameters are missing for quantized types.
            TypeError: If the dtype is not supported.
        """
        base_dtype = dtype.dtype if isinstance(dtype, TypeInfo) else dtype

        match base_dtype:
            case torch.bfloat16:
                return BF16Type.get(self._ctx)
            case torch.float16:
                return F16Type.get(self._ctx)
            case torch.float32:
                return F32Type.get(self._ctx)
            case torch.float64:
                return F64Type.get(self._ctx)
            case torch.int8:
                return IntegerType.get_signless(8, self._ctx)
            case torch.int16:
                return IntegerType.get_signless(16, self._ctx)
            case torch.int32:
                return IntegerType.get_signless(32, self._ctx)
            case torch.int64:
                return IntegerType.get_signless(64, self._ctx)
            case torch.uint8:
                return IntegerType.get_unsigned(8, self._ctx)
            case torch.uint16:
                return IntegerType.get_unsigned(16, self._ctx)
            case torch.uint32:
                return IntegerType.get_unsigned(32, self._ctx)
            case torch.uint64:
                return IntegerType.get_unsigned(64, self._ctx)
            case torch.qint32:
                if not isinstance(dtype, TypeInfo):
                    raise ValueError("TypeInfo required for qint32")
                if dtype.scale is None or dtype.zero_point is None:
                    raise ValueError("scale and zero_point required for qint32")
                return quant.UniformQuantizedType.get(
                    quant.UniformQuantizedType.FLAG_SIGNED,
                    IntegerType.get_signless(32, self._ctx),
                    F32Type.get(self._ctx),
                    dtype.scale,
                    dtype.zero_point,
                    torch.iinfo(torch.qint32).min,
                    torch.iinfo(torch.qint32).max,
                )
            case _:
                raise TypeError(f"Invalid Type {dtype}")

    # ----- Utility factories -----

    def ranked_tensor_type(
        self,
        shape: Shape,
        data_type: Optional[Type] = None,
        encoding: Optional[Attribute] = None,
    ) -> RankedTensorType:
        """Convenience wrapper constructing `RankedTensorType`."""
        dtype = data_type if data_type is not None else self._default_dtype

        with self._ctx, self._loc:
            return RankedTensorType.get(shape, dtype, encoding)

    def empty(
        self,
        shape: Shape,
        data_type: Optional[Type] = None,
    ) -> OpView:
        """Convenience wrapper constructing `ttir.EmptyOp`."""
        dtype = data_type if data_type is not None else self._default_dtype
        with self._ctx, self._loc:
            op = ttir.EmptyOp(RankedTensorType.get(shape, dtype))

            self.generate_and_store_random_golden(op)

            return op

    # ----- TTIR op factories -----
    def _organize_eltwise_ttir(
        self, inputs: List[Operand], output: OpView, output_shape: Optional[Shape]
    ):
        return ([self._get_type(output)], inputs, [output])

    def _organize_eltwise_golden(self, inputs: List[Operand]):
        return [self._get_golden_tensor(inp) for inp in inputs]

    def op_proxy(
        self,
        op_golden_function: Callable,
        op_ttir_function: Callable,
        inputs: List[Operand],
        unit_attrs: List[str] = None,
        organize_ttir_args: Optional[Callable] = None,
        organize_golden_args: Optional[Callable] = None,
        output_shape: Optional[Shape] = None,
        output_type: Optional[Type] = None,
        golden_kwargs: dict = {},
        ttir_kwargs: dict = {},
        use_zeros: bool = False,
    ) -> Any:
        """
        Provides a general interface for proxy-ing OPs and creating them.

        Parameters:
        - op_golden_function (Callable): A function that creates the OP using a golden approach.
        - op_ttir_function (Callable): A function that creates the OP using a TTIR approach.
        - inputs (List[Operand]): A list of operands serving as inputs to the OP.
        - organize_ttir_args (Callable): A function that organizes the inputs and other positional arguments for the TTIR approach.
            - Function signature:

                def organize_ttir_args(inputs: List[Operand], output: OpView, output_shape: Optional[Shape]) -> List/Tuple

                The list/tuple will then be unpacked as the positional arguments for the op_ttir_function

        - organize_golden_args (Callable): A function that organizes the inputs and other arguments for the golden approach.
            - Function signature:

                def organize_golden_args(inputs: List[Operand], output: OpView, output_shape: Optional[Shape]) -> List/Tuple

                The list/tuple will then be unpacked as the positional arugments for the op_golden_function
        - output_shape (Optional[Shape]): An optional argument specifying the shape of the output of the OP.
        - output_type (Optional[Type]): An optional argument specifying the type of the output of the OP.
        - golden_kwargs (dict): Additional keyword arguments for the `op_golden_function`.
        - ttir_kwargs (dict): Additional keyword arguments for the `op_ttir_function`.

        Returns:
        - OpView: The created op
        """
        # Snoop the location of the first caller outside of this file to
        # annotate the MLIR with. NOTE that this location is _NOT_ row:col, but
        # instead row:id, where id is a unique id given to all calls to builder
        # funcs. See `get_next_global_id` for more details
        stack = inspect.stack()
        # find the innermost frame outside of this file
        cur_filename = stack[0].filename

        while len(stack) > 0 and stack[0].filename == cur_filename:
            stack = stack[1:]

        assert (
            len(stack) > 0
        ), "Top of callstack to builder funcs must be outside this file"

        if organize_ttir_args is None:
            organize_ttir_args = self._organize_eltwise_ttir

        if organize_golden_args is None:
            organize_golden_args = self._organize_eltwise_golden

        with self._ctx, self._loc:
            # Compute the golden
            # Account for cases in which golden_arg organization is not needed:
            if (
                not isinstance(organize_golden_args(inputs), torch.Tensor)
                and organize_golden_args(inputs) == 0
            ):
                golden_output = op_golden_function(**golden_kwargs)
            else:
                golden_output = op_golden_function(
                    *(organize_golden_args(inputs)),
                    **golden_kwargs,
                )

            golden = (
                Golden(golden_output[0])
                if not isinstance(golden_output, torch.Tensor)
                else Golden(golden_output)
            )
            # Use the golden output to determine proper output shape and type unless otherwise specified.
            output_shape = golden.tensor.shape if not output_shape else output_shape
            if not output_type and inputs:
                output_type = self.get_type_from_torch_dtype(
                    self._get_golden_tensor(inputs[0]).dtype
                )
            elif not output_type:
                output_type = self._default_dtype

            if use_zeros:
                output = self.zeros(output_shape, output_type)
            else:
                output = self.empty(output_shape, output_type)
            id = self.get_next_global_id()
            loc = get_loc_of_extra_file_callee(id=id)
            # Account for cases in which ttir_arg organization is not needed:
            if (
                not isinstance(
                    organize_ttir_args(inputs, output, output_shape), torch.Tensor
                )
                and organize_ttir_args(inputs, output, output_shape) == 0
            ):
                op = op_ttir_function(loc=loc, **ttir_kwargs)
            else:
                op = op_ttir_function(
                    *organize_ttir_args(inputs, output, output_shape),
                    loc=loc,
                    **ttir_kwargs,
                )

            # Add unit attributes if specified
            if unit_attrs:
                from ttmlir.ir import UnitAttr

                for attr_name in unit_attrs:
                    op.operation.attributes[attr_name] = UnitAttr.get(self._ctx)

            self.id_golden_map[str(loc)] = golden
            self._store_golden(op, golden)
            self._override_golden(output, golden)
            return op

    def eltwise_proxy(
        self,
        op_golden_function: Callable,
        op_ttir_function: Callable,
        inputs: List[Operand],
    ) -> OpView:
        return self.op_proxy(op_golden_function, op_ttir_function, inputs)

    # TTIR top level ops

    def get_dimension_size(self, in0: Operand, dimension: int = 0) -> OpView:
        golden_dim = [self._get_golden_tensor(in0).size(dimension)]
        return self.op_proxy(
            torch.tensor,
            ttir.GetDimensionSizeOp,
            [in0],
            golden_kwargs={"data": golden_dim},
            ttir_kwargs={"dimension": dimension},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0]),
            organize_golden_args=lambda i: 0,
            output_type=self.get_type_from_torch_dtype(torch.int32),
        )

    def dot_general(
        self,
        in0: Operand,
        in1: Operand,
        batch_dims_lhs: List[int],
        contract_dims_lhs: List[int],
        batch_dims_rhs: List[int],
        contract_dims_rhs: List[int],
    ) -> OpView:
        # Configure inputs for golden function
        lhs_dims = contract_dims_lhs + batch_dims_lhs
        rhs_dims = contract_dims_rhs + batch_dims_rhs

        # Get output_shape from inputs' shapes and dimensions
        lhs = self._get_golden_tensor(in0)
        rhs = self._get_golden_tensor(in1)
        lhs_shape = self.get_shape(in0)
        rhs_shape = self.get_shape(in1)
        output_shape = [lhs_shape[i] for i in batch_dims_lhs]
        for d in range(len(lhs_shape)):
            if d not in batch_dims_lhs and d not in contract_dims_lhs:
                output_shape.append(lhs_shape[d])
        for d in range(len(rhs_shape)):
            if d not in batch_dims_rhs and d not in contract_dims_rhs:
                output_shape.append(rhs_shape[d])
        return self.op_proxy(
            torch.tensordot,
            ttir.DotGeneralOp,
            [in0, in1],
            golden_kwargs={"dims": (lhs_dims, rhs_dims)},
            ttir_kwargs={
                "batch_dims_lhs": batch_dims_lhs,
                "contract_dims_lhs": contract_dims_lhs,
                "batch_dims_rhs": batch_dims_rhs,
                "contract_dims_rhs": contract_dims_rhs,
            },
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1]),
            output_shape=output_shape,
            output_type=self.get_type_from_torch_dtype(
                self._get_golden_tensor(in0).dtype
            ),
        )

    # TTIR top level named ops
    # class TTIR_ElementwiseTernaryOp

    def where(
        self, in0: Operand, in1: Operand, in2: Operand, operandSegmentSizes=List[int]
    ) -> OpView:
        return self.op_proxy(
            torch.where,
            ttir.WhereOp,
            [in0, in1, in2],
            organize_golden_args=lambda i: (
                self._get_golden_tensor(i[0]).to(dtype=torch.bool),
                self._get_golden_tensor(i[1]),
                self._get_golden_tensor(i[2]),
            ),
        )

    # class TTIR_ElementwiseUnaryOp

    def abs(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.abs, ttir.AbsOp, [in0])

    def cbrt(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(lambda x: torch.pow(x, 1 / 3), ttir.CbrtOp, [in0])

    def ceil(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.ceil, ttir.CeilOp, [in0])

    def cos(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.cos, ttir.CosOp, [in0])

    def floor(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.floor, ttir.FloorOp, [in0])

    def gelu(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.nn.functional.gelu, ttir.GeluOp, [in0])

    def is_finite(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.isfinite, ttir.IsFiniteOp, [in0])

    def logical_not(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.logical_not, ttir.LogicalNotOp, [in0])

    def bitwise_not(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.bitwise_not, ttir.BitwiseNotOp, [in0])

    def neg(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.neg, ttir.NegOp, [in0])

    def tan(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.tan, ttir.TanOp, [in0])

    def atan(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.atan, ttir.AtanOp, [in0])

    def tanh(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.tanh, ttir.TanhOp, [in0])

    def reciprocal(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.reciprocal, ttir.ReciprocalOp, [in0])

    def relu(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.relu, ttir.ReluOp, [in0])

    def rsqrt(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.rsqrt, ttir.RsqrtOp, [in0])

    def sigmoid(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.sigmoid, ttir.SigmoidOp, [in0])

    def sign(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.sign, ttir.SignOp, [in0])

    def sin(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.sin, ttir.SinOp, [in0])

    def sqrt(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.sqrt, ttir.SqrtOp, [in0])

    def typecast(self, in0: Operand, out: Operand) -> OpView:
        output_type = self.get_type_from_torch_dtype(self._get_golden_tensor(out).dtype)
        return self.op_proxy(
            torch.Tensor.type,
            ttir.TypecastOp,
            [in0],
            golden_kwargs={"dtype": self._get_golden_tensor(out).type()},
            output_type=output_type,
        )

    def log(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.log, ttir.LogOp, [in0])

    def log1p(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.log1p, ttir.Log1pOp, [in0])

    def expm1(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.expm1, ttir.Expm1Op, [in0])

    # class TTIR_ElementwiseUnaryWithFloatParameterOp

    def leaky_relu(self, in0: Operand, parameter: float = 0.01) -> OpView:
        # TODO: reconcile this naming mismatch
        ttir_kwargs = {"parameter": parameter}
        golden_kwargs = {"negative_slope": parameter}
        return self.op_proxy(
            torch.nn.functional.leaky_relu,
            ttir.LeakyReluOp,
            [in0],
            golden_kwargs=golden_kwargs,
            ttir_kwargs=ttir_kwargs,
        )

    # class TTIR_ElementwiseBinaryOp

    def eq(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.eq, ttir.EqualOp, [in0, in1])

    def ne(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.ne, ttir.NotEqualOp, [in0, in1])

    def ge(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.ge, ttir.GreaterEqualOp, [in0, in1])

    def gt(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.gt, ttir.GreaterThanOp, [in0, in1])

    def le(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.le, ttir.LessEqualOp, [in0, in1])

    def lt(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.lt, ttir.LessThanOp, [in0, in1])

    def logical_and(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.logical_and, ttir.LogicalAndOp, [in0, in1])

    def logical_or(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.logical_or, ttir.LogicalOrOp, [in0, in1])

    def logical_xor(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.logical_xor, ttir.LogicalXorOp, [in0, in1])

    def bitwise_and(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.bitwise_and, ttir.BitwiseAndOp, [in0, in1])

    def bitwise_or(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.bitwise_or, ttir.BitwiseOrOp, [in0, in1])

    def bitwise_xor(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.bitwise_xor, ttir.BitwiseXorOp, [in0, in1])

    def minimum(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.minimum, ttir.MinimumOp, [in0, in1])

    def subtract(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.subtract, ttir.SubtractOp, [in0, in1])

    def remainder(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.remainder, ttir.RemainderOp, [in0, in1])

    def pow(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.pow, ttir.PowOp, [in0, in1])

    # class TTIR_ReductionOp

    def argmax(
        self, in0: Operand, dim_arg: List[int], keep_dim: bool = False
    ) -> OpView:
        return self.op_proxy(
            torch.argmax,
            ttir.ArgMaxOp,
            [in0],
            golden_kwargs={"dim": dim_arg[0], "keepdim": keep_dim},
            ttir_kwargs={"dim_arg": dim_arg, "keep_dim": keep_dim},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
            output_type=IntegerType.get_signless(32, self._ctx),
        )

    def sum(
        self, in0: Operand, dim_arg: List[int] = [0], keep_dim: bool = True
    ) -> OpView:
        return self.op_proxy(
            torch.sum,
            ttir.SumOp,
            [in0],
            golden_kwargs={"dim": dim_arg, "keepdim": keep_dim},
            ttir_kwargs={"dim_arg": dim_arg, "keep_dim": keep_dim},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def mean(
        self, in0: Operand, dim_arg: List[int] = [0], keep_dim: bool = True
    ) -> OpView:
        return self.op_proxy(
            torch.mean,
            ttir.MeanOp,
            [in0],
            golden_kwargs={"dim": dim_arg, "keepdim": keep_dim},
            ttir_kwargs={"dim_arg": dim_arg, "keep_dim": keep_dim},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def max(self, in0: Operand, dim_arg: int = None, keep_dim: bool = True) -> OpView:
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
            torch.max,
            ttir.MaxOp,
            [in0],
            golden_kwargs=golden_kwargs,
            ttir_kwargs=ttir_kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
            output_shape=output_shape,
        )

    def min(self, in0: Operand, dim_arg: int = None, keep_dim: bool = True) -> OpView:
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
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
            output_shape=output_shape,
        )

    # NOTE: Not useable. Boolean tensors are not supported by the runtime. Issue #1775
    def reduce_and(
        self, in0: Operand, keep_dim: bool = True, dim_args: Optional[List] = None
    ) -> OpView:
        return self.op_proxy(
            torch.all,
            ttir.ReduceAndOp,
            [in0],
            golden_kwargs={"dim": tuple(dim_args), "keepdim": keep_dim},
            ttir_kwargs={"dim_arg": dim_args, "keep_dim": keep_dim},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    # NOTE: Not useable. Boolean tensors are not supported by the runtime. Issue #1775
    def reduce_or(
        self, in0: Operand, keep_dim: bool = True, dim_args: Optional[List] = None
    ) -> OpView:
        return self.op_proxy(
            torch.any,
            ttir.ReduceOrOp,
            [in0],
            golden_kwargs={"dim": tuple(dim_args)},
            ttir_kwargs={"dim_arg": dim_args, "keep_dim": keep_dim},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def prod(self, in0: Operand, dim_arg: List[int], keep_dim: bool = False) -> OpView:
        g_kwargs = {}
        if len(dim_arg) == 1:
            g_kwargs["dim"] = dim_arg[0]
            g_kwargs["keepdim"] = keep_dim
            g_function = torch.prod
        else:
            g_function = lambda i: torch.tensor([torch.prod(i[0]).item()])
        return self.op_proxy(
            g_function,
            ttir.ProdOp,
            [in0],
            golden_kwargs=g_kwargs,
            ttir_kwargs={"keep_dim": keep_dim, "dim_arg": dim_arg},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def embedding(self, in0: Operand, in1: Operand) -> OpView:
        embedding = torch.nn.Embedding.from_pretrained(self._get_golden_tensor(in1))
        return self.op_proxy(
            embedding,
            ttir.EmbeddingOp,
            [in0, in1],
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1], o),
            organize_golden_args=lambda i: (
                torch.ones(self._get_golden_tensor(i[0]).size(), dtype=torch.long),
            ),
        )

    def cumsum(self, in0: Operand, in1: Operand, dim: int) -> OpView:
        return self.op_proxy(
            torch.cumsum,
            ttir.CumSumOp,
            [in0, in1],
            golden_kwargs={"dim": dim},
            ttir_kwargs={"dim": dim, "output": in1},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0]),
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
        )

    def softmax(self, in0: Operand, dimension: int = 1) -> OpView:
        return self.op_proxy(
            torch.softmax,
            ttir.SoftmaxOp,
            [in0],
            golden_kwargs={"dim": dimension},
            organize_ttir_args=lambda i, o, shape: (
                self._get_type(o),
                i[0],
                o,
                dimension,
            ),
        )

    def transpose(self, in0: Operand, dim0: int = 0, dim1: int = 1) -> OpView:
        kwargs = {"dim0": dim0, "dim1": dim1}
        return self.op_proxy(
            torch.transpose,
            ttir.TransposeOp,
            [in0],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def concat(self, ins: List[Operand], dim: int = 0) -> OpView:
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
        )

    def repeat(self, in0: Operand, dims: List[int]) -> OpView:
        return self.op_proxy(
            torch.Tensor.repeat,
            ttir.RepeatOp,
            [in0],
            golden_kwargs={"repeats": dims},
            ttir_kwargs={"repeat_dimensions": dims},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def repeat_interleave(
        self, in0: Operand, in1: Operand, repeats: int, dim: int
    ) -> OpView:
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
        )

    def fill_cache(self, in0: Operand, in1: Operand, batch_offset: int = 0) -> OpView:
        cache_tensor = self._get_golden_tensor(in0)
        input_tensor = self._get_golden_tensor(in1)
        a = torch.Tensor.repeat(
            self._get_golden_tensor(in1),
            [1, 1, cache_tensor.size()[2] // input_tensor.size()[2], 1],
        )
        b = input_tensor[:, :, 0 : (cache_tensor.size()[2] % input_tensor.size()[2]), :]
        return self.op_proxy(
            torch.cat,
            ttir.FillCacheOp,
            [in0, in1],
            golden_kwargs={"tensors": (a, b), "dim": 2},
            ttir_kwargs={"batch_offset": batch_offset},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1]),
            organize_golden_args=lambda i: 0,
        )

    def update_cache(
        self, in0: Operand, in1: Operand, in2: Operand, batch_offset: int = 0
    ) -> OpView:
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
        )

    def broadcast(
        self, in0: Operand, in1: Operand, broadcast_dimensions: List[int]
    ) -> OpView:
        return self.op_proxy(
            torch.broadcast_to,
            ttir.BroadcastOp,
            [in0],
            golden_kwargs={"size": self.get_shape(in1)},
            ttir_kwargs={"broadcast_dimensions": broadcast_dimensions},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def conv2d(
        self,
        in0: Operand,
        weight: Operand,
        bias: Optional[Operand],
        in1: Operand,
        stride: Union[IntegerAttr, DenseI32ArrayAttr],
        padding: Union[IntegerAttr, DenseI32ArrayAttr],
        dilation: Union[IntegerAttr, DenseI32ArrayAttr],
        groups: int,
    ) -> OpView:
        if not bias:
            bias = None
        return self.op_proxy(
            self.conv2d_golden_function,
            ttir.Conv2dOp,
            [in0, weight],
            golden_kwargs={
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
            },
            ttir_kwargs={
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
                "bias": bias,
            },
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1], o),
        )

    def conv2d_golden_function(
        self,
        input_tensor: Operand,
        weight: Operand,
        stride: Union[IntegerAttr, DenseI32ArrayAttr],
        padding: Union[IntegerAttr, DenseI32ArrayAttr],
        dilation: Union[IntegerAttr, DenseI32ArrayAttr],
        groups: int,
    ) -> Operand:
        # Reorganize ttir_kwargs into golden_kwargs
        stride = tuple(stride) if not isinstance(stride, IntegerAttr) else int(stride)
        padding = (
            tuple(padding) if not isinstance(padding, IntegerAttr) else int(padding)
        )
        dilation = (
            tuple(dilation) if not isinstance(dilation, IntegerAttr) else int(dilation)
        )
        golden_bias = torch.rand((weight.size()[0]), dtype=input_tensor.dtype)

        # Reorganize input and output tensors, golden and ttir functions have different expected tensor shapes
        input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
        result = torch.nn.functional.conv2d(
            input_tensor,
            weight,
            bias=golden_bias,
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
        stride: Union[IntegerAttr, DenseI32ArrayAttr],
        padding: Union[IntegerAttr, DenseI32ArrayAttr],
        output_padding: Union[IntegerAttr, DenseI32ArrayAttr],
        dilation: Union[IntegerAttr, DenseI32ArrayAttr],
        groups: int,
    ) -> OpView:
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
                "stride": stride,
                "padding": padding,
                "output_padding": output_padding,
                "dilation": dilation,
                "groups": groups,
                "bias": bias,
            },
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1], o),
        )

    def conv_transpose2d_golden_function(
        self,
        input_tensor: Operand,
        weight: Operand,
        stride: Union[IntegerAttr, DenseI32ArrayAttr],
        padding: Union[IntegerAttr, DenseI32ArrayAttr],
        output_padding: Union[IntegerAttr, DenseI32ArrayAttr],
        dilation: Union[IntegerAttr, DenseI32ArrayAttr],
        groups: int,
    ) -> Operand:
        # Reorganize ttir_kwargs into golden_kwargs
        stride = tuple(stride) if not isinstance(stride, IntegerAttr) else int(stride)
        padding = (
            tuple(padding) if not isinstance(padding, IntegerAttr) else int(padding)
        )
        output_padding = (
            tuple(output_padding)
            if not isinstance(output_padding, IntegerAttr)
            else int(output_padding)
        )
        dilation = (
            tuple(dilation) if not isinstance(dilation, IntegerAttr) else int(dilation)
        )
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
    ) -> OpView:
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
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

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
        maxpool_object = torch.nn.MaxPool2d(
            kernel_size, stride, padding, dilation, ceil_mode
        )
        input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
        result = maxpool_object(input_tensor)
        result = result.transpose(-3, -2).transpose(-2, -1)
        return result

    def reshape(self, in0: Operand, shape: Shape) -> OpView:
        kwargs = {"shape": shape}
        return self.op_proxy(
            torch.reshape,
            ttir.ReshapeOp,
            [in0],
            ttir_kwargs=kwargs,
            golden_kwargs=kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def pad(self, in0: Operand, padding: List[int], value: int) -> OpView:
        # Reformatting padding dimensions for golden tensor:
        golden_padding = []
        for i in range(int(len(padding) / 2)):
            golden_padding.append(padding[-((2 * i) + 2)])
            golden_padding.append(padding[-((2 * i) + 1)])
        return self.op_proxy(
            torch.nn.functional.pad,
            ttir.PadOp,
            [in0],
            golden_kwargs={"pad": golden_padding, "mode": "constant", "value": value},
            ttir_kwargs={"padding": padding, "value": value},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0]),
        )

    def select(
        self,
        in0: Operand,
        dim: int = 0,
        begin: int = 0,
        length: int = 2,
        stride: Optional[int] = None,
    ) -> OpView:
        end = begin + length - 1
        index = torch.tensor([begin, end])
        # TODO: handle stride. Issue #2488
        if stride:
            pass
        return self.op_proxy(
            torch.index_select,
            ttir.SelectOp,
            [in0],
            golden_kwargs={"dim": dim, "index": index},
            ttir_kwargs={
                "dim": dim,
                "begin": begin,
                "length": length,
                "stride": stride,
            },
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def index(
        self, in0: Operand, dim: int = 0, begin: int = 0, end: int = 3, step: int = 1
    ) -> OpView:
        index = torch.tensor([begin, end, step])
        return self.op_proxy(
            torch.index_select,
            ttir.IndexOp,
            [in0],
            golden_kwargs={"dim": dim, "index": index},
            ttir_kwargs={"dim": dim, "begin": begin, "end": end, "step": step},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def squeeze(self, in0: Operand, dim: Optional[int] = 0) -> OpView:
        kwargs = {"dim": dim}
        return self.op_proxy(
            torch.squeeze,
            ttir.SqueezeOp,
            [in0],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def unsqueeze(self, in0: Operand, dim: Optional[int] = 0) -> OpView:
        kwargs = {"dim": dim}
        return self.op_proxy(
            torch.unsqueeze,
            ttir.UnsqueezeOp,
            [in0],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def clamp_scalar(
        self,
        in0: Operand,
        min_arg: Optional[float] = None,
        max_arg: Optional[float] = None,
    ) -> OpView:
        kwargs = {"min": min_arg, "max": max_arg}
        return self.op_proxy(
            torch.clamp,
            ttir.ClampScalarOp,
            [in0],
            ttir_kwargs=kwargs,
            golden_kwargs=kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def clamp_tensor(
        self,
        in0: Operand,
        in1: Operand,
        in2: Operand,
        in3: Operand,
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
        )

    def zeros(self, shapes: List[Shape], data_type: Optional[Type] = None) -> OpView:
        output = self.ranked_tensor_type(shapes)
        dtype = data_type if data_type is not None else self._default_dtype
        return self.op_proxy(
            torch.zeros,
            ttir.ZerosOp,
            [],
            golden_kwargs={"size": shapes},
            ttir_kwargs={"result": output, "shape": shapes},
            organize_ttir_args=lambda i, o, shape: 0,
            output_type=dtype,
        )

    def ones(self, shapes: List[Shape]) -> OpView:
        output = self.ranked_tensor_type(shapes)
        return self.op_proxy(
            torch.ones,
            ttir.OnesOp,
            [],
            golden_kwargs={"size": shapes},
            ttir_kwargs={"result": output, "shape": shapes},
            organize_ttir_args=lambda i, o, shape: 0,
        )

    def reverse(self, in0: Operand, dims: List[int]) -> OpView:
        return self.op_proxy(
            torch.flip,
            ttir.ReverseOp,
            [in0],
            golden_kwargs={"dims": dims},
            ttir_kwargs={"dimensions": dims},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def linear(
        self,
        in0: Operand,
        in1: Operand,
        bias: Optional[Operand] = None,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ) -> OpView:
        kwargs = {"transpose_a": transpose_a, "transpose_b": transpose_b, "bias": bias}
        return self.op_proxy(
            self.linear_golden_function,
            ttir.LinearOp,
            [in0, in1],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            organize_ttir_args=lambda i, o, shape: (self._get_type(o), i[0], i[1], o),
        )

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
        self, in0: Operand, in1: Operand, bias: Optional[Operand] = None
    ) -> OpView:
        inputs = [in0, in1]
        if bias:
            inputs.append(bias)
        return self.op_proxy(
            torch.matmul,
            ttir.MatmulOp,
            inputs,
            organize_ttir_args=lambda i, o, shape: (self._get_type(o), i[0], i[1], o),
        )

    def permute(
        self, in0: Operand, in1: Operand, permutation: DenseI64ArrayAttr
    ) -> OpView:
        return self.op_proxy(
            torch.permute,
            ttir.PermuteOp,
            [in0, in1],
            golden_kwargs={"dims": tuple(permutation)},
            ttir_kwargs={"permutation": permutation},
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            organize_ttir_args=lambda i, o, _: (self._get_type(i[1]), i[0], i[1]),
        )

    def upsample2d(
        self,
        in0: Operand,
        in1: Operand,
        scale_factor: Union[SI32Attr, DenseI32ArrayAttr],
        mode: str = "nearest",
    ) -> OpView:
        golden_scale_factor = (
            tuple(scale_factor) if not isinstance(scale_factor, int) else scale_factor
        )
        upsample_obj = torch.nn.Upsample(scale_factor=golden_scale_factor, mode=mode)
        return self.op_proxy(
            upsample_obj,
            ttir.Upsample2dOp,
            [in0, in1],
            ttir_kwargs={"scale_factor": scale_factor, "mode": mode},
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            organize_ttir_args=lambda i, o, _: (self._get_type(i[1]), i[0], i[1]),
        )

    def arange(
        self, result=Operand, start=int, end=int, step=int, arange_dimension=int
    ) -> OpView:
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
        )

    # TTIR top level generic ops
    # class TTIR_GenericElementwiseUnaryOp

    def exp(self, in0: Operand) -> OpView:
        return self.eltwise_proxy(torch.exp, ttir.ExpOp, [in0])

    # class TTIR_GenericElementwiseBinaryOp

    def add(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.add, ttir.AddOp, [in0, in1])

    def multiply(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.multiply, ttir.MultiplyOp, [in0, in1])

    def div(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.div, ttir.DivOp, [in0, in1])

    def maximum(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.maximum, ttir.MaximumOp, [in0, in1])
