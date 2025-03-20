# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Callable, Dict, Any
from ttmlir.ir import *
from ttmlir.dialects import ttir, tt, tensor
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
            golden_info[name] = GoldenTensor(
                name,
                list(golden_tensor.tensor.shape),
                list(golden_tensor.tensor.stride()),
                DataType.Float32,
                golden_tensor.tensor.data_ptr(),
                golden_tensor.tensor.numel() * golden_tensor.tensor.dtype.itemsize,
            )
        return golden_info

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

    def get_type_from_torch_dtype(self, dtype: torch.dtype) -> Type:
        """
        Returns a MLIR `Type` obj corresponding to `dtype`
        """
        match dtype:
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
            case _:
                raise TypeError(f"Invalid Type {type}")

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
        """Convenience wrapper constructing `tensor.EmptyOp`."""
        dtype = data_type if data_type is not None else self._default_dtype
        with self._ctx, self._loc:
            op = tensor.EmptyOp(shape, dtype)

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

            # Use the golden output to determine proper output shape and type unless otherwise specified
            output_shape = golden.tensor.shape if not output_shape else output_shape
            output_type = self._default_dtype if not output_type else output_type
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
        return self.op_proxy(
            self.get_dimension_size_golden_function,
            ttir.GetDimensionSizeOp,
            [in0],
            golden_kwargs={"dimension": dimension},
            ttir_kwargs={"dimension": dimension},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0]),
            output_type=self.get_type_from_torch_dtype(torch.int32),
        )

    def get_dimension_size_golden_function(
        self, in0: Operand, dimension: int = 0
    ) -> torch.Tensor:
        return torch.tensor([in0.size(dimension)])

    # TTIR top level named ops
    # class TTIR_ElementwiseTernaryOp

    # NOTE torch.where doesn't take the same inputs as ttir.WhereOp, needs more extensive configuration
    def where(
        self, in0: Operand, in1: Operand, in2: Operand, operandSegmentSizes=List[int]
    ) -> OpView:
        return self.eltwise_proxy(torch.where, ttir.WhereOp, [in0, in1, in2])

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

    def typecast(self, in0: Operand, in1: Operand) -> OpView:
        return self.op_proxy(
            torch.Tensor.type,
            ttir.TypecastOp,
            [in0, in1],
            golden_kwargs={"dtype": self._get_golden_tensor(in0).type()},
            organize_ttir_args=lambda i, o, shape: ([self._get_type(o)], [i[0]], o),
            organize_golden_args=lambda i: [self._get_golden_tensor(i[1])],
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

    def power(self, in0: Operand, in1: Operand) -> OpView:
        return self.eltwise_proxy(torch.pow, ttir.PowerOp, [in0, in1])

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
        )

    def sum(
        self, in0: Operand, dim_arg: List[int] = [0], keep_dim: bool = True
    ) -> OpView:
        golden_kwargs = {"dim": dim_arg, "keepdim": keep_dim}
        ttir_kwargs = {"dim_arg": dim_arg, "keep_dim": keep_dim}
        return self.op_proxy(
            torch.sum,
            ttir.SumOp,
            [in0],
            golden_kwargs=golden_kwargs,
            ttir_kwargs=ttir_kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def mean(
        self, in0: Operand, dim_arg: List[int] = [0], keep_dim: bool = True
    ) -> OpView:
        golden_kwargs = {"dim": dim_arg, "keepdim": keep_dim}
        ttir_kwargs = {"dim_arg": dim_arg, "keep_dim": keep_dim}
        return self.op_proxy(
            torch.mean,
            ttir.MeanOp,
            [in0],
            golden_kwargs=golden_kwargs,
            ttir_kwargs=ttir_kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def max(self, in0: Operand, dim_arg: int = 0, keep_dim: bool = True) -> OpView:
        golden_kwargs = {"dim": dim_arg, "keepdim": keep_dim}
        ttir_kwargs = {"dim_arg": [dim_arg], "keep_dim": keep_dim}
        return self.op_proxy(
            torch.max,
            ttir.MaxOp,
            [in0],
            golden_kwargs=golden_kwargs,
            ttir_kwargs=ttir_kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
        )

    def min(self, in0: Operand, dim_arg: int = 0, keep_dim: bool = True) -> OpView:
        golden_kwargs = {"dim": dim_arg, "keepdim": keep_dim}
        ttir_kwargs = {"dim_arg": [dim_arg], "keep_dim": keep_dim}
        return self.op_proxy(
            torch.min,
            ttir.MinOp,
            [in0],
            golden_kwargs=golden_kwargs,
            ttir_kwargs=ttir_kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
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

    def prod(
        self, in0: Operand, in1: Operand, dim_arg: List[int], keep_dim: bool = False
    ) -> OpView:
        g_kwargs = {}
        if len(dim_arg) == 1:
            g_kwargs["dim"] = dim_arg[0]
            g_kwargs["keepdim"] = keep_dim
            g_function = torch.prod
        else:
            g_function = self.prod_golden_function
        return self.op_proxy(
            g_function,
            ttir.ProdOp,
            [in0],
            golden_kwargs=g_kwargs,
            ttir_kwargs={"keep_dim": keep_dim, "dim_arg": dim_arg},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
            output_type=self.get_type_from_torch_dtype(
                self._get_golden_tensor(in1).dtype
            ),
        )

    def prod_golden_function(self, in0: Operand):
        return torch.tensor([torch.prod(in0).item()])

    def cumsum(self, in0: Operand, in1: Operand, dim: int) -> OpView:
        return self.op_proxy(
            torch.cumsum,
            ttir.CumSumOp,
            [in0, in1],
            golden_kwargs={"dim": dim},
            ttir_kwargs={"dim": dim, "output": in1},
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0]),
            organize_golden_args=lambda i: [self._get_golden_tensor(i[0])],
            output_type=self.get_type_from_torch_dtype(
                self._get_golden_tensor(in1).dtype
            ),
        )

    def embedding(self, in0: Operand, weight: Operand, in1: Operand) -> OpView:
        embedding = torch.nn.Embedding.from_pretrained(self._get_golden_tensor(weight))
        return self.op_proxy(
            embedding,
            ttir.EmbeddingOp,
            [in0, weight, in1],
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], i[1], i[2]),
            organize_golden_args=lambda i: (
                torch.ones(self._get_golden_tensor(i[0]).size(), dtype=torch.long),
            ),
            output_type=self.get_type_from_torch_dtype(
                self._get_golden_tensor(weight).dtype
            ),
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
        output_type = self.get_type_from_torch_dtype(self._get_golden_tensor(in1).dtype)
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
            output_type=output_type,
        )

    def conv2d_golden_function(
        self,
        in0: Operand,
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
        bias = torch.rand((weight.size()[0]), dtype=in0.dtype)

        # Reorganize input and output tensors, golden and ttir functions have different expected tensor shapes
        n, h_out, w_out, c_out = list(in0.size())
        input_tensor = torch.rand((n, c_out, h_out, w_out), dtype=in0.dtype)
        output_unorganized = torch.nn.functional.conv2d(
            input_tensor,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        n, c_out, h_out, w_out = list(output_unorganized.size())
        return torch.rand((n, h_out, w_out, c_out), dtype=in0.dtype)

    def conv_transpose2d(
        self,
        in0: Operand,
        weight: Operand,
        bias: Optional[Operand],
        in1: Operand,
        stride: Union[int, IntegerAttr, DenseI32ArrayAttr],
        padding: Union[IntegerAttr, DenseI32ArrayAttr],
        output_padding: Union[IntegerAttr, DenseI32ArrayAttr],
        dilation: Union[IntegerAttr, DenseI32ArrayAttr],
        groups: int,
    ) -> OpView:
        output_type = self.get_type_from_torch_dtype(self._get_golden_tensor(in1).dtype)
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
            output_type=output_type,
        )

    def conv_transpose2d_golden_function(
        self,
        in0: Operand,
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
        bias = torch.rand((weight.size()[0]), dtype=in0.dtype)

        # Reorganize input and output tensors, golden and ttir functions have different expected tensor shapes
        n, h_out, w_out, c_out = list(in0.size())
        input_tensor = torch.rand((n, c_out, h_out, w_out), dtype=in0.dtype)
        output_unorganized = torch.nn.functional.conv_transpose2d(
            input_tensor,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )
        n, c_out, h_out, w_out = list(output_unorganized.size())
        return torch.rand((n, h_out, w_out, c_out), dtype=in0.dtype)

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
        maxpool_object = torch.nn.MaxPool2d(
            kernel_size, stride, padding, dilation, ceil_mode
        )
        return maxpool_object(input_tensor)

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
        golden_padding = []
        # Reformatting padding dimensions for golden tensor:
        if len(padding) == 4:
            golden_padding = padding.copy()
            golden_padding.reverse()
        if len(padding) > 4:
            for i in range(int(len(padding) / 2) - 4):
                i = i + 4
                golden_padding.append(padding[(2 * i) + 1])
                golden_padding.append(padding[2 * i])
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

    def clamp(
        self,
        in0: Operand,
        min_arg: Optional[float] = None,
        max_arg: Optional[float] = None,
    ) -> OpView:
        kwargs = {"min": min_arg, "max": max_arg}
        return self.op_proxy(
            torch.clamp,
            ttir.ClampOp,
            [in0],
            ttir_kwargs=kwargs,
            golden_kwargs=kwargs,
            organize_ttir_args=lambda i, o, _: (self._get_type(o), i[0], o),
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
