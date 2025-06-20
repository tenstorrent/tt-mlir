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


def get_loc_from_str(loc: Union[str, Location]) -> Location:
    if isinstance(loc, str):
        return Location.name(loc)
    else:
        return loc


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
    seed: Opional[int] = None

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


class GoldenCheckLevel(Enum):
    DISABLED = auto()  # Do not store golden.
    OP_LEVEL = auto()  # Check every single op level goldens
    GRAPH_LEVEL = auto()  # Check graph level goldens only


class TTIRBuilder:
    """Builder class providing API for creating TTIR ops."""

    def __init__(self, ctx: Context, location: Location):
        self._ctx = ctx
        self._loc = location

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

        # golden check level
        self._golden_check_level = GoldenCheckLevel.OP_LEVEL

    # ----- Public helpers -----

    @property
    def goldens(self) -> Dict:
        return self._goldens

    @property
    def golden_check_level(self) -> GoldenCheckLevel:
        return self._golden_check_level

    @golden_check_level.setter
    def golden_check_level(self, level: GoldenCheckLevel):
        if not isinstance(level, GoldenCheckLevel):
            raise ValueError("Invalid golden check level.")
        self._golden_check_level = level

    def get_context(self) -> Context:
        return self._ctx

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
        self, operand: Operand, dtype: Union[torch.dtype, TypeInfo] = torch.float32
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
        self,
        operand: Operand,
        dtype: Union[torch.dtype, TypeInfo],
        index: int,
        override: bool = False,
    ) -> None:
        """
        Generates random tensor of `dtype`s of `input`s shape, assigns it to a golden,
        and maps `input` to that golden.
        """
        if not override and f"input_{index}" in self.id_golden_map:
            return self.id_golden_map[f"input_{index}"]
        golden = self.generate_and_store_random_golden(operand, dtype)
        self.id_golden_map[f"input_{index}"] = golden
        return golden

    def get_golden_map(self) -> Dict:
        golden_info = {}
        if self.golden_check_level == GoldenCheckLevel.DISABLED:
            return golden_info
        for name, golden_tensor in self.id_golden_map.items():
            if self.golden_check_level == GoldenCheckLevel.GRAPH_LEVEL:
                if re.match(r"^(input|output)_[0-9]+$", name) is None:
                    # It means this is not graph level golden.
                    continue
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
        self,
        inputs: List[torch.Tensor],
        outputs: Optional[List[torch.Tensor]] = None,
        override: bool = False,
    ) -> None:
        """
        Records the input and output tensors for the graph.
        """
        for index, tensor in enumerate(inputs):
            input_key = f"input_{index}"
            if input_key in self.id_golden_map:
                assert self.id_golden_map[input_key].tensor.shape == tensor.shape
                assert self.id_golden_map[input_key].tensor.dtype == tensor.dtype
            if not override and input_key in self.id_golden_map:
                continue
            self.id_golden_map[input_key] = Golden(tensor)

        if outputs is not None:
            self.golden_check_level = GoldenCheckLevel.GRAPH_LEVEL
            for index, tensor in enumerate(outputs):
                output_key = f"output_{index}"
                if not override and output_key in self.id_golden_map:
                    continue
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
        shape: Shape, dtype: Union[torch.dtype, TypeInfo], seed: int
    ) -> torch.Tensor:
        """
        Generates random tensor of shape `shape`, with type `dtype`, using `seed` to seed torch
        random generator.
        """
        if isinstance(dtype, TypeInfo):
            # Generate float tensor and quantize it.
            float_tensor = torch.randn(
                shape, generator=torch.manual_seed(seed), dtype=torch.float32
            )
            return torch.quantize_per_tensor(
                float_tensor, dtype.scale, dtype.zero_point, dtype.dtype
            )
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
            case torch.int32 | torch.qint32:
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

    def metal_tensor_layout(
        self,
        shape: Shape,
        tilize=False,
        oobVal=ttcore.OOBVal.Undef,
        memorySpace=ttcore.MemorySpace.DeviceL1,
    ):
        ctx = self._ctx

        # Create layout with original logical shape.
        layout = ttcore.ir.MetalLayoutAttr.get(ctx, shape, oobVal, memorySpace)

        # Then shard the new shape by adding 1-filled grid dims.
        original_rank = len(shape)
        extended_shape = [1] * original_rank + list(shape)

        elemType = F32Type.get(ctx)

        if tilize:
            elemType = ttcore.ir.TileType.get(ctx, 32, 32, ttcore.DataType.Float32)
            extended_shape[-2] //= 32
            extended_shape[-1] //= 32

        return RankedTensorType.get(
            extended_shape, elemType, layout, Location.unknown(ctx)
        )

    def empty_from_tensor_type(
        self, shape: Shape, tensor_type: RankedTensorType
    ) -> OpView:
        """Convenience wrapper constructing `ttir.EmptyOp`."""
        with self._ctx, self._loc:
            op = ttir.EmptyOp(tensor_type)
            self.generate_and_store_random_golden(op)
            return op

    def _empty(self, shape: Shape, data_type: Optional[Type] = None) -> OpView:
        """Convenience wrapper constructing `ttir.EmptyOp`."""
        dtype = data_type if data_type is not None else self._default_dtype
        return self.empty_from_tensor_type(shape, self.ranked_tensor_type(shape, dtype))

    # ----- TTIR op factories -----
    def _organize_eltwise_ttir(
        self, inputs: List[Operand], output: OpView, _: Optional[Shape]
    ):
        return (self._get_type(output), *inputs, output)

    def _organize_eltwise_golden(self, inputs: List[Operand]):
        return [self._get_golden_tensor(inp) for inp in inputs]

    def op_proxy(
        self,
        op_golden_function: Callable,
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

            if output_create_fn:
                output = output_create_fn(output_shape, output_type)
            else:
                output = self._empty(output_shape, output_type)
            id = self.get_next_global_id()
            loc = (
                get_loc_from_str(loc)
                if loc is not None
                else get_loc_of_extra_file_callee(id=id)
            )
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
            if unit_attrs is not None:
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
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        return self.op_proxy(op_golden_function, op_ttir_function, inputs, unit_attrs)

    def ccl_proxy(
        self,
        op_golden_function: Callable,
        op_ttir_function: Callable,
        inputs: List[Operand],
        kwargs: dict = {},
    ) -> OpView:
        # Force GoldenCheckLevel to GRAPH_LEVEL when CCL Ops are used(phase 0)
        self.golden_check_level = GoldenCheckLevel.GRAPH_LEVEL
        return self.op_proxy(
            op_golden_function=op_golden_function,
            op_ttir_function=op_ttir_function,
            inputs=inputs,
            organize_golden_args=lambda i: (
                [self._get_golden_tensor(i[0]), self.mesh_shape]
            ),
            organize_ttir_args=lambda i, o, shape: (
                self._get_type(o),
                i[0],
                o,
            ),
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
        )

    # TTIR top level ops

    def get_dimension_size(
        self, in0: Operand, dimension: int = 0, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
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
        return self.eltwise_proxy(torch.abs, ttir.AbsOp, [in0], unit_attrs)

    def cbrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
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
        return self.eltwise_proxy(torch.ceil, ttir.CeilOp, [in0], unit_attrs)

    def cos(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.cos, ttir.CosOp, [in0], unit_attrs)

    def floor(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.floor, ttir.FloorOp, [in0], unit_attrs)

    def gelu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(
            torch.nn.functional.gelu, ttir.GeluOp, [in0], unit_attrs
        )

    def is_finite(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.isfinite, ttir.IsFiniteOp, [in0], unit_attrs)

    def logical_not(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        golden = self._get_golden_tensor(in0)
        golden_output = torch.empty(golden.shape, dtype=golden.dtype)
        return self.op_proxy(
            torch.logical_not,
            ttir.LogicalNotOp,
            [in0],
            golden_kwargs={"out": golden_output},
            unit_attrs=unit_attrs,
        )

    def bitwise_not(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        return self.eltwise_proxy(
            torch.bitwise_not, ttir.BitwiseNotOp, [in0], unit_attrs
        )

    def neg(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.neg, ttir.NegOp, [in0], unit_attrs)

    # NOTE: See issue #1719 for information on golden PCC fail
    def tan(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.tan, ttir.TanOp, [in0], unit_attrs)

    def atan(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.atan, ttir.AtanOp, [in0], unit_attrs)

    def tanh(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.tanh, ttir.TanhOp, [in0], unit_attrs)

    def reciprocal(
        self, in0: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        return self.eltwise_proxy(
            torch.reciprocal, ttir.ReciprocalOp, [in0], unit_attrs
        )

    def relu(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.relu, ttir.ReluOp, [in0], unit_attrs)

    def rsqrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.rsqrt, ttir.RsqrtOp, [in0], unit_attrs)

    def sigmoid(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.sigmoid, ttir.SigmoidOp, [in0], unit_attrs)

    def sign(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.sign, ttir.SignOp, [in0], unit_attrs)

    def sin(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.sin, ttir.SinOp, [in0], unit_attrs)

    def sqrt(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.sqrt, ttir.SqrtOp, [in0], unit_attrs)

    def typecast(
        self, in0: Operand, out: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
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
        return self.eltwise_proxy(torch.log, ttir.LogOp, [in0], unit_attrs)

    def log1p(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.log1p, ttir.Log1pOp, [in0], unit_attrs)

    def expm1(self, in0: Operand, unit_attrs: Optional[List[str]] = None) -> OpView:
        return self.eltwise_proxy(torch.expm1, ttir.Expm1Op, [in0], unit_attrs)

    # class TTIR_ElementwiseUnaryWithFloatParameterOp

    def leaky_relu(
        self,
        in0: Operand,
        parameter: float = 0.01,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
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
        return self.op_proxy(
            torch.ne,
            ttir.NotEqualOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def ge(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
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
        return self.eltwise_proxy(
            torch.bitwise_and, ttir.BitwiseAndOp, [in0, in1], unit_attrs=unit_attrs
        )

    def bitwise_or(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        return self.eltwise_proxy(
            torch.bitwise_or, ttir.BitwiseOrOp, [in0, in1], unit_attrs=unit_attrs
        )

    def bitwise_xor(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        return self.eltwise_proxy(
            torch.bitwise_xor, ttir.BitwiseXorOp, [in0, in1], unit_attrs=unit_attrs
        )

    def minimum(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        return self.eltwise_proxy(
            torch.minimum, ttir.MinimumOp, [in0, in1], unit_attrs=unit_attrs
        )

    def subtract(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        return self.eltwise_proxy(
            torch.subtract, ttir.SubtractOp, [in0, in1], unit_attrs=unit_attrs
        )

    def remainder(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        return self.eltwise_proxy(
            torch.remainder, ttir.RemainderOp, [in0, in1], unit_attrs=unit_attrs
        )

    def pow(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
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
    def reduce_and(
        self,
        in0: Operand,
        keep_dim: bool = True,
        dim_args: Optional[List] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        return self.op_proxy(
            torch.all,
            ttir.ReduceAndOp,
            [in0],
            golden_kwargs={"dim": tuple(dim_args), "keepdim": keep_dim},
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
        return self.op_proxy(
            # torch.softmax,
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
        stride: Optional[int] = None,
        unit_attrs: Optional[List[str]] = None,
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
        kwargs = {"transpose_a": transpose_a, "transpose_b": transpose_b, "bias": bias}
        return self.op_proxy(
            self.linear_golden_function,
            ttir.LinearOp,
            [in0, in1],
            golden_kwargs=kwargs,
            ttir_kwargs=kwargs,
            unit_attrs=unit_attrs,
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
        return self.eltwise_proxy(torch.exp, ttir.ExpOp, [in0], unit_attrs=unit_attrs)

    # class TTIR_GenericElementwiseBinaryOp

    def add(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        return self.eltwise_proxy(
            torch.add,
            ttir.AddOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def multiply(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        return self.eltwise_proxy(
            torch.multiply,
            ttir.MultiplyOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def subtract(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        return self.eltwise_proxy(
            torch.sub,
            ttir.SubtractOp,
            [in0, in1],
            unit_attrs=unit_attrs,
        )

    def div(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
        return self.eltwise_proxy(
            torch.div, ttir.DivOp, [in0, in1], unit_attrs=unit_attrs
        )

    def maximum(
        self, in0: Operand, in1: Operand, unit_attrs: Optional[List[str]] = None
    ) -> OpView:
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
        golden_kwargs = {"scale": scale, "zero_point": zero_point, "dtype": dtype}
        return self.op_proxy(
            lambda *args, **kwargs: torch.quantize_per_tensor(
                *args, **kwargs
            ).int_repr(),
            ttir.QuantizeOp,
            [in0],
            golden_kwargs=golden_kwargs,
            output_type=self.get_type_from_torch_dtype(
                TypeInfo(dtype=dtype, scale=scale, zero_point=zero_point)
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
        golden_kwargs = {"scale": scale, "zero_point": zero_point, "dtype": dtype}
        return self.op_proxy(
            lambda *args, **kwargs: torch.quantize_per_tensor(
                torch.dequantize(args[0]), **kwargs
            ),
            ttir.RequantizeOp,
            [in0],
            golden_kwargs=golden_kwargs,
            output_type=self.get_type_from_torch_dtype(
                TypeInfo(dtype=dtype, scale=scale, zero_point=zero_point)
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
        return self.op_proxy(
            lambda *args, **kwargs: args[0],
            ttir.ViewLayoutOp,
            [in0],
            ttir_kwargs={"reinterpretLayout": reinterpret_layout},
            output_type=output_type,
            output_create_fn=self.empty_from_tensor_type,
            organize_ttir_args=lambda i, o, _: (
                self._get_type(o),
                i[0],
            ),
            unit_attrs=unit_attrs,
        )

    def tilize(
        self,
        in0: Operand,
        output_type: RankedTensorType,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
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
        kwargs = {
            "source_target_pairs": source_target_pairs,
        }
        return self.ccl_proxy(
            collective_permute_golden,
            ttir.CollectivePermuteOp,
            [input],
            kwargs=kwargs,
        )
