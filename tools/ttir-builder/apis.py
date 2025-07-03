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
from .ops import TTIRBuilderOps

# Alias for operands of ops which can be either BlockArguments, Values, or other
# ops wrapped in OpView or Operation.
Operand = Union[Value, OpView, Operation]

# Convenience alias for shape
Shape = Union[List[int], Tuple[int, ...]]


def autodoc_skip(func):
    func.__autodoc_skip__ = True
    return func


def get_loc_of_extra_file_callee(id: int = 0) -> Location:
    """
    When called, this function returns a `Location` referring to first
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
    """
    Converts a string location or Location object to a Location object.

    Parameters
    ----------
    loc : *Union[str, Location]*
        Either a string representing a location or an existing Location object

    Returns
    -------
    Location
        The resulting Location object
    """
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
    seed: Optional[int] = None

    def __repr__(self) -> str:
        s = f"\nRandom seed: {self.seed}" if self.seed is not None else ""
        s += f"\nGolden tensor:\n{self.tensor}"
        return s

    def contiguous(self) -> Golden:
        return Golden(self.tensor.contiguous())


@dataclass
class TypeInfo:
    """
    Encapsulates type information for quantized tensors.

    Parameters
    ----------
        dtype : torch.dtype
            Base PyTorch data type (e.g. torch.float32, torch.qint32).
        scale : *Optional[float]*
            Scaling factor for quantization. Required for quantized types.
        zero_point : *Optional[int]*
            Zero point offset for quantization. Required for quantized types.
    """

    dtype: torch.dtype
    scale: Optional[float] = None
    zero_point: Optional[int] = None


class GoldenCheckLevel(Enum):
    DISABLED = auto()  # Do not store golden.
    OP_LEVEL = auto()  # Check every single op level goldens
    GRAPH_LEVEL = auto()  # Check graph level goldens only


class TTIRBuilder(TTIRBuilderOps):
    """Builder class providing APIs for creating TTIR ops."""

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
        """
        Returns
        -------
        Dict
            Dictionary mapping operands to their golden tensors
        """
        return self._goldens

    @property
    def golden_check_level(self) -> GoldenCheckLevel:
        """
        Returns
        -------
        GoldenCheckLevel
            Current golden check level
        """
        return self._golden_check_level

    @golden_check_level.setter
    def golden_check_level(self, level: GoldenCheckLevel):
        """
        Sets golden check level.
        Parameters
        ----------
        level : GoldenCheckLevel
            The validation level to set
        """
        if not isinstance(level, GoldenCheckLevel):
            raise ValueError("Invalid golden check level.")
        self._golden_check_level = level

    def get_context(self) -> Context:
        """
        Gets MLIR context.
        Returns
        -------
        Context
            The MLIR context
        """
        return self._ctx

    @autodoc_skip
    def get_next_global_id(self) -> int:
        """
        Gets next global identifier.
        Returns
        -------
        int
            The next global identifier
        """
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
        """
        Gets tensor shape.
        Parameters
        ----------
        input : Operand
            The operand whose shape to retrieve
        Returns
        -------
        Shape
            The shape of the operand
        """
        return self._get_type(input).shape

    def generate_and_store_random_golden(
        self, operand: Operand, dtype: Union[torch.dtype, TypeInfo] = torch.float32
    ) -> Golden:
        """
        Creates and stores a random golden tensor.
        Parameters
        ----------
        operand : Operand
            The operand to generate and store a golden for
        dtype : *Union[torch.dtype, TypeInfo]*, optional
            Data type of the golden tensor (default: torch.float32)
        Returns
        -------
        Golden
            The generated golden tensor
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
        Parameters
        ----------
        operand : Operand
            The input operand to generate a golden for
        dtype : *Union[torch.dtype, TypeInfo]*
            Data type of the golden tensor
        index : int
            Index to use for mapping the golden
        override : bool, optional
            Whether to override existing golden (default: False)
        Returns
        -------
        Golden
            The generated golden tensor
        """
        if not override and f"input_{index}" in self.id_golden_map:
            return self.id_golden_map[f"input_{index}"]
        golden = self.generate_and_store_random_golden(operand, dtype)
        self.id_golden_map[f"input_{index}"] = golden
        return golden

    def get_golden_map(self) -> Dict:
        """
        Gets the golden tensor mapping.
        Returns
        -------
        Dict
            Mapping of golden tensor names to their GoldenTensor objects
        """
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

    @autodoc_skip
    def set_mesh_shape(self, mesh_shape: Tuple[int, int]):
        """
        Sets the mesh shape for multi-device operations.
        Parameters
        ----------
        mesh_shape : *Tuple[int, int]*
            A tuple of (rows, columns) specifying the 2D mesh arrangement of devices
        """
        self.mesh_shape = mesh_shape

    def set_graph_input_output(
        self,
        inputs: List[torch.Tensor],
        outputs: Optional[List[torch.Tensor]] = None,
        override: bool = False,
    ) -> None:
        """
        Records the input and output tensors for the graph.
        Creates golden tensors for inputs and optionally for outputs.
        Can override existing golden tensors if specified.
        Parameters
        ----------
        inputs : *List[torch.Tensor]*
            List of input tensors for the graph
        outputs : *Optional[List[torch.Tensor]]*, optional
            List of output tensors for the graph (default: None)
        override : bool, optional
            Whether to override existing golden tensors (default: False)
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

    @autodoc_skip
    def get_datatype_from_torch_dtype(self, dtype: torch.dtype) -> DataType:
        """
        Returns a MLIR `DataType` obj corresponding to `dtype`.
        Parameters
        ----------
        dtype : torch.dtype
            The PyTorch data type to convert
        Returns
        -------
        DataType
            The corresponding MLIR data type
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

    @autodoc_skip
    def get_type_from_torch_dtype(
        self,
        dtype: Union[torch.dtype, TypeInfo],
        scale: Optional[float] = None,
        zero_point: Optional[float] = None,
    ) -> Type:
        """
        Converts PyTorch dtype or TypeInfo to corresponding MLIR Type.
        For quantized types (e.g. qint32), scale and zero_point must be provided via TypeInfo.
        For non-quantized types, a plain torch.dtype can be used.
        Args:
            dtype: Either a torch.dtype or TypeInfo containing dtype and quantization params.
        scale : *Optional[float]*
            Scaling factor for quantization. Required for quantized types.
        zero_point : *Optional[int]*
            Zero point offset for quantization. Required for quantized types.
        Returns:
            MLIR Type corresponding to the input dtype.
        """
        if scale and zero_point:
            dtype = TypeInfo(dtype=dtype, scale=scale, zero_point=zero_point)
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
        """
        Convenience wrapper constructing `RankedTensorType`
        Parameters
        ----------
        shape : Shape
            The shape of the tensor type
        data_type : *Optional[Type]*, optional
            The data type of the tensor (default: None)
        encoding : *Optional[Attribute]*, optional
            Optional encoding attribute (default: None)
        Returns
        -------
        RankedTensorType
            The created ranked tensor type
        """
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
        """
        Creates a metal tensor layout with attributes including grid,
        tiling, memory space, collapse intervals, and out-of-bounds value handling..
        Parameters
        ----------
        shape : Shape
            The shape of the tensor
        grid : *Union[List, Tuple, ttcore.ir.GridAttr]*
            Grid specification for the layout
        tiled : bool, optional
            Whether the layout is tiled (default: False)
        memorySpace : ttcore.MemorySpace, optional
            Memory space for the tensor (default: DeviceL1)
        collapseIntervals : *List[Tuple[int, int]]*, optional
            Intervals to collapse (default: [(0, -1)])
        oobVal : ttcore.OOBVal, optional
            Out-of-bounds value handling (default: Undef)
        Returns
        -------
        RankedTensorType
            Tensor type with metal layout attributes
        """
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

    @autodoc_skip
    def empty_from_tensor_type(
        self, shape: Shape, tensor_type: RankedTensorType
    ) -> OpView:
        """
        Convenience wrapper constructing ``ttir.EmptyOp``
        Parameters
        ----------
        shape : Shape
            The shape of the empty tensor
        tensor_type : RankedTensorType
            The type of the tensor to create
        Returns
        -------
        OpView
            The created empty tensor operation
        """
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

    @autodoc_skip
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
        Create and return a TTIR operation using the provided golden and TTIR functions.
        Parameters
        ----------
        op_golden_function : Callable
            Function that creates the operation using golden approach
        op_ttir_function : Callable
            Function that creates the operation using TTIR approach
        inputs : *List[Operand]*
            List of input operands for the operation
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes (default: None)
        organize_ttir_args : *Optional[Callable]*, optional
            Function to organize TTIR arguments (default: None)
        organize_golden_args : *Optional[Callable]*, optional
            Function to organize golden arguments (default: None)
        output_shape : *Optional[Shape]*, optional
            Shape of the output tensor (default: None)
        output_type : *Optional[Type]*, optional
            Type of the output tensor (default: None)
        output_create_fn : *Optional[Callable]*, optional
            Function to create output tensor (default: None)
        golden_kwargs : dict, optional
            Additional keyword arguments for golden function (default: {})
        ttir_kwargs : dict, optional
            Additional keyword arguments for TTIR function (default: {})
        loc : *Optional[Union[str, Location]]*, optional
            Source location information (default: None)
        Returns
        -------
        Any
            The created operation
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

    @autodoc_skip
    def eltwise_proxy(
        self,
        op_golden_function: Callable,
        op_ttir_function: Callable,
        inputs: List[Operand],
        unit_attrs: Optional[List[str]] = None,
    ) -> OpView:
        """
        Creates elementwise TTIR operations.
        Parameters
        ----------
        op_golden_function : Callable
            Function that creates the operation using golden approach
        op_ttir_function : Callable
            Function that creates the operation using TTIR approach
        inputs : *List[Operand]*
            List of input operands for the operation
        unit_attrs : *Optional[List[str]]*, optional
            Optional list of unit attributes (default: None)
        Returns
        -------
        OpView
            The created elementwise operation
        """
        return self.op_proxy(op_golden_function, op_ttir_function, inputs, unit_attrs)

    @autodoc_skip
    def ccl_proxy(
        self,
        op_golden_function: Callable,
        op_ttir_function: Callable,
        inputs: List[Operand],
        kwargs: dict = {},
    ) -> OpView:
        """
        Creates CCL TTIR operations. Forces
        golden check level to GRAPH_LEVEL and provides specialized argument
        organization for CCL operations.
        Parameters
        ----------
        op_golden_function : Callable
            Function that creates the operation using golden approach
        op_ttir_function : Callable
            Function that creates the operation using TTIR approach
        inputs : *List[Operand]*
            List of input operands for the operation
        kwargs : dict, optional
            Additional keyword arguments for both golden and TTIR functions (default: {})
        Returns
        -------
        OpView
            The created CCL operation
        """
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


# Remove autodoc_skip from Sphinx documentation
del autodoc_skip
