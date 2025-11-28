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

from ttmlir.ir import (
    Context,
    Location,
    Value,
    OpView,
    Operation,
    RankedTensorType,
    Type,
    Attribute,
    BF16Type,
    F16Type,
    F32Type,
    F64Type,
    IntegerType,
)
from ttmlir.dialects import tensor, quant
from ttmlir.passes import GoldenTensor, DataType
from golden import GoldenMapTensor, get_golden_function

# ----- Public APIs -----

# Type alias for MLIR operands representing values, operations and operation views
Operand = Union[Value, OpView, Operation]

# Type alias for tensor shapes, supporting both list and tuple representations
Shape = Union[List[int], Tuple[int, ...]]


@dataclass
class TypeInfo:
    """Type information for quantized tensors.

    This dataclass encapsulates the data type along with quantization parameters
    (scale and zero point) required for quantized tensor representations.

    Attributes
    ----------
    dtype : torch.dtype
        The torch data type (e.g., torch.qint8, torch.qint32)
    scale : Optional[float]
        Quantization scale factor for converting between quantized and dequantized values
    zero_point : Optional[int]
        Zero point offset for quantization
    """

    dtype: torch.dtype
    scale: Optional[float] = None
    zero_point: Optional[int] = None


class Builder:
    """Base class for MLIR builder implementations.

    Builder provides a foundation for constructing MLIR modules across different
    dialects (TTIR, StableHLO, TTNN, D2M). It manages:
    - MLIR context and location information
    - Golden tensor generation and verification for testing
    - Mesh configuration for distributed execution
    - Type conversions between PyTorch and MLIR type systems

    The golden tensor system tracks expected outputs for verification during
    compilation and runtime, enabling automated correctness checking.

    Attributes
    ----------
    _ctx : Context
        MLIR context for creating types and operations
    _loc : Location
        Source location for debugging
    _global_id : int
        Counter for generating unique operation identifiers
    _disable_golden_check : bool
        Flag to disable golden tensor generation and verification
    _force_graph_level_check : bool
        Flag to force verification only at graph boundaries (inputs/outputs)
    _ordered_inputs : List[Operand]
        Ordered list of function input operands
    _ordered_outputs : List[Operand]
        Ordered list of function output operands
    _goldens_to_store : List[Operand]
        Operands whose golden tensors should be stored for verification
    _goldens : Dict[Operand, GoldenMapTensor]
        Map from operands to their golden reference tensors
    _operand_to_loc : Dict[Operand, str]
        Map from operands to their source location strings
    _meshes : Dict[str, OrderedDict[str, int]]
        Map from mesh names to their dimension specifications
    _mesh_shape : Tuple[int, int]
        Shape of the default mesh for distributed execution
    """

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
        """Initialize the Builder with MLIR context and mesh configuration.

        Parameters
        ----------
        ctx : Context
            MLIR context for creating types, attributes, and operations
        location : Location
            Source location for debugging information
        mesh_name : Union[List[str], str], optional
            Name(s) of mesh(es) for distributed execution (default: "mesh")
        mesh_dict : Union[List[OrderedDict[str, int]], OrderedDict[str, int]], optional
            Mesh dimension specifications, e.g., OrderedDict([("x", 1), ("y", 1)])
            (default: 1x1 mesh)
        disable_golden_check : bool, optional
            If True, disables golden tensor generation and verification
            (default: False)

        Raises
        ------
        ValueError
            If mesh_name and mesh_dict lengths don't match when both are lists
        """
        self._ctx = ctx
        self._loc = location
        self._global_id = -1
        self._disable_golden_check = disable_golden_check
        self._force_graph_level_check = False

        # Store ordered inputs and outputs for deterministic golden map generation.
        self._ordered_inputs: List[Operand] = []
        self._ordered_outputs: List[Operand] = []

        # Track which operands should have their golden tensors stored.
        # If empty, all goldens are stored by default.
        self._goldens_to_store: List[Operand] = []

        # Golden tensor storage for verification.
        self._goldens: Dict[Operand, GoldenMapTensor] = {}

        # Location tracking for error reporting and debugging.
        self._operand_to_loc: Dict[Operand, str] = {}

        # Map from location string to the operand at that location.
        self._loc_to_operand: Dict[str, Operand] = {}

        # Set torch seed for reproducibility.
        torch.manual_seed(0)

        # Normalize mesh configuration to lists for uniform handling.
        if not isinstance(mesh_name, List):
            mesh_name = [mesh_name]
        if not isinstance(mesh_dict, List):
            mesh_dict = [mesh_dict]
        if len(mesh_name) != len(mesh_dict):
            raise ValueError(
                f"mesh_name length {len(mesh_name)} must match mesh_dict length {len(mesh_dict)}"
            )
        
        # Build mesh registry for distributed execution configuration.
        self._meshes = {}
        for name, mesh in zip(mesh_name, mesh_dict):
            self._meshes[name] = mesh

        # Extract default mesh shape from the first mesh configuration.
        self._mesh_shape = tuple(mesh_dict[0].values())

    # ----- Public methods -----

    @property
    def context(self) -> Context:
        """Get the MLIR context used by this builder.

        Returns
        -------
        Context
            The MLIR context for creating types and operations
        """
        return self._ctx

    @property
    def location(self) -> Location:
        """Get the source location used for debugging.

        Returns
        -------
        Location
            The MLIR location for error reporting
        """
        return self._loc

    @property
    def mesh_shape(self) -> Tuple[int, int]:
        """Get the shape of the default mesh for distributed execution.

        Returns
        -------
        Tuple[int, int]
            The mesh shape as (rows, columns), e.g., (1, 1) for single device
        """
        return self._mesh_shape

    @property
    def golden_map(self) -> Dict[str, Dict[int, GoldenTensor]]:
        """Generate golden tensor map for verification.

        This property builds a mapping of location strings to device-specific golden
        tensors. The map includes:
        - All input tensors (always included)
        - Output tensors (if marked for checking)
        - Intermediate operands (if marked for checking and not in graph-level mode)

        The golden tensors are used by the runtime to verify correctness of compiled
        operations by comparing actual outputs against expected values.

        Returns
        -------
        Dict[str, Dict[int, GoldenTensor]]
            Nested dictionary mapping location strings to device IDs to golden tensors.
            Format: {location_string: {device_id: GoldenTensor}}
        """
        golden_info: Dict[str, Dict[int, GoldenTensor]] = {}

        # Early return if golden checking is disabled.
        if self._disable_golden_check:
            return golden_info

        # Default behavior: store all goldens if none explicitly marked.
        if len(self._goldens_to_store) == 0:
            self._goldens_to_store = list(self._goldens.keys())

        # Always include inputs in golden map for verification.
        for index, input in enumerate(self._ordered_inputs):
            loc = f"input_{index}"
            golden_info[loc] = self._get_golden_tensor(input)

        # Include outputs if they are marked for checking.
        for index, output in enumerate(self._ordered_outputs):
            if output not in self._goldens_to_store:
                continue

            loc = f"output_{index}"
            golden_info[loc] = self._get_golden_tensor(output)

        # Skip intermediate operands if graph-level checking is enforced.
        if self._force_graph_level_check:
            return golden_info

        # Store other operands into golden map if they are marked to be stored.
        for operand, golden_map_tensor in self._goldens.items():
            if operand not in self._goldens_to_store:
                continue

            if not (isinstance(operand, OpView) or isinstance(operand, Operation)):
                continue

            loc = self._operand_to_loc.get(operand, None)
            self._loc_to_operand[loc] = operand
            golden_info[loc] = golden_map_tensor

        return golden_info

    def get_shape(self, input: Operand) -> Shape:
        """Get the shape of an operand.

        Parameters
        ----------
        input : Operand
            The operand to query

        Returns
        -------
        Shape
            The shape as a tuple or list of dimensions
        """
        return self._get_type(input).shape

    def get_type(self, input: Operand) -> Type:
        return self._get_type(input).element_type

    def set_goldens(
        self,
        inputs: Dict[Operand, Union[Callable, torch.tensor, Dict[int : torch.tensor]]],
        outputs: Dict[Operand, Union[torch.tensor, Dict[int : torch.tensor]]] = None,
        set_all_outputs: bool = True,
    ):
        """Set golden tensors for inputs and optionally outputs.

        This method allows manual specification of golden reference tensors for
        verification. Inputs can be specified as:
        - Torch tensors (directly)
        - Shard maps (Dict[device_id, tensor])
        - Callables that generate tensors given a shape

        Parameters
        ----------
        inputs : Dict[Operand, Union[Callable, torch.tensor, Dict[int, torch.tensor]]]
            Map from input operands to their golden values
        outputs : Dict[Operand, Union[torch.tensor, Dict[int, torch.tensor]]], optional
            Map from output operands to their golden values. If provided, these
            outputs will be marked for verification
        """
        self._set_goldens(self._create_builder_golden_from_torch_tensor(inputs))

        if outputs != None:
            self._set_goldens(self._create_builder_golden_from_torch_tensor(outputs))
            if set_all_outputs:
                self.set_goldens_to_check(outputs.keys())

    def set_goldens_from_builder_tensor(
        self,
        inputs: Dict[Operand, GoldenMapTensor],
        outputs: Dict[Operand, GoldenMapTensor] = None,
    ):
        """Set golden tensors using GoldenMapTensor objects directly.

        This method is similar to set_goldens but accepts GoldenMapTensor objects
        instead of torch tensors, allowing more control over sharding and device
        placement.

        Parameters
        ----------
        inputs : Dict[Operand, GoldenMapTensor]
            Map from input operands to their golden map tensors
        outputs : Dict[Operand, GoldenMapTensor], optional
            Map from output operands to their golden map tensors. If provided,
            these outputs will be marked for verification
        """
        self._set_goldens(inputs)

        if outputs != None:
            self.set_goldens_to_check(outputs.keys())
            self._set_goldens(outputs)

    def set_operand_goldens(
        self, operands: Dict[Operand, Union[torch.tensor, Dict[int : torch.tensor]]]
    ):
        """Set golden tensors for intermediate operands and mark them for checking.

        This method is used to specify golden values for intermediate operations
        (not just inputs/outputs) and automatically marks them for verification.

        Parameters
        ----------
        operands : Dict[Operand, Union[torch.tensor, Dict[int, torch.tensor]]]
            Map from operands to their golden values
        """
        self._set_goldens(self._create_builder_golden_from_torch_tensor(operands))
        self.set_goldens_to_check(operands.keys())

    def set_goldens_to_check(self, operands: List[Operand], override: bool = False):
        """Mark specific operands for golden tensor verification.

        Parameters
        ----------
        operands : List[Operand]
            List of operands to mark for checking
        override : bool, optional
            If True, replaces the existing list. If False, extends it (default: False)
        """
        if override:
            self._goldens_to_store = operands
        else:
            self._goldens_to_store.extend(operands)

    def set_graph_level_check(self, check: bool):
        """Control whether verification is limited to graph boundaries.

        When enabled, only input and output tensors are checked, skipping
        intermediate operations. This is useful for performance or when
        intermediate operations don't have stable golden implementations.

        Parameters
        ----------
        check : bool
            If True, enables graph-level checking only
        """
        self._force_graph_level_check = check

    # ----- Private methods -----

    def _get_output_shape_and_type(
        self,
        organize_golden_args: Callable,
        inputs: List[Operand],
        op_function: Callable,
        golden_kwargs: dict = {},
    ):
        """Infer output shape and type by executing the golden function.

        This helper method runs the golden reference implementation to determine
        the output shape and dtype, which is needed because TTIR ops don't have
        MLIR shape inference traits.

        Parameters
        ----------
        organize_golden_args : Callable
            Function to organize input operands into golden function arguments
        inputs : List[Operand]
            Input operands to the operation
        op_function : Callable
            The MLIR operation function
        golden_kwargs : dict, optional
            Additional keyword arguments for the golden function

        Returns
        -------
        Optional[Tuple[Shape, torch.dtype]]
            The inferred output shape and dtype, or None if no golden function exists
        """
        op_golden_function = get_golden_function(op_function, **golden_kwargs)
        if op_golden_function is None:
            return

        # Execute golden function with appropriate arguments.
        if len(inputs) == 0:
            # Operations with no inputs (e.g., ttir.zeros) use only kwargs.
            golden_output = op_golden_function(**golden_kwargs)
        else:
            golden_output = op_golden_function(
                *(organize_golden_args(inputs)), **golden_kwargs
            )

        return golden_output.shape, golden_output.dtype

    def _get_datatype_from_torch_dtype(self, dtype: torch.dtype) -> DataType:
        """Convert PyTorch dtype to MLIR DataType enum.

        Parameters
        ----------
        dtype : torch.dtype
            PyTorch data type

        Returns
        -------
        DataType
            Corresponding MLIR DataType enum value
        """
        match dtype:
            case torch.float16:
                return DataType.Float16
            case torch.bfloat16:
                return DataType.BFloat16
            case torch.uint8:
                return DataType.UInt8
            case torch.uint16:
                return DataType.UInt16
            case torch.uint32:
                return DataType.UInt32
            case torch.int32 | torch.qint32:
                return DataType.Int32
            case torch.int64:
                return DataType.Int32
            case torch.float32 | None:
                return DataType.Float32

    def _get_type(self, input: Operand) -> RankedTensorType:
        """Extract the MLIR type from an operand.

        Parameters
        ----------
        input : Operand
            The operand (Value, OpView, or Operation)

        Returns
        -------
        RankedTensorType
            The MLIR ranked tensor type

        Raises
        ------
        TypeError
            If input is not a valid operand type
        """
        if isinstance(input, Value):
            typ = input.type
        elif isinstance(input, OpView):
            typ = input.operation.result.type
        elif isinstance(input, Operation):
            typ = input.result.type
        else:
            raise TypeError(f"Invalid input {type(input)}")

        return typ

    def _get_type_from_torch_dtype(
        self,
        dtype: Union[torch.dtype, TypeInfo],
        scale: Optional[float] = None,
        zero_point: Optional[float] = None,
    ) -> Type:
        """Convert PyTorch dtype to MLIR Type.

        Supports both regular and quantized types. For quantized types, either
        pass a TypeInfo object or provide scale and zero_point parameters.

        Parameters
        ----------
        dtype : Union[torch.dtype, TypeInfo]
            PyTorch data type or TypeInfo for quantized types
        scale : Optional[float], optional
            Quantization scale (only used if dtype is a regular torch.dtype)
        zero_point : Optional[float], optional
            Quantization zero point (only used if dtype is a regular torch.dtype)

        Returns
        -------
        Type
            Corresponding MLIR type

        Raises
        ------
        ValueError
            If quantized type requires missing scale or zero_point
        TypeError
            If dtype is not a recognized type
        """
        if scale is not None and zero_point is not None:
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
            case torch.qint8:
                if not isinstance(dtype, TypeInfo):
                    raise ValueError("TypeInfo required for qint8")
                if dtype.scale is None or dtype.zero_point is None:
                    raise ValueError("scale and zero_point required for qint8")
                return quant.UniformQuantizedType.get(
                    quant.UniformQuantizedType.FLAG_SIGNED,
                    IntegerType.get_signless(8, self._ctx),
                    F32Type.get(self._ctx),
                    dtype.scale,
                    dtype.zero_point,
                    torch.iinfo(torch.qint8).min,
                    torch.iinfo(torch.qint8).max,
                )
            case torch.quint8:
                if not isinstance(dtype, TypeInfo):
                    raise ValueError("TypeInfo required for quint8")
                if dtype.scale is None or dtype.zero_point is None:
                    raise ValueError("scale and zero_point required for quint8")
                return quant.UniformQuantizedType.get(
                    0,
                    IntegerType.get_unsigned(8, self._ctx),
                    F32Type.get(self._ctx),
                    dtype.scale,
                    dtype.zero_point,
                    torch.iinfo(torch.quint8).min,
                    torch.iinfo(torch.quint8).max,
                )
            case torch.bool:
                return IntegerType.get_signless(1, self._ctx)
            case _:
                raise TypeError(f"Invalid Type {dtype}")

    def _get_torch_dtype_from_type(self, mlir_type: Type) -> torch.dtype:
        """Convert MLIR Type to torch.dtype.
        Parameters
        ----------
        mlir_type : Type
            MLIR type to convert
        Returns
        -------
        torch.dtype
            Corresponding torch dtype
        """
        type_str = str(mlir_type)

        if isinstance(mlir_type, BF16Type) or type_str == "bf16":
            return torch.bfloat16
        elif isinstance(mlir_type, F16Type) or type_str == "f16":
            return torch.float16
        elif isinstance(mlir_type, F32Type) or type_str == "f32":
            return torch.float32
        elif isinstance(mlir_type, F64Type) or type_str == "f64":
            return torch.float64
        elif isinstance(mlir_type, IntegerType):
            width = mlir_type.width
            is_signed = mlir_type.is_signed
            is_unsigned = mlir_type.is_unsigned

            if width == 1:
                return torch.bool
            elif width == 8:
                if is_unsigned:
                    return torch.uint8
                else:
                    return torch.int8
            elif width == 16:
                if is_unsigned:
                    return torch.uint16
                else:
                    return torch.int16
            elif width == 32:
                if is_unsigned:
                    return torch.uint32
                else:
                    return torch.int32
            elif width == 64:
                if is_unsigned:
                    return torch.uint64
                else:
                    return torch.int64
            else:
                raise TypeError(f"Unsupported integer width: {width}")
        else:
            raise TypeError(f"Unsupported MLIR type: {mlir_type}")

    def _get_next_global_id(self) -> int:
        """Generate the next unique global ID for operations.

        Returns
        -------
        int
            The next global ID
        """
        self._global_id += 1
        return self._global_id

    def _get_loc_of_extra_file_callee(self, id: int = 0) -> Location:
        """Generate a Location from the calling stack outside this file.

        This method walks the call stack to find the first frame outside the
        current builder file and creates an MLIR Location from it. This is used
        for better error reporting and debugging.

        Parameters
        ----------
        id : int, optional
            Additional ID to include in location string (default: 0)

        Returns
        -------
        Location
            MLIR location object referencing the caller

        Raises
        ------
        RuntimeError
            If cannot find a caller outside the current file
        """
        stack = inspect.stack()
        caller_filename = stack[1].filename

        # Walk up the stack until we find a frame from a different file.
        while len(stack) > 0 and stack[0].filename == caller_filename:
            stack = stack[1:]

        if len(stack) == 0:
            raise RuntimeError(
                "Top of callstack to builder funcs must be outside the caller's file"
            )

        return Location.name(
            f"{stack[0].filename}:{str(stack[0].lineno)}:id({str(id)})"
        )

    def _get_loc_from_str(self, loc: Union[str, Location]) -> Location:
        """Convert a string or Location to a Location object.

        Parameters
        ----------
        loc : Union[str, Location]
            String location or Location object

        Returns
        -------
        Location
            MLIR Location object
        """
        if isinstance(loc, str):
            return Location.name(loc)
        else:
            return loc

    def _create_ranked_tensor_type(
        self,
        shape: Shape,
        data_type: Optional[Type] = None,
        encoding: Optional[Attribute] = None,
    ) -> RankedTensorType:
        """Create an MLIR RankedTensorType with specified shape and type.

        Parameters
        ----------
        shape : Shape
            The tensor shape
        data_type : Optional[Type], optional
            The element type (default: F32Type)
        encoding : Optional[Attribute], optional
            Optional encoding attribute for layout information

        Returns
        -------
        RankedTensorType
            The created ranked tensor type
        """
        with self._ctx, self._loc:
            dtype = data_type if data_type is not None else F32Type.get(self._ctx)
            return RankedTensorType.get(shape, dtype, encoding)

    def _organize_eltwise_golden(self, inputs: List[Operand]) -> List[GoldenMapTensor]:
        """Organize elementwise operation inputs for golden function execution.

        Extracts golden tensors from input operands in the order required by
        golden functions.

        Parameters
        ----------
        inputs : List[Operand]
            List of input operands

        Returns
        -------
        List[GoldenMapTensor]
            List of golden tensors corresponding to inputs
        """
        return [self._goldens[inp] for inp in inputs]

    def _generate_random_tensor(
        self, shape: Shape, dtype: Union[torch.dtype, TypeInfo]
    ) -> torch.Tensor:
        """Generate a random tensor for golden input initialization.

        Creates random tensors appropriate for the data type:
        - Floating point: normal distribution (torch.randn)
        - Boolean: random 0/1
        - Integer: uniform distribution across full type range
        - Quantized: normal distribution, then quantized

        Parameters
        ----------
        shape : Shape
            The desired tensor shape
        dtype : Union[torch.dtype, TypeInfo]
            The data type (or TypeInfo for quantized types)

        Returns
        -------
        torch.Tensor
            A randomly initialized tensor
        """
        if isinstance(dtype, TypeInfo):
            # For quantized types, generate float tensor then quantize.
            float_tensor = torch.randn(shape, dtype=torch.float32)
            return torch.quantize_per_tensor(
                float_tensor, dtype.scale, dtype.zero_point, dtype.dtype
            )
        if dtype.is_floating_point:
            return torch.randn(shape, dtype=dtype)
        elif dtype == torch.bool:
            return torch.randint(0, 2, shape, dtype=torch.bool)
        else:
            # For integer types, generate random values within the type's range.
            min_int = torch.iinfo(dtype).min
            max_int = torch.iinfo(dtype).max
            return torch.randint(
                low=min_int,
                high=max_int,
                size=shape,
                dtype=dtype,
            )

    def _generate_golden_tensor(
        self, operand: Operand, dtype: Union[torch.dtype, TypeInfo]
    ) -> GoldenMapTensor:
        """Generate a random golden tensor for an operand.

        Parameters
        ----------
        operand : Operand
            The operand to generate golden for
        dtype : Union[torch.dtype, TypeInfo]
            The data type

        Returns
        -------
        GoldenMapTensor
            Golden map tensor with random initialization
        """
        random_tensor = self._generate_random_tensor(self.get_shape(operand), dtype)
        return GoldenMapTensor({0: random_tensor}, mesh_shape=self._mesh_shape)

    def _generate_golden_device_tensor(
        self, loc: str, golden_map_tensor: GoldenMapTensor
    ) -> Dict[int, GoldenTensor]:
        """Convert a GoldenMapTensor to device-specific GoldenTensor format.

        This method prepares golden tensors for embedding in the flatbuffer by
        converting them to the runtime's GoldenTensor format with proper data
        pointers and size information.

        Parameters
        ----------
        loc : str
            Location string for the tensor (e.g., "input_0", "output_1")
        builder_golden_tensor : GoldenMapTensor
            The builder's golden tensor representation

        Returns
        -------
        Dict[int, GoldenTensor]
            Map from device ID to GoldenTensor objects ready for flatbuffer embedding
        """
        device_golden_info: Dict[int, GoldenTensor] = {}
        contiguous_tensor = golden_map_tensor.contiguous()
        for device_id, device_golden in contiguous_tensor.shard_map.items():
            data_type = self._get_datatype_from_torch_dtype(device_golden.dtype)
            device_golden_info[device_id] = GoldenTensor(
                loc,
                list(device_golden.shape),
                list(device_golden.stride()),
                data_type if data_type is not None else DataType.Float32,
                device_golden.data_ptr(),
                device_golden.numel() * device_golden.dtype.itemsize,
            )

        return device_golden_info

    def _create_builder_golden_from_torch_tensor(
        self,
        inputs: Dict[Operand, Union[Callable, torch.Tensor, Dict[int, torch.Tensor]]],
    ) -> Dict[Operand, GoldenMapTensor]:
        """Convert torch tensors or callables to GoldenMapTensor format.

        This method handles three input formats:
        1. Callable: A function that generates a tensor given a shape
        2. torch.Tensor: A single tensor to replicate across devices
        3. Dict[int, torch.Tensor]: Explicit per-device shard map

        Parameters
        ----------
        inputs : Dict[Operand, Union[Callable, torch.Tensor, Dict[int, torch.Tensor]]]
            Map from operands to tensor specifications

        Returns
        -------
        Dict[Operand, GoldenMapTensor]
            Map from operands to builder golden tensors

        Raises
        ------
        TypeError
            If callable doesn't return a torch.Tensor
        RuntimeError
            If callable execution fails
        """
        input_goldens: Dict[Operand, GoldenMapTensor] = {}
        for operand, tensor_or_shard_map_or_callable in inputs.items():
            if callable(tensor_or_shard_map_or_callable):
                # Handle callable initialization functions.
                operand_shape = self.get_shape(operand)
                try:
                    generated_tensor = tensor_or_shard_map_or_callable(operand_shape)
                    if not isinstance(generated_tensor, torch.Tensor):
                        raise TypeError(
                            f"Callable must return a torch.Tensor, got {type(generated_tensor)}"
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Error calling initialization function for operand {operand}: {e}"
                    )
                golden_tensor = GoldenMapTensor(
                    {0: generated_tensor}, mesh_shape=self._mesh_shape
                )
            elif isinstance(tensor_or_shard_map_or_callable, torch.Tensor):
                # Handle direct tensor specification.
                golden_tensor = GoldenMapTensor(
                    {0: tensor_or_shard_map_or_callable}, mesh_shape=self._mesh_shape
                )
            else:
                # Handle explicit shard map.
                golden_tensor = GoldenMapTensor(
                    tensor_or_shard_map_or_callable, mesh_shape=self._mesh_shape
                )
            input_goldens[operand] = golden_tensor

        return input_goldens

    def _set_golden_tensor(
        self,
        operand: Operand,
        golden: GoldenMapTensor,
    ):
        """Store a golden tensor for an operand and record its location.

        Parameters
        ----------
        operand : Operand
            The operand to associate with the golden
        golden : GoldenMapTensor
            The golden tensor to store
        """
        self._goldens[operand] = golden

        # Record location for debugging and verification.
        if isinstance(operand, OpView):
            loc = str(operand.operation.location)
            self._operand_to_loc[operand] = loc
        elif isinstance(operand, Operation):
            loc = str(operand.location)
            self._operand_to_loc[operand] = loc

    def _set_goldens(
        self,
        goldens: Dict[Operand, GoldenMapTensor],
    ):
        """Batch store golden tensors for multiple operands.

        Parameters
        ----------
        goldens : Dict[Operand, GoldenMapTensor]
            Map from operands to their golden tensors
        """
        for operand, golden in goldens.items():
            self._set_golden_tensor(operand, golden)

    def _get_golden_tensor(
        self,
        operand: Operand,
    ) -> GoldenMapTensor:
        """Retrieve the golden tensor for an operand.

        Parameters
        ----------
        operand : Operand
            The operand to query

        Returns
        -------
        GoldenMapTensor
            The stored golden tensor
        """
        return self._goldens[operand]

    def _get_golden_tensors(
        self,
        operands: List[Operand],
    ) -> List[GoldenMapTensor]:
        return [self._goldens[operand] for operand in operands]

    def _set_input_ordering(self, inputs: List[Operand]):
        """Record the ordered list of function inputs.

        This ordering is used when generating the golden map to ensure
        deterministic input naming (input_0, input_1, etc.).

        Parameters
        ----------
        inputs : List[Operand]
            Ordered list of input operands
        """
        self._ordered_inputs = inputs

    def _set_output_ordering(self, outputs: List[Operand]):
        """Record the ordered list of function outputs.

        This ordering is used when generating the golden map to ensure
        deterministic output naming (output_0, output_1, etc.).

        Parameters
        ----------
        outputs : List[Operand]
            Ordered list of output operands
        """
        self._ordered_outputs = outputs

    # ----- Shared Empty Operations -----

    def _empty(self, shape: Shape, data_type: Optional[Type] = None) -> OpView:
        """Create an empty tensor operation.

        Creates an uninitialized tensor of the specified shape and type using
        the dialect-specific empty operation. The actual operation is provided
        by subclasses via _get_empty_op.

        Parameters
        ----------
        shape : Shape
            The desired tensor shape
        data_type : Optional[Type], optional
            The element type (default: F32Type)

        Returns
        -------
        OpView
            The empty operation view
        """
        dtype = data_type if data_type is not None else F32Type.get(self._ctx)
        return self._create_empty_from_tensor_type(
            shape, self._create_ranked_tensor_type(shape, dtype)
        )

    def _create_empty_from_tensor_type(
        self, shape: Shape, tensor_type: RankedTensorType
    ) -> OpView:
        """Create empty operation from a ranked tensor type.

        Helper method that delegates to the dialect-specific empty operation
        implementation.

        Parameters
        ----------
        shape : Shape
            The tensor shape (used for validation)
        tensor_type : RankedTensorType
            The complete tensor type including shape, dtype, and encoding

        Returns
        -------
        OpView
            The empty operation view
        """
        with self._ctx, self._loc:
            op = self._get_empty_op(tensor_type)
            return op

    def _get_empty_op(self, tensor_type: RankedTensorType) -> OpView:
        """Get dialect-specific empty operation.

        This method must be implemented by subclasses to provide the appropriate
        empty operation for their dialect (e.g., ttir.EmptyOp, tensor.EmptyOp).

        Parameters
        ----------
        tensor_type : RankedTensorType
            The tensor type for the empty operation

        Returns
        -------
        OpView
            The dialect-specific empty operation

        Raises
        ------
        NotImplementedError
            If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement _get_empty_op")

    # ----- Shared Metal Tensor Layout -----

    def get_metal_tensor_layout(
        self,
        logical_shape: Shape,
        tiled=False,
        oobVal=None,
        memorySpace=None,
        grid: Optional[Tuple[int, int]] = None,
        index_map: Optional[AffineMap] = None,
        memory_layout=None,
    ):
        """Create a metal tensor layout using shared implementation.

        This method wraps the shared get_metal_tensor_layout utility function,
        providing convenient defaults and access to the builder's context.

        For detailed parameter documentation, see builder_utils.get_metal_tensor_layout.

        Parameters
        ----------
        logical_shape : Shape
            Logical shape of the tensor
        tiled : bool, optional
            Whether to use 32x32 tiled layout (default: False)
        oobVal : optional
            Out-of-bounds value handling (default: ttcore.OOBVal.Undef)
        memorySpace : optional
            Memory space (default: ttcore.MemorySpace.DeviceL1)
        grid : Optional[Tuple[int, int]], optional
            Grid shape for sharding
        index_map : Optional[AffineMap], optional
            Optional affine map for layout transformation
        memory_layout : optional
            Memory layout strategy (default: ttcore.TensorMemoryLayout.Sharded)

        Returns
        -------
        RankedTensorType
            Metal tensor type with specified layout
        """
        from builder.base.builder_utils import get_metal_tensor_layout
        from ttmlir.dialects import ttcore

        # Set defaults if not provided
        if oobVal is None:
            oobVal = ttcore.OOBVal.Undef
        if memorySpace is None:
            memorySpace = ttcore.MemorySpace.DeviceL1
        if memory_layout is None:
            memory_layout = ttcore.TensorMemoryLayout.Sharded

        return get_metal_tensor_layout(
            self._ctx,
            logical_shape,
            tiled,
            oobVal,
            memorySpace,
            grid,
            index_map,
            memory_layout,
        )
