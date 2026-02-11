# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import inspect
from typing import List, Optional, Union, Tuple, Callable, Dict, Any
import torch
from enum import Enum, auto
import re
from collections import OrderedDict

from ttmlir.ir import *
from ttmlir.dialects import tensor, quant, func, ttir, ttcore, stablehlo, ttnn, debug
from ttmlir.passes import GoldenTensor, DataType
from golden import GoldenMapTensor, get_golden_function, apply_sharding

from builder.base.builder_utils import (
    process_multi_return_result,
    TypeInfo,
    tag,
    parse,
    split,
)


class BuilderMeta(type):
    """Metaclass for the Builder class that automatically builds operation view maps.

    This metaclass ensures that when a Builder subclass is created, the necessary
    operation view mapping dictionaries are populated automatically. These maps
    are used to translate between MLIR operation views and their corresponding
    builder methods, parsers, and split operations.
    """

    def __new__(mcls, name, bases, namespace):
        """Create a new class instance and build the operation view maps.

        Args:
            mcls: The metaclass instance.
            name: Name of the class being created.
            bases: Base classes of the new class.
            namespace: Namespace dictionary containing class attributes.

        Returns:
            The newly created class with populated operation view maps.
        """
        cls = super().__new__(mcls, name, bases, namespace)
        cls.build_opview_to_builder_map()
        cls.build_opview_to_parser_map()
        cls.build_opview_to_split(map)
        return cls


class Builder(metaclass=BuilderMeta):
    """Base builder class for constructing MLIR operations with golden testing support.

    This class provides the foundation for building MLIR operations across different
    dialects. It manages operation view mappings, golden tensor comparisons for
    testing, and mesh-based distributed computation configurations.

    Attributes:
        opview_to_builder_map: Maps operation views to their builder methods.
        opview_to_parser_map: Maps operation views to their parser methods.
        opview_to_split_map: Maps operation views to their split operations.
    """

    opview_to_builder_map: Dict[OpView, Callable] = {}
    opview_to_parser_map: Dict[OpView, Callable] = {}
    opview_to_split_map: Dict[OpView, Callable] = {}

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
        """Initialize the Builder with context, location, and configuration.

        Args:
            ctx: MLIR context for operation creation.
            location: Source location for debugging and error reporting.
            mesh_name: Name or list of names for mesh configurations.
            mesh_dict: Dictionary or list of dictionaries defining mesh dimensions.
            disable_golden_check: If True, disables golden tensor validation.

        Raises:
            ValueError: If mesh_name and mesh_dict lengths don't match.
        """
        self._ctx = ctx
        self._loc = location
        self._global_id = -1
        self._disable_golden_check = disable_golden_check
        self._force_graph_level_check = False

        # Keep a list of inputs and outputs in order so we know how to store them in golden map.
        # ordered dict determines program order when comparing goldens during runtime
        # func_op: [[ordered_inputs], [ordered_outputs]]
        self._func_ops_generated: Dict[func.FuncOp, List[List[Operand]]] = {}

        # Explicity set goldens to store. If empty, store all goldens.
        self._goldens_to_store: List[Operand] = []

        # Map from operand to its golden tensor.
        self._goldens: Dict[Operand, GoldenMapTensor] = {}

        # Map from operand to its location string.
        self._operand_to_loc: Dict[Operand, str] = {}

        # Map from location string to the operand at that location.
        self._loc_to_operand: Dict[str, Operand] = {}

        # List of op locations to bypass golden comparison.
        self._bypass_ops: List[str] = []

        # Set torch seed for reproducibility.
        torch.manual_seed(0)

        if not isinstance(mesh_name, List):
            mesh_name = [mesh_name]
        if not isinstance(mesh_dict, List):
            mesh_dict = [mesh_dict]
        if len(mesh_name) != len(mesh_dict):
            raise ValueError(
                f"mesh_name length {len(mesh_name)} must match mesh_dict length {len(mesh_dict)}"
            )
        self._meshes = {}
        for name, mesh in zip(mesh_name, mesh_dict):
            self._meshes[name] = mesh

        self._mesh_shape = tuple(mesh_dict[0].values())
        self._mesh_name = mesh_name[0]

        # Internal values to keep track
        self._root_module_insertion_point = None
        self._current_module_insertion_point = None
        self._cpu_module_insertion_point = None
        self._hoisted_cpu_functions: List[str] = []
        self._nested_funcs: List[str] = []
        self._func_name_to_op: Dict[str, func.FuncOp] = {}

    # ----- Class helper methods -----

    @classmethod
    def build_opview_to_builder_map(cls):
        """Build the mapping from operation views to their builder methods.

        This method scans all class attributes to find methods tagged with
        operation view information and populates the opview_to_builder_map
        dictionary. Tagged methods are identified by the presence of a '_tag'
        attribute.
        """
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            func = attr

            if callable(attr) and hasattr(func, "_tag"):
                cls.opview_to_builder_map[func._tag] = attr

    @classmethod
    def build_opview_to_parser_map(cls):
        """Build the mapping from operation views to their parser methods.

        This method scans all class attributes to find methods tagged with
        operation view information and populates the opview_to_parser_map
        dictionary. Parser methods are used to parse operation representations
        from various formats into MLIR operation views.

        The method identifies tagged methods by checking for the '_parse' attribute
        on class methods. Only methods with OpView tags are included in the parser map.
        """
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            func = attr

            if callable(attr) and hasattr(func, "_parse"):
                cls.opview_to_parser_map[func._parse] = attr

    @classmethod
    def build_opview_to_split(cls, map):
        """Build the mapping from operation views to their split operation methods.

        This method scans all class attributes to find methods tagged with
        operation view information and populates the opview_to_split_map
        dictionary. Split operation methods are used to decompose complex
        operations into simpler sub-operations for optimization or execution.

        The method identifies tagged methods by checking for the '_split' attribute
        on class methods. Only methods with OpView tags are included in the split map.
        """
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            func = attr

            if callable(attr) and hasattr(func, "_split"):
                cls.opview_to_split_map[func._split] = attr

    def get_opview_from_method(self, method: func) -> OpView:
        """Get the operation view associated with a builder method.

        Args:
            method: The builder method to inspect.

        Returns:
            The OpView associated with the method, or None if not tagged.
        """
        return getattr(method, "_tag", None)

    def get_opview_from_parser(self, parser: func) -> OpView:
        """Get the operation view associated with a parser method.

        Args:
            parser: The parser method to inspect.

        Returns:
            The OpView associated with the parser, or None if not tagged.
        """
        return getattr(parser, "_parse", None)

    def get_opview_from_split(self, split: func) -> OpView:
        """Get the operation view associated with a split operation method.

        Args:
            split: The split operation method to inspect.

        Returns:
            The OpView associated with the split method, or None if not tagged.
        """
        return getattr(split, "_split", None)

    def get_builder_from_opview(self, opview: OpView) -> Callable:
        if opview not in self.opview_to_builder_map:
            assert False, f"No builder found for opview {opview}"
        return self.opview_to_builder_map.get(opview)

    def get_parser_from_opview(self, opview: OpView) -> Callable:
        if opview not in self.opview_to_parser_map:
            assert False, f"No parser found for opview {opview}"
        return self.opview_to_parser_map.get(opview)

    def get_split_from_opview(self, opview: OpView) -> Callable:
        if opview not in self.opview_to_split_map:
            assert False, f"No split function found for opview {opview}"
        return self.opview_to_split_map.get(opview)

    # ----- Public methods -----

    @property
    def context(self) -> Context:
        """Get the MLIR context associated with this builder.

        Returns:
            The MLIR context used for operation creation.
        """
        return self._ctx

    @property
    def location(self) -> Location:
        """Get the source location associated with this builder.

        Returns:
            The source location used for debugging and error reporting.
        """
        return self._loc

    @property
    def mesh_shape(self) -> Tuple[int, int]:
        """Get the shape of the computation mesh.

        Returns:
            A tuple representing the dimensions of the computation mesh.
        """
        return self._mesh_shape

    @property
    def golden_map(
        self,
    ) -> Tuple[
        Dict[int, Dict[str, Dict[int, GoldenMapTensor]]],
        Dict[str, Dict[int, GoldenMapTensor]],
    ]:
        """Get the golden tensor map for validation and testing.

        This property constructs a comprehensive map of golden tensors used for
        validating computation results. It organizes tensors by program index,
        location, and device ID to support distributed computation validation.

        Returns:
            A tuple containing two dictionaries:
            1. input_output_golden_info: Maps program indices to location-based
               device-specific golden tensors for inputs and outputs.
            2. intermediate_golden_info: Maps location strings to device-specific
               golden tensors for intermediate computation results.

        Note:
            Golden tensors are reference values used to validate that MLIR
            operations produce correct results during testing.
        """
        # { program_index: {loc: {device_id: GoldenMapTensor} } }
        input_output_golden_info: Dict[int, Dict[str, Dict[int, GoldenMapTensor]]] = {}
        intermediate_golden_info: Dict[str, Dict[int, GoldenMapTensor]] = {}

        if self._disable_golden_check:
            return input_output_golden_info, intermediate_golden_info

        # If no specific golden is marked to be stored, store all goldens.
        if len(self._goldens_to_store) == 0:
            self._goldens_to_store = list(self._goldens.keys())

        # Iterate through all functions generated to collect their inputs and outputs.
        programs = []
        for func_op, ordered_values in self._func_ops_generated.items():
            if (
                func_op.name.value in self._hoisted_cpu_functions
                or func_op.name.value in self._nested_funcs
            ):
                continue

            programs.append(ordered_values)

        for program_index, ordered_values in enumerate(programs):
            input_output_golden_info[program_index] = {}
            ordered_inputs, ordered_outputs = ordered_values

            # Always store inputs into golden map.
            for index, input in enumerate(ordered_inputs):
                loc = f"input_{index}"
                input_output_golden_info[program_index][loc] = self._get_golden_tensor(
                    input
                )

            # Store outputs into golden map if they are marked to be stored.
            for index, output in enumerate(ordered_outputs):
                if output not in self._goldens_to_store:
                    continue

                loc = f"output_{index}"
                input_output_golden_info[program_index][loc] = self._get_golden_tensor(
                    output
                )

        if self._force_graph_level_check:
            return input_output_golden_info, intermediate_golden_info

        # Store other operands into golden map if they are marked to be stored.
        for operand, golden_map_tensor in self._goldens.items():
            if operand not in self._goldens_to_store:
                continue

            if not isinstance(operand, OpResult):
                continue

            loc = self._operand_to_loc.get(operand, None)
            self._loc_to_operand[loc] = operand
            intermediate_golden_info[loc] = golden_map_tensor

        return input_output_golden_info, intermediate_golden_info

    def get_shape(self, input: Operand) -> Shape:
        """Get the shape of an input operand.

        Args:
            input: The operand to inspect.

        Returns:
            The shape of the operand as a Shape object.
        """
        return self._get_type(input).shape

    def get_type(self, input: Operand) -> Type:
        """Get the element type of an input operand.

        Args:
            input: The operand to inspect.

        Returns:
            The element type of the operand as a Type object.
        """
        return self._get_type(input).element_type

    def set_goldens(
        self,
        inputs: Dict[Operand, Union[Callable, torch.tensor, Dict[int : torch.tensor]]],
        outputs: Dict[Operand, Union[torch.tensor, Dict[int : torch.tensor]]] = None,
        set_all_outputs: bool = True,
    ):
        """Set golden tensors for input and output validation.

        Golden tensors are reference values used to validate that MLIR operations
        produce correct results. This method sets golden tensors for both inputs
        and outputs of operations.

        Args:
            inputs: Dictionary mapping input operands to their golden values.
                   Values can be torch tensors, callables that generate tensors,
                   or dictionaries mapping device IDs to tensors for distributed computation.
            outputs: Optional dictionary mapping output operands to their golden values.
                    If provided, these outputs will be validated during golden checks.
            set_all_outputs: If True, all outputs in the outputs dictionary will be
                           marked for validation during golden checks.

        Note:
            Golden tensor validation is essential for ensuring computational correctness
            in MLIR operation testing and debugging.
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
        self._set_goldens(inputs)

        if outputs != None:
            self.set_goldens_to_check(outputs.keys())
            self._set_goldens(outputs)

    def set_operand_goldens(
        self, operands: Dict[Operand, Union[torch.tensor, Dict[int : torch.tensor]]]
    ):
        self._set_goldens(self._create_builder_golden_from_torch_tensor(operands))
        self.set_goldens_to_check(operands.keys())

    def set_goldens_to_check(self, operands: List[Operand], override: bool = False):
        if override:
            self._goldens_to_store = operands
        else:
            self._goldens_to_store.extend(operands)

    def set_graph_level_check(self, check: bool):
        self._force_graph_level_check = check

    def bypass(self, operand: Operand):
        if isinstance(operand, BlockArgument):
            raise TypeError("Cannot bypass BlockArgument")

        loc = str(operand.owner.location)
        self._bypass_ops.append(loc)

    def set_arg_attribute(
        self, operand: Operand, new_attr_name: str, new_attr: Attribute
    ):
        func_op = operand.owner.owner

        arg_attr_list = func_op.arg_attrs
        new_arg_attr_list = []
        for arg_number, arg_attrs in enumerate(arg_attr_list):
            if arg_number == operand.arg_number:
                new_arg_attr = {}
                for attr in arg_attrs:
                    new_arg_attr[attr.name] = attr.attr
                new_arg_attr[new_attr_name] = new_attr
                new_arg_attr_list.append(DictAttr.get(new_arg_attr))
            else:
                new_arg_attr_list.append(arg_attrs)

        func_op.arg_attrs = ArrayAttr.get(new_arg_attr_list)

    def preshard_arg(self, operand: Operand, shard_dims: List[int]):
        golden_tensor = self._get_golden_tensor(operand)
        sharded_golden_tensor = apply_sharding(
            golden_tensor, self._mesh_shape, shard_dims
        )

        # Generate new multi-device golden if it's presharded
        if not self._disable_golden_check:
            self._set_golden_tensor(operand, sharded_golden_tensor)

        local_shape = sharded_golden_tensor.shape
        local_shape_attr = RankedTensorType.get(local_shape, F32Type.get(self._ctx))
        new_attr_name: str = "ttcore.runtime_tensor_sharding"
        shard_status_attr = ttcore.ir.ShardStatusAttr.get(
            self._ctx, ttcore.ir.ShardStatus.Presharded
        )
        new_attr = ttcore.ir.RuntimeTensorShardingAttr.get(
            self._ctx, shard_status_attr, local_shape_attr
        )

        self.set_arg_attribute(operand, new_attr_name, new_attr)

    # ----- Private methods -----

    def _get_output_shape_and_type(
        self,
        organize_golden_args: Callable,
        inputs: List[Operand],
        op_function: Callable,
        golden_kwargs: dict = {},
    ):
        op_golden_function = get_golden_function(op_function, **golden_kwargs)
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

    def _get_datatype_from_torch_dtype(self, dtype: torch.dtype) -> DataType:
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
        return input.type

    def _get_type_from_torch_dtype(
        self,
        dtype: Union[torch.dtype, TypeInfo],
        scale: Optional[float] = None,
        zero_point: Optional[float] = None,
    ) -> Type:
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
        self._global_id += 1
        return self._global_id

    def _get_loc_of_extra_file_callee(self, id: int = 0) -> Location:
        stack = inspect.stack()
        caller_filename = stack[1].filename

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
        if isinstance(loc, str):
            return Location.name(loc)
        else:
            return loc

    def _create_ranked_tensor_type(
        self,
        shape: Shape,
        data_type: Optional[Union[Type, torch.dtype]] = None,
        encoding: Optional[Attribute] = None,
    ) -> RankedTensorType:
        with self._ctx, self._loc:
            if isinstance(data_type, torch.dtype):
                dtype = self._get_type_from_torch_dtype(data_type)
            else:
                dtype = data_type if data_type is not None else F32Type.get(self._ctx)
            return RankedTensorType.get(shape, dtype, encoding)

    def _organize_eltwise_golden(self, inputs: List[Operand]) -> List[GoldenMapTensor]:
        return [self._goldens[inp] for inp in inputs]

    def _generate_random_tensor(
        self, shape: Shape, dtype: Union[torch.dtype, TypeInfo]
    ) -> torch.Tensor:
        if isinstance(dtype, TypeInfo):
            float_tensor = torch.randn(shape, dtype=torch.float32)
            return torch.quantize_per_tensor(
                float_tensor, dtype.scale, dtype.zero_point, dtype.dtype
            )
        if dtype.is_floating_point:
            return torch.randn(shape, dtype=dtype)
        elif dtype == torch.bool:
            return torch.randint(0, 2, shape, dtype=torch.bool)
        else:
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
        random_tensor = self._generate_random_tensor(self.get_shape(operand), dtype)
        return GoldenMapTensor({0: random_tensor}, mesh_shape=self._mesh_shape)

    def _generate_golden_device_tensor(
        self, loc: str, golden_map_tensor: GoldenMapTensor
    ) -> Dict[int, GoldenTensor]:
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
        input_goldens: Dict[Operand, GoldenMapTensor] = {}
        for operand, tensor_or_shard_map_or_callable in inputs.items():
            if callable(tensor_or_shard_map_or_callable):
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
                golden_tensor = GoldenMapTensor(
                    {0: tensor_or_shard_map_or_callable}, mesh_shape=self._mesh_shape
                )
            else:
                golden_tensor = GoldenMapTensor(
                    tensor_or_shard_map_or_callable, mesh_shape=self._mesh_shape
                )
            input_goldens[operand] = golden_tensor

        return input_goldens

    def _set_golden_tensor(
        self,
        operand: Operand,
        goldens: List[GoldenMapTensor],
    ):
        self._goldens[operand] = goldens
        self._operand_to_loc[operand] = str(operand.location)

    def _set_goldens(
        self,
        goldens: Dict[Operand, GoldenMapTensor],
    ):
        for operand, golden in goldens.items():
            self._set_golden_tensor(operand, golden)

    def _get_golden_tensor(
        self,
        operand: Operand,
    ) -> GoldenMapTensor:
        return self._goldens[operand]

    def _get_golden_tensors(
        self,
        operands: List[Operand],
    ) -> List[GoldenMapTensor]:
        return [self._goldens[operand] for operand in operands]

    def _get_location(self) -> Location:
        stack = inspect.stack()
        caller_frame = stack[2]
        filename = caller_frame.filename
        lineno = caller_frame.lineno
        return Location.name(f"{filename}:{lineno}")

    # ----- Shared Empty Operations -----

    def _empty(self, shape: Shape, data_type: Optional[Type] = None) -> OpView:
        """Create an empty operation using the dialect-specific EmptyOp."""
        dtype = data_type if data_type is not None else F32Type.get(self._ctx)
        return self._create_empty_from_tensor_type(
            shape, self._create_ranked_tensor_type(shape, dtype)
        )

    def _create_empty_from_tensor_type(
        self, shape: Shape, tensor_type: RankedTensorType
    ) -> OpView:
        """Create empty operation from tensor type using dialect-specific EmptyOp."""
        with self._ctx, self._loc:
            op = self._get_empty_op(tensor_type)
            return op

    def _get_empty_op(self, tensor_type: RankedTensorType) -> OpView:
        """Get dialect-specific empty operation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_empty_op")

    def create_tensor_encoding(
        self, shape: Shape, element_type: Union[torch.dtype, TypeInfo]
    ) -> ttnn.ir.TTNNLayoutAttr:
        raise NotImplementedError("Subclasses must implement create_tensor_encoding")

    # ----- Shared Metal Tensor Layout -----

    def get_metal_tensor_layout(
        self,
        logical_shape: Shape,
        tiled=False,
        oobVal=None,  # Will default to ttcore.OOBVal.Undef in the utility
        memorySpace=None,  # Will default to ttcore.MemorySpace.DeviceL1 in the utility
        grid: Optional[Tuple[int, int]] = None,
        index_map: Optional[AffineMap] = None,
        memory_layout=None,  # Will default to ttcore.TensorMemoryLayout.Sharded in the utility
        dim_alignments: Optional[Tuple[int, ...]] = None,
    ):
        """Create a metal tensor layout using the shared implementation."""
        from builder.base.builder_apis import get_metal_tensor_layout
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
            dim_alignments,
        )

    # ----- Operations -----

    def call(
        self,
        nested_func: Callable,
        original_inputs: List[Operand],
        loc: Optional[str] = None,
    ):
        fn_input_types = []
        for operand in original_inputs:
            fn_input_types.append(operand.type)

        ordered_inputs = []
        ordered_outputs = []

        with InsertionPoint(self._current_module_insertion_point):

            @func.func(*fn_input_types, name=nested_func.__name__)
            def decorated_func(*inputs):
                input_goldens: Dict[Operand, GoldenMapTensor] = {}
                for index, (new_operand, original_operand) in enumerate(
                    zip(inputs, original_inputs)
                ):
                    input_goldens[new_operand] = self._get_golden_tensor(
                        original_operand
                    )

                self._set_goldens(input_goldens)
                ordered_inputs.extend(inputs)

                result = nested_func(*inputs, self)

                outputs = result if hasattr(result, "__iter__") else [result]
                output_goldens: Dict[Operand, GoldenMapTensor] = {}
                for op in outputs:
                    output_goldens[op] = self._get_golden_tensor(op)
                self._set_goldens(output_goldens)
                ordered_outputs.extend(outputs)

                return process_multi_return_result(result)

        new_func_op = decorated_func.func_op
        new_func_op.sym_visibility = StringAttr.get("private")
        self._func_ops_generated[new_func_op] = [ordered_inputs, ordered_outputs]

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        call_op = func.CallOp(new_func_op, original_inputs, loc=loc)
        for index, output in enumerate(ordered_outputs):
            self._set_golden_tensor(
                call_op.results[index], self._get_golden_tensor(output)
            )

        return (
            call_op.results[0] if len(call_op.results) == 1 else tuple(call_op.results)
        )

    # ----- Parse module ----

    def _build_op_from_parsed_op(
        self,
        parsed_op: Operation,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        if isinstance(parsed_op, func.CallOp):
            return self.parse_call_op(parsed_op, global_dict)

        parsed_function = self.get_parser_from_opview(type(parsed_op))
        return parsed_function(self, parsed_op, global_dict)

    def get_input_types(self, func_op: func.FuncOp):
        inputs_types = []
        inputs_shapes = []
        input_encodings = []
        for arg in func_op.type.inputs:
            if isinstance(arg, RankedTensorType):
                inputs_types.append(arg.element_type)
                inputs_shapes.append(arg.shape)
                input_encodings.append(arg.encoding)
            else:
                raise ValueError("Only ranked tensor types are supported")

        return [
            self._create_ranked_tensor_type(shape, dtype, encoding)
            for (shape, dtype, encoding) in zip(
                inputs_shapes, inputs_types, input_encodings
            )
        ]

    def generate_golden_tensors(
        self, parsed_func: func.FuncOp
    ) -> List[Dict[int, torch.Tensor]]:
        golden_inputs = []

        arg_attr_list = parsed_func.arg_attrs
        for arg_number, arg_attrs in enumerate(arg_attr_list):
            found_runtime_tensor_sharding_attr = False
            for named_attr in arg_attrs:
                if named_attr.name == "ttcore.runtime_tensor_sharding":
                    runtime_tensor_sharding_attr = (
                        ttcore.ir.RuntimeTensorShardingAttr.maybe_downcast(
                            named_attr.attr
                        )
                    )
                    arg = parsed_func.arguments[arg_number]
                    ranked_tensor_type = arg.type
                    local_shape = ranked_tensor_type.shape

                    if (
                        runtime_tensor_sharding_attr.shard_status.value
                        == ttcore.ir.ShardStatus.Presharded
                    ):
                        local_shape = runtime_tensor_sharding_attr.local_shape

                    device_golden_info = {}
                    for device_id in range(self._mesh_shape[0] * self._mesh_shape[1]):
                        device_golden_info[device_id] = self.generate_random_tensor(
                            local_shape, ranked_tensor_type.element_type
                        )
                    golden_inputs.append(device_golden_info)
                    found_runtime_tensor_sharding_attr = True
                    break

            if not found_runtime_tensor_sharding_attr:
                arg = parsed_func.arguments[arg_number]
                ranked_tensor_type = arg.type
                golden_input = self.generate_random_tensor(
                    ranked_tensor_type.shape, ranked_tensor_type.element_type
                )
                golden_inputs.append({0: golden_input})

        return golden_inputs

    def generate_random_tensor(self, shape: Shape, dtype: Type) -> torch.Tensor:
        torch_dtype = self._get_torch_dtype_from_type(dtype)

        if torch_dtype.is_floating_point or torch_dtype.is_complex:
            if len(shape) == 0:
                return torch.randn(1, dtype=torch_dtype).squeeze()
            else:
                return torch.randn(*shape, dtype=torch_dtype)
        elif torch_dtype == torch.bool:
            if len(shape) == 0:
                return torch.randint(0, 2, (), dtype=torch.bool)
            else:
                return torch.randint(0, 2, shape, dtype=torch.bool)
        else:
            if len(shape) == 0:
                return torch.randint(0, 256, (), dtype=torch_dtype)
            else:
                return torch.randint(0, 256, shape, dtype=torch_dtype)

    def parse_root_module(
        self,
        parsed_root_module: Module,
        golden_inputs: Dict[str, [List[Dict[int, torch.tensor]]]],
    ):
        found_cpu_module = False

        for entry in parsed_root_module.body.operations:
            if isinstance(entry, ttcore.CPUModuleOp):
                found_cpu_module = True
                builtin_module = entry.regions[0].blocks[0].operations[0]
                for op in builtin_module.regions[0].blocks[0].operations:
                    if isinstance(op, func.FuncOp):
                        self._hoisted_cpu_functions.append(op.name.value)
                        self._func_name_to_op[op.name.value] = op
            elif isinstance(entry, ttcore.DeviceModuleOp):
                builtin_module = entry.regions[0].blocks[0].operations[0]
                for op in builtin_module.regions[0].blocks[0].operations:
                    if isinstance(op, func.FuncOp):
                        # Only add functions with bodies, not declarations
                        if not op.is_external:
                            self._func_name_to_op[op.name.value] = op

                        for block in op.body:
                            for inner_op in block.operations:
                                if isinstance(inner_op, func.CallOp):
                                    self._nested_funcs.append(inner_op.callee.value)
            elif isinstance(entry, func.FuncOp):
                self._func_name_to_op[entry.name.value] = entry

                for block in entry.body:
                    for inner_op in block.operations:
                        if isinstance(inner_op, func.CallOp):
                            self._nested_funcs.append(inner_op.callee.value)

        new_root_module = Module.create()
        self._root_module_insertion_point = new_root_module.body
        self._current_module_insertion_point = new_root_module.body

        with InsertionPoint(new_root_module.body):
            if found_cpu_module:
                cpu_module_op = ttcore.CPUModuleOp()
                region = cpu_module_op.regions[0]
                block = Block.create_at_start(region)
                new_module = Module.create()
                cloned_op = new_module.operation.clone()
                cpu_module_op.regions[0].blocks[0].append(cloned_op.operation)
                self._cpu_module_insertion_point = cloned_op.regions[0].blocks[0]

            for entry in parsed_root_module.body.operations:
                if isinstance(entry, ttcore.DeviceModuleOp):
                    device_module_op = ttcore.DeviceModuleOp()
                    region = device_module_op.regions[0]
                    block = Block.create_at_start(region)
                    new_builtin_module = self.parse_builtin_module(
                        entry.regions[0].blocks[0].operations[0], golden_inputs
                    )
                    device_module_op.regions[0].blocks[0].append(
                        new_builtin_module.operation
                    )
                elif isinstance(entry, func.FuncOp):
                    if entry.name.value in self._nested_funcs:
                        continue
                    self.parse_func(entry, golden_inputs)

        return new_root_module

    def parse_builtin_module(
        self,
        parsed_builtin_module: Module,
        golden_inputs: Dict[str, [List[Dict[int, torch.tensor]]]],
    ):
        new_builtin_module = Module.create()
        cloned_op = new_builtin_module.operation.clone()
        self._current_module_insertion_point = cloned_op.regions[0].blocks[0]

        with InsertionPoint(cloned_op.regions[0].blocks[0]):
            for entry in parsed_builtin_module.regions[0].blocks[0].operations:
                if isinstance(entry, func.FuncOp):
                    if entry.name.value in self._nested_funcs:
                        continue
                    new_func = self.parse_func(entry, golden_inputs)

        return cloned_op

    def parse_func(
        self,
        parsed_func: func.FuncOp,
        golden_inputs: Dict[str, [List[Dict[int, torch.tensor]]]],
    ):
        fn_input_types = self.get_input_types(parsed_func)

        parsed_func_golden_inputs = []
        if parsed_func.name.value in golden_inputs.keys():
            parsed_func_golden_inputs.extend(golden_inputs[parsed_func.name.value])
        else:
            parsed_func_golden_inputs.extend(self.generate_golden_tensors(parsed_func))

        ordered_inputs = []
        ordered_outputs = []

        @func.func(*fn_input_types, name=parsed_func.name.value)
        def decorated_func(*inputs):
            golden_dict = {}
            for operand, torch_golden_dictionary in zip(
                inputs, parsed_func_golden_inputs
            ):
                golden_dict[operand] = torch_golden_dictionary

            input_goldens: Dict[Operand, GoldenMapTensor] = (
                self._create_builder_golden_from_torch_tensor(golden_dict)
            )
            self._set_goldens(input_goldens)
            ordered_inputs.extend(inputs)

            global_dict = {}
            for i, arg in enumerate(parsed_func.arguments):
                global_dict[arg] = inputs[i]

            global_result = None
            for block in parsed_func.body:
                for op in block.operations:
                    if isinstance(op, func.ReturnOp):
                        global_result = tuple(
                            global_dict[operand] for operand in op.operands
                        )
                    elif isinstance(op, ttir.EmptyOp):
                        continue
                    else:
                        (
                            parsed_op,
                            op_golden_dictionary,
                        ) = self._build_op_from_parsed_op(op, global_dict)
                        global_dict.update(op_golden_dictionary)

            outputs = (
                global_result
                if hasattr(global_result, "__iter__")
                else (global_result,)
            )
            output_goldens: Dict[Operand, GoldenMapTensor] = {}
            for op in outputs:
                output_goldens[op] = self._get_golden_tensor(op)
            self._set_goldens(output_goldens)
            ordered_outputs.extend(outputs)

            return process_multi_return_result(global_result)

        new_func_op = decorated_func.func_op
        self._func_ops_generated[new_func_op] = [ordered_inputs, ordered_outputs]

        parsed_func_op_arg_attr_list = parsed_func.arg_attrs
        new_func_op_arg_attr_list = []
        for arg_number, arg_attrs in enumerate(parsed_func_op_arg_attr_list):
            new_arg_attr = {}
            for attr in arg_attrs:
                new_arg_attr[attr.name] = attr.attr
            new_func_op_arg_attr_list.append(DictAttr.get(new_arg_attr))
        new_func_op.arg_attrs = ArrayAttr.get(new_func_op_arg_attr_list)

        return new_func_op

    def parse_nested_func(
        self, parsed_func: func.FuncOp, golden_inputs: List[GoldenMapTensor]
    ):
        fn_input_types = self.get_input_types(parsed_func)

        ordered_inputs = []
        ordered_outputs = []

        @func.func(*fn_input_types, name=parsed_func.name.value)
        def decorated_func(*inputs):
            input_goldens = {}
            for operand, golden_map_tensor in zip(inputs, golden_inputs):
                input_goldens[operand] = golden_map_tensor

            self._set_goldens(input_goldens)
            ordered_inputs.extend(inputs)

            global_dict = {}
            for i, arg in enumerate(parsed_func.arguments):
                global_dict[arg] = inputs[i]

            global_result = None
            for block in parsed_func.body:
                for op in block.operations:
                    if isinstance(op, func.ReturnOp):
                        global_result = tuple(
                            global_dict[operand] for operand in op.operands
                        )
                    else:
                        (
                            parsed_op,
                            op_golden_dictionary,
                        ) = self._build_op_from_parsed_op(op, global_dict)
                        global_dict.update(op_golden_dictionary)

            outputs = (
                global_result
                if hasattr(global_result, "__iter__")
                else (global_result,)
            )
            output_goldens: Dict[Operand, GoldenMapTensor] = {}
            for op in outputs:
                output_goldens[op] = self._get_golden_tensor(op)
            self._set_goldens(output_goldens)
            ordered_outputs.extend(outputs)

            return process_multi_return_result(global_result)

        new_func_op = decorated_func.func_op
        new_func_op.attributes["tt.function_type"] = StringAttr.get(
            "forward_cpu", self._ctx
        )
        self._func_ops_generated[new_func_op] = [ordered_inputs, ordered_outputs]
        return new_func_op

    def parse_call_op(
        self,
        parsed_op: func.CallOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[Operand, GoldenMapTensor]]:
        is_hoisted = False
        parsed_op_attributes = parsed_op.attributes
        parsed_op_callee_value = parsed_op.callee.value
        parsed_op_operands = parsed_op.operands

        for attr in parsed_op_attributes:
            if attr.name == "ttir.cpu_hoisted_call":
                is_hoisted = True
                break

        if is_hoisted:
            insertion_point = self._cpu_module_insertion_point
            hoisted_func_name = parsed_op_callee_value
            nested_func_op = self._func_name_to_op[hoisted_func_name]

            new_golden_inputs = []
            for operand in parsed_op_operands:
                owner0 = operand.owner
                if isinstance(owner0, Block):
                    queried_operand = operand
                else:
                    queried_operand = owner0.result

                new_golden_inputs.append(
                    self._get_golden_tensor(global_dict[queried_operand])
                )

            with InsertionPoint(insertion_point):
                new_func_op = self.parse_nested_func(nested_func_op, new_golden_inputs)

            with InsertionPoint(self._current_module_insertion_point):
                private_func_op = func.FuncOp(
                    type=new_func_op.type, name=hoisted_func_name, visibility="private"
                )
                private_func_op.attributes["tt.function_type"] = StringAttr.get(
                    "forward_cpu_declaration", self._ctx
                )

            self._nested_funcs.append(private_func_op.name.value)

            new_operands = [global_dict[operand] for operand in parsed_op_operands]
            call_op = func.CallOp(private_func_op, new_operands)
            call_op_result = call_op.results
            call_op.attributes["ttir.cpu_hoisted_call"] = UnitAttr.get(self._ctx)

            ordered_inputs, ordered_outputs = self._func_ops_generated[new_func_op]
            for index, output in enumerate(ordered_outputs):
                self._set_golden_tensor(
                    call_op_result[index], self._get_golden_tensor(output)
                )

            op_map_dictionary = {}
            parsed_op_results = parsed_op.results
            for old_result, new_result in zip(parsed_op_results, call_op.results):
                op_map_dictionary[old_result] = new_result

            return call_op, op_map_dictionary

        else:
            insertion_point = self._current_module_insertion_point
            nested_func_op = self._func_name_to_op[parsed_op.callee.value]
            new_golden_inputs = []
            for operand in parsed_op.operands:
                owner0 = operand.owner
                if isinstance(owner0, Block):
                    queried_operand = operand
                else:
                    queried_operand = owner0.result

                new_golden_inputs.append(
                    self._get_golden_tensor(global_dict[queried_operand])
                )

            with InsertionPoint(insertion_point):
                new_func_op = self.parse_nested_func(nested_func_op, new_golden_inputs)

            new_operands = [global_dict[operand] for operand in parsed_op.operands]
            call_op = func.CallOp(new_func_op, new_operands)

            ordered_inputs, ordered_outputs = self._func_ops_generated[new_func_op]
            for index, output in enumerate(ordered_outputs):
                self._set_golden_tensor(
                    call_op.results[index], self._get_golden_tensor(output)
                )

            op_map_dictionary = {}
            for old_result, new_result in zip(parsed_op.results, call_op.results):
                op_map_dictionary[old_result] = new_result

        return call_op, op_map_dictionary

    def split_call_op(
        self,
        old_op: func.CallOp,
    ) -> Tuple[Module, TTIRBuilder]:
        is_hoisted = False
        for attr in old_op.attributes:
            if attr.name == "ttir.cpu_hoisted_call":
                is_hoisted = True
                break

        if is_hoisted:
            sub_modules_and_builders = []
        else:
            nested_func_op = self._func_name_to_op[old_op.callee.value]
            sub_modules_and_builders = []

            for block in nested_func_op.body:
                for op in block.operations:
                    if isinstance(op, func.ReturnOp) or isinstance(
                        op,
                        ttir.EmptyOp,
                    ):
                        continue
                    elif isinstance(op, func.CallOp):
                        sub_op_module_builder = self.split_call_op(op)
                        sub_modules_and_builders.append(sub_op_module_builder)
                    else:
                        sub_op_module_builder = self.split_op(op)
                        sub_modules_and_builders.append(sub_op_module_builder)

        return sub_modules_and_builders

    # ----- Helper decorator functions ----

    def func(self, input_shapes: List[List[int]], input_types: List[torch.dtype]):
        def wrapper(fn):
            encoding_fn = self.create_tensor_encoding
            fn_input_types = [
                self._create_ranked_tensor_type(
                    shape,
                    self._get_type_from_torch_dtype(dtype),
                    encoding_fn(shape, dtype) if encoding_fn else None,
                )
                for shape, dtype in zip(input_shapes, input_types)
            ]

            ordered_inputs = []
            ordered_outputs = []

            @func.func(*fn_input_types, name=fn.__name__)
            def decorated_func(*inputs):
                if not self._disable_golden_check:
                    input_goldens: Dict[Operand, GoldenMapTensor] = {}
                    for index, (operand, dtype) in enumerate(zip(inputs, input_types)):
                        input_goldens[operand] = self._generate_golden_tensor(
                            operand, dtype
                        )
                    self._set_goldens(input_goldens)
                ordered_inputs.extend(inputs)

                result = fn(*inputs, self)

                outputs = result if hasattr(result, "__iter__") else [result]

                if not self._disable_golden_check:
                    output_goldens: Dict[Operand, GoldenMapTensor] = {}
                    for op in outputs:
                        output_goldens[op] = self._get_golden_tensor(op)
                    self._set_goldens(output_goldens)
                ordered_outputs.extend(outputs)

                return process_multi_return_result(result)

            new_func_op = decorated_func.func_op
            self._func_ops_generated[new_func_op] = [ordered_inputs, ordered_outputs]
            return new_func_op

        return wrapper

    def device_module(self, root_func: Callable):
        def wrapper(self):
            device_module_op = ttcore.DeviceModuleOp()
            region = device_module_op.regions[0]
            block = Block.create_at_start(region)
            new_module = Module.create()
            cloned_op = new_module.operation.clone()
            self._current_module_insertion_point = cloned_op.regions[0].blocks[0]

            with InsertionPoint(cloned_op.regions[0].blocks[0]):
                new_func = root_func(self)

            device_module_op.regions[0].blocks[0].append(cloned_op.operation)
            return device_module_op

        return wrapper(self)

    def cpu_module(self, root_func: Callable):
        def wrapper(self):
            cpu_module_op = ttcore.CPUModuleOp()
            region = cpu_module_op.regions[0]
            block = Block.create_at_start(region)
            new_module = Module.create()
            cloned_op = new_module.operation.clone()
            self._current_module_insertion_point = cloned_op.regions[0].blocks[0]

            with InsertionPoint(cloned_op.regions[0].blocks[0]):
                new_func = root_func(self)

            cpu_module_op.regions[0].blocks[0].append(cloned_op.operation)
            return cpu_module_op

        return wrapper(self)

    # ----- Debug dialect operations ----

    @tag(debug.AnnotateOp)
    def annotate(
        self,
        operand: Operand,
        annotation: str,
        loc: Optional[str] = None,
    ) -> OpResult:
        debug_op = self.get_opview_from_method(Builder.annotate)
        annotation_attr = StringAttr.get(annotation)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = debug_op(
            operand,
            annotation=annotation_attr,
            loc=loc,
        )
        op_result = op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(debug_op)
            golden_output = op_golden_function(input0, annotation_attr)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @tag(debug.BreakpointOp)
    def breakpoint(
        self,
        operand: Operand,
        loc: Optional[str] = None,
    ) -> OpResult:
        debug_op = self.get_opview_from_method(Builder.breakpoint)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = debug_op(
            operand,
            loc=loc,
        )
        op_result = op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(debug_op)
            golden_output = op_golden_function(input0)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @tag(debug.MemorySnapshotOp)
    def memory_snapshot(
        self,
        operand: Operand,
        file_path: str,
        loc: Optional[str] = None,
    ) -> OpResult:
        debug_op = self.get_opview_from_method(Builder.memory_snapshot)
        file_path_attr = StringAttr.get(file_path)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = debug_op(
            operand,
            file_path=file_path_attr,
            loc=loc,
        )
        op_result = op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(debug_op)
            golden_output = op_golden_function(input0, file_path_attr)
            self._set_golden_tensor(op_result, golden_output)

        return op_result
