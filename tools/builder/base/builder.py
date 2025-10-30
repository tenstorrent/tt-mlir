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
from golden import GoldenMapTensor

# ----- Public APIs -----

Operand = Union[Value, OpView, Operation]
Shape = Union[List[int], Tuple[int, ...]]


@dataclass
class TypeInfo:
    dtype: torch.dtype
    scale: Optional[float] = None
    zero_point: Optional[int] = None


class Builder:
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
        self._ctx = ctx
        self._loc = location
        self._global_id = -1
        self._disable_golden_check = disable_golden_check
        self._force_graph_level_check = False

        # Keep a list of inputs and outputs in order so we know how to store them in golden map.
        self._ordered_inputs: List[Operand] = []
        self._ordered_outputs: List[Operand] = []

        # Explicity set goldens to store. If empty, store all goldens.
        self._goldens_to_store: List[Operand] = []

        # Map from operand to its golden tensor.
        self._goldens: Dict[Operand, GoldenMapTensor] = {}

        # Map from operand to its location string.
        self._operand_to_loc: Dict[Operand, str] = {}

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

    # ----- Public methods -----

    @property
    def context(self) -> Context:
        return self._ctx

    @property
    def location(self) -> Location:
        return self._loc

    @property
    def mesh_shape(self) -> Tuple[int, int]:
        return self._mesh_shape

    @property
    def golden_map(self) -> Dict[str, Dict[int, GoldenTensor]]:
        golden_info: Dict[str, Dict[int, GoldenTensor]] = {}

        if self._disable_golden_check:
            return golden_info

        # If no specific golden is marked to be stored, store all goldens.
        if len(self._goldens_to_store) == 0:
            self._goldens_to_store = list(self._goldens.keys())

        # Always store inputs into golden map.
        for index, input in enumerate(self._ordered_inputs):
            loc = f"input_{index}"
            golden_info[loc] = self._generate_golden_device_tensor(
                loc, self._get_golden_tensor(input)
            )

        # Store outputs into golden map if they are marked to be stored.
        for index, output in enumerate(self._ordered_outputs):
            if output not in self._goldens_to_store:
                continue

            loc = f"output_{index}"
            golden_info[loc] = self._generate_golden_device_tensor(
                loc, self._get_golden_tensor(output)
            )

        if self._force_graph_level_check:
            return golden_info

        # Store other operands into golden map if they are marked to be stored.
        for operand, builder_golden_tensor in self._goldens.items():
            if operand not in self._goldens_to_store:
                continue

            if not (isinstance(operand, OpView) or isinstance(operand, Operation)):
                continue

            loc = self._operand_to_loc.get(operand, None)
            golden_info[loc] = self._generate_golden_device_tensor(
                loc, builder_golden_tensor
            )

        return golden_info

    def get_shape(self, input: Operand) -> Shape:
        return self._get_type(input).shape

    def set_goldens(
        self,
        inputs: Dict[Operand, Union[Callable, torch.tensor, Dict[int : torch.tensor]]],
        outputs: Dict[Operand, Union[torch.tensor, Dict[int : torch.tensor]]] = None,
    ):
        self._set_goldens(self._create_builder_golden_from_torch_tensor(inputs))

        if outputs != None:
            self.set_goldens_to_check(outputs.keys())
            self._set_goldens(self._create_builder_golden_from_torch_tensor(outputs))

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

    # ----- Private methods -----

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
            case _:
                raise TypeError(f"Invalid Type {dtype}")

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
        data_type: Optional[Type] = None,
        encoding: Optional[Attribute] = None,
    ) -> RankedTensorType:
        with self._ctx, self._loc:
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
        self, loc: str, builder_golden_tensor: GoldenMapTensor
    ) -> Dict[int, GoldenTensor]:
        device_golden_info: Dict[int, GoldenTensor] = {}
        contiguous_tensor = builder_golden_tensor.contiguous()
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
        golden: GoldenMapTensor,
    ):
        self._goldens[operand] = golden

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
        for operand, golden in goldens.items():
            self._set_golden_tensor(operand, golden)

    def _get_golden_tensor(
        self,
        operand: Operand,
    ) -> GoldenMapTensor:
        return self._goldens[operand]

    def _set_input_ordering(self, inputs: List[Operand]):
        self._ordered_inputs = inputs

    def _set_output_ordering(self, outputs: List[Operand]):
        self._ordered_outputs = outputs

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
    ):
        """Create a metal tensor layout using the shared implementation."""
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
