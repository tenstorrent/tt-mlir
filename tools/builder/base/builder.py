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

import ttmlir
from ttmlir.ir import *
from ttmlir.dialects import *
from ttmlir.passes import GoldenTensor, DataType
from builder.base.builder_golden import BuilderGoldenTensor

# ----- Public APIs -----

Operand = Union[Value, OpView, Operation]
Shape = Union[List[int], Tuple[int, ...]]


@dataclass
class TypeInfo:
    dtype: torch.dtype
    scale: Optional[float] = None
    zero_point: Optional[int] = None


class GoldenCheckLevel(Enum):
    DISABLED = auto()
    MANUAL = auto()
    AUTOMATIC = auto()


class Builder:
    # ----- Methods -----

    def __init__(
        self,
        ctx: Context,
        location: Location,
        mesh_shape: Tuple[int, int] = (1, 1),
        golden_check_level: GoldenCheckLevel = GoldenCheckLevel.AUTOMATIC,
    ):
        ttmlir._mlir_libs._ttmlir.register_dialect(ctx)
        self._ctx = ctx
        self._loc = location
        self._global_id = -1
        self._golden_check_level = golden_check_level
        self._mesh_shape = mesh_shape
        self._goldens: Dict[Operand, BuilderGoldenTensor] = {}

        # We need a separate dictionary for input/output mapping because mlir Values don't have location info.
        self._input_mapping: Dict[Operand, str] = {}
        self._output_mapping: Dict[Operand, str] = {}

    # ----- Public methods -----

    @property
    def golden_check_level(self) -> GoldenCheckLevel:
        return self._golden_check_level

    @property
    def context(self) -> Context:
        return self._ctx

    @property
    def golden_map(self) -> Dict[str, Dict[int, GoldenTensor]]:
        print("taps0")
        golden_info: Dict[str, Dict[int, GoldenTensor]] = {}

        print("HEREEEE")
        print(self._goldens)
        print("HEREEEE")



        print(self._goldens)
        with self._ctx, self._loc:

            if self._golden_check_level == GoldenCheckLevel.DISABLED:
                return golden_info

            # Handle all operation outputs.
            for operand, builder_golden_tensor in self._goldens.items():
                print("yoyo")
                print(type(operand))
                print("yoyo")
                if isinstance(operand, Value):
                    continue
                elif isinstance(operand, OpView):
                    print("omg0")
                    #print(operand)
                    loc = str(operand.operation.location)
                    print(loc)
                    print("omg1")
                elif isinstance(operand, Operation):
                    print("skdhdsh1")
                    loc = str(operand.location)
                    print("skdhdsh2")

                print("WOWOW")

                golden_info[loc] = self._generate_device_golden_info(
                    loc, builder_golden_tensor
                )
            print("taps1")

            # Handle all inputs.
            for operand, loc in self._input_mapping.items():
                builder_golden_tensor = self._goldens[operand]
                golden_info[loc] = self._generate_device_golden_info(
                    loc, builder_golden_tensor
                )

            print("taps2")

            # Handle all outputs.
            for operand, loc in self._output_mapping.items():
                builder_golden_tensor = self._goldens[operand]
                golden_info[loc] = self._generate_device_golden_info(
                    loc, builder_golden_tensor
                )

            print("taps3")

            return golden_info

    def get_shape(self, input: Operand) -> Shape:
        return self._get_type(input).shape

    def set_input_goldens(
        self, inputs: List[Operand], goldens: List[BuilderGoldenTensor]
    ):
        for index, (input, golden) in enumerate(zip(inputs, goldens)):
            self._set_golden_tensor(input, golden)
            self._set_input_mapping(input, index)

    def set_output_goldens(
        self, outputs: List[Operand], goldens: List[BuilderGoldenTensor]
    ):
        for index, (output, golden) in enumerate(zip(outputs, goldens)):
            self._set_output_mapping(output, index)

    def set_operand_golden(self, operand: Operand, golden: BuilderGoldenTensor):
        self._set_golden_tensor(operand, golden)

    # ----- Private methods -----

    def _generate_device_golden_info(
        self, loc: str, builder_golden_tensor: BuilderGoldenTensor
    ):
        device_golden_info: Dict[int:GoldenTensor] = {}
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

    def _get_datatype_from_torch_dtype(self, dtype: torch.dtype) -> DataType:
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

    def _get_next_global_id(self) -> int:
        self._global_id += 1
        return self._global_id

    # Generates a random PyTorch tensor with the specified shape, dtype, and seed for testing.
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

    # Extracts a RankedTensorType from a Value, OpView, or Operation, ensuring the type is ranked.
    def _get_type(self, input: Operand):
        with self._ctx:
            if isinstance(input, Value):
                typ = input.type
            elif isinstance(input, OpView):
                typ = input.operation.result.type
            elif isinstance(input, Operation):
                typ = input.result.type
            else:
                raise TypeError(f"Invalid input {type(input)}")

            if not isinstance(typ, RankedTensorType):
                raise TypeError("Only ranked tensors are supported")

            return typ

    def _get_loc_of_extra_file_callee(self, id: int = 0) -> Location:
        stack = inspect.stack()
        caller_filename = stack[1].filename

        # Skip all frames from the current file
        while len(stack) > 0 and stack[0].filename == caller_filename:
            stack = stack[1:]

        if len(stack) == 0:
            raise RuntimeError(
                "Top of callstack to builder funcs must be outside the caller's file"
            )

        # Build a location string
        loc_str = f"{stack[0].filename}:{stack[0].lineno}:id({id})"

        # Create a Location tied to the MLIR context
        with self._ctx:
            return Location.name(loc_str, context=self._ctx)


    # Creates an MLIR RankedTensorType from a shape, optional data type, and optional encoding.
    def _create_ranked_tensor_type(
        self,
        shape: Shape,
        data_type: Optional[Type] = None,
        encoding: Optional[Attribute] = None,
    ) -> RankedTensorType:
        dtype = data_type if data_type is not None else F32Type.get(self._ctx)

        loc = Location.unknown(self._ctx)

        return RankedTensorType.get(shape, dtype, loc=loc)

    # Converts a torch.dtype or TypeInfo (with optional scale and zero_point) into the corresponding MLIR type.
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
                print("DKSHKDHJKSHDJKSHDJHSDSD")
                print(self._ctx)
                print("DKSHKDHJKSHDJKSHDJHSDSD")
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

    def _generate_golden_tensor(
        self, operand: Operand, dtype: Union[torch.dtype, TypeInfo]
    ) -> BuilderGoldenTensor:
        random_tensor = self._generate_random_tensor(self.get_shape(operand), dtype)
        golden = BuilderGoldenTensor({0: random_tensor}, mesh_shape=self._mesh_shape)
        return golden

    def _organize_eltwise_golden(
        self, inputs: List[Operand]
    ) -> List[BuilderGoldenTensor]:
        return [self._goldens[inp] for inp in inputs]

    def _get_golden_tensor(
        self,
        operand: Operand,
    ) -> BuilderGoldenTensor:
        return self._goldens[operand]

    def _set_golden_tensor(
        self,
        operand: Operand,
        golden: BuilderGoldenTensor,
    ):
        
        self._goldens[operand] = golden

        print("OPERANDDDDDDDD")
        print(self._goldens)
        print("OPERANDDDDDDDD")

    def _set_input_mapping(
        self,
        operand: Operand,
        index: int,
    ):
        self._input_mapping[operand] = f"input_{index}"

    def _set_output_mapping(
        self,
        operand: Operand,
        index: int,
    ):
        self._output_mapping[operand] = f"output_{index}"
