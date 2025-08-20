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

from ttmlir.ir import *
from ttmlir.dialects import tensor, quant
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
        self._ctx = ctx
        self._loc = location
        self._global_id = -1
        self._golden_check_level = golden_check_level
        self._mesh_shape = mesh_shape
        self._goldens: Dict[Operand, BuilderGoldenTensor] = {}
        self._module_dictionary: Dict[str, BuilderGoldenTensor] = {}

        """
        The reason why we need _goldens and _module_dictionary is:
        1. _goldens is used to store the Operand which is used when looking up the input golden tensor for an operation from an Operand type.
        2. _module_dictionary is used to store the location string to BuilderGoldenTensor mapping which is used when generating the golden map for the module.
        These 2 dictionaries have to be separate because BlockArguments do not have location information
        """

    # ----- Public methods -----

    @property
    def golden_check_level(self) -> GoldenCheckLevel:
        return self._golden_check_level

    def set_golden_check_level(self, level: GoldenCheckLevel):
        self._golden_check_level = level

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

        if self._golden_check_level == GoldenCheckLevel.DISABLED:
            return golden_info

        for loc, builder_golden_tensor in self._module_dictionary.items():
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

            golden_info[loc] = device_golden_info

        return golden_info

    def get_shape(self, input: Operand) -> Shape:
        return self._get_type(input).shape

    def set_input_goldens(
        self, inputs: List[Operand], goldens: List[BuilderGoldenTensor]
    ):
        """
        Provide a list of golden tensors for the given list of input operands.
        The location of each input operand will be calculated internally to preserve ordering when loading these during runtime.
        """
        for index, (input, golden) in enumerate(zip(inputs, goldens)):
            self._set_module_dictionary(f"input_{index}", golden)
            self._set_golden_tensor(input, golden)

    def set_output_goldens(
        self, outputs: List[Operand], goldens: List[BuilderGoldenTensor]
    ):
        """
        Provide a list of golden tensors for the given list of output operands.
        The location of each output operand will be calculated internally to preserve ordering when loading these during runtime.
        """
        for index, (output, golden) in enumerate(zip(outputs, goldens)):
            self._set_module_dictionary(f"output_{index}", golden)
            self._set_golden_tensor(output, golden)

    def set_operand_golden(
        self, operand: Operand, golden: BuilderGoldenTensor, manual: bool = False
    ):
        """
        Provide a golden tensor for the given operand (OpView or Operation).
        The location of the operand is used as the key in the golden map.
        """
        if isinstance(operand, OpView):
            loc = str(operand.operation.location)
        elif isinstance(operand, Operation):
            loc = str(operand.location)
        else:
            raise TypeError(
                "Operand must be OpView or Operation to set golden tensor. Call set_input_goldens or set_output_goldens for inputs/outputs."
            )

        self._set_golden_tensor(operand, golden)
        if (
            self._golden_check_level == GoldenCheckLevel.AUTOMATIC
            or self._golden_check_level == GoldenCheckLevel.MANUAL
            and manual
        ):
            self._set_module_dictionary(loc, golden)

    # ----- Private methods -----

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

    def _generate_golden_tensor(
        self, operand: Operand, dtype: Union[torch.dtype, TypeInfo]
    ) -> BuilderGoldenTensor:
        random_tensor = self._generate_random_tensor(self.get_shape(operand), dtype)
        return BuilderGoldenTensor({0: random_tensor}, mesh_shape=self._mesh_shape)

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

    def _set_module_dictionary(
        self,
        loc: str,
        golden: BuilderGoldenTensor,
    ):
        self._module_dictionary[loc] = golden
