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
from builder.base.sharded_tensor import ShardedTensor

# ----- Public APIs -----

Operand = Union[Value, OpView, Operation]
Shape = Union[List[int], Tuple[int, ...]]


@dataclass
class TypeInfo:
    dtype: torch.dtype
    scale: Optional[float] = None
    zero_point: Optional[int] = None


@dataclass(frozen=True)
class Golden:
    tensor: Union[torch.Tensor, ShardedTensor]
    seed: Optional[int] = None

    def contiguous(self) -> Golden:
        return Golden(self.tensor.contiguous(), self.seed)


class GoldenCheckLevel(Enum):
    DISABLED = auto()
    OP_LEVEL = auto()
    GRAPH_LEVEL = auto()


class Builder:
    # ----- Methods -----

    def __init__(self, ctx: Context, location: Location):
        self._ctx = ctx
        self._loc = location
        self._seed = 0
        self._goldens: Dict[Operand, Golden] = {}
        self._global_id = -1
        self._id_golden_map = {}
        self._golden_check_level = GoldenCheckLevel.OP_LEVEL

    # ----- Public methods -----

    @property
    def golden_check_level(self) -> GoldenCheckLevel:
        return self._golden_check_level

    @golden_check_level.setter
    def golden_check_level(self, level: GoldenCheckLevel):
        if not isinstance(level, GoldenCheckLevel):
            raise ValueError("Invalid golden check level.")
        self._golden_check_level = level

    @property
    def context(self) -> Context:
        return self._ctx

    @property
    def golden_map(self) -> Dict:
        golden_info = {}
        if self.golden_check_level == GoldenCheckLevel.DISABLED:
            return golden_info
        for name, golden_tensor in self._id_golden_map.items():
            if self.golden_check_level == GoldenCheckLevel.GRAPH_LEVEL:
                if re.match(r"^(input|output)_[0-9]+$", name) is None:
                    # It means this is not graph level golden.
                    continue
            if isinstance(golden_tensor.tensor, ShardedTensor):
                # Skip multi-device tensors (i.e., ShardedTensor).
                # We cannot bring them back to the host until we unshard/collect,
                # so we skip them when building the golden map.
                continue
            golden_tensor = golden_tensor.contiguous()
            data_type = self._get_datatype_from_torch_dtype(golden_tensor.tensor.dtype)
            golden_info[name] = GoldenTensor(
                name,
                list(golden_tensor.tensor.shape),
                list(golden_tensor.tensor.stride()),
                data_type if data_type is not None else DataType.Float32,
                golden_tensor.tensor.data_ptr(),
                golden_tensor.tensor.numel() * golden_tensor.tensor.dtype.itemsize,
            )
        return golden_info

    def set_graph_input_output(
        self,
        inputs: List[torch.Tensor],
        outputs: Optional[List[torch.Tensor]] = None,
        override: bool = False,
    ) -> None:
        for index, tensor in enumerate(inputs):
            input_key = f"input_{index}"

            if input_key in self._id_golden_map:
                if self._id_golden_map[input_key].tensor.shape != tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for tensor '{input_key}': "
                        f"expected {self._id_golden_map[input_key].tensor.shape}, got {tensor.shape}"
                    )

                if self._id_golden_map[input_key].tensor.dtype != tensor.dtype:
                    raise ValueError(
                        f"Dtype mismatch for tensor '{input_key}': "
                        f"expected {self._id_golden_map[input_key].tensor.dtype}, got {tensor.dtype}"
                    )

            if not override and input_key in self._id_golden_map:
                continue
            self._id_golden_map[input_key] = Golden(tensor)

        if outputs is not None:
            self.golden_check_level = GoldenCheckLevel.GRAPH_LEVEL
            for index, tensor in enumerate(outputs):
                output_key = f"output_{index}"
                if not override and output_key in self._id_golden_map:
                    continue
                self._id_golden_map[output_key] = Golden(tensor)

    def get_shape(self, input: Operand) -> Shape:
        return self._get_type(input).shape

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

    def _get_next_global_id(self) -> int:
        self._global_id += 1
        return self._global_id

    def _get_name(operand: Operand) -> str:
        name = getattr(operand, "get_name", lambda: None)() or getattr(
            operand, "name", None
        )

        if name is None:
            raise ValueError(
                f"Couldn't retrieve name for operand {operand}. Check if this "
                f"operand type is properly supported."
            )

        return name

    def _operand_is_mlir_func_arg(operand: Operand) -> bool:
        return isinstance(operand, BlockArgument) and "arg" in Builder._get_name(
            operand
        )

    def _get_seed(self) -> int:
        seed = self._seed
        self._seed += 1
        return seed

    # Generates a random PyTorch tensor with the specified shape, dtype, and seed for testing.
    def _generate_random_tensor(
        self, shape: Shape, dtype: Union[torch.dtype, TypeInfo], seed: int
    ) -> torch.Tensor:
        if isinstance(dtype, TypeInfo):
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

    @property
    def _default_type(self) -> Type:
        return F32Type.get(self._ctx)

    # Extracts a RankedTensorType from a Value, OpView, or Operation, ensuring the type is ranked.
    def _get_type(self, input: Operand):
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

    # Creates an MLIR RankedTensorType from a shape, optional data type, and optional encoding.
    def _create_ranked_tensor_type(
        self,
        shape: Shape,
        data_type: Optional[Type] = None,
        encoding: Optional[Attribute] = None,
    ) -> RankedTensorType:
        dtype = data_type if data_type is not None else self._default_type

        with self._ctx, self._loc:
            return RankedTensorType.get(shape, dtype, encoding)

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

    def _generate_and_store_random_golden(
        self, operand: Operand, dtype: Union[torch.dtype, TypeInfo] = torch.float32
    ) -> Golden:
        seed = self._get_seed()
        random_tensor = self._generate_random_tensor(
            self.get_shape(operand), dtype, seed
        )
        golden = Golden(random_tensor, seed)
        self._store_golden(operand, golden)
        return golden

    def _generate_input_golden(
        self,
        operand: Operand,
        dtype: Union[torch.dtype, TypeInfo],
        index: int,
        override: bool = False,
    ) -> None:
        if not override and f"input_{index}" in self._id_golden_map:
            return self._id_golden_map[f"input_{index}"]
        golden = self._generate_and_store_random_golden(operand, dtype)
        self._id_golden_map[f"input_{index}"] = golden
        return golden

    def _get_golden(self, operand: Operand) -> Golden:
        golden = self._goldens.get(operand)

        if golden is None:
            raise ValueError(f"Expected to have a golden stored for {operand}")

        return golden

    def _store_golden(self, operand: Operand, golden: Golden) -> None:
        if self._goldens.get(operand) is not None:
            raise ValueError(f"Golden for {operand} already exists.")

        self._goldens[operand] = golden

    def _override_golden(self, operand: Operand, golden: Golden) -> None:
        if self._goldens.get(operand) is None:
            raise ValueError(
                f"Expected golden for {operand} to already exist before overriding it."
            )

        self._goldens[operand] = golden

    def _get_golden_tensor(self, operand: Operand) -> torch.Tensor:
        return self._get_golden(operand).tensor

    def _organize_eltwise_golden(self, inputs: List[Operand]):
        return [self._get_golden_tensor(inp) for inp in inputs]
