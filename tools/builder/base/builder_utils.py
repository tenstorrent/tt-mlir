# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextvars import ContextVar
import os
import inspect
import time
import re
import torch
from functools import reduce
import operator
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict, Any
from collections import OrderedDict
import json
from dataclasses import dataclass

from ttmlir.ir import *
from ttmlir.dialects import func, ttcore, ttnn, ttir
from ttmlir.passmanager import PassManager
from ttmlir.passes import (
    tt_populate_argument_types,
    ttir_to_ttnn_backend_pipeline,
    ttnn_to_flatbuffer_file,
    ttir_to_ttmetal_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    translate_to_cpp,
    translate_to_python,
    MLIRModuleLogger,
    stablehlo_pipeline,
    stablehlo_to_ttir_pipeline,
    ttir_to_emitpy_pipeline,
)

# ----- Typedefs -----

Operand = Union[BlockArgument, OpResult]
Shape = Union[List[int], Tuple[int, ...]]


@dataclass
class TypeInfo:
    dtype: torch.dtype
    scale: Optional[float] = None
    zero_point: Optional[int] = None


# ----- Shared Helper Functions -----


def _torch_dtype_from_mlir_token(token: str) -> torch.dtype:
    match token.strip():
        case "bf16":
            return torch.bfloat16
        case "f16":
            return torch.float16
        case "f32":
            return torch.float32
        case "f64":
            return torch.float64
        case "i8":
            return torch.int8
        case "i16":
            return torch.int16
        case "i32":
            return torch.int32
        case "i64":
            return torch.int64
        case "ui8":
            return torch.uint8
        case "ui16":
            return torch.uint16
        case "ui32":
            return torch.uint32
        case "ui64":
            return torch.uint64
        case _:
            raise TypeError(f"Unsupported MLIR type token: {token}")


def _torch_quant_dtype_from_storage(storage: str) -> torch.dtype:
    storage = storage.strip()
    match storage:
        case "i8":
            return torch.qint8
        case "ui8":
            return torch.quint8
        case "i32":
            return torch.qint32
        case _:
            raise TypeError(f"Unsupported quantized storage type: {storage}")


def parse_quantized_type(mlir_type: Type) -> Optional[Dict[str, Any]]:
    type_str = str(mlir_type).strip()
    match = re.fullmatch(
        r"!quant\.uniform<(?P<storage>[^:>]+):(?P<expressed>[^,:>]+)"
        r"(?::(?P<axis>-?\d+))?, (?P<params>\{.*\}|[^>]+)>",
        type_str,
    )
    if match is None:
        return None

    params = match.group("params").strip()
    if params.startswith("{"):
        param_entries = [entry.strip() for entry in params[1:-1].split(",") if entry]
    else:
        param_entries = [params]

    scales = []
    zero_points = []
    for entry in param_entries:
        scale_text, zero_point_text = (
            entry.split(":", 1) if ":" in entry else (entry, None)
        )
        scales.append(float(scale_text))
        zero_points.append(0 if zero_point_text is None else int(zero_point_text))

    quantized_dimension = match.group("axis")
    return {
        "storage_dtype": _torch_quant_dtype_from_storage(match.group("storage")),
        "expressed_dtype": _torch_dtype_from_mlir_token(match.group("expressed")),
        "quantized_dimension": (
            None if quantized_dimension is None else int(quantized_dimension)
        ),
        "scales": scales,
        "zero_points": zero_points,
    }


def normalize_quantized_dimension(quantized_dimension: int, rank: int) -> int:
    if quantized_dimension < 0:
        quantized_dimension += rank

    if quantized_dimension < 0 or quantized_dimension >= rank:
        raise ValueError(
            "Per-axis quantized type dimension must be within the tensor rank. "
            f"Got axis {quantized_dimension} for rank {rank}."
        )

    return quantized_dimension


def tag(name):
    def decorator(func):
        func._tag = name
        return func

    return decorator


def parse(name):
    def decorator(func):
        func._parse = name
        return func

    return decorator


def split(name):
    def decorator(func):
        func._split = name
        return func

    return decorator


def get_target_path(output_path, builder_dir, filename, target):
    target_dir = os.path.join(output_path, builder_dir, target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return os.path.join(target_dir, filename)


def get_artifact_dir(output_root, builder_type, test_base, make_dir=True):
    artifact_path = os.path.join(
        output_root, "builder-artifacts", builder_type, test_base
    )
    if make_dir and not os.path.exists(artifact_path):
        os.makedirs(artifact_path)
    return artifact_path


def emitc_to_executable(module):
    return translate_to_cpp(module)


def emitpy_to_executable(module):
    return translate_to_python(module)


def _convert_to_mlir_value(obj):
    if hasattr(obj, "operation") and hasattr(obj.operation, "results"):
        results = obj.operation.results
        if len(results) == 1:
            return results[0]
        else:
            return results
    elif hasattr(obj, "type"):
        return obj
    else:
        return obj


def process_multi_return_result(result):
    if hasattr(result, "__iter__") and not isinstance(result, str):
        converted_results = []
        for item in result:
            converted = _convert_to_mlir_value(item)
            if hasattr(converted, "__iter__") and not hasattr(converted, "type"):
                converted_results.extend(converted)
            else:
                converted_results.append(converted)
        return tuple(converted_results)
    else:
        return _convert_to_mlir_value(result)


def create_custom_ttir_pipeline_fn(
    pipeline: str, verify: bool = True, print_ir: Union[bool, str] = False
) -> Callable:
    def wrapper(module, device_register_options):
        register_device = "ttcore-register-device"
        if device_register_options:
            register_device = f"{register_device}{{{device_register_options}}}"

        pipeline_str = f"builtin.module({','.join([register_device, pipeline])})"
        with module.context:
            pm = PassManager.parse(pipeline_str)
            pm.enable_verifier(verify)
            print("Running custom pipeline:", pm)
            if print_ir:
                print_ir_path = print_ir if isinstance(print_ir, str) else None
                pm.enable_ir_printing(tree_printing_dir_path=print_ir_path)
            pm.run(module.operation)

    return wrapper


def run_ttir_pipeline(
    module,
    pipeline_fn: Callable,
    pipeline_options: Optional[List[str]] = None,
    save_artifacts: bool = False,
    output_file_name: str = "test.mlir",
    system_desc_path: Optional[str] = None,
    mesh_dict: OrderedDict[str, int] = OrderedDict([("x", 1), ("y", 1)]),
    argument_types_string: Optional[str] = None,
):
    if pipeline_options is None:
        pipeline_options = []

    if argument_types_string:
        tt_populate_argument_types(module, argument_types_string)
        pipeline_options.append("enable-const-eval=true")

    # Default to the `SYSTEM_DESC_PATH` envvar.
    if system_desc_path is None:
        system_desc_path = os.getenv("SYSTEM_DESC_PATH", "")
    pipeline_options.append(f"system-desc-path={system_desc_path}")

    mesh_shape = tuple(mesh_dict.values())
    if len(mesh_shape) != 2:
        raise ValueError(f"Mesh shape must be a tuple of length 2, got: {mesh_shape}")

    pipeline_options.append(f"mesh-shape={mesh_shape[0]},{mesh_shape[1]}")

    # Now, pass it through the pipeline. Module gets modified in place.
    pipeline_fn(module, " ".join(pipeline_options))

    # Optionally dump to file.
    if save_artifacts:
        with open(output_file_name, "w") as f:
            f.write(str(module))

    return module


def get_metal_tensor_layout(
    ctx: Context,
    logical_shape: Shape,
    tiled=False,
    element_dtype: torch.dtype = torch.float32,
    oobVal=ttcore.OOBVal.Undef,
    memorySpace=ttcore.MemorySpace.DeviceL1,
    grid: Optional[Tuple[int, int]] = None,
    index_map: Optional[AffineMap] = None,
    memory_layout: Optional[
        ttcore.TensorMemoryLayout
    ] = ttcore.TensorMemoryLayout.Sharded,
    dim_alignments: Optional[Tuple[int, ...]] = None,
) -> RankedTensorType:
    """
    Create a metal tensor layout.

    This function creates metal tensor layouts for both TTIR and D2M operations.
    Previously duplicated between TTIRBuilder and D2MBuilder.

    Parameters
    ----------
    ctx : Context
        MLIR context
    logical_shape : Shape
        Logical shape of the tensor
    tiled : bool
        Whether to use tiled layout (32x32 tiles)
    oobVal : ttcore.OOBVal
        Out-of-bounds value handling
    memorySpace : ttcore.MemorySpace
        Memory space (L1, DRAM, etc.)
    grid : Optional[Tuple[int, int]]
        Grid shape for sharding
    index_map : Optional[AffineMap]
        Deprecated. Remapping is now carried by view/stream ops, not layouts.
    dim_alignments : Optional[Tuple[int, ...]]
        Optional explicit dimension alignments. When specified, the tensor
        will be padded to these alignments regardless of tile size. Useful
        for testing masking of complete out-of-bounds tiles.
    element_dtype : torch.dtype
        Element dtype for the resulting tensor. Supports torch.float32,
        torch.int32, torch.uint16, and torch.bfloat16.

    Returns
    -------
    RankedTensorType
        The metal tensor type with layout
    """
    import numpy as np

    # Create grid shape by 1s filling logical rank.
    if grid is None:
        original_rank = len(logical_shape)
        grid_shape = [1] * original_rank
    else:
        grid_shape = list(grid)

    # Create layout with original logical shape.
    if dim_alignments is not None:
        # Use 8-arg overload with explicit dim_alignments.
        # Create collapse_intervals as [[0, rank-1]] (collapse all dims).
        rank = len(logical_shape)
        intervals_np = np.array([[0, rank - 1]], dtype=np.int64)
        collapse_intervals = DenseElementsAttr.get(intervals_np)

        layout = ttcore.ir.MetalLayoutAttr.get(
            ctx,
            logical_shape,
            oobVal,
            memorySpace,
            memory_layout,
            collapse_intervals,
            list(dim_alignments),
        )
    else:
        layout = ttcore.ir.MetalLayoutAttr.get(
            ctx, logical_shape, oobVal, memorySpace, memory_layout
        )

    shard_shape = []
    for l, g in zip(logical_shape, grid_shape):
        assert l % g == 0, f"Logical shape {l} must be divisible by grid shape {g}"
        shard_shape.append(l // g)

    # Get sharded shape w/ proper collapse & alignment logic.
    typed_layout = ttcore.ir.MetalLayoutAttr.maybe_downcast(layout)
    if typed_layout is None:
        raise RuntimeError("Failed to downcast MetalLayoutAttr")
    device_shape = typed_layout.getDeviceShape(
        grid_shape, [32, 32] if tiled else [1, 1]
    )

    if element_dtype == torch.float32:
        elemType = F32Type.get(ctx)
        tile_elem_dtype = ttcore.DataType.Float32
    elif element_dtype == torch.int32:
        elemType = IntegerType.get_signed(32, ctx)
        tile_elem_dtype = ttcore.DataType.Int32
    elif element_dtype == torch.uint16:
        elemType = IntegerType.get_unsigned(16, ctx)
        tile_elem_dtype = ttcore.DataType.UInt16
    elif element_dtype == torch.bfloat16:
        elemType = BF16Type.get(ctx)
        tile_elem_dtype = ttcore.DataType.BFloat16
    else:
        raise ValueError(f"Unsupported dtype for metal layout: {element_dtype}")

    # For tiled layouts, ensure the device shape accounts for tiles.
    if tiled:
        elemType = ttcore.ir.TileType.get(ctx, 32, 32, tile_elem_dtype)
        if grid is None or grid == (1, 1):
            # For default 1x1 grid, use tile count based on aligned shape.
            # If dim_alignments is specified, use that; otherwise use logical_shape.
            if dim_alignments is not None:
                aligned_h = dim_alignments[-2]
                aligned_w = dim_alignments[-1]
            else:
                aligned_h = logical_shape[-2]
                aligned_w = logical_shape[-1]
            tile_count_h = (aligned_h + 31) // 32
            tile_count_w = (aligned_w + 31) // 32
            device_shape[-2] = tile_count_h
            device_shape[-1] = tile_count_w
        else:
            # For explicit grids, calculate proper sharded tile count.
            shard_h, shard_w = shard_shape[-2], shard_shape[-1]
            tiles_per_shard_h = (shard_h + 31) // 32
            tiles_per_shard_w = (shard_w + 31) // 32
            device_shape[-2] = tiles_per_shard_h
            device_shape[-1] = tiles_per_shard_w

    return RankedTensorType.get(device_shape, elemType, layout, Location.unknown(ctx))


def affine_map_from_lambda(fn):
    class Dim:
        def __init__(self, position, name):
            self.position = position
            self.name = name

    dims = tuple(
        Dim(i, name) for i, name in enumerate(inspect.signature(fn).parameters)
    )
    num_dims = len(dims)
    results = fn(*dims)
    exprs = []
    for result in results:
        if isinstance(result, Dim):
            exprs.append(AffineDimExpr.get(result.position))
        elif isinstance(result, int):
            assert (
                result == 0
            ), "The only integer constant allowed in an indexing_map is 0"
            exprs.append(AffineConstantExpr.get(result))
        else:
            raise TypeError(
                f"Unsupported indexing_map result type `{type(result)}` for result `{result}`"
            )
    num_syms = 0
    return AffineMap.get(num_dims, num_syms, exprs)
