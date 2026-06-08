# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextvars import ContextVar
import os
import inspect
import time
import torch
from functools import reduce
import operator
from typing import Callable, List, Optional, Tuple, Union, Literal, Dict
from collections import OrderedDict
import json
from dataclasses import dataclass

from ttmlir.ir import *
from ttmlir.dialects import func, ttcore, ttnn, ttir
from ttmlir.passmanager import PassManager
from ttmlir.passes import (
    tt_populate_argument_types,
    ttir_to_ttnn_runtime_pipeline,
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

import _ttmlir_runtime as tt_runtime

# ----- Typedefs -----

Operand = Union[BlockArgument, OpResult]
Shape = Union[List[int], Tuple[int, ...]]


@dataclass
class TypeInfo:
    dtype: torch.dtype
    scale: Optional[float] = None
    zero_point: Optional[int] = None


@dataclass
class GridShapes:
    """Container for worker and DRAM grid shapes."""

    worker_grid_shape: List[int]
    dram_grid_shape: List[int]


# ----- Shared Helper Functions -----


def load_grid_shapes_from_system_desc(
    system_desc_path: str,
) -> GridShapes:
    """
    Load worker and DRAM grid shapes from a system descriptor file.

    This function loads the system descriptor once and extracts the grid
    information. The result can be cached to avoid repeated file I/O.

    Parameters
    ----------
    system_desc_path : str
        Path to system descriptor file. Must be provided.

    Returns
    -------
    GridShapes
        Object containing worker_grid_shape and dram_grid_shape as [rows, cols] lists

    Raises
    ------
    FileNotFoundError
        If the system descriptor file does not exist
    RuntimeError
        If the system descriptor cannot be parsed or does not contain required grid information
    """
    if not os.path.exists(system_desc_path):
        raise FileNotFoundError(f"System descriptor file not found: {system_desc_path}")

    try:
        system_desc_fbs = tt_runtime.binary.load_system_desc_from_path(system_desc_path)
        system_desc_json = json.loads(system_desc_fbs.as_json())

        chip_descs = system_desc_json["system_desc"]["chip_descs"]
        chip_desc = chip_descs[0]

        # Get worker grid dimensions (grid_size field)
        grid_size = chip_desc["grid_size"]
        worker_grid_shape = [grid_size["y"], grid_size["x"]]

        # Get DRAM grid dimensions (dram_grid_size field)
        dram_grid_size = chip_desc["dram_grid_size"]
        dram_grid_shape = [dram_grid_size["y"], dram_grid_size["x"]]

    except Exception as e:
        raise RuntimeError(
            f"Unexpected error loading system descriptor from {system_desc_path}: {e}"
        )

    return GridShapes(
        worker_grid_shape=worker_grid_shape,
        dram_grid_shape=dram_grid_shape,
    )


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
    def wrapper(module, options):
        # Split options: ttcore-register-device only accepts device-related
        # options (system-desc-path, mesh-shape, etc.). All other options
        # belong to the pipeline itself.
        device_option_prefixes = (
            "system-desc-path=",
            "mesh-shape=",
            "mock-system-desc-arch=",
            "mesh-topology=",
        )
        device_opts = []
        pipeline_opts = []
        if options:
            for opt in options.split():
                if opt.startswith(device_option_prefixes):
                    device_opts.append(opt)
                else:
                    pipeline_opts.append(opt)

        register_device = "ttcore-register-device"
        if device_opts:
            register_device = f"{register_device}{{{' '.join(device_opts)}}}"

        pipeline_with_opts = pipeline
        if pipeline_opts:
            pipeline_with_opts = f"{pipeline}{{{' '.join(pipeline_opts)}}}"

        pipeline_str = (
            f"builtin.module({','.join([register_device, pipeline_with_opts])})"
        )
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
        torch.int32, torch.uint16, torch.uint32 and torch.bfloat16.

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
            memorySpace,
            memory_layout,
            collapse_intervals,
            list(dim_alignments),
        )
    else:
        layout = ttcore.ir.MetalLayoutAttr.get(
            ctx, logical_shape, memorySpace, memory_layout
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
    elif element_dtype == torch.uint32:
        elemType = IntegerType.get_unsigned(32, ctx)
        tile_elem_dtype = ttcore.DataType.UInt32
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


def derive_canonical_core_range_set(
    ctx: Context,
    buffer_type: ttnn.ir.BufferTypeAttr,
    tensor_memory_layout: ttnn.ir.TensorMemoryLayoutAttr,
    grid_shape: List[int],
    worker_grid_shape: List[int],
    dram_grid_shape: List[int],
) -> Optional[ttnn.ir.CoreRangeSetAttr]:
    """
    Derive the canonical CoreRangeSet for a sharded TTNN layout.

    This function replicates the logic from TTNNLayoutAttr::Builder::buildWithCanonicalCorePlacement
    in the C++ codebase. It derives core placements based on the buffer type (L1 or DRAM) and
    memory layout (BlockSharded, HeightSharded, or WidthSharded).

    This is a pure computation function that requires grid shapes to be provided by the caller.
    Use Builder._get_grid_shapes() to obtain cached grid shapes from the system descriptor.

    Parameters
    ----------
    ctx : Context
        MLIR context
    buffer_type : ttnn.ir.BufferType
        Buffer type (L1, DRAM, or SystemMemory)
    tensor_memory_layout : ttnn.ir.TensorMemoryLayout
        Memory layout (Interleaved, BlockSharded, HeightSharded, WidthSharded)
    grid_shape : List[int]
        Grid shape as [height, width]
    worker_grid_shape : List[int]
        Worker grid shape as [rows, cols]. Required for L1-sharded layouts.
    dram_grid_shape : List[int]
        DRAM grid shape as [rows, cols]. Required for DRAM-sharded layouts.

    Returns
    -------
    Optional[ttnn.ir.CoreRangeSetAttr]
        The canonical CoreRangeSet, or None if not sharded
    """
    # Only calculate for sharded layouts
    if tensor_memory_layout == ttnn.TensorMemoryLayout.Interleaved:
        return None

    # Calculate CoreRangeSet based on buffer type and memory layout
    if buffer_type == ttnn.BufferType.DRAM:
        # DRAM-sharded: single rectangle covering first N banks
        return _derive_canonical_dram_core_range_set(
            ctx, tensor_memory_layout, grid_shape, dram_grid_shape
        )
    elif buffer_type == ttnn.BufferType.L1:
        # L1-sharded: map virtual grid to physical cores
        return _derive_canonical_l1_core_range_set(
            ctx, tensor_memory_layout, grid_shape, worker_grid_shape
        )

    raise ValueError(
        f"Unsupported sharded buffer type {buffer_type!r}; expected one of "
        f"{ttnn.BufferType.DRAM!r} or {ttnn.BufferType.L1!r}."
    )


def _derive_canonical_l1_core_range_set(
    ctx: Context,
    mem_layout: ttnn.ir.TensorMemoryLayoutAttr,
    grid_shape: List[int],
    worker_grid_shape: List[int],
) -> ttnn.ir.CoreRangeSetAttr:
    """
    Derive canonical CoreRangeSet for L1-sharded layout.

    Mirrors the C++ deriveCanonicalL1CoreRangeSet function.
    """
    assert len(grid_shape) == 2, "Grid shape must be 2D"
    assert len(worker_grid_shape) == 2, "Worker grid shape must be 2D"

    worker_grid_volume = worker_grid_shape[0] * worker_grid_shape[1]

    if mem_layout == ttnn.TensorMemoryLayout.BlockSharded:
        # Virtual [H, W] maps identity onto physical cores (0,0)-(W-1, H-1)
        assert (
            grid_shape[0] <= worker_grid_shape[0]
            and grid_shape[1] <= worker_grid_shape[1]
        ), f"BlockSharded grid {grid_shape} does not fit in worker grid {worker_grid_shape}"

        ranges = [
            ttnn.ir.CoreRangeAttr.get(
                ctx,
                ttnn.ir.CoreCoordAttr.get(ctx, 0, 0),
                ttnn.ir.CoreCoordAttr.get(ctx, grid_shape[1] - 1, grid_shape[0] - 1),
            )
        ]
    elif mem_layout == ttnn.TensorMemoryLayout.HeightSharded:
        # Virtual [M, 1] row-major flattens onto (m / W, m % W)
        assert grid_shape[1] == 1, "HeightSharded expects [M, 1] grid"
        assert (
            grid_shape[0] <= worker_grid_volume
        ), f"HeightSharded count {grid_shape[0]} exceeds worker volume {worker_grid_volume}"

        ranges = _build_row_major_core_ranges(ctx, grid_shape[0], worker_grid_shape)
    elif mem_layout == ttnn.TensorMemoryLayout.WidthSharded:
        # Virtual [1, M] row-major flattens onto (m / W, m % W)
        assert grid_shape[0] == 1, "WidthSharded expects [1, M] grid"
        assert (
            grid_shape[1] <= worker_grid_volume
        ), f"WidthSharded count {grid_shape[1]} exceeds worker volume {worker_grid_volume}"

        ranges = _build_row_major_core_ranges(ctx, grid_shape[1], worker_grid_shape)
    else:
        raise ValueError(f"Unexpected memory layout: {mem_layout}")

    return ttnn.ir.CoreRangeSetAttr.get(ctx, ranges)


def _derive_canonical_dram_core_range_set(
    ctx: Context,
    mem_layout: ttnn.ir.TensorMemoryLayoutAttr,
    grid_shape: List[int],
    dram_grid_shape: List[int],
) -> ttnn.ir.CoreRangeSetAttr:
    """
    Derive canonical CoreRangeSet for DRAM-sharded layout.

    Mirrors the C++ deriveCanonicalDramCoreRangeSet function.
    """
    assert mem_layout in [
        ttnn.TensorMemoryLayout.HeightSharded,
        ttnn.TensorMemoryLayout.WidthSharded,
    ], f"DRAM-sharded only supports HeightSharded/WidthSharded, got {mem_layout}"

    assert len(dram_grid_shape) == 2, "DRAM grid must be 2D"
    assert dram_grid_shape[0] == 1, f"DRAM grid expected [1, N], got {dram_grid_shape}"

    shard_volume = grid_shape[0] * grid_shape[1]
    dram_volume = dram_grid_shape[0] * dram_grid_shape[1]

    assert (
        shard_volume <= dram_volume
    ), f"Shard volume {shard_volume} exceeds DRAM volume {dram_volume}"
    assert shard_volume >= 1, "Shard volume must be at least 1"

    ranges = [
        ttnn.ir.CoreRangeAttr.get(
            ctx,
            ttnn.ir.CoreCoordAttr.get(ctx, 0, 0),
            ttnn.ir.CoreCoordAttr.get(ctx, shard_volume - 1, 0),
        )
    ]

    return ttnn.ir.CoreRangeSetAttr.get(ctx, ranges)


def _build_row_major_core_ranges(
    ctx: Context,
    shard_count: int,
    physical_grid_shape: List[int],
) -> List[ttnn.ir.CoreRangeAttr]:
    """
    Build CoreRangeAttr list for row-major flattening of shards onto physical grid.

    This implementation matches the C++ buildRowMajorCoreRanges function, which
    coalesces the core ranges into at most two rectangles:
    1. One large W x H block covering all full rows
    2. Optionally one tail 1 x W' strip for any partial row

    Maps linear shard indices to physical cores in row-major order.

    Parameters
    ----------
    ctx : Context
        MLIR context
    shard_count : int
        Total number of shards to map
    physical_grid_shape : List[int]
        Physical grid shape as [rows, cols]

    Returns
    -------
    List[ttnn.ir.CoreRangeAttr]
        List of at most 2 CoreRangeAttr covering the shards in row-major order
    """
    ranges = []
    grid_width = physical_grid_shape[1]

    # Calculate how many full rows and remaining cores
    full_rows = shard_count // grid_width
    tail_cores = shard_count % grid_width

    # Add the main block covering all full rows (if any)
    if full_rows > 0:
        ranges.append(
            ttnn.ir.CoreRangeAttr.get(
                ctx,
                ttnn.ir.CoreCoordAttr.get(ctx, 0, 0),
                ttnn.ir.CoreCoordAttr.get(ctx, grid_width - 1, full_rows - 1),
            )
        )

    # Add the tail strip for the partial row (if any)
    if tail_cores > 0:
        tail_y = full_rows
        ranges.append(
            ttnn.ir.CoreRangeAttr.get(
                ctx,
                ttnn.ir.CoreCoordAttr.get(ctx, 0, tail_y),
                ttnn.ir.CoreCoordAttr.get(ctx, tail_cores - 1, tail_y),
            )
        )

    return ranges


class DeferredDevice:
    """Device that opens after compilation, not before.

    The optimizer pipeline uses OpModel's internal mock device during
    compilation. If a real device is already open, mock device creation
    fails. Pass ``DeferredDevice(request)`` as the ``device`` argument to
    ``compile_and_execute_ttir`` so the real device is opened only for
    execution.
    """

    def __init__(self, request):
        self._request = request

    def prepare(self):
        """Close any cached device from prior tests so compilation can use
        the mock device without conflict."""
        # Expected to be used through pytest
        from conftest import _current_device, clear_device_cache
        import _ttmlir_runtime as tt_runtime

        if _current_device is not None:
            tt_runtime.runtime.close_mesh_device(_current_device)
            tt_runtime.runtime.set_fabric_config(
                tt_runtime.runtime.FabricConfig.DISABLED
            )
            clear_device_cache()

    def open(self):
        return self._request.getfixturevalue("device")

    def close(self, device):
        """Close the device and clear the fixture cache so the next test
        can compile with a mock device."""
        import _ttmlir_runtime as tt_runtime

        tt_runtime.runtime.close_mesh_device(device)
        tt_runtime.runtime.set_fabric_config(tt_runtime.runtime.FabricConfig.DISABLED)
        from conftest import clear_device_cache

        clear_device_cache()
