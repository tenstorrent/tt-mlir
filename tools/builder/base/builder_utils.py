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
"""An MLIR value that can serve as an input to an operation (block argument or op result)."""

Shape = Union[List[int], Tuple[int, ...]]
"""Tensor shape expressed as a list or tuple of dimension sizes."""


@dataclass
class TypeInfo:
    """Extended dtype descriptor carrying optional quantisation parameters.

    When ``scale`` and ``zero_point`` are provided, the builder creates a
    ``quant.UniformQuantizedType`` instead of a plain integer/float type.

    Parameters
    ----------
    dtype : torch.dtype
        Base PyTorch dtype (e.g. ``torch.qint32``).
    scale : float, optional
        Quantisation scale factor.
    zero_point : int, optional
        Quantisation zero point.
    """

    dtype: torch.dtype
    scale: Optional[float] = None
    zero_point: Optional[int] = None


# ----- Shared Helper Functions -----


def tag(name):
    """Decorator that registers a builder method as the constructor for a given ``OpView``.

    The decorated method is later discovered by :meth:`BuilderMeta.__new__`
    and stored in :attr:`Builder.opview_to_builder_map`.
    """

    def decorator(func):
        func._tag = name
        return func

    return decorator


def parse(name):
    """Decorator that registers a method as the parser (round-trip re-emitter) for an ``OpView``."""

    def decorator(func):
        func._parse = name
        return func

    return decorator


def split(name):
    """Decorator that registers a method as the split handler for an ``OpView``."""

    def decorator(func):
        func._split = name
        return func

    return decorator


def get_target_path(output_path, builder_dir, filename, target):
    """Build the output path for a target artefact, creating directories as needed.

    Parameters
    ----------
    output_path : str
        Root output directory.
    builder_dir : str
        Subdirectory for the builder type.
    filename : str
        Name of the artefact file.
    target : str
        Backend target name (e.g. ``"ttnn"``).

    Returns
    -------
    str
        Full filesystem path to the artefact.
    """
    target_dir = os.path.join(output_path, builder_dir, target)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return os.path.join(target_dir, filename)


def get_artifact_dir(output_root, builder_type, test_base, make_dir=True):
    """Compute the artefact directory for a test and optionally create it.

    Parameters
    ----------
    output_root : str
        Root directory for all builder artefacts.
    builder_type : str
        Builder name (e.g. ``"TTIRBuilder"``).
    test_base : str
        Base name of the test (used as the leaf directory).
    make_dir : bool
        If ``True``, create the directory when it does not exist.

    Returns
    -------
    str
        Path to the artefact directory.
    """
    artifact_path = os.path.join(
        output_root, "builder-artifacts", builder_type, test_base
    )
    if make_dir and not os.path.exists(artifact_path):
        os.makedirs(artifact_path)
    return artifact_path


def emitc_to_executable(module):
    """Translate an MLIR module to a C++ source string via the EmitC backend."""
    return translate_to_cpp(module)


def emitpy_to_executable(module):
    """Translate an MLIR module to a Python source string via the EmitPy backend."""
    return translate_to_python(module)


def _convert_to_mlir_value(obj):
    """Normalise an object to its MLIR ``Value`` representation.

    Handles ``OpView`` (extracts ``operation.results``), raw ``Value`` objects,
    and passes through anything else unchanged.
    """
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
    """Flatten a multi-return result into a tuple of MLIR values.

    Builder functions may return a single ``OpView``, a single value, or an
    iterable of mixed types.  This function normalises all of these to either
    a single MLIR value (for one-result ops) or a flat tuple of values.
    """
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
    """Create a callable that runs a custom MLIR pass pipeline string.

    Parameters
    ----------
    pipeline : str
        Comma-separated pass names (e.g. ``"ttir-lower-to-layout,ttir-bufferization-pipeline"``).
    verify : bool
        Enable the MLIR verifier after each pass.
    print_ir : Union[bool, str]
        ``True`` to print IR to stdout after each pass, or a directory path
        for per-pass file dumps.

    Returns
    -------
    Callable
        ``(module, device_register_options) -> None`` that runs the pipeline
        in-place on *module*.
    """
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
    """Run a compilation pipeline on an MLIR module in-place.

    Configures system-descriptor path, mesh shape, and optional constant-eval
    settings, then delegates to *pipeline_fn*.  The modified module is
    optionally saved to disk.

    Parameters
    ----------
    module : Module
        The MLIR module to compile (modified in place).
    pipeline_fn : Callable
        ``(module, options_str) -> None`` that runs the pass pipeline.
    pipeline_options : List[str], optional
        Additional ``key=value`` options forwarded to the pipeline.
    save_artifacts : bool
        When ``True``, write the final module to *output_file_name*.
    output_file_name : str
        File path for the saved module.
    system_desc_path : str, optional
        Path to the system descriptor.  Falls back to the
        ``SYSTEM_DESC_PATH`` environment variable.
    mesh_dict : OrderedDict[str, int]
        Device mesh shape.
    argument_types_string : str, optional
        Type string for constant evaluation (if constant-eval is enabled).

    Returns
    -------
    Module
        The (mutated) input module.
    """
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
        Optional affine map for layout transformation
    dim_alignments : Optional[Tuple[int, ...]]
        Optional explicit dimension alignments. When specified, the tensor
        will be padded to these alignments regardless of tile size. Useful
        for testing masking of complete out-of-bounds tiles.

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

        if index_map is None:
            index_map = AffineMap.get_identity(2 * rank, ctx)

        layout = ttcore.ir.MetalLayoutAttr.get(
            ctx,
            logical_shape,
            oobVal,
            memorySpace,
            memory_layout,
            collapse_intervals,
            list(dim_alignments),
            index_map,
        )
    elif index_map is None:
        layout = ttcore.ir.MetalLayoutAttr.get(
            ctx, logical_shape, oobVal, memorySpace, memory_layout
        )
    else:
        layout = ttcore.ir.MetalLayoutAttr.get(
            ctx,
            logical_shape,
            oobVal,
            memorySpace,
            memory_layout,
            index_map,
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

    elemType = F32Type.get(ctx)

    # For tiled layouts, ensure the device shape accounts for tiles.
    if tiled:
        elemType = ttcore.ir.TileType.get(ctx, 32, 32, ttcore.DataType.Float32)
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
