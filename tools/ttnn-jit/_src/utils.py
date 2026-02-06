# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import textwrap
import inspect
import importlib
import ttnn
from typing import Callable
from ttmlir.ir import *
from ttnn_jit._src import DispatchCoreType


def discover_dialect_ops(dialect, denylist=None):
    """
    Return a mapping Dict[str, Callable] of available pybounded dialect ops.
    """
    # Convert string dialect names to their corresponding objects
    if isinstance(dialect, str):
        dialect = importlib.import_module(f"ttmlir.dialects.{dialect}")

    denylist = set() if denylist is None else denylist
    op_map = {}
    ns = dialect.__name__.split("ttmlir.dialects.")[-1]
    for attr_name in dir(dialect):
        if attr_name.startswith("_"):
            continue
        op_obj = getattr(dialect, attr_name, None)
        if (
            op_obj is None
            or not hasattr(op_obj, "OPERATION_NAME")
            or not inspect.isclass(op_obj)
        ):
            continue

        func_name = getattr(op_obj, "OPERATION_NAME")
        name = func_name.removeprefix(ns + ".")
        if name in denylist:
            continue
        func = getattr(dialect, name, None)

        # must be the module-level function, and not the class
        if inspect.isfunction(func):
            op_map[name] = func

    return op_map


def cleanup_source_code(f: Callable):
    source_code = inspect.getsource(f)
    source_code = textwrap.dedent(source_code)

    # Find the line that starts the function definition and keep from there
    lines = source_code.splitlines()
    def_line_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("def "):
            def_line_idx = i
            break

    if def_line_idx is None:
        # Fallback to old behavior if we can't find def line
        cleaned = [line for line in lines if not line.strip().startswith("@")]
        source_code = "\n".join(cleaned)
    else:
        # Keep only from the def line onwards (removes all decorator lines)
        source_code = "\n".join(lines[def_line_idx:])

    return source_code


def get_num_pos_args(func: Callable):
    sig = inspect.signature(func)
    num_pos_args = len(
        [
            p
            for p in sig.parameters.values()
            if p.default == inspect.Parameter.empty
            and p.kind != inspect.Parameter.VAR_KEYWORD
        ]
    )
    return num_pos_args


def _get_cluster_type():
    return ttnn.cluster.get_cluster_type()


def get_dispatch_core_type():
    cluster_type = _get_cluster_type()
    match cluster_type:
        case ttnn.cluster.ClusterType.N150:
            dispatch_core_type = DispatchCoreType.ETH
        case ttnn.cluster.ClusterType.N300:
            dispatch_core_type = DispatchCoreType.ETH
        case ttnn.cluster.ClusterType.P150:
            dispatch_core_type = DispatchCoreType.WORKER
        case ttnn.cluster.ClusterType.T3K:
            dispatch_core_type = DispatchCoreType.ETH
        case _:
            raise ValueError(f"Unsupported cluster type: {cluster_type}")
    return dispatch_core_type


def get_maximal_block_sharding_grid(shape, core_grid):
    """Infer a TTNN grid/end coord for block sharding the given logical tensor shape and device core grid"""

    # Collapse dims [0, -1)
    if len(shape) > 2:
        collapsed_dim = 1
        for i in range(len(shape) - 1):
            collapsed_dim *= shape[i]
    else:
        collapsed_dim = shape[0]
    tile_shape = [collapsed_dim // 32, shape[-1] // 32]

    grid = []
    for dim, max_grid in zip(tile_shape, core_grid):
        for grid_dim in reversed(range(max_grid)):
            if dim % (grid_dim + 1) == 0:
                grid.append(grid_dim)
                break
    return list(reversed(grid))


def get_core_grid_from_tensor_args(tensor_args):
    """Get the core grid from the device of the first tensor argument"""

    if not tensor_args:
        raise ValueError("No tensor arguments provided")
    tensor_arg = next(iter(tensor_args.values()))
    device = tensor_arg.device
    return (device.core_grid.x, device.core_grid.y)
