# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import functools
import logging
import math
import os
import traceback
from typing import Optional, Tuple

import torch

from _ttmlir_runtime import runtime as tt_runtime
from _ttmlir_runtime.binary import Binary
from _ttmlir_runtime.runtime import CallbackContext, Tensor, TensorRef

from golden import GoldenMapTensor
from golden.mapping import mlir_datatype_to_torch_dtype

logger = logging.getLogger("chisel")


def get_torch_tensor(tensor: Tensor) -> torch.Tensor:
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    dtype = mlir_datatype_to_torch_dtype(rt_dtype)
    shape = tensor.get_shape()
    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=dtype)
    return torch_tensor.reshape(shape)


def retrieve_tensor(
    rt_program_context: CallbackContext,
    rt_tensor_ref: TensorRef,
    mesh_shape: Tuple[int, ...],
) -> GoldenMapTensor:
    """Pull the pool entry for `rt_tensor_ref` and wrap it as a GoldenMapTensor.

    The runtime returns one host tensor per device shard (single entry for single-device tensors).
    Shards are keyed 0..N-1 and `mesh_shape` reflects the binary's compiled mesh so downstream
    shape/dtype/PCC checks operate over the right grid.
    """
    tensor = tt_runtime.retrieve_tensor_from_pool(rt_program_context, rt_tensor_ref)
    if tensor is None:
        raise RuntimeError(
            "retrieve_tensor_from_pool returned no tensor for the requested tensor reference"
        )

    shards = tt_runtime.to_host(tensor, untilize=True)
    if shards is None:
        raise RuntimeError(
            "retrieve_tensor_from_pool returned no shards for the requested tensor"
        )

    # Making sure we fail if submesh program appear.
    expected_shards = math.prod(mesh_shape) if mesh_shape else -1
    if len(shards) != expected_shards:
        raise RuntimeError(
            f"retrieve_tensor_from_pool shard count ({len(shards)}) does not "
            f"match mesh_shape {mesh_shape} (expected {expected_shards})"
        )

    shard_map = {i: get_torch_tensor(t) for i, t in enumerate(shards)}
    return GoldenMapTensor(shard_map, mesh_shape)


def cached_retrieve_tensor(
    ctx, ssa: str, rt_tensor_ref: TensorRef, mesh_shape: Tuple[int, ...]
) -> GoldenMapTensor:
    """Return a host copy of `rt_tensor_ref`, caching by `ssa` in the device pool.

    On a cache miss the tensor is pulled from device and stored; subsequent
    requests for the same SSA reuse the cached copy.
    """
    pool = ctx.device_tensor_pool
    cached = pool.get(ssa)
    if cached is not None:
        return cached
    tensor = retrieve_tensor(ctx.rt_program_context, rt_tensor_ref, mesh_shape)
    pool[ssa] = tensor
    return tensor


def invalidate_device_cache(ctx, ssa: str) -> None:
    """Drop any cached host copy for `ssa` so the next read re-pulls from device."""
    ctx.device_tensor_pool.pop(ssa, None)


def get_op_asm(op) -> str:
    # Mirror OpPrintingFlags from FuncOpToProgram so the asm string matches
    # the flatbuffer debug_info.
    return op.get_asm(
        enable_debug_info=True,
        large_elements_limit=16,
        large_resource_limit=64,
        skip_regions=True,
        assume_verified=True,
    ).strip()


def debug_wrap(fn):
    # Checked per-call so exporting CHISEL_DEBUG mid-session takes effect
    # without re-import.

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if os.environ.get("CHISEL_DEBUG"):
                import pdb

                traceback.print_exc()
                pdb.post_mortem()
            raise

    return wrapper
