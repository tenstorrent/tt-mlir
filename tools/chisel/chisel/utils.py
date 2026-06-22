# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import functools
import logging
import math
import os
import traceback
from typing import Dict, Optional, Tuple

import torch

from _ttmlir_runtime import runtime as tt_runtime
from _ttmlir_runtime.binary import Binary
from _ttmlir_runtime.runtime import CallbackContext, Tensor, TensorRef

from golden import GoldenMapTensor
from golden.mapping import mlir_datatype_to_torch_dtype

from .report import (
    ChiselBugPayload,
    ChiselRecord,
    GoldenPromotedPayload,
    GoldenPromotionSource,
)

logger = logging.getLogger("chisel")


def get_torch_tensor(tensor: Tensor) -> torch.Tensor:
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    dtype = mlir_datatype_to_torch_dtype(rt_dtype)
    shape = tensor.get_shape()
    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=dtype)
    return torch_tensor.reshape(shape)


def _get_live_tensor(
    rt_program_context: CallbackContext, tensor_ref: TensorRef
) -> Tensor:
    tensor = tt_runtime.retrieve_tensor_from_pool(rt_program_context, tensor_ref)
    if tensor is None:
        raise RuntimeError(
            "retrieve_tensor_from_pool returned None during op callback - "
            "tensor should be live"
        )
    return tensor


# Inverse of golden.mapping.mlir_datatype_to_torch_dtype, narrowed to the torch
# dtypes goldens produce. Used to construct runtime host tensors for write-back.
# TODO(ndrakulic): merge this with same builder dict
_TORCH_TO_RUNTIME_DTYPE = {
    torch.float32: tt_runtime.DataType.Float32,
    torch.float16: tt_runtime.DataType.Float16,
    torch.bfloat16: tt_runtime.DataType.BFloat16,
    torch.float64: tt_runtime.DataType.Float64,
    torch.int8: tt_runtime.DataType.Int8,
    torch.int16: tt_runtime.DataType.Int16,
    torch.int32: tt_runtime.DataType.Int32,
    torch.int64: tt_runtime.DataType.Int64,
    torch.uint8: tt_runtime.DataType.UInt8,
    torch.uint16: tt_runtime.DataType.UInt16,
    torch.uint32: tt_runtime.DataType.UInt32,
    torch.uint64: tt_runtime.DataType.UInt64,
    torch.bool: tt_runtime.DataType.Bool,
}


def torch_dtype_to_runtime_dtype(dtype: torch.dtype):
    try:
        return _TORCH_TO_RUNTIME_DTYPE[dtype]
    except KeyError:
        raise ValueError(f"no runtime DataType for torch dtype {dtype}")


def tensor_to_golden(
    tensor: Tensor, mesh_shape: Tuple[int, ...]
) -> GoldenMapTensor:
    """Move `tensor` to host (untilized, row-major) and wrap its shards as a
    GoldenMapTensor, asserting the shard count matches `mesh_shape`.

    `tensor` may live on device or already on host: to_host short-circuits the
    host case (no device transfer), so this also wraps CPU-op outputs.
    """
    shards = tt_runtime.to_host(tensor, untilize=True)
    if shards is None:
        raise RuntimeError("to_host returned no shards for the requested tensor")

    # Fail loudly if a submesh / unexpected shard count appears.
    expected_shards = math.prod(mesh_shape) if mesh_shape else -1
    if len(shards) != expected_shards:
        raise RuntimeError(
            f"to_host shard count ({len(shards)}) does not match mesh_shape "
            f"{mesh_shape} (expected {expected_shards})"
        )

    shard_map = {i: get_torch_tensor(t) for i, t in enumerate(shards)}
    return GoldenMapTensor(shard_map, mesh_shape)


def retrieve_tensor(
    rt_program_context: CallbackContext,
    rt_tensor_ref: TensorRef,
    mesh_shape: Tuple[int, ...],
) -> GoldenMapTensor:
    """Pull the pool entry for `rt_tensor_ref` and wrap it as a GoldenMapTensor."""
    tensor = _get_live_tensor(rt_program_context, rt_tensor_ref)
    return tensor_to_golden(tensor, mesh_shape)


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


def publish_to_session_pool(
    ctx, tensor_ref: TensorRef, golden: GoldenMapTensor
) -> None:
    """Write `golden` into the session pool keyed by the stored runtime
    tensor's globalId, and register a destroy-callback evictor so the entry
    dies with the underlying TTNNTensorWrapper."""
    rt_program_context = ctx.rt_program_context
    global_id = _get_live_tensor(rt_program_context, tensor_ref).get_global_id()

    session_pool = ctx.session_pool
    session_pool[global_id] = golden

    def _evict(
        _tensor,
        _gid: int = global_id,
        _pool: Dict[int, GoldenMapTensor] = session_pool,
    ) -> None:
        _pool.pop(_gid, None)

    registered = tt_runtime.register_pool_tensor_destroy_callback(
        rt_program_context, tensor_ref, _evict
    )
    if registered:
        return

    # No evictor installed: drop the entry so it can't leak or go stale.
    # The accumulation chain breaks at this tensor; surface it in the report.
    session_pool.pop(global_id, None)
    logger.warning(
        "session pool publish failed for globalId %s: destroy-callback "
        "registration failed, golden not chained",
        global_id,
    )
    ctx.write_record(
        ChiselRecord(
            op=ctx.op.name,
            check="session_pool_publish_failed",
            payload=ChiselBugPayload(
                traceback=(
                    f"destroy-callback registration failed for globalId "
                    f"{global_id}: tensor was live but not in session pool, golden not chained"
                ),
            ),
        )
    )


def lookup_from_session_pool(ctx, tensor_ref: TensorRef) -> Optional[GoldenMapTensor]:
    """Return the session-pool golden for the given tensor_ref, or None if not yet published"""
    global_id = _get_live_tensor(ctx.rt_program_context, tensor_ref).get_global_id()
    return ctx.session_pool.get(global_id)


def promote_golden(
    ctx,
    op,
    ssa: str,
    tensor_ref: TensorRef,
) -> None:
    """Seed pool[ssa] from the session pool or fall back to device; write golden_promoted record."""
    cross_golden = lookup_from_session_pool(ctx, tensor_ref)
    if cross_golden is not None:
        ctx.golden_tensor_pool[ssa] = cross_golden
        source = GoldenPromotionSource.SESSION_POOL
    else:
        ctx.golden_tensor_pool[ssa] = cached_retrieve_tensor(
            ctx, ssa, tensor_ref, ctx.mesh_shape
        )
        source = GoldenPromotionSource.DEVICE
    ctx.write_record(
        ChiselRecord(
            op=op.name,
            check="golden_promoted",
            ssa=ssa,
            payload=GoldenPromotedPayload(source=source),
        )
    )


def golden_to_runtime_tensor(golden: GoldenMapTensor) -> Tensor:
    """Build a host runtime.Tensor from a GoldenMapTensor"""
    shard_ids = sorted(golden.shard_map)
    shards = [golden.shard_map[i].contiguous() for i in shard_ids]
    first = shards[0]
    shape = list(first.shape)
    stride = list(first.stride())
    itemsize = first.element_size()
    data_type = torch_dtype_to_runtime_dtype(first.dtype)

    if len(shards) == 1:
        return tt_runtime.create_owned_host_tensor(
            shards[0].data_ptr(), shape, stride, itemsize, data_type
        )
    return tt_runtime.create_multi_device_host_tensor(
        [s.data_ptr() for s in shards],
        shape,
        stride,
        itemsize,
        data_type,
        {},  # strategy: unused for explicit per-shard data
        list(golden.mesh_shape),
    )


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
