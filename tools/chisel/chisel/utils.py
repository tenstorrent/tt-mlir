# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for Chisel: runtime tensor conversion, debug decorator.
"""
import functools
import logging
import os
import traceback
from typing import Optional

import torch

from _ttmlir_runtime import runtime as tt_runtime

from golden import GoldenMapTensor
from golden.mapping import mlir_datatype_to_torch_dtype, mlir_type_to_torch_dtype

logger = logging.getLogger("chisel")


def get_torch_tensor(tensor) -> torch.Tensor:
    """Convert a runtime tensor to a PyTorch tensor (copies data to host)."""
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    dtype = mlir_datatype_to_torch_dtype(rt_dtype)
    shape = tensor.get_shape()
    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=dtype)
    return torch_tensor.reshape(shape).clone()


def _retrieve_device_tensors(program_context, tensor_ref, untilize=True):
    """Return a list of per-device runtime tensors from the pool.

    Handles both the old Optional[Tensor] API and the new List[Tensor] API
    so the rest of chisel is insulated from the API migration.
    """
    result = tt_runtime.retrieve_tensor_from_pool(program_context, tensor_ref, untilize)
    if result is None:
        return []
    return result if isinstance(result, list) else [result]


def _update_device_tensors(program_context, tensor_ref, rt_tensors):
    """Write per-device runtime tensors back to the pool.

    Handles both the old single-Tensor API and the new List[Tensor] API.
    """
    try:
        tt_runtime.update_tensor_in_pool(program_context, tensor_ref, rt_tensors)
    except TypeError:
        # Old API expects a single Tensor — only safe for single-device.
        if len(rt_tensors) == 1:
            tt_runtime.update_tensor_in_pool(program_context, tensor_ref, rt_tensors[0])
        else:
            raise


def retrieve_torch_tensor(
    program_context,
    tensor_ref,
    *,
    checker=None,
    slot: str = "",
    check: str = "retrieve",
) -> Optional[GoldenMapTensor]:
    """Retrieve tensor(s) from the runtime pool as a GoldenMapTensor.

    For single-device programs the result wraps one shard ({0: tensor}, (1,1)).
    For multi-device programs each shard corresponds to one device in mesh order.
    On failure records an "error" entry via `checker` (if provided) and returns None.
    """
    try:
        device_tensors = _retrieve_device_tensors(program_context, tensor_ref)
        if not device_tensors:
            return None
        shard_map = {i: get_torch_tensor(t) for i, t in enumerate(device_tensors)}
        return GoldenMapTensor(shard_map, (1, len(shard_map)))
    except Exception:
        tb = traceback.format_exc()
        logger.error(f"{slot} [{check}]: retrieve_tensor error\n{tb}")
        if checker is not None:
            checker.record(slot, check, "error", traceback=tb)
        return None


def write_torch_tensor_to_pool(
    program_context,
    tensor_ref,
    golden: GoldenMapTensor,
    *,
    checker=None,
    slot: str = "",
    check: str = "skip_on_device",
) -> bool:
    """Overwrite tensor(s) in the runtime pool with values from a GoldenMapTensor.

    Each shard of `golden` is written to the corresponding per-device tensor
    in the pool.  Shape, stride, and dtype are taken from the existing pool
    tensors so the substitute matches the layout downstream ops expect.

    Records "applied" on success and "error" on failure via `checker`.
    Returns True on success, False on failure.
    """
    try:
        dst_tensors = _retrieve_device_tensors(program_context, tensor_ref)
        if not dst_tensors:
            if checker is not None:
                checker.record(slot, check, "error", traceback="no tensors in pool")
            return False
        golden_shards = golden.golden_map_tensor_as_torch_tensors()
        rt_list = []
        for i, dst in enumerate(dst_tensors):
            src = golden_shards[i].contiguous()
            rt = tt_runtime.create_owned_host_tensor(
                src.data_ptr(),
                dst.get_shape(),
                dst.get_stride(),
                src.numel(),
                dst.get_dtype(),
            )
            rt_list.append(rt)
        _update_device_tensors(program_context, tensor_ref, rt_list)
    except Exception:
        tb = traceback.format_exc()
        logger.error(f"{slot} [{check}]: write_tensor error\n{tb}")
        if checker is not None:
            checker.record(slot, check, "error", traceback=tb)
        return False
    if checker is not None:
        checker.record(slot, check, "applied")
    return True


def debug_wrap(fn):
    """Drop into pdb post-mortem on exception when CHISEL_DEBUG is set; always re-raises.

    Checked per-call, so exporting CHISEL_DEBUG mid-session takes effect without
    re-import.
    """

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


def chisel_safe(fn):
    """Safety net for top-level DebugHooks callbacks.

    Swallows any uncaught exception so a chisel bug never kills ttrt execution.
    Emitter functions are expected to catch and record their own failures; if
    anything still escapes, this wrapper best-effort records a `chisel_bug`
    entry against the current op so the failure is visible in the report.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"chisel callback {fn.__name__} crashed\n{tb}")
            try:
                from .checker import ChiselChecker
                from .context import ChiselContext
                ctx = ChiselContext.get_instance()
                op = ctx.current_program.current_op if ctx.current_program else None
                if op is not None:
                    ChiselChecker(ctx, op.name).record(
                        "<callback>", fn.__name__, "chisel_bug", traceback=tb,
                    )
            except Exception:
                pass

    return wrapper
