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

import torch

from golden.mapping import mlir_datatype_to_torch_dtype, mlir_type_to_torch_dtype

from .exceptions import TensorRetrievalError, TensorWriteError

logger = logging.getLogger("chisel")


def get_torch_tensor(tensor) -> torch.Tensor:
    """Convert a runtime tensor to a PyTorch tensor (copies data to host)."""
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    dtype = mlir_datatype_to_torch_dtype(rt_dtype)
    shape = tensor.get_shape()
    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=dtype)
    return torch_tensor.reshape(shape).clone()


def retrieve_torch_tensor(program_context, tensor_ref) -> torch.Tensor:
    """Retrieve a tensor from the runtime pool and convert it to a PyTorch tensor.

    Raises TensorRetrievalError on any underlying failure.
    """
    from ttrt import runtime as tt_runtime

    try:
        device_tensor = tt_runtime.retrieve_tensor_from_pool(
            program_context, tensor_ref
        )
        return get_torch_tensor(device_tensor)
    except Exception as e:
        raise TensorRetrievalError(str(e)) from e


def write_torch_tensor_to_pool(
    program_context, tensor_ref, torch_tensor: torch.Tensor
) -> None:
    """Overwrite a tensor in the runtime pool with a host-side torch tensor.

    Shape/stride/dtype are taken from the existing pool tensor so the substitute
    matches the layout downstream ops expect. The source tensor is made
    contiguous first so its data_ptr points at a valid linear buffer.

    Raises TensorWriteError on any underlying failure.
    """
    from ttrt import runtime as tt_runtime

    try:
        dst = tt_runtime.retrieve_tensor_from_pool(program_context, tensor_ref)
        src = torch_tensor.contiguous()
        rt = tt_runtime.create_owned_host_tensor(
            src.data_ptr(),
            dst.get_shape(),
            dst.get_stride(),
            src.numel(),
            dst.get_dtype(),
        )
        tt_runtime.update_tensor_in_pool(program_context, tensor_ref, rt)
    except Exception as e:
        raise TensorWriteError(str(e)) from e


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

    Swallows any uncaught exception (typed or not) so a chisel bug never kills
    the ttrt execution. ChiselErrors are expected to be caught at per-slot
    level inside the callback body via record_check — if one reaches this
    wrapper it means we forgot to guard something.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"chisel callback {fn.__name__} crashed\n{tb}")

    return wrapper
