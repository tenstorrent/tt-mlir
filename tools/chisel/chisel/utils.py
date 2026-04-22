# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for Chisel: runtime tensor conversion, debug decorator.
"""
import functools

import torch

from golden.mapping import mlir_datatype_to_torch_dtype, mlir_type_to_torch_dtype


def get_torch_tensor(tensor) -> torch.Tensor:
    """Convert a runtime tensor to a PyTorch tensor (copies data to host)."""
    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    dtype = mlir_datatype_to_torch_dtype(rt_dtype)
    shape = tensor.get_shape()
    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=dtype)
    return torch_tensor.reshape(shape).clone()


def retrieve_torch_tensor(program_context, tensor_ref) -> torch.Tensor:
    """Retrieve a tensor from the runtime pool and convert it to a PyTorch tensor."""
    from ttmlir_runtime import runtime as tt_runtime

    device_tensor = tt_runtime.retrieve_tensor_from_pool(
        program_context, tensor_ref
    )
    return get_torch_tensor(device_tensor)


def debug_wrap(*, debug: bool = False):
    """Decorator factory for runtime callbacks — drops into pdb on exception if debug=True."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                if debug:
                    import pdb
                    import traceback

                    traceback.print_exc()
                    pdb.set_trace()
                raise

        return wrapper

    return decorator
