# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for Chisel: dtype maps, runtime tensor conversion, debug decorator.
"""
import functools

import torch


ttir_dtype_maps = {
    "i32": torch.int32,
    "i64": torch.int64,
    "f32": torch.float32,
    "f64": torch.float64,
    "si32": torch.int32,
    "i1": torch.bool,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "ui32": torch.uint32,
}


ttrt_dtype_maps = {
    "DataType.Float32": torch.float32,
    "DataType.BFloat16": torch.bfloat16,
    "DataType.UInt32": torch.uint32,
    "DataType.UInt16": torch.uint16,
    "DataType.UInt8": torch.uint8,
    "DataType.Int32": torch.int32,
}


def get_torch_tensor(tensor) -> torch.Tensor:
    """Convert a runtime tensor to a PyTorch tensor (copies data to host)."""
    from ttrt.runtime import Tensor as RtTensor  # noqa: F401 — lazy import

    rt_data_ptr = tensor.get_data_buffer()
    rt_dtype = tensor.get_dtype()
    dtype = ttrt_dtype_maps[str(rt_dtype)]
    shape = tensor.get_shape()
    torch_tensor = torch.frombuffer(rt_data_ptr, dtype=dtype)
    return torch_tensor.reshape(shape).clone()


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
