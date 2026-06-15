# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Process-level debug knobs for the lazy builder.

Usage:
    from d2m_jit import config
    config.print_ir_before_pipeline = True
    config.print_ir_after_each_pass = True
    out = lt.to_host()

Each flag also reads an environment variable of the same name prefixed
with `D2M_JIT_` (resolved once at import time). True values: "1", "true",
"yes", "on" (case-insensitive).
"""

import os
from dataclasses import dataclass
from typing import Optional


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class _Config:
    # Print the full pass-pipeline string before invoking it.
    print_pipeline: bool = _env_bool("D2M_JIT_PRINT_PIPELINE")
    # Dump the constructed module before the pass pipeline runs.
    print_ir_before_pipeline: bool = _env_bool("D2M_JIT_PRINT_IR_BEFORE")
    # Dump the lowered module after the pass pipeline finishes.
    print_ir_after_pipeline: bool = _env_bool("D2M_JIT_PRINT_IR_AFTER")
    # Use PassManager.enable_ir_printing to dump after every pass.
    # NOTE: forces ctx.enable_multithreading(False), so it's slower.
    print_ir_after_each_pass: bool = _env_bool("D2M_JIT_PRINT_IR_AFTER_ALL")
    # Include locations / debug info in printed IR.
    print_ir_debug_info: bool = _env_bool("D2M_JIT_PRINT_IR_DEBUG_INFO")
    # PassManager.enable_verifier toggle. Default True.
    verify_passes: bool = _env_bool("D2M_JIT_VERIFY", default=True)
    # Convert all tensor inputs and out-params to DRAM before each kernel call.
    kernel_io_in_dram: bool = _env_bool("D2M_JIT_KERNEL_IO_IN_DRAM")
    # If set, write the post-pipeline flatbuffer to this path before
    # device submit. Useful for offline inspection with ttrt.
    save_flatbuffer_path: Optional[str] = os.environ.get("D2M_JIT_SAVE_FLATBUFFER_PATH")


# Process-level singleton. Mutate fields directly.
config = _Config()
