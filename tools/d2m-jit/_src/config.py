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
    # Keep explicit d2m.tile_matmul loops instead of replacing them with
    # d2m.tile_matmul_block in the backend pipeline.
    use_tile_matmul: bool = _env_bool("D2M_JIT_USE_TILE_MATMUL")
    # Use the d2m-split-unified-thread-v2 rewrite (passed through to
    # d2m-be-pipeline as use-split-unified-thread-v2=1). Default on: v1 asserts
    # on multi-synchronizable-op kernels (CCL: device_synchronize + remote_store
    # + semaphore_wait), and v2 lowers the eltwise/matmul/CCL paths. Set
    # D2M_JIT_SPLIT_UNIFIED_THREAD_V2=0 to fall back to the legacy split.
    use_split_unified_thread_v2: bool = _env_bool(
        "D2M_JIT_SPLIT_UNIFIED_THREAD_V2", default=True
    )
    # Use TensorAccessor-based DMA lowering (passed through to d2m-be-pipeline
    # and d2m-to-ttkernel-pre-emitc-pipeline as use-tensor-accessor-dma=1):
    # lowers plain shard-level dma_read/write to TTKernel TensorAccessor ops
    # (multicast and local-destination DMAs still go through
    # D2MLowerDMAToFullyIndexedForm). Default ON; opt out with
    # D2M_JIT_USE_TENSOR_ACCESSOR_DMA=0.
    use_tensor_accessor_dma: bool = _env_bool(
        "D2M_JIT_USE_TENSOR_ACCESSOR_DMA", default=True
    )
    # If set, write the post-pipeline flatbuffer to this path before
    # device submit. Useful for offline inspection with ttrt.
    save_flatbuffer_path: Optional[str] = os.environ.get("D2M_JIT_SAVE_FLATBUFFER_PATH")
    # Insert TTKernel device-zone profiler scopes so the compiled kernels
    # auto-instrument their ops (DeviceZoneScopedN markers picked up by the
    # device profiler / tracy). Mirrors the `insert-profiler-traces` option of
    # createTTIRToTTMetalPipeline. Requires a perf-trace runtime build.
    insert_profiler_traces: bool = _env_bool("D2M_JIT_INSERT_PROFILER_TRACES")
    # Comma-separated TTKernel traits to instrument when insert_profiler_traces
    # is on (e.g. "device-zone", "fpu,sfpu", "all"). Empty -> "device-zone".
    profiler_traits: str = os.environ.get("D2M_JIT_PROFILER_TRAITS", "")
    # Enable runtime device-profiler collection during execution: flips the
    # perf::Env singleton so the ttmetal executor reads device profiler results
    # after each workload (-> $TT_METAL_HOME/generated/profiler/.logs/
    # profile_log_device.csv). Requires a TT_RUNTIME_ENABLE_PERF_TRACE=ON build
    # and TT_METAL_DEVICE_PROFILER=1 (set automatically if unset). Pairs with
    # insert_profiler_traces to capture the inserted DeviceZoneScopedN zones.
    enable_perf_trace: bool = _env_bool("D2M_JIT_ENABLE_PERF_TRACE")


# Process-level singleton. Mutate fields directly.
config = _Config()
