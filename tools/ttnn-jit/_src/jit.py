# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
from typing import Literal

import ttnn

from ttmlir.ir import *
from ttmlir.passes import (
    ttnn_to_flatbuffer_file,
    ttnn_to_flatbuffer_bin,
    ttnn_to_ttmetal_pipeline,
    ttkernel_to_cpp_file,
)

from ttnn_jit._src.utils import cleanup_source_code, get_dispatch_core_type
from ttnn_jit._src.dispatch_op import run_binary, run_binary_from_capsule
from ttnn_jit._src import JitCache
from ttnn_jit._src.ir_generator import generate_ir
from ttnn_jit._src import (
    get_current_system_desc,
    create_runtime_device_from_ttnn,
)
from ttnn_jit._src.memory_analyzer import MemoryAnalyzer


class JitFunction:
    """A JIT-compiled function with its own cache."""

    def __init__(
        self,
        func,
        compile_only: bool,
        debug: bool,
        enable_cache: bool,
        math_fidelity: ttnn.MathFidelity,
        enable_l1_acc: bool,
        use_tile_matmul: bool,
        memory_config: ttnn.MemoryConfig,
    ):
        self.func = func
        self.source_code = cleanup_source_code(func)
        self.compile_only = compile_only
        self.debug = debug or compile_only
        self.out_dir = os.path.join("generated", "ttnn-jit", func.__name__)
        self.math_fidelity = math_fidelity
        self.enable_l1_acc = enable_l1_acc
        self.use_tile_matmul = use_tile_matmul
        self.memory_config = memory_config
        os.makedirs(self.out_dir, exist_ok=True)

        self.system_desc_path = os.getenv("SYSTEM_DESC_PATH")

        if self.debug:
            os.environ["TTRT_LOGGER_LEVEL"] = "DEBUG"
            os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "TRACE"

        # Each JitFunction hold its own cache.
        # Hashing based off runtime tensor metadata.
        self.cache = JitCache(64) if enable_cache else None

    def _validate_arguments(self, args, kwargs):
        """Validate arguments (handles defaults and kwargs)."""
        sig = inspect.signature(self.func)
        try:
            sig.bind(*args, **kwargs)
        except TypeError as e:
            raise ValueError(
                f"Invalid arguments for function {self.func.__name__}: {str(e)}"
            ) from e
        return sig

    def _query_and_save_system_desc(self, ttnn_device=None):
        """Query system descriptor from device and save it to a file.
        Uses the MLIR runtime bindings directly, replicating the logic from
        ttrt query --save-artifacts.
        """
        dispatch_core_type = get_dispatch_core_type()
        try:
            # Use input tensor device to query if available.
            if ttnn_device:
                runtime_device = create_runtime_device_from_ttnn(ttnn_device)
                system_desc = get_current_system_desc(
                    dispatch_core_type, runtime_device
                )
                if self.debug:
                    print(f"System descriptor queried using existing device.")
            else:
                system_desc = get_current_system_desc(dispatch_core_type)
                if self.debug:
                    print(f"System descriptor queried by creating new device")

            system_desc_path = os.path.join(self.out_dir, "system_desc.ttsys")
            system_desc.store(system_desc_path)
            os.environ["SYSTEM_DESC_PATH"] = system_desc_path

            if self.debug:
                print(f"System descriptor saved to: {system_desc_path}")

            return system_desc_path
        except Exception as e:
            raise RuntimeError(
                f"Failed to query system descriptor. Ensure device is available.\n"
                f"Exception: {str(e)}"
            )

    def __call__(self, *args, **kwargs):
        """Execute the JIT-compiled function."""
        device = args[0].device() if args else None
        assert device is not None, "Device is required"
        if not self.system_desc_path:
            self.system_desc_path = self._query_and_save_system_desc(device)

        sig = self._validate_arguments(args, kwargs)
        param_names = list(sig.parameters.keys())
        kwargs["_tensor_args"] = {param_names[i]: args[i] for i in range(len(args))}

        # Cache hit, no need to compile.
        if self.cache and self.cache.contains(*args):
            fb_binary = self.cache.get(*args)
            return run_binary(fb_binary, args)

        ir, output_type = generate_ir(
            self.func,
            self.debug,
            self.memory_config,
            *args,
            **kwargs,
        )

        # Analyze memory: get available L1/DRAM ranges and output tensor requirements
        memory_analyzer = MemoryAnalyzer(device, output_type)
        if self.debug:
            memory_analyzer.print_stats()

        options = (
            f"system-desc-path={self.system_desc_path}"
            f" ttnn-mode=true"
            f" set-math-fidelity={self.math_fidelity.name}"
            f" matmul-interchange=2,0,1"
            f" enable-l1-acc={self.enable_l1_acc}"
            f" use-tile-matmul={self.use_tile_matmul}"
        )
        options += memory_analyzer.get_l1_range_str()
        print("Compiling with options: ", options)
        if self.compile_only:
            ttnn_to_ttmetal_pipeline(ir, options)
            print("---- IR Dump after ttnn_to_ttmetal_pipeline ----")
            print(ir)

            # Dump kernels to C++ files in generated/ttnn-jit
            ttkernel_to_cpp_file(ir, self.out_dir)

            # Generate and dump flatbuffer in generated/ttnn-jit
            flatbuffer_file = os.path.join(self.out_dir, self.func.__name__ + ".ttnn")
            ttnn_to_flatbuffer_file(ir, flatbuffer_file, {}, [])
            return ir

        if self.cache:
            fb_binary = self.cache.compile_and_insert(
                str(ir), options, self.debug, *args
            )
            return run_binary(fb_binary, args)

        ttnn_to_ttmetal_pipeline(ir, options)
        if self.debug:
            print("---- IR Dump after ttnn_to_ttmetal_pipeline ----")
            print(ir)
        fb_capsule = ttnn_to_flatbuffer_bin(ir)
        return run_binary_from_capsule(fb_capsule, args)

    @property
    def num_entries(self):
        """Return the number of cache entries."""
        assert self.cache, "Cache is not enabled"
        return self.cache.num_entries()
