# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import ast
import inspect
import functools
from typing import Literal

from ttmlir.ir import *
from ttmlir.passes import (
    ttir_to_ttmetal_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    ttnn_to_flatbuffer_file,
    ttnn_to_ttmetal_pipeline,
    ttnn_to_flatbuffer_bin,
)

from ttnn_jit._src.ttir_ast import TTIRCompiler
from ttnn_jit._src.utils import _cleanup_source_code
from ttnn_jit._src.dispatch_op import _run_binary_from_capsule
from ttnn_jit._ttnn_jit import JitCache


class JitFunction:
    """A JIT-compiled function with its own cache."""

    def __init__(
        self,
        func,
        backend: Literal["ttnn", "metal"],
        max_grid: tuple[int, int],
        perf: bool,
        compile_only: bool,
        debug: bool,
    ):
        self.func = func
        self.source_code = _cleanup_source_code(func)
        self.backend = backend
        self.max_grid = max_grid
        self.perf = perf
        self.compile_only = compile_only
        self.debug = debug

        self.out_dir = os.path.join("generated", "pykernels")
        os.makedirs(self.out_dir, exist_ok=True)

        self.system_desc_path = os.getenv("SYSTEM_DESC_PATH")
        assert self.system_desc_path, "SYSTEM_DESC_PATH must be set."

        if self.debug:
            os.environ["TTRT_LOGGER_LEVEL"] = "DEBUG"

        # Each JitFunction hold its own cache.
        # Hashing based off runtime tensor metadata.
        self.cache = JitCache(64)

    def __call__(self, *args, **kwargs):
        """Execute the JIT-compiled function."""
        tensor_args = {}
        param_names = list(inspect.signature(self.func).parameters.keys())
        if len(param_names) != len(args):
            raise ValueError(
                f"Passed {len(args)} args, but function expects {len(param_names)}"
            )

        for i, arg in enumerate(args):
            tensor_args[param_names[i]] = arg
        kwargs["_tensor_args"] = tensor_args
        kwargs["_backend"] = self.backend
        kwargs["_max_grid"] = self.max_grid

        # Parse AST and compile to IR
        m = ast.parse(self.source_code)
        if self.debug:
            print(ast.dump(m, indent=2) + "\n")

        compiler = TTIRCompiler(None, *args, **kwargs)
        compiler.visit(m)

        ir = compiler.module
        if self.debug:
            print(ir)
        ir.operation.verify()

        if self.backend == "ttnn":
            options = f"system-desc-path={self.system_desc_path} ttnn-mode=true"
            if self.compile_only:
                ir = ttnn_to_ttmetal_pipeline(ir, options)
                flatbuffer_bin = os.path.join(self.out_dir, self.func.__name__ + ".ttn")
                ttnn_to_flatbuffer_file(ir, flatbuffer_bin, {}, [])
                return ir

            fb_capsule = self.cache.get(str(ir), options, *args)
            return _run_binary_from_capsule(fb_capsule, args)

        elif self.backend == "metal":
            if not self.compile_only:
                raise NotImplementedError("Metal runtime is not implemented yet")
            ttir_to_ttmetal_backend_pipeline(
                ir,
                f"system-desc-path={self.system_desc_path} override-device-shape=1,1",
            )
            if self.debug:
                print("---- After ttir_to_ttmetal_backend_pipeline ----")
                print(ir)

            flatbuffer_bin = os.path.join(self.out_dir, self.func.__name__ + ".ttm")
            ttmetal_to_flatbuffer_file(ir, flatbuffer_bin, {}, [])
            return ir
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @property
    def cache_hits(self):
        """Return the number of cache hits."""
        return self.cache.cache_hits()
