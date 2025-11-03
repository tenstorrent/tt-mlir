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
    ttnn_to_flatbuffer_file,
    ttnn_to_ttmetal_pipeline,
)

from ttnn_jit._src.ttir_ast import TTIRCompiler
from ttnn_jit._src.utils import _cleanup_source_code
from ttnn_jit._src.dispatch_op import _run_binary
from ttnn_jit._src import JitCache


class JitFunction:
    """A JIT-compiled function with its own cache."""

    def __init__(
        self,
        func,
        max_grid: tuple[int, int],
        compile_only: bool,
        debug: bool,
    ):
        self.func = func
        self.source_code = _cleanup_source_code(func)
        self.max_grid = max_grid
        self.compile_only = compile_only
        self.debug = debug

        self.out_dir = os.path.join("generated", "pykernels")
        os.makedirs(self.out_dir, exist_ok=True)

        self.system_desc_path = os.getenv("SYSTEM_DESC_PATH")
        assert self.system_desc_path, "SYSTEM_DESC_PATH must be set."

        if self.debug:
            os.environ["TTRT_LOGGER_LEVEL"] = "DEBUG"
            os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "TRACE"

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
        kwargs["_max_grid"] = self.max_grid

        # Cache hit, no need to compile.
        if self.cache.contains(*args):
            fb_binary = self.cache.get(*args)
            return _run_binary(fb_binary, args)

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

        options = f"system-desc-path={self.system_desc_path} ttnn-mode=true"
        if self.compile_only:
            ir = ttnn_to_ttmetal_pipeline(ir, options)
            flatbuffer_bin = os.path.join(self.out_dir, self.func.__name__ + ".ttn")
            ttnn_to_flatbuffer_file(ir, flatbuffer_bin, {}, [])
            return ir

        fb_binary = self.cache.compile_and_insert(str(ir), options, self.debug, *args)
        return _run_binary(fb_binary, args)

    @property
    def num_entries(self):
        """Return the number of cache entries."""
        return self.cache.num_entries()
