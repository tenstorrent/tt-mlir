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

_cache = JitCache(1024)


def jit(
    backend: Literal["ttnn", "metal"] = "ttnn",
    max_grid: tuple[int, int] = (8, 8),
    # max_dest_size: int = 1,
    perf: bool = False,
    compile_only: bool = False,
    debug: bool = False,
):
    def _decorator(f):
        source_code = _cleanup_source_code(f)

        @functools.wraps(f)
        def _wrapper(*args, **kwargs):

            # Pass the actual tensors through as kwargs
            tensor_args = {}
            sig = inspect.signature(f)
            param_names = list(sig.parameters.keys())
            if len(param_names) != len(args):
                raise ValueError(
                    f"Passed {len(args)} args, but function expects {len(param_names)}"
                )

            for i, arg in enumerate(args):
                tensor_args[param_names[i]] = arg
            kwargs["_tensor_args"] = tensor_args
            kwargs["_backend"] = backend
            kwargs["_max_grid"] = max_grid

            # Parse and compile
            m = ast.parse(source_code)
            if debug:
                print(ast.dump(m, indent=2) + "\n")

            # TODO (#5043): emit ttnn IR instead of TTIR, TTIR should be fallback.
            b = TTIRCompiler(None, *args, **kwargs)
            b.visit(m)

            # Check if generated IR is valid
            ir = b.module
            if debug:
                print(ir)
            ir.operation.verify()

            system_desc_path = os.getenv("SYSTEM_DESC_PATH")
            assert system_desc_path, "SYSTEM_DESC_PATH must be set."
            if debug:
                os.environ["TTRT_LOGGER_LEVEL"] = "DEBUG"

            out_dir = os.path.join("generated", "pykernels")
            os.makedirs(out_dir, exist_ok=True)
            if backend == "metal":
                ttir_to_ttmetal_backend_pipeline(
                    ir, f"system-desc-path={system_desc_path} override-device-shape=1,1"
                )
                if debug:
                    print("---- After ttir_to_ttmetal_backend_pipeline ----")
                    print(ir)

                if compile_only:
                    flatbuffer_bin = os.path.join(out_dir, f.__name__ + ".ttm")
                    ttmetal_to_flatbuffer_file(ir, flatbuffer_bin, {}, [])
                    return ir
                else:
                    # TODO: hook up metal runtime here
                    raise NotImplementedError("Metal runtime is not implemented yet")
            elif backend == "ttnn":
                options = f"system-desc-path={system_desc_path} ttnn-mode=true"
                if compile_only:
                    ir = ttnn_to_ttmetal_pipeline(ir, options)
                    print("---- After ttnn_to_ttmetal_pipeline ----")
                    print(ir)
                    flatbuffer_bin = os.path.join(out_dir, f.__name__ + ".ttn")
                    ttnn_to_flatbuffer_file(ir, flatbuffer_bin, {}, [])
                    print(f"Flatbuffer created successfully at: {flatbuffer_bin}")
                    return ir

                func_sig = f"{f.__name__}{sig}"
                print(f"func_sig: {func_sig}")
                fb_capsule = _cache.get(
                    func_sig, str(ir), options, backend, max_grid, *args
                )
                print(f"Cache hits: {_cache.cache_hits()}\n")
                return _run_binary_from_capsule(fb_capsule, args)
            else:
                raise ValueError(f"Unsupported backend: {backend}")

        if inspect.ismethod(f):
            return staticmethod(_wrapper)
        return _wrapper

    return _decorator
