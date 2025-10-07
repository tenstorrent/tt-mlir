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
)

from ttnn_jit._src.ttir_ast import TTIRCompiler
from ttnn_jit._src.utils import _cleanup_source_code
from ttnn_jit._src.dispatch_op import _run_binary


def jit(
    backend: Literal["ttnn", "metal"] = "ttnn",
    max_grid: tuple[int, int] = (8, 8),
    # max_dest_size: int = 1,
    perf: bool = False,
    compile_only: bool = False,
    debug: bool = False,
):
    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            source_code = _cleanup_source_code(f)

            # Pass the actual tensors through as kwargs
            tensor_args = {}
            sig = inspect.signature(f)
            param_names = list(sig.parameters.keys())
            if len(param_names) != len(args):
                raise ValueError(f"How is this even possible???")

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

                flatbuffer_bin = os.path.join(out_dir, f.__name__ + ".ttm")
                if compile_only:
                    ttmetal_to_flatbuffer_file(ir, flatbuffer_bin, {}, [])
                    return ir
                else:
                    # TODO: hook up metal runtime here
                    raise NotImplementedError("Metal runtime is not implemented yet")
            elif backend == "ttnn":
                # TODO (#5224): use pipeline once hooked up
                ttnn_to_ttmetal_pipeline(
                    ir, f"system-desc-path={system_desc_path} ttnn-mode=true"
                )
                if debug:
                    print("---- After ttnn_to_ttmetal_pipeline ----")
                    print(ir)

                flatbuffer_bin = os.path.join(out_dir, f.__name__ + ".ttnn")
                if compile_only:
                    ttnn_to_flatbuffer_file(ir, flatbuffer_bin, {}, [])
                    return ir
                else:
                    # TODO (#5055): always dump flatbuffer to disk for now, in the future we want to run flatbuffer from memory and only dump to disk if debug=True.
                    ttnn_to_flatbuffer_file(ir, flatbuffer_bin, {}, [])

                    return _run_binary(flatbuffer_bin, args)
            else:
                raise ValueError(f"Unsupported backend: {backend}")

        if inspect.ismethod(f):
            return staticmethod(_wrapper)
        return _wrapper

    return _decorator
