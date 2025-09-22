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
    ttir_to_ttnn_backend_pipeline,
    ttnn_to_flatbuffer_file,
)

from ttnn_jit._src.ttir_ast import TTIRCompiler
from ttnn_jit._src.utils import _cleanup_source_code
from ttnn_jit._src.dispatch_op import _run_binary


def jit(
    backend: Literal["ttnn", "metal"] = "ttnn",
    perf: bool = False,
    dump_flatbuffer: bool = True,
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

            if backend == "metal":
                ttir_to_ttmetal_backend_pipeline(
                    ir, f"system-desc-path={system_desc_path} override-device-shape=1,1"
                )
                if debug:
                    print("---- After ttir_to_ttmetal_backend_pipeline ----")
                    print(ir)

                if dump_flatbuffer:
                    out_dir = os.path.join("generated", "pykernels")
                    os.makedirs(out_dir, exist_ok=True)
                    ttmetal_to_flatbuffer_file(
                        ir, os.path.join(out_dir, f.__name__ + ".ttm"), {}, []
                    )
                # TODO: hook up metal runtime here
                return ir
            elif backend == "ttnn":
                ttir_to_ttnn_backend_pipeline(
                    ir, f"system-desc-path={system_desc_path}"
                )
                if debug:
                    print("---- After ttir_to_ttnn_backend_pipeline ----")
                    print(ir)

                out_dir = os.path.join("generated", "pykernels")
                os.makedirs(out_dir, exist_ok=True)
                flatbuffer_bin = os.path.join(out_dir, f.__name__ + ".ttnn")

                # TODO: remove once we can run flatbuffer from memory
                if not dump_flatbuffer:
                    raise RuntimeError("dump_flatbuffer must be True for ttnn backend")
                ttnn_to_flatbuffer_file(ir, flatbuffer_bin, {}, [])

                return _run_binary(flatbuffer_bin, args)
            else:
                raise ValueError(f"Unsupported backend: {backend}")

        if inspect.ismethod(f):
            return staticmethod(_wrapper)
        return _wrapper

    return _decorator
