# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import ast
import inspect
import functools

from ttmlir.ir import *
from ttmlir.passes import ttir_to_ttmetal_backend_pipeline, ttmetal_to_flatbuffer_file

from pykernel._src.ttir_ast import TTIRCompiler
from pykernel._src.utils import _cleanup_source_code


def jit(
    perf: bool = False,
    to_flatbuffer_file: bool = False,
    debug: bool = False,
):
    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            source_code = _cleanup_source_code(f)

            # Pass the actual tensors as kwargs
            tensor_args = {}
            sig = inspect.signature(f)
            param_names = list(sig.parameters.keys())
            if len(param_names) != len(args):
                raise ValueError(f"How is this even possible???")

            for i, arg in enumerate(args):
                tensor_args[param_names[i]] = arg
            kwargs["_tensor_args"] = tensor_args
            kwargs["_verbose"] = debug
            kwargs["_source_code"] = source_code.splitlines() if debug else ""

            # Parse and compile
            m = ast.parse(source_code)
            if debug:
                print(ast.dump(m, indent=2) + "\n")
            b = TTIRCompiler(None, *args, **kwargs)
            b.visit(m)

            # Check if generated IR is valid
            ir = b.module
            if debug:
                print(ir)
            ir.operation.verify()

            system_desc_path = os.getenv("SYSTEM_DESC_PATH")
            assert system_desc_path, "SYSTEM_DESC_PATH must be set."

            ttir_to_ttmetal_backend_pipeline(
                ir, f"system-desc-path={system_desc_path} override-device-shape=1,1"
            )
            if debug:
                print("---- After ttir_to_ttmetal_backend_pipeline ----")
                print(ir)

            if to_flatbuffer_file:
                ttmetal_to_flatbuffer_file(ir, to_flatbuffer_file, {}, [])

            return ir

        if inspect.ismethod(f):
            return staticmethod(_wrapper)
        return _wrapper

    return _decorator
