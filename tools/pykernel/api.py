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
    ttkernel_to_cpp,
    pykernel_compile_pipeline,
)

from pykernel._src.ttir_ast import TTIRCompiler
from pykernel._src.kernel_ast import TTKernelCompiler
from pykernel._src.utils import _cleanup_source_code


def jit(
    backend: Literal["ttnn", "metal"] = "ttnn",
    perf: bool = False,
    dump_flatbuffer: bool = False,
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
            elif backend == "ttnn":
                ttir_to_ttnn_backend_pipeline(
                    ir, f"system-desc-path={system_desc_path}"
                )
                if debug:
                    print("---- After ttir_to_ttnn_backend_pipeline ----")
                    print(ir)

                if dump_flatbuffer:
                    out_dir = os.path.join("generated", "pykernels")
                    os.makedirs(out_dir, exist_ok=True)
                    ttnn_to_flatbuffer_file(
                        ir, os.path.join(out_dir, f.__name__ + ".ttnn"), {}, []
                    )

            return ir

        if inspect.ismethod(f):
            return staticmethod(_wrapper)
        return _wrapper

    return _decorator


def ttkernel_compile(
    kernel_type=None, verbose: bool = False, optimize: bool = False, thread_type=""
):
    def _decorator(f):
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            # Code to deal with identation issues
            source_code = _cleanup_source_code(f)

            if verbose is True:
                # Create easily index-able object to store source code:
                kwargs["_source_code"] = source_code.splitlines()
                kwargs["_verbose"] = True
            m = ast.parse(source_code)
            b = TTKernelCompiler(f.__name__, kernel_type, *args, **kwargs)
            print(ast.dump(m, indent=4) + "\n")
            b.visit(m)

            # Check if generated IR is valid
            print(b.module)
            b.module.operation.verify()

            # Run the PyKernel Compile Pipeline to fit model for Translation
            if optimize:
                pykernel_compile_pipeline(b.module)
                print("---- Optimized PyKernel Module ----", b.module, sep="\n\n")

            if kernel_type:
                print("---- Kernel String ----", b.module, sep="\n\n")
                kernel_string = ttkernel_to_cpp(b.module)
                return kernel_string

        # Make the decorator apply staticmethod for class methods defined using op.py
        _wrapper._decorator_name = thread_type + "_thread"
        if inspect.ismethod(f):
            return staticmethod(_wrapper)
        return _wrapper

    return _decorator


def compute_thread(verbose: bool = False, optimize: bool = False):
    return ttkernel_compile(
        kernel_type="compute", verbose=verbose, optimize=optimize, thread_type="compute"
    )


def reader_thread(verbose: bool = False, optimize: bool = False):
    return ttkernel_compile(
        kernel_type="noc", verbose=verbose, optimize=optimize, thread_type="reader"
    )


def writer_thread(verbose: bool = False, optimize: bool = False):
    return ttkernel_compile(
        kernel_type="noc", verbose=verbose, optimize=optimize, thread_type="writer"
    )


def ttkernel_tensix_compile(verbose: bool = False, optimize: bool = False):
    return ttkernel_compile(kernel_type="compute", verbose=verbose, optimize=optimize)


def ttkernel_noc_compile(verbose: bool = False, optimize: bool = False):
    return ttkernel_compile(kernel_type="noc", verbose=verbose, optimize=optimize)
