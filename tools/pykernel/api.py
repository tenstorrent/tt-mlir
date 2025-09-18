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
    ttkernel_to_cpp,
    pykernel_compile_pipeline,
)

from pykernel._src.kernel_ast import TTKernelCompiler
from pykernel._src.utils import _cleanup_source_code


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
