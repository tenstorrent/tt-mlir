# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Literal

from ttnn_jit._src.jit import JitFunction


def jit(
    max_grid: tuple[int, int] = (7, 7),
    compile_only: bool = False,
    debug: bool = False,
    use_ast_compiler: bool = False,
):
    """
    Sets up the decorated function to be JIT compiled through D2M.

    Args:
        max_grid: The maximum grid size for the JIT-compiled function.
        compile_only: If True, only compile the function to a flatbuffer.
        debug: If True, print debug information during compilation and execution.
        use_ast_compiler: If True, use the TTIR compiler to generate the IR. Otherwise, uses graph trace

    Returns:
        A wrapped version of the function that when invoked, will JIT compile through D2M and execute the resulting flatbuffer.
    """

    def _decorator(f):
        jit_func = JitFunction(
            f,
            max_grid,
            compile_only,
            debug,
            use_ast_compiler,
        )

        if inspect.ismethod(f):
            return staticmethod(jit_func)
        return jit_func

    return _decorator
