# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect

from d2m_jit._src.ast_compiler import AstCompiler

def jit(
    grid: tuple[int, int] = (1, 1),
    compile_only: bool = False,
    debug: bool = False,
    math_fidelity = None, # Make this untyped for now to avoid ttnn dependency in parsing
):
    """
    Sets up the decorated function to be AST parsed into a d2m.generic region.

    Args:
        grid: Tuple defining the generic op core grid.
        compile_only: If True, only compile the function to a flatbuffer.
        debug: If True, print debug information during compilation and execution.
        math_fidelity: Set the math fidelity level for computations. Options are "LoFi", "HiFi2", "HiFi3", "HiFi4".

    Returns:
        A wrapped version of the function that when invoked, will JIT compile through D2M.
    """

    def _decorator(f):
        # We delay compilation until the first call so we can capture tensor argument metadata (shapes/dtypes)
        def wrapper(*args, **kwargs):
            compiler = AstCompiler(
                f,
                grid,
                compile_only,
                debug,
                math_fidelity,
            )
            return compiler.compile_and_run(*args, **kwargs)

        if inspect.ismethod(f):
            return staticmethod(wrapper)
        return wrapper

    return _decorator
