# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Literal

import ttnn
from ttnn_jit._src.jit import JitFunction


def jit(
    compile_only: bool = False,
    debug: bool = False,
    enable_cache: bool = False,
    graph_capture: bool = True,
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
):
    """
    Sets up the decorated function to be JIT compiled through D2M.

    Args:
        compile_only: If True, only compile the function to a flatbuffer.
        debug: If True, print debug information during compilation and execution.
        graph_capture: If True, use graph trace compiler to generate the IR. Otherwise, uses AST compiler (TTIR).
        math_fidelity: Set the math fidelity level for computations. Options are "LoFi", "HiFi2", "HiFi3", and "HiFi4".

    Returns:
        A wrapped version of the function that when invoked, will JIT compile through D2M and execute the resulting flatbuffer.
    """

    def _decorator(f):
        jit_func = JitFunction(
            f,
            compile_only,
            debug,
            enable_cache,
            graph_capture,
            math_fidelity,
        )

        if inspect.ismethod(f):
            return staticmethod(jit_func)
        return jit_func

    return _decorator
