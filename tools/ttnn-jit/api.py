# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect

import ttnn
from ttnn_jit._src.jit import JitFunction


def jit(
    compile_only: bool = False,
    debug: bool = False,
    enable_cache: bool = False,
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
    enable_l1_acc: bool = False,
    use_tile_matmul: bool = False,
    memory_config: ttnn.MemoryConfig = None,
):
    """
    Sets up the decorated function to be JIT compiled through D2M.

    Args:
        compile_only: If True, only compile the function to a flatbuffer.
        debug: If True, print debug information during compilation and execution.
        enable_cache: If True, enables caching for compiled JIT graphs.
        math_fidelity: Set the math fidelity level for computations. Options are "LoFi", "HiFi2", "HiFi3", and "HiFi4".
        memory_config: Output memory configuration for the function. If specified, the output tensor will use this exact layout.
                      If unspecified (None), the output will use a maximally L1 block sharded layout.

    Returns:
        A wrapped version of the function that when invoked, will JIT compile through D2M and execute the resulting flatbuffer.
    """

    def _decorator(f):
        jit_func = JitFunction(
            f,
            compile_only,
            debug,
            enable_cache,
            math_fidelity,
            enable_l1_acc,
            use_tile_matmul,
            memory_config,
        )

        if inspect.ismethod(f):
            return staticmethod(jit_func)
        return jit_func

    return _decorator
