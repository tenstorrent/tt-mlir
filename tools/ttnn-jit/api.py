# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Literal

from ttnn_jit._src.jit import JitFunction


def jit(
    backend: Literal["ttnn", "metal"] = "ttnn",
    max_grid: tuple[int, int] = (8, 8),
    compile_only: bool = False,
    debug: bool = False,
):
    def _decorator(f):
        jit_func = JitFunction(
            f,
            backend,
            max_grid,
            compile_only,
            debug,
        )

        if inspect.ismethod(f):
            return staticmethod(jit_func)
        return jit_func

    return _decorator
