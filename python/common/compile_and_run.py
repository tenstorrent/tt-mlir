# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Convenience python wrappers around some commonly used compiler and runtime calls.

All functions exposed in this file are segfault-safe, i.e. they are resistant to any
unpredictable errors that might happen inside pybound compiler calls, like segfaults,
which would crash the main python process otherwise. They handle them gracefully by
raising RuntimeError that can be handled with a try-except block in caller.
"""

import os

from ttmlir.ir import Module

from .compile_and_run_internal import *


def stablehlo_to_ttir(module: Module | str) -> Module:
    """
    Runs `stablehlo-to-ttir-pipeline` compiler pass on `module` in a safe way.

    This is a segfault resistant function. It runs the pybound compiler pass in a
    separate process, thus protecting the caller of this function from any unpredictable
    (those that cannot be caught with a try-except) errors.

    Returns
    -------
    Module produced by the pass.

    Raises
    ------
    RuntimeError if any errors happen.
    """
    m = module if isinstance(module, str) else str(module)
    return run_compilation_process(stablehlo_to_ttir_pipeline_worker, (m,))


def _resolve_system_desc(system_desc: str | None) -> str:
    if system_desc is not None:
        return system_desc
    return os.getenv("SYSTEM_DESC_PATH", "ttrt-artifacts/system_desc.ttsys")


def ttir_to_ttnn(
    module: Module | str,
    system_desc: str | None = None,
) -> Module:
    """
    Runs `ttir-to-ttnn-runtime-pipeline` compiler pass on `module` in a safe way.

    This is a segfault resistant function. It runs the pybound compiler pass in a
    separate process, thus protecting the caller of this function from any unpredictable
    (those that cannot be caught with a try-except) errors.

    Returns
    -------
    Module produced by the pass.

    Raises
    ------
    RuntimeError if any errors happen.
    """
    m = module if isinstance(module, str) else str(module)
    return run_compilation_process(
        ttir_to_ttnn_runtime_pipeline_worker, (m, _resolve_system_desc(system_desc))
    )


def ttir_to_ttmetal(
    module: Module | str,
    system_desc: str | None = None,
) -> Module:
    """
    Runs `ttir-to-ttmetal-pipeline` compiler pass on `module` in a safe way.

    This is a segfault resistant function. It runs the pybound compiler pass in a
    separate process, thus protecting the caller of this function from any unpredictable
    (those that cannot be caught with a try-except) errors.

    Returns
    -------
    Module produced by the pass.

    Raises
    ------
    RuntimeError if any errors happen.
    """
    m = module if isinstance(module, str) else str(module)
    return run_compilation_process(
        ttir_to_ttmetal_backend_pipeline_worker, (m, _resolve_system_desc(system_desc))
    )


def ttnn_to_flatbuffer(module: Module, output_file_name: str = "ttnn_fb.ttnn") -> str:
    """
    Converts TTNN module to flatbuffer in a safe way and saves to file.

    This is a segfault resistant function. It runs the pybound translation pass in a
    separate process, thus protecting the caller of this function from any unpredictable
    (those that cannot be caught with a try-except) errors.

    Returns
    -------
    Path to the generated flatbuffer file.

    Raises
    ------
    RuntimeError if any errors happen.
    """
    m = module if isinstance(module, str) else str(module)
    return run_translation_process(
        ttnn_to_flatbuffer_file_worker, (m, output_file_name)
    )


def ttmetal_to_flatbuffer(
    module: Module, output_file_name: str = "ttmetal_fb.ttm"
) -> str:
    """
    Converts ttmetal module to flatbuffer in a safe way and saves to file.

    This is a segfault resistant function. It runs the pybound translation pass in a
    separate process, thus protecting the caller of this function from any unpredictable
    (those that cannot be caught with a try-except) errors.

    Returns
    -------
    Path to the generated flatbuffer file.

    Raises
    ------
    RuntimeError if any errors happen.
    """
    m = module if isinstance(module, str) else str(module)
    return run_translation_process(
        ttmetal_to_flatbuffer_file_worker, (m, output_file_name)
    )


def run_flatbuffer(flatbuffer_path: str) -> int:
    """
    Runs flatbuffer at `flatbuffer_path` on device in a safe way.

    This is a segfault resistant function. It runs the device execution in a
    separate subprocess, thus protecting the caller of this function from any
    unpredictable (those that cannot be caught with a try-except) errors.

    Returns
    -------
    Return code of the device run subprocess (0 on success).

    Raises
    ------
    RuntimeError if any errors happen.
    """
    return run_flatbuffer_execution_process(flatbuffer_path)
