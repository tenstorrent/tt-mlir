# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Convenience python wrappers around some commonly used compiler and runtime calls.
"""

import os
import subprocess

from ttmlir.ir import Module
from ttmlir.passes import (
    stablehlo_to_ttir_pipeline,
    ttir_to_ttmetal_backend_pipeline,
    ttir_to_ttnn_backend_pipeline,
    ttmetal_to_flatbuffer_file,
    ttnn_to_flatbuffer_file,
)
from ttrt.common.util import Binary, FileManager, Logger


def stablehlo_to_ttir(module: Module) -> Module:
    """
    Runs `stablehlo-to-ttir-pipeline` compiler pass on `module` in-place.

    Wrapper around `stablehlo_to_ttir_pipeline` pybound pass.

    Returns `module` for convenince.
    """
    stablehlo_to_ttir_pipeline(module)
    return module


def ttir_to_ttnn(
    module: Module,
    system_desc: str = os.getenv(
        "SYSTEM_DESC_PATH",
        "ttrt-artifacts/system_desc.ttsys",
    ),
) -> Module:
    """
    Runs `ttir-to-ttnn-backend-pipeline` compiler pass on `module` in-place.

    Wrapper around `ttir_to_ttnn_backend_pipeline` pybound pass.

    Returns `module` for convenince.
    """
    assert os.path.exists(system_desc), f"System desc file {system_desc} not found."

    ttir_to_ttnn_backend_pipeline(module, f"system-desc-path={system_desc}")
    return module


def ttir_to_ttmetal(
    module: Module,
    system_desc: str = os.getenv(
        "SYSTEM_DESC_PATH",
        "ttrt-artifacts/system_desc.ttsys",
    ),
) -> Module:
    """
    Runs `ttir-to-ttmetal-backend-pipeline` compiler pass on `module` in-place.

    Wrapper around `ttir_to_ttmetal_backend_pipeline` pybound pass.

    Returns `module` for convenince.
    """
    assert os.path.exists(system_desc), f"System desc file {system_desc} not found."

    ttir_to_ttmetal_backend_pipeline(module, f"system-desc-path={system_desc}")
    return module


def ttnn_to_flatbuffer(
    module: Module, output_file_name: str = "ttnn_fb.ttnn"
) -> Binary:
    """
    Converts TTNN module to flatbuffer and saves to file.

    Wrapper around `ttnn_to_flatbuffer_file` pybound pass.

    Returns flatbuffer `Binary` instance for convenience.
    """
    ttnn_to_flatbuffer_file(module, output_file_name)

    logger = Logger()
    file_manager = FileManager(logger)
    return Binary(logger, file_manager, output_file_name)


def ttmetal_to_flatbuffer(
    module: Module, output_file_name: str = "ttmetal_fb.ttm"
) -> Binary:
    """
    Converts TTMetal module to flatbuffer and saves to file.

    Wrapper around `ttmetal_to_flatbuffer_file` pybound pass.

    Returns flatbuffer `Binary` instance for convenience.
    """
    ttmetal_to_flatbuffer_file(module, output_file_name)

    logger = Logger()
    file_manager = FileManager(logger)
    return Binary(logger, file_manager, output_file_name)


def run_flatbuffer(flatbuffer: Binary) -> int:
    """
    Runs `flatbuffer` on device.

    Returns return code of `ttrt run flatbuffer.file_path` bash process.

    NOTE This could be written using runtime's python API like
    ```
    API.initialize_apis()
    run_instance = API.Run(args={"binary": flatbuffer.file_path})
    return_code, _ = run_instance()
    return return_code
    ```
    Done this way since the only thing we need is the return code and it protects us
    from segfaults and such impossible-to-catch errors.
    """
    result = subprocess.run(["ttrt", "run", flatbuffer.file_path])
    return result.returncode
