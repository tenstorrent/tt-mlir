# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from mlir.ir import Module
from ttmlir.passes import (
    stablehlo_to_ttir_pipeline,
    ttir_to_ttnn_backend_pipeline,
    ttnn_to_flatbuffer_file,
)
from ttrt.common.util import Binary, FileManager, Logger


def stablehlo_to_ttir(module: Module) -> Module:
    stablehlo_to_ttir_pipeline(module)
    return module


def ttir_to_ttnn(module: Module) -> Module:
    ttir_to_ttnn_backend_pipeline(module)
    return module


def ttnn_to_flatbuffer(
    module: Module, output_file_name: str = "ttnn_fb.ttnn"
) -> Binary:
    ttnn_to_flatbuffer_file(module, output_file_name)

    logger = Logger()
    file_manager = FileManager(logger)
    return Binary(logger, file_manager, output_file_name)
