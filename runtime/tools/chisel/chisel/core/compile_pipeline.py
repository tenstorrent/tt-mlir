# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple
from pathlib import Path
from copy import copy
from ttmlir.ir import Context, Module, Location
from ttmlir.passmanager import PassManager
from ttmlir.passes import ttir_to_ttnn_backend_pipeline, ttnn_to_flatbuffer_file


def run_ttir_decomposition_with_passmanager(module_path: str) -> Module:
    """Run ttir-to-ttir-decomposition using PassManager"""
    with Context() as ctx:
        module = Module.parseFile(str(module_path))
        pm = PassManager.parse(
            "builtin.module(ttir-implicit-broadcast-fold, ttir-to-ttir-decomposition,canonicalize)"
        )
        pm.enable_ir_printing()
        pm.run(module.operation)

    return module


def run_ttir_to_ttnn(module_str: str, ctx) -> Module:
    module = Module.parse(module_str, ctx)
    pm = PassManager.parse(
        "builtin.module(ttir-to-ttnn-backend-pipeline{enable-erase-inverse-ops-pass=false enable-const-eval=false system-desc-path=ttrt-artifacts/system_desc.ttsys})",
        context=ctx,
    )
    pm.run(module.operation)
    return module


def chisel_pipeline(ttir_path: Path) -> Tuple[Module, Module]:
    ttir_module = run_ttir_decomposition_with_passmanager(ttir_path)
    ctx = Context()
    ttir_module = Module.parse(str(ttir_module), ctx)
    ttnn_module = run_ttir_to_ttnn(str(ttir_module), ctx)

    return ttir_module, ttnn_module
