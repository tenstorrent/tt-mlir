# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Tuple

from ttmlir.ir import Context, Module
from ttmlir.passmanager import PassManager


def run_ttir_decomposition_with_passmanager(module_path: str) -> Module:
    """
    Run ttir-to-ttir-decomposition using PassManager
    Following passes are executed:
    - ttir-implicit-broadcast-fold:
        This is executed before TTIR Golden Module as it can be tricky to compare tensors
        in golden which are not implicit broadcasted and device which are implicit broadcasted
    - ttir-to-ttir-decomposition:
        This is executed to decompose the TTIR OPs so that we can reduce number of ops for which golden are needed.
        Also allows for easier comparison of tensors in golden and device.
    - canonicalize: This is executed to canonicalize the module
    """
    with Context():
        module = Module.parseFile(str(module_path))
        pre_passes = [
            "ttir-fusing{ttnn-enable-conv2d-with-multiply-pattern=true}",
            "ttir-implicit-broadcast-fold",
            "ttir-to-ttir-decomposition",
            "canonicalize",
        ]
        pm = PassManager.parse(
            "builtin.module({})".format(",".join(pre_passes)),
        )
        pm.run(module.operation)

    return module


def run_ttir_to_ttnn(module_str: str, ctx) -> Module:
    """
    Run ttir-to-ttnn backend pipeline
    Following options are used:
    - enable-erase-inverse-ops-pass: False
        This interferes with the comparison of tensors as golden and device might have different shapes.
    - enable-const-eval: False
        This can be enabled but it makes comparison of tensors more easier if disabled.
    - system-desc-path
    """

    pass_params = {
        # "enable-bfp8-conversion": "true",
        "enable-fusing-conv2d-with-multiply-pattern": "true",
        "enable-erase-inverse-ops-pass": "false",
        "enable-const-eval": "false",
        "system-desc-path": "ttrt-artifacts/system_desc.ttsys",  # TODO: make this configurable as in the rest of the code
    }
    pm = PassManager.parse(
        "builtin.module(ttir-to-ttnn-backend-pipeline{{{}}})".format(
            " ".join([f"{k}={v}" for k, v in pass_params.items()])
        ),
        context=ctx,
    )
    module = Module.parse(module_str, ctx)
    pm.run(module.operation)
    return module


def chisel_pipeline(ttir_path: Path) -> Tuple[Module, Module]:
    ttir_module = run_ttir_decomposition_with_passmanager(ttir_path)
    ctx = Context()
    # This resets the location so that ttir_module location is just the line and column number where that op is defined
    # and ttnn_module locations points to the location from where that op originates in ttir_module
    ttir_module = Module.parse(str(ttir_module), ctx)
    ttnn_module = run_ttir_to_ttnn(str(ttir_module), ctx)

    return ttir_module, ttnn_module
