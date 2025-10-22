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
        "enable-erase-inverse-ops-pass": "false",
        "enable-const-eval": "false",
        "enable-fusing-conv2d-with-multiply-pattern": "true",
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


def chisel_pipeline(
    ttir_path: Path, dump_ttir: bool = False, dump_ttnn: bool = False
) -> Tuple[Module, Module]:
    ctx = Context()
    with ctx:
        ttir_module = Module.parseFile(str(ttir_path))
    # This resets the location so that ttir_module location is just the line and column number where that op is defined
    # and ttnn_module locations points to the location from where that op originates in ttir_module
    ttir_module = Module.parse(str(ttir_module), ctx)
    print(str(ttir_module))

    # Dump TTIR module to file
    if dump_ttir:
        chisel_mlir_path = Path("chisel_ttir.mlir")
        with open(chisel_mlir_path, "w") as f:
            f.write(str(ttir_module))
        print(f"TTIR module dumped to: {chisel_mlir_path}")

    ttnn_module = run_ttir_to_ttnn(str(ttir_module), ctx)

    if dump_ttnn:
        chisel_mlir_path = Path("chisel_ttnn.mlir")
        with open(chisel_mlir_path, "w") as f:
            f.write(str(ttnn_module))
        print(f"TTNN module dumped to: {chisel_mlir_path}")

    return ttir_module, ttnn_module
