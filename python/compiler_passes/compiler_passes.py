# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


from mlir.ir import Module


def stablehlo_to_ttir(module: Module) -> Module:
    # TODO
    # stablehlo_to_ttir_pipeline(module)
    # return module
    pass


def ttir_to_ttnn(module: Module) -> Module:
    # Note that it modifies module in place. Make a module.clone() if that is not desired.
    # ttir_to_ttnn_backend_pipeline(module, f"system-desc-path={os.getenv("SYSTEM_DESC_PATH")}")
    # return module
    pass


# TODO should return Binary, check out tools/explorer/tt_adapter/src/tt_adapter/utils.py
# it uses Binary
def ttnn_to_flatbuffer(module, output_file_name: str = "ttnn_fb.ttnn") -> None:
    # ttnn_to_flatbuffer_file(module, output_file_name)
    pass
