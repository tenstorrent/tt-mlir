# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# TODO this file is simulating ttmlir lib with pybound compiler calls.

from mlir.ir import Module


def stablehlo_to_ttir(module: Module) -> Module:
    pass


def ttir_to_ttnn(module: Module) -> Module:
    pass


class Binary:
    # Representing flatbuffer.
    pass


def ttnn_to_flatbuffer(module: Module) -> Binary:
    pass
