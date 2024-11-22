# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttmlir


def parse_mlir_file(model_path):
    with ttmlir.ir.Context() as ctx, open(model_path, "r") as model_file:
        ttmlir.dialects.ttir.register_dialect(ctx)
        ttmlir.dialects.tt.register_dialect(ctx)
        ttmlir.dialects.ttnn.register_dialect(ctx)
        module = ttmlir.ir.Module.parse(model_file.read(), ctx)
        return module
