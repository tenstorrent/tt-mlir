# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttmlir
from dataclasses import make_dataclass


def parse_mlir_file(model_path):
    with ttmlir.ir.Context() as ctx, open(model_path, "r") as model_file:
        ttmlir.dialects.ttir.register_dialect(ctx)
        ttmlir.dialects.tt.register_dialect(ctx)
        ttmlir.dialects.ttnn.register_dialect(ctx)
        module = ttmlir.ir.Module.parse(model_file.read(), ctx)
        return module


def to_dataclass(obj: dict, dc_name: str = "tempClass"):
    return make_dataclass(dc_name, ((k, type(v)) for k, v in obj.items()))(**obj)


def to_adapter_format(obj: dict):
    return {"graphs": [to_dataclass(obj)]}
