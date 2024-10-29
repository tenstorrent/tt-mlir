# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict
import model_explorer
import ttmlir
from . import ttir


class TTAdapter(model_explorer.Adapter):
    metadata = model_explorer.AdapterMetadata(
        id="tt_adapter",
        name="Tenstorrent MLIR Adapter",
        description="Adapter for Tenstorrent MLIR dialects used in the Forge compiler.",
        source_repo="https://github.com/tenstorrent/tt-mlir/tree/main/tools/explorer/tt_adapter",
        fileExts=["mlir", "ttir"],
    )

    # Required.
    def __init__(self):
        super().__init__()

    def convert(
        self, model_path: str, settings: Dict
    ) -> model_explorer.ModelExplorerGraphs:
        with ttmlir.ir.Context() as ctx, open(model_path, "r") as model_file:
            ttmlir.dialects.ttkernel.register_dialect(ctx)
            ttmlir.dialects.ttir.register_dialect(ctx)
            ttmlir.dialects.tt.register_dialect(ctx)
            module = ttmlir.ir.Module.parse("".join(model_file.readlines()), ctx)

        # Convert TTIR to Model Explorer Graphs and Display/Return
        graph = ttir.ttir_to_graph(module, ctx)
        return {"graphs": [graph]}
