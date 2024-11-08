# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict
import model_explorer
from . import ttir, runner, utils


class TTAdapter(model_explorer.Adapter):
    metadata = model_explorer.AdapterMetadata(
        id="tt_adapter",
        name="Tenstorrent MLIR Adapter",
        description="Adapter for Tenstorrent MLIR dialects used in the Forge compiler.",
        source_repo="https://github.com/tenstorrent/tt-mlir/tree/main/tools/explorer/tt_adapter",
        fileExts=["mlir", "ttir"],
    )
    model_runner = None

    # Required.
    def __init__(self):
        super().__init__()
        self.model_runner = runner.ModelRunner()

    def convert(
        self, model_path: str, settings: Dict
    ) -> model_explorer.ModelExplorerGraphs:
        module = utils.parse_mlir_file(model_path)

        # Convert TTIR to Model Explorer Graphs and Display/Return
        graph = ttir.ttir_to_graph(module)
        return {"graphs": [graph]}

    def execute(
        self, model_path: str, settings: Dict
    ) -> model_explorer.ModelExplorerGraphs:
        # TODO(odjuricic, #1178) settings need to be parsed.
        # Waiting on override class for this.
        ttnn_ir = self.model_runner.run(model_path)

        # TODO(odjuricic, #933) Parse TTNN IR and return the post optimized graph.
        return {"graphs": []}
