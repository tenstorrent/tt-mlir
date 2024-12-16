# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict
import model_explorer
from . import runner, utils, mlir
import dataclasses
import enum


class OptimizationPolicy(enum.Enum):
    DFSharding = "DF Sharding"
    L1Interleaved = "L1 Interleaved"
    OptimizerDisabled = "Optimizer Disabled"


OPTIMIZATION_POLICIES = [member.value for member in OptimizationPolicy]


@dataclasses.dataclass
class TTAdapterMetadata(model_explorer.AdapterMetadata):
    settings: Dict[str, list] = dataclasses.field(default_factory=dict)


class TTAdapter(model_explorer.Adapter):
    metadata = TTAdapterMetadata(
        id="tt_adapter",
        name="Tenstorrent MLIR Adapter",
        description="Adapter for Tenstorrent MLIR dialects used in the Forge compiler.",
        source_repo="https://github.com/tenstorrent/tt-mlir/tree/main/tools/explorer/tt_adapter",
        fileExts=["mlir", "ttir"],
        settings={
            "optimizationPolicies": OPTIMIZATION_POLICIES,
        },
    )
    model_runner = None

    # Required.
    def __init__(self):
        super().__init__()
        self.model_runner = runner.ModelRunner()

    def convert(
        self, model_path: str, settings: Dict
    ) -> model_explorer.ModelExplorerGraphs:
        perf_trace = None
        if optimized_model_path := self.model_runner.get_optimized_model_path():
            print(f"Using optimized model: {optimized_model_path}")
            model_path = optimized_model_path

            # Get performance results.
            perf_trace = self.model_runner.get_perf_trace()

        module = utils.parse_mlir_file(model_path)

        # Convert TTIR to Model Explorer Graphs and Display/Return
        graph, perf_data = mlir.build_graph(module, perf_trace)
        if perf_data:
            graph = utils.add_to_dataclass(graph, "perf_data", perf_data.graphsData)
        return {"graphs": [graph]}

    def execute(
        self, model_path: str, settings: Dict
    ) -> model_explorer.ModelExplorerGraphs:
        # TODO(odjuricic, #1178) settings need to be parsed.
        # Waiting on override class for this.

        # Parse optimization policy from settings.
        optimization_policy = settings.get("optimizationPolicy")
        if optimization_policy not in OPTIMIZATION_POLICIES:
            raise ValueError(
                f"Invalid optimization policy selected: {optimization_policy}"
            )
        optimization_policy = OptimizationPolicy(optimization_policy)

        memory_layout_analysis_enabled = True
        memory_layout_analysis_policy = optimization_policy.name

        if optimization_policy == OptimizationPolicy.OptimizerDisabled:
            memory_layout_analysis_enabled = False
            memory_layout_analysis_policy = None

        self.model_runner.run(
            model_path, memory_layout_analysis_enabled, memory_layout_analysis_policy
        )

        return {"graphs": []}

    def status_check(self, model_path: str, settings: Dict) -> bool:
        done = not self.model_runner.is_busy()
        logs = self.model_runner.get_logs()
        progress = self.model_runner.get_progress()
        error = self.model_runner.get_error()

        return utils.to_adapter_format(
            [
                {
                    "isDone": done,
                    "progress": progress,
                    "total": 100,
                    "error": error,
                    "stdout": logs,
                }
            ]
        )
