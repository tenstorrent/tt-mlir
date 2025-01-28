# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict
import model_explorer
from . import runner, utils, mlir
import dataclasses
import enum
from ttmlir import optimizer_overrides

OPTIMIZER_DISABLED_POLICY = "Optimizer Disabled"

OPTIMIZATION_POLICIES = {
    "DF Sharding": optimizer_overrides.MemoryLayoutAnalysisPolicyType.DFSharding,
    "Greedy L1 Interleaved": optimizer_overrides.MemoryLayoutAnalysisPolicyType.GreedyL1Interleaved,
    "BF Interleaved": optimizer_overrides.MemoryLayoutAnalysisPolicyType.BFInterleaved,
    OPTIMIZER_DISABLED_POLICY: False,
}


@dataclasses.dataclass
class TTAdapterMetadata(model_explorer.AdapterMetadata):
    settings: Dict[str, list] = dataclasses.field(default_factory=dict)


def settings_to_overrides(settings, artifacts_dir):
    override_handler = optimizer_overrides.OptimizerOverridesHandler()
    override_handler.set_system_desc_path(f"{artifacts_dir}/system_desc.ttsys")

    # Parse optimization policy from settings.
    optimization_policy = settings.get("optimizationPolicy")
    if optimization_policy not in OPTIMIZATION_POLICIES:
        raise ValueError(f"Invalid optimization policy selected: {optimization_policy}")

    if optimization_policy == OPTIMIZER_DISABLED_POLICY:
        override_handler.set_enable_optimizer(False)
    else:
        override_handler.set_enable_optimizer(True)
        override_handler.set_enable_memory_layout_analysis(True)
        override_handler.set_memory_layout_analysis_policy(
            OPTIMIZATION_POLICIES[optimization_policy]
        )

    # Convert settings to output layout overrides.
    if settings.get("overrides"):
        for op_id, overrides in settings["overrides"].items():
            output_layout_override = optimizer_overrides.OutputLayoutOverrideParams()
            op_loc = overrides["named_location"]
            for attr in overrides["attributes"]:
                match attr["key"]:
                    case "data_type":
                        output_layout_override.set_data_type_from_str(attr["value"])
                    case "memory_layout":
                        output_layout_override.set_memory_layout_from_str(attr["value"])
                    case "buffer_type":
                        output_layout_override.set_buffer_type_from_str(attr["value"])
                    case "tensor_memory_layout":
                        output_layout_override.set_tensor_memory_layout_from_str(
                            attr["value"]
                        )
                    case "grid_shape":
                        output_layout_override.grid = [
                            int(x) for x in attr["value"].strip("[]").split(",")
                        ]
                    case _:
                        raise ValueError(f"Invalid override attribute: {attr['key']}")
            override_handler.add_output_layout_override(op_loc, output_layout_override)
    return override_handler


class TTAdapter(model_explorer.Adapter):
    metadata = TTAdapterMetadata(
        id="tt_adapter",
        name="Tenstorrent MLIR Adapter",
        description="Adapter for Tenstorrent MLIR dialects used in the Forge compiler.",
        source_repo="https://github.com/tenstorrent/tt-mlir/tree/main/tools/explorer/tt_adapter",
        fileExts=["mlir", "ttir", "ttnn"],
        settings={
            "optimizationPolicies": list(OPTIMIZATION_POLICIES.keys()),
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
        if optimized_model_path := self.model_runner.get_optimized_model_path(
            model_path
        ):
            print(f"Using optimized model: {optimized_model_path}")
            # Get performance results.
            perf_trace = self.model_runner.get_perf_trace(model_path)

            with open(optimized_model_path, "r") as model_file:
                module = utils.parse_mlir_str(model_file.read())

            # Convert TTIR to Model Explorer Graphs and Display/Return
            graph, perf_data = mlir.build_graph(module, perf_trace)
            if perf_data:
                # TODO(odjuricic) We should replace the perf_data with overlays once this is fixed on FE.
                graph = utils.add_to_dataclass(graph, "perf_data", perf_data.graphsData)

            if overrides := self.model_runner.get_overrides(model_path):
                graph = utils.add_to_dataclass(graph, "overrides", overrides)
        else:
            if model_path.endswith(".ttnn"):
                # Executing on a Flatbuffer so we should parse through that path
                module_str = utils.parse_flatbuffer_file(
                    model_path, at_pass="ConvertTTIRToTTNN"
                )
                assert module_str is not None, "Failed to parse flatbuffer"
                if module_str:
                    module = utils.parse_mlir_str(module_str)
            else:
                with open(model_path, "r") as model_file:
                    module = utils.parse_mlir_str(model_file.read())

            # Convert TTIR to Model Explorer Graphs and Display/Return
            graph, _ = mlir.build_graph(module)

        return {"graphs": [graph]}

    def execute(
        self, model_path: str, settings: Dict
    ) -> model_explorer.ModelExplorerGraphs:
        override_handler = settings_to_overrides(
            settings, self.model_runner.get_artifacts_dir()
        )
        self.model_runner.run(
            model_path, override_handler.to_string(), settings.get("overrides", None)
        )

        return {"graphs": []}

    def status_check(self, model_path: str, settings: Dict) -> bool:
        done = not self.model_runner.is_busy()
        logs = self.model_runner.get_logs()
        progress = self.model_runner.get_progress()
        error = self.model_runner.get_error()

        return utils.to_adapter_format(
            {
                "isDone": done,
                "progress": progress,
                "total": 100,
                "error": error,
                "stdout": logs,
            }
        )
