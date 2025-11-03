# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict
import model_explorer
from . import runner, utils, mlir
import dataclasses
import logging
import os
from ttmlir import optimizer_overrides

OVERRIDE_PARAMETER_DISABLED_STR = "None"

OPTIMIZER_DISABLED_POLICY = "Optimizer Disabled"

OPTIMIZATION_POLICIES = {
    OPTIMIZER_DISABLED_POLICY: False,
    "DF Sharding": optimizer_overrides.MemoryLayoutAnalysisPolicyType.DFSharding,
    # "Greedy L1 Interleaved": optimizer_overrides.MemoryLayoutAnalysisPolicyType.GreedyL1Interleaved,
    # "BF Interleaved": optimizer_overrides.MemoryLayoutAnalysisPolicyType.BFInterleaved,
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
        override_handler.set_enable_memory_layout_analysis(False)
        override_handler.set_enable_l1_interleaved_fallback_analysis(False)
        override_handler.set_memory_layout_analysis_policy(
            OPTIMIZATION_POLICIES[optimization_policy]
        )

    # Convert settings to output layout overrides.
    if settings.get("overrides"):
        for op_id, overrides in settings["overrides"].items():
            op_name_loc = overrides["named_location"]
            output_layout_override = optimizer_overrides.OutputLayoutOverrideParams()
            conv2d_config_override = optimizer_overrides.Conv2dConfigOverrideParams()
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
                            int(x) for x in attr["value"].strip("[]").split("x")
                        ]
                    case "dtype":
                        conv2d_config_override.set_dtype_from_str(attr["value"])
                    case "weights_dtype":
                        conv2d_config_override.set_weights_dtype_from_str(attr["value"])
                    case "activation":
                        conv2d_config_override.set_activation_from_str(
                            "none"
                            if attr["value"] == OVERRIDE_PARAMETER_DISABLED_STR
                            else attr["value"]
                        )
                    case "deallocate_activation":
                        conv2d_config_override.set_deallocate_activation_from_str(
                            attr["value"]
                        )
                    case "reallocate_halo_output":
                        conv2d_config_override.set_reallocate_halo_output_from_str(
                            attr["value"]
                        )
                    case "act_block_h_override":
                        conv2d_config_override.set_act_block_h_override_from_str(
                            attr["value"].strip("[]")
                        )
                    case "act_block_w_div":
                        conv2d_config_override.set_act_block_w_div_from_str(
                            attr["value"].strip("[]")
                        )
                    case "reshard_if_not_optimal":
                        conv2d_config_override.set_reshard_if_not_optimal_from_str(
                            attr["value"]
                        )
                    case "override_sharding_config":
                        conv2d_config_override.set_override_sharding_config_from_str(
                            attr["value"]
                        )
                    case "shard_layout":
                        if attr["value"] != OVERRIDE_PARAMETER_DISABLED_STR:
                            conv2d_config_override.set_shard_layout_from_str(
                                attr["value"]
                            )
                    case "core_grid":
                        conv2d_config_override.set_core_grid_from_str(attr["value"])
                    case "transpose_shards":
                        conv2d_config_override.set_transpose_shards_from_str(
                            attr["value"]
                        )
                    case "output_layout":
                        conv2d_config_override.set_output_layout_from_str(attr["value"])
                    case "enable_act_double_buffer":
                        conv2d_config_override.set_enable_act_double_buffer_from_str(
                            attr["value"]
                        )
                    case "enable_weights_double_buffer":
                        conv2d_config_override.set_enable_weights_double_buffer_from_str(
                            attr["value"]
                        )
                    case _:
                        raise ValueError(f"Invalid override attribute: {attr['key']}")
            if not output_layout_override.empty():
                override_handler.add_output_layout_override(
                    op_name_loc, output_layout_override
                )
            if not conv2d_config_override.empty():
                override_handler.add_conv2d_config_override(
                    op_name_loc, conv2d_config_override
                )

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
            "supportsPreload": True,
        },
    )
    model_runner = None

    # Required.
    def __init__(self):
        super().__init__()
        self.model_runner = runner.ModelRunner()

    def preload(self, model_path: str, settings: Dict):
        if not os.path.exists(utils.IR_DUMPS_DIR) or not os.path.isdir(
            utils.IR_DUMPS_DIR
        ):
            return utils.to_adapter_format({"graphPaths": []})

        ir_paths = utils.list_ir_files(utils.IR_DUMPS_DIR)

        graph_paths = []
        for path in ir_paths:
            full_path = os.path.abspath(path)
            collections_path = str(utils.get_collection_path(path))

            if (
                os.access(full_path, os.R_OK)
                and os.access(collections_path, os.R_OK)
                and collections_path not in graph_paths
            ):
                graph_paths.append(collections_path)

        return utils.to_adapter_format({"graphPaths": graph_paths})

    def __convert_model(self, model_path: str, settings: Dict):
        if optimized_model_path := self.model_runner.get_optimized_model_path(
            model_path
        ):
            logging.info(f"Using optimized model: {optimized_model_path}")
            # Get performance results.
            perf_trace = self.model_runner.get_perf_trace(model_path)
            memory_trace = self.model_runner.get_memory_usage(model_path)
            golden_results = self.model_runner.get_golden_results(model_path)
            cpp_code = self.model_runner.get_cpp_code(model_path)

            with open(optimized_model_path, "r") as model_file:
                module = utils.parse_mlir_str(model_file.read())

            # Convert TTIR to Model Explorer Graphs and Display/Return
            graph_handler = mlir.GraphHandler()
            graph, overlays = graph_handler.build_graph(
                model_path,
                module,
                self.model_runner,
                perf_trace,
                memory_trace,
                golden_results,
            )

            if overlays:
                graph = utils.add_to_dataclass(graph, "overlays", overlays)

            if overrides := self.model_runner.get_overrides(model_path):
                graph = utils.add_to_dataclass(graph, "overrides", overrides)

            if cpp_code:
                graph = utils.add_to_dataclass(graph, "cppCode", cpp_code)
        else:
            if model_path.endswith(".ttnn"):
                # Executing on a Flatbuffer so we should parse through that path
                module_str = utils.parse_flatbuffer_file(
                    model_path, at_pass="PRE-PIPELINE"
                )

                if module_str:
                    module = utils.parse_mlir_str(module_str)
                elif module_str is None:
                    raise Exception("Failed to parse flatbuffer")
            else:
                with open(model_path, "r") as model_file:
                    module = utils.parse_mlir_str(model_file.read())

            # Convert TTIR to Model Explorer Graphs and Display/Return
            graph_handler = mlir.GraphHandler()
            graph, _ = graph_handler.build_graph(model_path, module, self.model_runner)

            return graph

    def convert(
        self, model_path: str, settings: Dict
    ) -> model_explorer.ModelExplorerGraphs:
        if os.path.isdir(model_path):
            ir_paths = utils.list_ir_files(model_path)
            graphs = []
            for ir_path in ir_paths:
                graphs.append(self.__convert_model(ir_path, settings))

            return utils.to_adapter_collection_format(
                *graphs, label=utils.get_collection_label(model_path)
            )
        else:
            graph = self.__convert_model(model_path, settings)
            return utils.to_adapter_format(graph)

    def execute(
        self, model_path: str, settings: Dict
    ) -> model_explorer.ModelExplorerGraphs:
        override_handler = settings_to_overrides(
            settings, self.model_runner.get_artifacts_dir()
        )
        self.model_runner.run(model_path, override_handler.to_string(), settings)

        return {"graphs": []}

    def status_check(self, model_path: str, settings: Dict):
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
