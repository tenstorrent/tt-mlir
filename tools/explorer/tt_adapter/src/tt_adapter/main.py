# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Dict
import model_explorer
from . import runner, utils, mlir
import dataclasses
import logging
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
        settings={"optimizationPolicies": list(OPTIMIZATION_POLICIES.keys())},
    )
    model_runner = None

    # Required.
    def __init__(self):
        super().__init__()
        self.model_runner = runner.ModelRunner()

    def convert(
        self, model_path: str, settings: Dict
    ) -> model_explorer.ModelExplorerGraphs:
        import os
        import logging
        
        # Log to file AND console
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        optimized_model_path = self.model_runner.get_optimized_model_path(model_path)
        
        # Write to log file for debugging subprocess issues
        log_file = "/tmp/tt_adapter_convert_debug.log"
        with open(log_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"DEBUG [Adapter.convert]: model_path = {model_path}\n")
            f.write(f"DEBUG [Adapter.convert]: optimized_model_path = {optimized_model_path}\n")
            f.write(f"DEBUG [Adapter.convert]: model_path in model_state = {model_path in self.model_runner.model_state}\n")
            f.write(f"DEBUG [Adapter.convert]: All model_state keys = {list(self.model_runner.model_state.keys())}\n")
        
        logger.info(f"DEBUG [Adapter.convert]: model_path = {model_path}")
        logger.info(f"DEBUG [Adapter.convert]: optimized_model_path = {optimized_model_path}")
        logger.info(f"DEBUG [Adapter.convert]: model_path in model_state = {model_path in self.model_runner.model_state}")
        logger.info(f"DEBUG [Adapter.convert]: All model_state keys = {list(self.model_runner.model_state.keys())}")
        
        if model_path in self.model_runner.model_state:
            state = self.model_runner.model_state[model_path]
            print(f"DEBUG [Adapter.convert]: model_state[model_path].optimized_model_path = {state.optimized_model_path}")
            print(f"DEBUG [Adapter.convert]: model_state[model_path].model_output_dir = {state.model_output_dir}")
            
            # Check if paths exist
            if state.optimized_model_path:
                print(f"DEBUG [Adapter.convert]: optimized_model_path exists = {os.path.exists(state.optimized_model_path)}")
            if state.model_output_dir:
                print(f"DEBUG [Adapter.convert]: model_output_dir exists = {os.path.exists(state.model_output_dir)}")
                perf_dir = os.path.join(state.model_output_dir, "perf")
                csv_path = os.path.join(perf_dir, "ops_perf_results.csv")
                print(f"DEBUG [Adapter.convert]: perf_dir exists = {os.path.exists(perf_dir)}")
                print(f"DEBUG [Adapter.convert]: ops_perf_results.csv exists = {os.path.exists(csv_path)}")
                if os.path.exists(csv_path):
                    print(f"DEBUG [Adapter.convert]: ops_perf_results.csv size = {os.path.getsize(csv_path)} bytes")
                    print(f"DEBUG [Adapter.convert]: ops_perf_results.csv readable = {os.access(csv_path, os.R_OK)}")
                
                # Check if we can create test files in model_output_dir
                if os.path.exists(state.model_output_dir):
                    print(f"DEBUG [Adapter.convert]: model_output_dir writable = {os.access(state.model_output_dir, os.W_OK)}")
                    test_file = os.path.join(state.model_output_dir, ".test_write")
                    try:
                        with open(test_file, "w") as f:
                            f.write("test")
                        os.remove(test_file)
                        print(f"DEBUG [Adapter.convert]: Test file creation in model_output_dir = SUCCESS")
                    except Exception as e:
                        print(f"DEBUG [Adapter.convert]: Test file creation in model_output_dir = FAILED: {e}")
        
        if optimized_model_path:
            logging.info(f"Using optimized model: {optimized_model_path}")
            # Get performance results.
            print(f"DEBUG [Adapter.convert]: About to call get_perf_trace")
            try:
                perf_trace = self.model_runner.get_perf_trace(model_path)
                print(f"DEBUG [Adapter.convert]: get_perf_trace returned successfully, rows = {len(perf_trace) if perf_trace is not None else 'None'}")
            except FileNotFoundError as e:
                print(f"DEBUG [Adapter.convert]: get_perf_trace raised FileNotFoundError: {e}")
                raise
            except Exception as e:
                print(f"DEBUG [Adapter.convert]: get_perf_trace raised unexpected exception: {type(e).__name__}: {e}")
                raise
            
            memory_trace = self.model_runner.get_memory_usage(model_path)
            golden_results = self.model_runner.get_golden_results(model_path)
            cpp_code = self.model_runner.get_cpp_code(model_path)

            with open(optimized_model_path, "r") as model_file:
                module = utils.parse_mlir_str(model_file.read())

            # Convert TTIR to Model Explorer Graphs and Display/Return
            graph_handler = mlir.GraphHandler()
            
            log_file = "/tmp/tt_adapter_convert_debug.log"
            with open(log_file, "a") as f:
                f.write(f"DEBUG [Adapter.convert]: About to call build_graph\n")
                f.write(f"DEBUG [Adapter.convert]: perf_trace type = {type(perf_trace)}, rows = {len(perf_trace) if perf_trace is not None else 'None'}\n")
            
            graph, overlays = graph_handler.build_graph(
                model_path,
                module,
                self.model_runner,
                perf_trace,
                memory_trace,
                golden_results,
            )
            
            with open(log_file, "a") as f:
                f.write(f"DEBUG [Adapter.convert]: build_graph returned\n")
                f.write(f"DEBUG [Adapter.convert]: overlays type = {type(overlays)}\n")
                f.write(f"DEBUG [Adapter.convert]: overlays = {overlays}\n")
                f.write(f"DEBUG [Adapter.convert]: overlays is truthy = {bool(overlays)}\n")
                if isinstance(overlays, dict):
                    f.write(f"DEBUG [Adapter.convert]: overlays keys = {list(overlays.keys())}\n")

            if overlays:
                with open(log_file, "a") as f:
                    f.write(f"DEBUG [Adapter.convert]: Adding overlays to graph\n")
                graph = utils.add_to_dataclass(graph, "overlays", overlays)
            else:
                with open(log_file, "a") as f:
                    f.write(f"DEBUG [Adapter.convert]: NOT adding overlays (empty or falsy)\n")

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

        return {"graphs": [graph]}

    def execute(
        self, model_path: str, settings: Dict
    ) -> model_explorer.ModelExplorerGraphs:
        override_handler = settings_to_overrides(
            settings, self.model_runner.get_artifacts_dir()
        )
        self.model_runner.run(model_path, override_handler.to_string(), settings)

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
