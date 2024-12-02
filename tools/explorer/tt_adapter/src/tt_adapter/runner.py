# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
import tempfile

# TODO(odjuricic) Cleaner to implement ttrt --quiet flag.
# os.environ["TTRT_LOGGER_LEVEL"] = "ERROR"
from ttrt import API as ttrt
import ttmlir.passes
from . import utils, mlir
import pandas as pd
from model_explorer import node_data_builder


class ModelRunner:
    """
    ModelRunner is a singleton class used for compilation and running of models. Ensuring only one can be run at a time.
    This is necessary because the adaptor class is reinitialized on every request from the frontend, so it cannot keep state.
    """

    _instance = None
    _explorer_artifacts_dir = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            print("Creating a new ModelRunner instance.")
            cls._instance = super(ModelRunner, cls).__new__(cls, *args, **kwargs)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        # Initialize machine to generate SystemDesc and load up functionality to begin
        print("Running ttrt initialization.")
        ttrt.initialize_apis()

        if "TT_MLIR_HOME" not in os.environ:
            raise RuntimeError("TT_MLIR_HOME not set. Did you run source env/activate?")

        # TODO(odjuricic, #1200) ttrt perf breaks if artifacts dir is changed from default.
        # self._explorer_artifacts_dir = os.environ['TT_MLIR_HOME'] + '/explorer-artifacts'
        self._explorer_artifacts_dir = os.environ["TT_MLIR_HOME"] + "/ttrt-artifacts"
        os.makedirs(self._explorer_artifacts_dir, exist_ok=True)

        # Save the system descriptor.
        ttrt.Query(
            args={
                "--save-artifacts": True,
                "--artifact-dir": self._explorer_artifacts_dir,
            }
        )()

    def run(
        self, model_path, memory_layout_analysis_enabled, memory_layout_analysis_policy
    ):
        # TODO(odjuricic, #1174) This should be in a separete thread later.
        model_name = os.path.basename(model_path).split(".")[0]

        options = [
            f'system-desc-path={f"{self._explorer_artifacts_dir}/system_desc.ttsys"}',
            "enable-optimizer=true",
            f"memory-layout-analysis-enabled={memory_layout_analysis_enabled}",
        ]
        if memory_layout_analysis_policy:
            options.append(
                f"memory-layout-analysis-policy={memory_layout_analysis_policy}"
            )
        options_string = " ".join(options)

        module = utils.parse_mlir_file(model_path)

        # Collect unique locations
        name_dict = mlir.get_locs(module)

        try:
            print("Running MLIR compile: TTIR to TTNN Backend Pipeline")
            print("With options: ", options_string)
            # TODO(odjuricic) When we hit compiler assert it terminates the process. We should catch this and return an error to the frontend.
            ttmlir.passes.ttir_to_ttnn_backend_pipeline(module, options_string)
        except Exception as e:
            print("Error running MLIR compile: TTIR to TTNN Backend Pipeline")
            raise e

        # TODO(odjuricic) Move this file somewhere else, but keep the name.
        flatbuffer_file = model_name + ".ttnn"
        try:
            print("Running TTNN to Flatbuffer File")
            ttmlir.passes.ttnn_to_flatbuffer_file(module, flatbuffer_file, {})
        except Exception as e:
            print("Error running TTNN to Flatbuffer File")
            raise e

        # TODO(odjuricic) validate that the module was converted to TTNN without fail

        if os.path.exists(f"{self._explorer_artifacts_dir}/{flatbuffer_file}"):
            print("Removing artifacts of previous run.")
            os.system(f"rm -rf {self._explorer_artifacts_dir}/{flatbuffer_file}")

        ttrt_perf_command = " ".join(
            [
                "ttrt",
                "perf",
                flatbuffer_file,
                f"--artifact-dir={self._explorer_artifacts_dir}",
            ]
        )

        print("Running", ttrt_perf_command)
        process = subprocess.Popen(
            ttrt_perf_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for line in process.stdout:
            print(line, end="")

        process.stdout.close()
        process.wait()

        if process.returncode != 0:
            print(f"Error: TTRT process exited with code {process.returncode}")
            raise RuntimeError("Error running TTRT")

        op_perf_file = f"{self._explorer_artifacts_dir}/{flatbuffer_file}/perf/ops_perf_results.csv"
        if not os.path.exists(op_perf_file):
            raise FileNotFoundError(f"Performance file {op_perf_file} not found.")
        perf = pd.read_csv(op_perf_file)
        columns = [
            "GLOBAL CALL COUNT",
            "OP CODE",
            "DEVICE FW DURATION [ns]",
            "CORE COUNT",
            "OUTPUT_0_MEMORY",
            "LOC",
        ]
        perf = perf[columns]

        # Create the node_data type here
        timing_data = list(zip(perf["LOC"], perf["DEVICE FW DURATION [ns]"]))
        results = {}
        for loc, duration in timing_data:
            loc = mlir.get_loc_str(loc).replace("'", '"')
            if loc in name_dict:
                for i in range(name_dict[loc]):
                    results[f"{loc}__{i}"] = node_data_builder.NodeDataResult(
                        value=duration
                    )
            else:
                print(
                    f"Location {loc} not found in graph, ops data for this op was not reported."
                )

        gradient = [
            node_data_builder.GradientItem(stop=0, bgColor="yellow"),
            node_data_builder.GradientItem(stop=1, bgColor="red"),
        ]

        data = node_data_builder.GraphNodeData(results=results, gradient=gradient)

        res = node_data_builder.ModelNodeData(graphsData={"tt-graph": data})
        return res
