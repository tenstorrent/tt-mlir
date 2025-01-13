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
import threading
import queue


class ExplorerRunException(Exception):
    pass


class ModelState:
    """
    After a model is compiled and executed we keep track of all additional data that was created.
    """

    # Path to the compiled TTNN IR file.
    optimized_model_path = None
    # Path to the output directory where ttrt dumps all model files (perf trace, memory state, etc)
    model_output_dir = None
    # Overrides, changes that the user made to op configurations.
    overrides = None


class ModelRunner:
    """
    ModelRunner is a singleton class used for compilation and running of models. Ensuring only one can be run at a time.
    This is necessary because the adaptor class is reinitialized on every request from the frontend, so it cannot keep state.
    """

    # Global static runner state. Initialized once.
    _instance = None
    _explorer_artifacts_dir = None
    _build_dir = None

    # Singleton runner state. Initialized on every run.
    runner_thread = None
    runner_error = None
    log_queue = queue.Queue()
    # progress should be a number between 0 and 100.
    progress = 0

    # State for models that have been executed.
    # Contains a mapping from model path to ModelState.
    model_state = dict()

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
        self._build_dir = os.environ["TT_MLIR_HOME"] + "/build"
        os.makedirs(self._explorer_artifacts_dir, exist_ok=True)

        # Save the system descriptor.
        ttrt.Query(
            args={
                "--save-artifacts": True,
                "--artifact-dir": self._explorer_artifacts_dir,
                "--quiet": True,
            }
        )()

        print("ModelRunner initialized.")

    def get_optimized_model_path(self, model_path):
        if model_path in self.model_state:
            return self.model_state[model_path].optimized_model_path
        return None

    def get_output_dir(self, model_path):
        return self.model_state[model_path].model_output_dir

    def get_overrides(self, model_path):
        if model_path in self.model_state:
            return self.model_state[model_path].overrides
        return None

    def get_error(self):
        return self.runner_error

    def get_progress(self):
        return self.progress

    def get_artifacts_dir(self):
        return self._explorer_artifacts_dir

    def is_busy(self):
        return self.runner_thread and self.runner_thread.is_alive()

    def get_logs(self):
        logs = []
        while not self.log_queue.empty():
            logs.append(self.log_queue.get())
        return "\n".join(logs)

    def reset_state(self, model_path):
        assert not self.is_busy()
        self.runner_thread = None
        self.runner_error = None
        self.progress = 0
        self.log_queue.queue.clear()

        if model_path in self.model_state:
            del self.model_state[model_path]

    def log(self, message):
        print(message)
        self.log_queue.put(message)

    def get_perf_trace(self, model_path):
        op_perf_file = (
            f"{self.model_state[model_path].model_output_dir}/perf/ops_perf_results.csv"
        )
        if not os.path.exists(op_perf_file):
            raise FileNotFoundError(f"Performance file {op_perf_file} not found.")

        return pd.read_csv(op_perf_file)

    def run_in_subprocess(self, command):
        self.log(f"Running command:\n{' '.join(command)}\n")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        for line in process.stdout:
            self.log(line.strip())

        process.stdout.close()
        process.wait()

        return process

    def compile_and_run_wrapper(self, model_path, overrides_string):
        try:
            self.compile_and_run(model_path, overrides_string)
        except ExplorerRunException as e:
            self.runner_error = str(e)
            raise e
        except Exception as e:
            self.runner_error = "An unexpected error occurred: " + str(e)
            self.log(self.runner_error)
            raise e
        finally:
            self.progress = 100

    def compile_and_run(self, model_path, overrides_string):
        model_name = os.path.basename(model_path)
        flatbuffer_file = model_name + ".ttnn"
        state = self.model_state[model_path]

        state.model_output_dir = self._explorer_artifacts_dir + "/" + flatbuffer_file

        if os.path.exists(state.model_output_dir):
            self.log("Removing artifacts of previous run.")
            os.system(f"rm -rf {state.model_output_dir}")

        os.makedirs(state.model_output_dir)
        # Copy the model to the run directory.
        os.system(f"cp {model_path} {state.model_output_dir}")

        self.progress = 10

        ############################### Compile ##################################

        ttnn_ir_file = (
            f"{state.model_output_dir}/{model_name.replace('.mlir', '_ttnn.mlir')}"
        )
        compile_command = [
            f"{self._build_dir}/bin/ttmlir-opt",
            f"--ttir-to-ttnn-backend-pipeline={overrides_string}",
            model_path,
            "-o",
            ttnn_ir_file,
            "--mlir-print-debuginfo",
        ]

        self.log("Running compile TTIR to TTNN Backend Pipeline")
        self.log("With options: " + overrides_string)

        compile_process = self.run_in_subprocess(compile_command)
        if compile_process.returncode != 0:
            error = "Error running compile TTIR to TTNN Backend Pipeline"
            self.log(error)
            raise ExplorerRunException(error)
        self.progress = 20

        ############################## Translate #################################

        to_flatbuffer_command = [
            f"{self._build_dir}/bin/ttmlir-translate",
            "--ttnn-to-flatbuffer",
            ttnn_ir_file,
            "-o",
            flatbuffer_file,
        ]

        self.log("Running TTNN to Flatbuffer File")
        translate_process = self.run_in_subprocess(to_flatbuffer_command)
        if translate_process.returncode != 0:
            error = "Error while running TTNN to Flatbuffer File"
            self.log(error)
            raise ExplorerRunException(error)
        self.progress = 30

        ############################## TTRT Perf #################################

        ttrt_perf_command = [
            "ttrt",
            "perf",
            flatbuffer_file,
            f"--artifact-dir={self._explorer_artifacts_dir}",
        ]

        ttrt_process = self.run_in_subprocess(ttrt_perf_command)

        if ttrt_process.returncode != 0:
            error = "Error while running TTRT perf"
            self.log(error)
            raise ExplorerRunException(error)

        perf = self.get_perf_trace(model_path)
        columns = [
            "GLOBAL CALL COUNT",
            "OP CODE",
            "DEVICE FW DURATION [ns]",
            "CORE COUNT",
            "OUTPUT_0_MEMORY",
            "LOC",
        ]
        perf = perf[columns]
        print(perf)

        print("Total device duration: ", perf["DEVICE FW DURATION [ns]"].sum(), "ns")

        state.optimized_model_path = ttnn_ir_file
        self.progress = 100

    def run(self, model_path, compile_options, overrides):
        # Check if a run is already in progress
        if self.is_busy():
            raise RuntimeError(
                "A model is already being processed. Please wait for it to finish."
            )
        self.reset_state(model_path)
        self.model_state[model_path] = ModelState()
        self.model_state[model_path].overrides = overrides

        # Start compile and run in a new thread
        self.runner_thread = threading.Thread(
            target=self.compile_and_run_wrapper, args=(model_path, compile_options)
        )
        self.runner_thread.start()
