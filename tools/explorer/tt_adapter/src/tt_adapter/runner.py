# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import os
import logging

# TODO(odjuricic) Cleaner to implement ttrt --quiet flag.
# os.environ["TTRT_LOGGER_LEVEL"] = "ERROR"
from ttmlir import passes
from . import utils, mlir, ttrt_loader
import pandas as pd
from datetime import datetime, timezone
import threading
import queue
import json
import glob


class ExplorerRunException(Exception):
    pass


class ModelRun:
    """
    The run information for this model, containing metadata about the run.
    """

    # The execution UTC timestamp, in ISO 8601 format
    timestamp: str = None
    # Overrides, changes that the user made to op configurations.
    overrides = None
    # EmitC Call to Generate C Code
    generate_cpp_code = False

    def __init__(self):
        self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ModelState:
    """
    After a model is compiled and executed we keep track of all additional data that was created.
    """

    # Path to the compiled TTNN IR file.
    optimized_model_path = None
    # Path to the output directory where ttrt dumps all model files (perf trace, memory state, etc)
    model_output_dir = None
    # List of previous runs for this model
    runs: list[ModelRun] = []


class ModelRunner:
    """
    ModelRunner is a singleton class used for compilation and running of models. Ensuring only one can be run at a time.
    This is necessary because the adaptor class is reinitialized on every request from the frontend, so it cannot keep state.
    """

    # TTRT module. dynamically imported to enable or disable module execution.
    _ttrt = None

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
    model_state: dict[str, ModelState] = dict()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            logging.info("Creating a new ModelRunner instance.")
            cls._instance = super(ModelRunner, cls).__new__(cls, *args, **kwargs)
            cls._instance._load_ttrt()
            cls._instance.initialize()
        return cls._instance

    def _load_ttrt(self):
        self._ttrt = ttrt_loader.load_ttrt()
        pass

    def initialize(self):
        # Initialize machine to generate SystemDesc and load up functionality to begin
        logging.info("Running ttrt initialization.")

        if "TT_MLIR_HOME" not in os.environ:
            raise RuntimeError("TT_MLIR_HOME not set. Did you run source env/activate?")

        # TODO(odjuricic, #1200) ttrt perf breaks if artifacts dir is changed from default.
        # self._explorer_artifacts_dir = os.environ['TT_MLIR_HOME'] + '/explorer-artifacts'
        self._explorer_artifacts_dir = os.environ["TT_MLIR_HOME"] + "/ttrt-artifacts"
        self._build_dir = os.environ["TT_MLIR_HOME"] + "/build"
        os.makedirs(self._explorer_artifacts_dir, exist_ok=True)

        if self._ttrt is not None:
            self._ttrt.API.initialize_apis()

            # Save the system descriptor.
            self._ttrt.API.Query(
                args={
                    "--save-artifacts": True,
                    "--artifact-dir": self._explorer_artifacts_dir,
                    "--quiet": True,
                }
            )()

        logging.info("ModelRunner initialized.")

    def get_optimized_model_path(self, model_path):
        if model_path in self.model_state:
            return self.model_state[model_path].optimized_model_path
        return None

    def get_output_dir(self, model_path):
        return self.model_state[model_path].model_output_dir

    def get_last_run(self, model_path):
        if model_path in self.model_state:
            if hasattr(self.model_state[model_path], "runs"):
                return self.model_state[model_path].runs[-1]
        return None

    def get_attr_from_last_run(self, model_path: str, attr: str):
        last_run = self.get_last_run(model_path)

        if last_run:
            return getattr(last_run, attr, None)
        return None

    def get_overrides(self, model_path):
        return self.get_attr_from_last_run(model_path, "overrides")

    def get_generate_cpp_code(self, model_path):
        last_run = self.get_last_run(model_path)

        if last_run:
            return getattr(last_run, "generate_cpp_code", False)
        return False

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

    def reset_state(self):
        assert not self.is_busy()
        self.runner_thread = None
        self.runner_error = None
        self.progress = 0
        self.log_queue.queue.clear()

    def init_model_state(self, model_path: str):
        if model_path not in self.model_state:
            self.model_state[model_path] = ModelState()

    def update_model_state(self, model_path: str):
        self.init_model_state(model_path)

        self.model_state[model_path].runs.append(ModelRun())

        return self.model_state[model_path].runs[-1]

    def log(self, message, severity=logging.info):
        severity(message)
        self.log_queue.put(message)

    def get_perf_trace(self, model_path):
        op_perf_file = (
            f"{self.model_state[model_path].model_output_dir}/perf/ops_perf_results.csv"
        )
        if not os.path.exists(op_perf_file):
            raise FileNotFoundError(f"Performance file {op_perf_file} not found.")

        csv_data = pd.read_csv(op_perf_file)
        
        # Log CSV and check tracy_ops_data.csv for MLIR analysis data
        log_file = "/tmp/tt_adapter_convert_debug.log"
        perf_dir = os.path.dirname(op_perf_file)
        tracy_ops_data = os.path.join(perf_dir, "tracy_ops_data.csv")
        
        with open(log_file, "a") as f:
            f.write(f"DEBUG [get_perf_trace]: tracy_ops_data.csv exists = {os.path.exists(tracy_ops_data)}\n")
            if os.path.exists(tracy_ops_data):
                try:
                    import csv
                    # tracy_ops_data.csv uses semicolon delimiter (see perf.py:488)
                    with open(tracy_ops_data, 'r') as tracy_file:
                        reader = csv.DictReader(tracy_file, delimiter=';')
                        tracy_rows = list(reader)
                        f.write(f"DEBUG [get_perf_trace]: tracy_ops_data.csv has {len(tracy_rows)} rows\n")
                        if len(tracy_rows) > 0:
                            f.write(f"DEBUG [get_perf_trace]: tracy_ops_data columns: {list(tracy_rows[0].keys())}\n")
                            f.write(f"DEBUG [get_perf_trace]: First 3 rows:\n")
                            for row in tracy_rows[:3]:
                                f.write(f"  {row}\n")
                            
                            # Count MLIR_OP_LOCATION entries
                            mlir_loc_count = sum(1 for row in tracy_rows if 'MLIR_OP_LOCATION' in str(row.get('MessageName', '')))
                            tt_dnn_op_count = sum(1 for row in tracy_rows if 'TT_DNN_DEVICE_OP' in str(row.get('MessageName', '')))
                            f.write(f"DEBUG [get_perf_trace]: MLIR_OP_LOCATION entries: {mlir_loc_count}\n")
                            f.write(f"DEBUG [get_perf_trace]: TT_DNN_DEVICE_OP entries: {tt_dnn_op_count}\n")
                            if mlir_loc_count == 0:
                                f.write(f"DEBUG [get_perf_trace]: WARNING - No MLIR_OP_LOCATION entries found! Tracy may not be capturing mlir location messages.\n")
                                f.write(f"DEBUG [get_perf_trace]: Sample message types in first 10 rows:\n")
                                for i, row in enumerate(tracy_rows[:10]):
                                    f.write(f"DEBUG [get_perf_trace]:   Row {i}: {row.get('MessageName', 'NO_NAME')[:100]}\n")
                except Exception as e:
                    f.write(f"DEBUG [get_perf_trace]: Error reading tracy_ops_data.csv: {e}\n")
        
        return csv_data

    def get_memory_usage(self, model_path):
        memory_trace = {}
        mem_file = f"{self.model_state[model_path].model_output_dir}/run/program_0/memory_results.json"
        if not os.path.exists(mem_file):
            error = (
                "Error while fetching memory file,",
                mem_file,
                "may not have been created ... Continuing Explorer Execution",
            )
            self.log(error, severity=logging.error)
        else:
            with open(mem_file, "r") as file:
                memory_trace = json.load(file)
        return memory_trace

    def get_golden_results(self, model_path):
        accuracy_res = f"{self.model_state[model_path].model_output_dir}/run/program_0/golden_results.json"

        if not os.path.exists(accuracy_res):
            raise FileNotFoundError(f"Golden results not found @ {accuracy_res}")

        with open(accuracy_res, "r") as f:
            res = json.load(f)

        return res

    def get_cpp_code(self, model_path):
        cpp_code = f"{self.model_state[model_path].model_output_dir}/**/*.cpp"
        cpp_files = glob.glob(cpp_code, recursive=True)

        if cpp_files:
            with open(cpp_files[0], "r") as f:
                return f.read()
        return None

    def run_in_subprocess(self, command):
        self.log(f"Running command:\n{' '.join(command)}\n")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.environ.get("TT_MLIR_HOME", os.getcwd()),
        )

        for line in process.stdout:
            self.log(line.strip())

        process.stdout.close()
        process.wait()

        return process

    def compile_and_run_wrapper(self, model_path, overrides_string):
        try:
            if not ttrt_loader.get_is_ttrt_available():
                raise Exception("TTRT not available. Model execution is disabled.")

            self.compile_and_run(model_path, overrides_string)
        except ExplorerRunException as e:
            self.runner_error = str(e)
            raise e
        except Exception as e:
            self.runner_error = "An unexpected error occurred: " + str(e)
            self.log(self.runner_error, severity=logging.error)
            raise e
        finally:
            self.progress = 100

    def compile_and_run(self, model_path, overrides_string):
        FLATBUFFER = False
        if model_path.endswith(".ttnn"):
            # This is being run from a Flatbuffer. Need To Render TTIR from Flatbuffer
            FLATBUFFER = True
            # Write the TTIR from this file into a temporary file to run through the compiler
            ttir_module_str = utils.parse_flatbuffer_file(
                model_path, at_pass="PRE-PIPELINE"
            )
            ttir_module_path = f"{model_path}_ttir.mlir"
            with open(ttir_module_path, "w+") as temp_module:
                temp_module.write(ttir_module_str)

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

        # Compile to StableHLO or Shardy module to TTIR
        if utils.needs_stablehlo_pass(model_path):
            ttir_ir_file = (
                f"{state.model_output_dir}/{model_name.replace('.mlir', '_ttir.mlir')}"
            )
            compile_command = [
                f"{self._build_dir}/bin/ttmlir-opt",
                "--stablehlo-to-ttir-pipeline",
                model_path,
                "-o",
                ttir_ir_file,
            ]

            self.log("Running StableHLO / Shardy to TTIR Pipeline")

            compile_process = self.run_in_subprocess(compile_command)
            if compile_process.returncode != 0:
                error = "Error running StableHLO / Shardy to TTIR Pipeline"
                self.log(error, severity=logging.error)
                raise ExplorerRunException(error)
        else:
            ttir_ir_file = model_path

        ttnn_ir_file = (
            f"{state.model_output_dir}/{model_name.replace('.mlir', '_ttnn.mlir')}"
        )

        if FLATBUFFER:
            ttnn_ir_file = f"{state.model_output_dir}/{model_name}.mlir"

        compile_command = [
            f"{self._build_dir}/bin/ttmlir-opt",
            f"--ttir-to-ttnn-backend-pipeline={overrides_string}",
            ttir_ir_file if not FLATBUFFER else ttir_module_path,
            "-o",
            ttnn_ir_file,
            "--mlir-print-debuginfo",
        ]

        self.log("Running compile TTIR to TTNN Backend Pipeline")
        self.log("With options: " + overrides_string)

        compile_process = self.run_in_subprocess(compile_command)
        if compile_process.returncode != 0:
            error = "Error running compile TTIR to TTNN Backend Pipeline"
            self.log(error, severity=logging.error)
            raise ExplorerRunException(error)
        self.progress = 20

        ########################### EmitC Generation #############################

        if self.get_generate_cpp_code(model_path):
            # Backend Pipeline has already been run, the file is stored to ttnn_ir_file
            # The translation to flatbuffer will happen, need to run the pass to emitc and then store the artifact
            self.log("Enabled C++ Code Generation w/ EmitC")

            emitc_ir_file = "/".join(
                ttnn_ir_file.split("/")[:-1] + [f"{model_name}_emitc.mlir"]
            )
            emitc_command = [
                f"{self._build_dir}/bin/ttmlir-opt",
                "--ttnn-backend-to-emitc-pipeline",
                ttnn_ir_file,
                "-o",
                emitc_ir_file,
            ]

            self.log("Running compile TTNN IR to EmitC IR")

            compile_process = self.run_in_subprocess(emitc_command)
            if compile_process.returncode != 0:
                error = "Error compiling TTNN IR to EmitC IR"
                self.log(error, severity=logging.error)
                raise ExplorerRunException(error)

            # Translate EmitC IR to C++ File
            self.log("Translating EmitC IR to C++ Code")
            # Write to C Code file directly in ttrt-artifacts-dir
            c_code_file = state.model_output_dir + f"/{model_name}.cpp"
            emitc_translate_command = [
                f"{self._build_dir}/bin/ttmlir-translate",
                "--mlir-to-cpp",
                emitc_ir_file,
                "-o",
                c_code_file,
            ]

            compile_process = self.run_in_subprocess(emitc_translate_command)
            if compile_process.returncode != 0:
                error = "Error translating EmitC IR to C++"
                self.log(error, severity=logging.error)
                raise ExplorerRunException(error)

        ############################## Translate #################################

        # Need this flatbuffer file to inherit the golden data
        if FLATBUFFER:
            golden_map = utils.golden_map_from_flatbuffer(model_path)
            # need to parse this golden_map
            kept_alive_data_arrs = []
            rendered_golden_map = {}

            for entry in golden_map:
                data = entry["value"]
                # Turn this into a Torch Tensor to easily format it for the GoldenMap
                # data is a uint8_t buffer type that contains the data in the format of dtype
                # We will need to render this data as a buffer reference for the GoldenTensor constructor
                import array

                # B is unsigned char in the array library
                # This will parse the data as a 1D Buffer of uint8_t, exactly the pointer type expected
                data_arr = array.array("B", data["data"])

                rendered_golden_map[entry["key"]] = passes.GoldenTensor(
                    data["name"],
                    data["shape"],
                    data["stride"],
                    passes.lookup_dtype(data["dtype"]),
                    data_arr.buffer_info()[0],
                    data_arr.buffer_info()[1],
                )

            # Get module from file
            with open(ttnn_ir_file, "r") as f:
                ttnn_module = utils.parse_mlir_str(f.read())

            self.log("Running TTNN to Flatbuffer File")

            # Run through pybound translation so we can pass golden_map
            try:
                if golden_map:
                    passes.ttnn_to_flatbuffer_file(
                        ttnn_module, flatbuffer_file, rendered_golden_map
                    )
                else:
                    passes.ttnn_to_flatbuffer_file(ttnn_module, flatbuffer_file)
            except:
                self.log("Error while running TTNN to Flatbuffer File")
                raise ExplorerRunException()
        else:
            # Translate to Flatbuffer normally.
            to_flatbuffer_command = [
                f"{self._build_dir}/bin/ttmlir-translate",
                "--ttnn-to-flatbuffer",
                ttnn_ir_file,
                "-o",
                flatbuffer_file,
            ]

            translate_process = self.run_in_subprocess(to_flatbuffer_command)
            if translate_process.returncode != 0:
                error = "Error while running TTNN to Flatbuffer File"
                self.log(error, severity=logging.error)
                raise ExplorerRunException(error)

        self.progress = 30

        ############################## TTRT Perf #################################

        ttrt_perf_command = [
            "ttrt",
            "perf",
            flatbuffer_file,
            f"--artifact-dir={self._explorer_artifacts_dir}",
            "--memory",
        ]

        ttrt_process = self.run_in_subprocess(ttrt_perf_command)

        if ttrt_process.returncode != 0:
            # 42 is the specific code for a test error instead of ttrt
            if ttrt_process.returncode == 42:
                error = (
                    "Error while running TTRT Tests... Continuing Explorer Execution"
                )
                self.log(error, severity=logging.error)
            else:
                error = "Error while running TTRT perf"
                self.log(error, severity=logging.error)
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
        logging.info(perf)

        logging.info(
            "Total device duration: ", perf["DEVICE FW DURATION [ns]"].sum(), "ns"
        )

        # TTNN_IR_FILE from flatbuffer is still relevant since model_path is the FB with golden data and it will rented optimized_model_path instead
        state.optimized_model_path = ttnn_ir_file
        self.progress = 100

    def run(self, model_path, compile_options, settings):
        # Check if a run is already in progress
        if self.is_busy():
            raise RuntimeError(
                "A model is already being processed. Please wait for it to finish."
            )
        
        self.reset_state()
        last_run = self.update_model_state(model_path)

        # Set the EmitC State
        generate_cpp_code = settings.get("generateCppCode", False)
        last_run.generate_cpp_code = generate_cpp_code

        # Set overrides state from settings
        overrides = settings.get("overrides", None)
        last_run.overrides = overrides

        # Start compile and run in a new thread
        self.runner_thread = threading.Thread(
            target=self.compile_and_run_wrapper, args=(model_path, compile_options)
        )
        self.runner_thread.start()
