# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import json
import importlib.machinery
import sys
import signal
import os
import io
import subprocess
import time
import socket
from pkg_resources import get_distribution
import sys
import shutil
import atexit
import logging
from io import StringIO

from ttrt.common.util import *

#######################################################################################
########################################**API**########################################
#######################################################################################
class API:
    registered_apis = {}
    registered_functions = ["__init__", "_preprocess", "check_constraints", "execute", "postprocess", "__str__", "__getitem__", "__setitem__", "__call__", "register_arg", "generate_subparser"]

    @staticmethod
    def initialize_api():
        pass

    @staticmethod
    def register_api(api_class):
        missing_methods = [
            func for func in API.registered_functions 
            if not hasattr(api_class, func)
        ]

        if missing_methods:
            raise TypeError(f"API class is missing methods: {missing_methods}")

        API.registered_apis[api_class.__name__] = api_class

    class query:
        registered_args = {}

        def __init__(self, logging, args):
            self.system_desc = None
            self.device_ids = None
            self.logging = logging
            self.artifacts = Artifacts.prepare_artifact()
            
            for name, _ in API.query.registered_args.items():
                self[name] = args[name]
          
        def preprocess(self):
            if self["clean_artifacts"]:
                self.artifacts.clean_artifacts()

            if self["save_artifacts"]:
                self.artifacts.create_artifacts()

        def check_constraints(self):
            pass

        def execute(self):
            import ttrt.runtime

            try:
                self.system_desc, self.device_ids = ttrt.runtime.get_current_system_desc()
            except Exception as e:
                raise Exception(f"an unexpected error occurred: {e}")

        def postprocess(self):
            if self["save_artifacts"]:
                self.artifacts.save_system_desc(self.system_desc)

        def __str__(self):
            pass 

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key):
            setattr(self, key, value)

        def __call__(self):
            self.preprocess()
            self.check_constraints()
            self.execute()
            self.postprocess()
            
        def get_system_desc(self):
            return StringIO(json.loads(self.system_desc))
        
        def get_system_desc_as_dict(self):
            return json.loads(self.system_desc.as_json())

        @staticmethod
        def register_arg(name, type, default, choices, help):
            API.query.registered_args[name] = {"type": type, "default": default, "choices": choices, "help": help}

        @staticmethod
        def generate_subparser(subparsers):
            query_parser = subparsers.add_parser(
                "query", help="query information about the current system"
            )
            query_parser.set_defaults(func=API.query)

            for name, attributes in API.query.registered_args.items():
                query_parser.add_argument(
                    f"--{name}",
                    type=attributes["type"],
                    default=attributes["default"],
                    choices=choices["choices"]
                    help=attributes["help"],
                )

            return query_parser

    class read:
        registered_args = {}
        read_actions = {
            "all": Binary: API.read.all,
            "version": Binary: API.read.version,
            "system-desc": Binary: API.read.system_desc,
            "mlir": Binary: API.read.mlir,
            "cpp": Binary: API.read.cpp,
            "inputs": Binary: API.read.inputs,
            "outputs": Binary: API.read.outputs,
        }
        
        def __init__(self, logging, args):
            self.logging = logging
            self.artifacts = Artifacts.prepare_artifact()
            self.ttnn_binaries = None
            self.ttmetal_binaries = None
            
            for name, _ in API.read.registered_args.items():
                self[name] = args[name]
          
        def preprocess(self):
            if self["clean_artifacts"]:
                self.artifacts.clean_artifacts()

            if self["save_artifacts"]:
                self.artifacts.create_artifacts()

            ttnn_binary_paths = BinaryTTNN.find_ttnn_binary_paths(self["binary"])
            ttmetal_binary_paths = BinaryTTMetal.find_ttmetal_binary_paths(self["binary"])

        def check_constraints(self):
            for path in ttnn_binary_paths:
                bin = BinaryTTNN.prepare_binary(path)
                if bin.check_version():
                    self.ttnn_binaries.append(bin)

            for path in ttmetal_binary_paths:
                bin = BinaryTTMetal.prepare_binary(path)
                if bin.check_version():
                    self.ttmetal_binaries.append(bin)

        def execute(self):
            for binary in self.ttnn_binaries
                API.read[self["section"]](binary)

            for binary in self.ttmetal_binaries
                API.read[self["section"]](binary)

        def postprocess(self):
            if self["save_artifacts"]:
                for binary in self.ttnn_binaries:
                    self.artifacts.save_binary(binary)

                for binary in self.ttmetal_binaries:
                    self.artifacts.save_binary(binary)

        def __str__(self):
            pass

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key):
            setattr(self, key, value)

        def __call__(self):
            self.preprocess()
            self.check_constraints()
            self.execute()
            self.postprocess()

        @staticmethod
        def register_arg(name, type, default, choices, help):
            API.read.registered_args[name] = {"type": type, "default": default, "choices": choices, "help": help}

        @staticmethod
        def generate_subparser(subparsers):
            read_parser = subparsers.add_parser(
                "read", help="read information from flatbuffer binary"
            )
            read_parser.set_defaults(func=API.read)
            read_parser.add_argument("binary", help="flatbuffer binary file")

            for name, attributes in API.read.registered_args.items():
                read_parser.add_argument(
                    f"--{name}",
                    type=attributes["type"],
                    default=attributes["default"],
                    choices=choices["choices"]
                    help=attributes["help"],
                )
            
            return read_parser

        @staticmethod
        def all(Binary binary):
            return logging.info(binary.fbb.as_json())

        @staticmethod
        def version(Binary binary):
            return logging.info(f"version: {binary.fbb.version}\ntt-mlir git hash: {binary.fbb.ttmlir_git_hash}")

        @staticmethod
        def system_desc(Binary binary):
            import ttrt.binary
            bin_dict = ttrt.binary.as_dict(binary.fbb)
            return logging.info(json.dumps(bin_dict["system_desc"], indent=2))

        @staticmethod
        def mlir(Binary binary):
            import ttrt.binary
            bin_dict = ttrt.binary.as_dict(binary.fbb)
            
            for i, program in enumerate(d["programs"]):
                if "debug_info" not in program:
                    logging.info("// no debug info found for program:", program["name"])
                    continue
                logging.info(
                    f"// program[{i}]:",
                    program["name"],
                    "-",
                    program["debug_info"]["mlir"]["name"],
                )
                logging.info(program["debug_info"]["mlir"]["source"], end="")

        @staticmethod
        def cpp(Binary binary):
            import ttrt.binary
            bin_dict = ttrt.binary.as_dict(binary.fbb)
            
            for i, program in enumerate(d["programs"]):
                if "debug_info" not in program:
                    logging.info("// no debug info found for program:", program["name"])
                    continue
                logging.info(f"// program[{i}]:", program["name"])
                logging.info(program["debug_info"]["cpp"], end="")

        @staticmethod
        def inputs(Binary binary):
            import ttrt.binary
            bin_dict = ttrt.binary.as_dict(binary.fbb)

            for program in bin_dict["programs"]:
                logging.info("program:", program["name"])
                logging.info(json.dumps(program["inputs"], indent=2))

        @staticmethod
        def outputs(Binary binary):
          import ttrt.binary
            bin_dict = ttrt.binary.as_dict(binary.fbb)
            
            for program in bin_dict["programs"]:
                logging.info("program:", program["name"])
                logging.info(json.dumps(program["outputs"], indent=2))

            return outputs_string

    class run:
        registered_args = {}

        def __init__(self, logging):
            self.logging = logging
            self.artifacts = Artifacts.prepare_artifact()
            self.ttnn_binaries = None
            self.ttmetal_binaries = None
            self.query = API.Query()
            
            for name, _ in API.run.registered_args.items():
                self[name] = args[name]
          
        def preprocess(self):
            self.query()

            if self["clean_artifacts"]:
                self.artifacts.clean_artifacts()

            if self["save_artifacts"]:
                self.artifacts.create_artifacts()

            ttnn_binary_paths = BinaryTTNN.find_ttnn_binary_paths(self["binary"])
            ttmetal_binary_paths = BinaryTTMetal.find_ttmetal_binary_paths(self["binary"])

        def check_constraints(self):
            for path in ttnn_binary_paths:
                bin = BinaryTTNN.prepare_binary(path)
                if not bin.check_version():
                    continue

                if not bin.check_system_desc(self.query)
                    continue

                if self["program_index"] != "all":
                    if not bin.check_program_index_exists(int(self["program_index"])):
                        continue
                
                self.ttnn_binaries.append(bin)

            for path in ttmetal_binary_paths:
                bin = BinaryTTMetal.prepare_binary(path)
                if not bin.check_version():
                    continue

                if not bin.check_system_desc(self.query)
                    continue

                if self["program_index"] != "all":
                    if not bin.check_program_index_exists(int(self["program_index"])):
                        continue

                self.ttmetal_binaries.append(bin)

        def execute(self):
            def _execute(binaries):
                ttrt.runtime.set_compatible_runtime(self.binaries.fbb)
                device = ttrt.runtime.open_device([self.query.device_ids[0]])
                atexit.register(lambda: ttrt.runtime.close_device(device))

                for bin in binaries:
                    program_indices = []
                    if self["program_index"] == "all":
                        program_indices.extend(range(bin.get_num_programs()))
                    else:
                        program_indices.append(int(self["program_index"]))

                    for program_index in program_indices:
                        program = bin.get_program(program_index)
                        program.populate_inputs(API.run.TorchInitilizer.get_initilizer(self[init]), program_index)
                        program.populate_outputs(API.run.TorchInitilizer.get_initilizer("zeros"), program_index)

                        total_inputs = []
                        total_outputs = []
                        for loop in range(self["loops"]):
                            inputs = []
                            outputs = []
                            for i in program.get_inputs():
                                inputs.append(
                                    ttrt.runtime.create_tensor(
                                        i.data_ptr(),
                                        list(i.shape),
                                        list(i.stride()),
                                        i.element_size(),
                                        toDataType(i.dtype),
                                    )
                                )

                            for i in program.get_outputs():
                                outputs.append(
                                    ttrt.runtime.create_tensor(
                                        i.data_ptr(),
                                        list(i.shape),
                                        list(i.stride()),
                                        i.element_size(),
                                        toDataType(i.dtype),
                                    )
                                )

                            total_inputs.append(inputs)
                            total_outputs.append(outputs)

                        event = None
                        for loop in range(self["loops"]):
                            event = ttrt.runtime.submit(device, self.bin.fbb, program_index, total_inputs[loop], total_outputs[loop])
                        ttrt.runtime.wait(event)

                        if self["identity"]:
                            for i, o in zip(program.get_inputs(), program.get_outputs()):
                                if not torch.allclose(i, o):
                                    logging.error(f"Failed: inputs and outputs do not match in binary")
                                    logging.error(i - o)

            import ttrt.runtime
            import torch

            torch.manual_seed(self[seed])

            if len(self.ttnn_binaries) != 0:
                _execute(self.ttnn_binaries)
            
            if len(self.ttmetal_binaries) != 0:
                _execute(self.ttnn_binaries)

        def postprocess(self):
            if self["save_artifacts"]:
                for bin in self.ttnn_binaries:
                    bin.save(self.artifacts, self.query)

                for bin in self.ttmetal_binaries:
                    bin.save(self.artifacts, self.query)

        def __str__(self):
            pass

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key):
            setattr(self, key, value)

        def __call__(self):
            self.preprocess()
            self.check_constraints()
            self.execute()
            self.postprocess()

        @staticmethod
        def toDataType(dtype):
            import torch
            import ttrt.runtime

            if dtype == torch.float32:
                return ttrt.runtime.DataType.Float32
            if dtype == torch.float16:
                return ttrt.runtime.DataType.Float16
            if dtype == torch.bfloat16:
                return ttrt.runtime.DataType.BFloat16
            if dtype == torch.uint32:
                return ttrt.runtime.DataType.UInt32
            if dtype == torch.uint16:
                return ttrt.runtime.DataType.UInt16
            if dtype == torch.uint8:
                return ttrt.runtime.DataType.UInt8
            raise ValueError(f"unsupported dtype: {dtype}")

        @staticmethod
        def fromDataType(dtype):
            import torch

            if dtype == "Float32":
                return torch.float32
            if dtype == "Float16":
                return torch.float16
            if dtype == "BFloat16":
                return torch.bfloat16
            if dtype == "UInt32":
                return torch.uint32
            if dtype == "UInt16":
                return torch.uint16
            if dtype == "UInt8":
                return torch.uint8
            raise ValueError(f"unsupported dtype: {dtype}")

        @staticmethod
        def register_arg(name, type, default, choices, help):
            API.run.registered_args[name] = {"type": type, "default": default, "choices": choices, "help": help}

        @staticmethod
        def generate_subparser(subparsers):
            run_parser = subparsers.add_parser(
                "run", help="run a flatbuffer binary"
            )
            run_parser.set_defaults(func=API.run)

            for name, attributes in API.run.registered_args.items():
                run_parser.add_argument(
                    f"--{name}",
                    type=attributes["type"],
                    default=attributes["default"],
                    choices=choices["choices"]
                    help=attributes["help"],
                )

            return run_parser

        class TorchInitilizer():
            init_fns_map = {
                "randn": TorchInitilizer.randn,
                "arange": TorchInitilizer.arange,
                "zeros": TorchInitilizer.zeros,
            }
            init_fns = sorted(list(init_fns_map.keys()))

            @staticmethod
            def get_initilizer(name):
                return TorchInitilizer.init_fns_map[name]
            
            @staticmethod
            def get_init_fns():
                return TorchInitilizer.init_fns

            @staticmethod
            def randn(shape, dtype):
                import torch
                return torch.randn(shape, dtype=dtype)

            @staticmethod
            def arange(shape, dtype):
                import torch
                def volume(shape):
                    v = 1
                    for i in shape:
                        v *= i
                    return v

                return torch.arange(volume(shape), dtype=dtype).reshape(shape)

            @staticmethod
            def zeros(shape, dtype):
                import torch
                return torch.zeros(shape, dtype=dtype)

    class perf:
      def __init__(self, logging):
            pass
          
        def preprocess(self):
            pass

        def check_constraints(self):
            pass

        def execute(self):
            pass

        def postprocess(self):
            pass

        def __str__(self):
            pass

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key):
            setattr(self, key, value)

        def __call__(self):
            self.preprocess()
            self.check_constraints()
            self.execute()
            self.postprocess()

        @staticmethod
        def register_arg():
            pass

        @staticmethod
        def generate_subparser():
            pass


"""
API: perf
  - run flatbuffer on device in performance mode
"""


def perf(args, logger):
    import ttrt.common.perf_trace as perf_trace

    # initialization
    binaries = []

    # acquire parameters
    arg_binary = args.binary
    arg_program_index = args.program_index
    arg_clean_artifacts = args.clean_artifacts
    arg_perf_csv = args.perf_csv
    arg_loops = int(args.loops)
    arg_save_artifacts = args.save_artifacts
    arg_device = args.device
    arg_generate_params = args.generate_params

    # preprocessing
    if os.path.isdir(arg_binary):
        print("provided directory of flatbuffers")
        binaries = find_ttnn_files(arg_binary)
    else:
        binaries.append(arg_binary)

    if arg_clean_artifacts:
        print("cleaning artifacts")
        clean_artifacts()

    if arg_save_artifacts:
        print("setting up artifact directories")
        setup_artifacts(binaries)

    # constraint checking
    if arg_generate_params:
        check_file_exists(arg_perf_csv)

    for binary in binaries:
        check_file_exists(binary)

    # execution
    if arg_generate_params:
        perf_trace.generate_params_dict(arg_perf_csv)
    else:
        for binary in binaries:
            # get available port for tracy client and server to communicate on
            port = perf_trace.get_available_port()

            if not port:
                print("No available port found")
                sys.exit(1)
            print(f"Using port {port}")

            # setup environment flags
            envVars = dict(os.environ)
            envVars["TRACY_PORT"] = port
            envVars["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"

            if arg_device:
                envVars["TT_METAL_DEVICE_PROFILER"] = "1"

            # run perf artifact setup
            captureProcess = perf_trace.run_perf_artifact_setup(port)

            # generate test command to execute
            testCommandOptions = ""
            if arg_save_artifacts:
                testCommandOptions += f"--save-artifacts "

            testCommand = f"python -m tracy -p {TTMLIR_VENV_DIR}/bin/ttrt run {binary} --loops {arg_loops} --program-index {arg_program_index} {testCommandOptions}"
            testProcess = subprocess.Popen(
                [testCommand], shell=True, env=envVars, preexec_fn=os.setsid
            )
            print(f"Test process started")

            # setup multiprocess signal handler
            def signal_handler(sig, frame):
                os.killpg(os.getpgid(testProcess.pid), signal.SIGTERM)
                captureProcess.terminate()
                captureProcess.communicate()
                sys.exit(3)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            testProcess.communicate()

            binary_name = os.path.splitext(os.path.basename(binary))[0]
            binary_perf_folder = f"{TTRT_ARTIFACTS}/{binary_name}/perf"

            try:
                captureProcess.communicate(timeout=15)
                perf_trace.generate_report(binary_perf_folder)
            except subprocess.TimeoutExpired as e:
                captureProcess.terminate()
                captureProcess.communicate()
                print(
                    f"No profiling data could be captured. Please make sure you are on the correct build. Use scripts/build_scripts/build_with_profiler_opt.sh to build if you are not sure."
                )
                sys.exit(1)

            # save artifacts
            perf_trace.save_perf_artifacts(binary_perf_folder)
