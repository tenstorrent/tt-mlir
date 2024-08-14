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

from ttrt.common.util import *

#######################################################################################
########################################**API**########################################
#######################################################################################
class API:
    registered_apis = {}
    registered_functions = ["__init__", "preprocess", "check_constraints", "execute", "postprocess", "__str__", "__getitem__", "__setitem__", "__call__", "register_arg", "generate_subparser"]

    @staticmethod
    def initialize_apis():
        # name, type, default, choices, help
        # register all query arguments
        API.Query.register_arg(name="--clean-artifacts", type=bool, default=False, choices=[True, False], help="clean all artifacts from previous runs")
        API.Query.register_arg(name="--save-artifacts", type=bool, default=False, choices=[True, False], help="save all artifacts during run")

        # register all read arguments
        API.Read.register_arg(name="--section", type=str, default="all", choices=sorted(API.Read.read_actions), help="output sections of the fb")
        API.Read.register_arg(name="--clean-artifacts", type=bool, default=False, choices=[True, False], help="clean all artifacts from previous runs")
        API.Read.register_arg(name="--save-artifacts", type=bool, default=False, choices=[True, False], help="save all artifacts during run")

        # register all run arguments
        API.Run.register_arg(name="--clean-artifacts", type=bool, default=False, choices=[True, False], help="clean all artifacts from previous runs", api_only=False)
        API.Run.register_arg(name="--save-artifacts", type=bool, default=False, choices=[False, False], help="save all artifacts during run", api_only=False)
        API.Run.register_arg(name="--program-index", type=str, default="all", choices=["all"] + [str(i) for i in range(1, 101)], help="the program inside the fbb to run", api_only=False)
        API.Run.register_arg(name="--loops", type=int, default=1, choices=None, help="number of loops", api_only=False)
        API.Run.register_arg(name="--init", type=str, default="randn", choices=API.Run.TorchInitilizer.init_fns, help="function to initialize tensors with", api_only=False)
        API.Run.register_arg(name="--identity", type=bool, default=False, choices=[False, False], help="do a golden identity test on the output tensors", api_only=False)
        API.Run.register_arg(name="--rtol", type=float, default=1e-05, choices=None, help="rtol for golden test", api_only=False)
        API.Run.register_arg(name="--atol", type=float, default=1e-08, choices=None, help="atol for golden test", api_only=False)
        API.Run.register_arg(name="--seed", type=int, default=0, choices=None, help="seed for random number generator", api_only=False)

        # register all perf arguments
        API.Perf.register_arg(name="--device", type=bool, default=False, choices=[False, False], help="collect performance trace on both host and device")
        up_stream_apis = API.Run.get_upstream_apis()
        for api in up_stream_apis:
            API.Perf.register_arg(name=api["name"], type=api["type"], default=api["default"], choices=api["choices"], help=api["help"])

        # register apis
        API.register_api(API.Query)
        API.register_api(API.Read)
        API.register_api(API.Run)
        API.register_api(API.Perf)

    @staticmethod
    def register_api(api_class):
        missing_methods = [
            func for func in API.registered_functions 
            if not hasattr(api_class, func)
        ]

        if missing_methods:
            raise TypeError(f"API class={api_class.__name__} is missing methods={missing_methods}")

        API.registered_apis[api_class.__name__] = api_class

    class Query:
        registered_args = {}
        api_only_arg = []

        def __init__(self, args, logging=None, globals=None, file_manager=None, artifacts=None):
            self.logger = logging if logging != None else Logger("ttrt.log")
            self.logging = self.logger.get_logger()
            self.globals = globals if globals != None else Globals(self.logging)
            self.file_manager = file_manager if file_manager != None else FileManager(self.logging)
            self.artifacts = artifacts if artifacts != None else Artifacts(self.logging, self.file_manager)
            self.system_desc = None
            self.device_ids = None
            
            for name, _ in API.Query.registered_args.items():
                name = name if not name.startswith('-') else name.lstrip('-')
                name = name.replace('-', '_')
                self[name] = getattr(args, name)
          
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
                self.logging.info(self.get_system_desc_as_dict())
            except Exception as e:
                raise Exception(f"an unexpected error occurred: {e}")

        def postprocess(self):
            if self["save_artifacts"]:
                self.artifacts.save_system_desc(self.system_desc)

        def __str__(self):
            pass 

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, value):
            setattr(self, key, value)

        def __call__(self):
            self.preprocess()
            self.check_constraints()
            self.execute()
            self.postprocess()
        
        def get_system_desc_as_dict(self):
            return json.loads(self.system_desc.as_json())

        @staticmethod
        def register_arg(name, type, default, choices, help, api_only=True):
            API.Query.registered_args[name] = {"type": type, "default": default, "choices": choices, "help": help}

            if api_only:
                API.Query.api_only_arg.append(name)

        @staticmethod
        def get_upstream_apis():
            upstream_apis = []
            for arg_name, arg_value in API.Query.registered_args.items():
                if arg_name not in API.Query.api_only_arg:
                    upstream_apis.append({"name": arg_name, "type": arg_value["type"], "default": arg_value["default"], "choices": arg_value["choices"], "help": arg_value["help"]})

            return upstream_apis

        @staticmethod
        def generate_subparser(subparsers):
            query_parser = subparsers.add_parser(
                "query", help="query information about the current system"
            )
            query_parser.set_defaults(api=API.Query)

            for name, attributes in API.Query.registered_args.items():
                if attributes["type"] == bool:
                    query_parser.add_argument(
                        f"{name}",
                        action="store_true", 
                        help=attributes["help"],
                    )
                else:
                    query_parser.add_argument(
                        f"{name}",
                        type=attributes["type"],
                        default=attributes["default"],
                        choices=attributes["choices"],
                        help=attributes["help"],
                    )

            return query_parser

    class Read:
        registered_args = {}
        api_only_arg = []
        read_actions = ["all", "version", "system-desc", "mlir", "cpp", "inputs", "outputs"]
        
        def __init__(self, args, logging=None, globals=None, file_manager=None, artifacts=None):
            self.logger = logging if logging != None else Logger("ttrt.log")
            self.logging = self.logger.get_logger()
            self.globals = globals if globals != None else Globals(self.logging)
            self.file_manager = file_manager if file_manager != None else FileManager(self.logging)
            self.artifacts = artifacts if artifacts != None else Artifacts(self.logging, self.file_manager)
            self.ttnn_binaries = None
            self.ttmetal_binaries = None
            
            for name, _ in API.Read.registered_args.items():
                name = name if not name.startswith('-') else name.lstrip('-')
                name = name.replace('-', '_')
                self[name] = getattr(args, name)
          
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
            for binary in self.ttnn_binaries:
                API.Read[self["section"]](binary)

            for binary in self.ttmetal_binaries:
                API.Read[self["section"]](binary)

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

        def __setitem__(self, key, value):
            setattr(self, key, value)

        def __call__(self):
            self.preprocess()
            self.check_constraints()
            self.execute()
            self.postprocess()

        @staticmethod
        def register_arg(name, type, default, choices, help, api_only=True):
            API.Read.registered_args[name] = {"type": type, "default": default, "choices": choices, "help": help}

            if api_only:
                API.Read.api_only_arg.append(name)

        @staticmethod
        def get_upstream_apis():
            upstream_apis = []
            for arg_name, arg_value in API.Read.registered_args.items():
                if arg_name not in API.Read.api_only_arg:
                    upstream_apis.append({"name": arg_name, "type": arg_value["type"], "default": arg_value["default"], "choices": arg_value["choices"], "help": arg_value["help"]})

            return upstream_apis

        @staticmethod
        def generate_subparser(subparsers):
            read_parser = subparsers.add_parser(
                "read", help="read information from flatbuffer binary"
            )
            read_parser.set_defaults(api=API.Read)
            read_parser.add_argument("binary", help="flatbuffer binary file")

            for name, attributes in API.Read.registered_args.items():
                if attributes["type"] == bool:
                    read_parser.add_argument(
                        f"{name}",
                        action="store_true", 
                        help=attributes["help"],
                    )
                else:
                    read_parser.add_argument(
                        f"{name}",
                        type=attributes["type"],
                        default=attributes["default"],
                        choices=attributes["choices"],
                        help=attributes["help"],
                    )
            
            return read_parser

        @staticmethod
        def all(binary):
            return self.logging.info(binary.fbb.as_json())

        @staticmethod
        def version(binary):
            return self.logging.info(f"version: {binary.fbb.version}\ntt-mlir git hash: {binary.fbb.ttmlir_git_hash}")

        @staticmethod
        def system_desc(binary):
            import ttrt.binary
            bin_dict = ttrt.binary.as_dict(binary.fbb)
            return self.logging.info(json.dumps(bin_dict["system_desc"], indent=2))

        @staticmethod
        def mlir(binary):
            import ttrt.binary
            bin_dict = ttrt.binary.as_dict(binary.fbb)
            
            for i, program in enumerate(d["programs"]):
                if "debug_info" not in program:
                    self.logging.info("// no debug info found for program:", program["name"])
                    continue
                self.logging.info(
                    f"// program[{i}]:",
                    program["name"],
                    "-",
                    program["debug_info"]["mlir"]["name"],
                )
                self.logging.info(program["debug_info"]["mlir"]["source"], end="")

        @staticmethod
        def cpp(binary):
            import ttrt.binary
            bin_dict = ttrt.binary.as_dict(binary.fbb)
            
            for i, program in enumerate(d["programs"]):
                if "debug_info" not in program:
                    self.logging.info("// no debug info found for program:", program["name"])
                    continue
                self.logging.info(f"// program[{i}]:", program["name"])
                self.logging.info(program["debug_info"]["cpp"], end="")

        @staticmethod
        def inputs(binary):
            import ttrt.binary
            bin_dict = ttrt.binary.as_dict(binary.fbb)

            for program in bin_dict["programs"]:
                self.logging.info("program:", program["name"])
                self.logging.info(json.dumps(program["inputs"], indent=2))

        @staticmethod
        def outputs(binary):
          import ttrt.binary
          bin_dict = ttrt.binary.as_dict(binary.fbb)
          
          for program in bin_dict["programs"]:
              self.logging.info("program:", program["name"])
              self.logging.info(json.dumps(program["outputs"], indent=2))

          return outputs_string

    class Run:
        registered_args = {}
        api_only_arg = []

        def __init__(self, args, logging=None, globals=None, file_manager=None, artifacts=None):
            self.logger = logging if logging != None else Logger("ttrt.log")
            self.logging = self.logger.get_logger()
            self.globals = globals if globals != None else Globals(self.logging)
            self.file_manager = file_manager if file_manager != None else FileManager(self.logging)
            self.artifacts = artifacts if artifacts != None else Artifacts(self.logging, self.file_manager)
            self.query = API.Query()
            self.ttnn_binaries = None
            self.ttmetal_binaries = None
            
            for name, _ in API.Run.registered_args.items():
                name = name if not name.startswith('-') else name.lstrip('-')
                name = name.replace('-', '_')
                self[name] = getattr(args, name)
          
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

                if not bin.check_system_desc(self.query):
                    continue

                if self["program_index"] != "all":
                    if not bin.check_program_index_exists(int(self["program_index"])):
                        continue
                
                self.ttnn_binaries.append(bin)

            for path in ttmetal_binary_paths:
                bin = BinaryTTMetal.prepare_binary(path)
                if not bin.check_version():
                    continue

                if not bin.check_system_desc(self.query):
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
                        program.populate_inputs(API.Run.TorchInitilizer.get_initilizer(self[init]), program_index)
                        program.populate_outputs(API.Run.TorchInitilizer.get_initilizer("zeros"), program_index)

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
                                    self.logging.error(f"Failed: inputs and outputs do not match in binary")
                                    self.logging.error(i - o)

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

        def __setitem__(self, key, value):
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
        def register_arg(name, type, default, choices, help, api_only=True):
            API.Run.registered_args[name] = {"type": type, "default": default, "choices": choices, "help": help}

            if api_only:
                API.Run.api_only_arg.append(name)

        @staticmethod
        def get_upstream_apis():
            upstream_apis = []
            for arg_name, arg_value in API.Run.registered_args.items():
                if arg_name not in API.Run.api_only_arg:
                    upstream_apis.append({"name": arg_name, "type": arg_value["type"], "default": arg_value["default"], "choices": arg_value["choices"], "help": arg_value["help"]})

            return upstream_apis

        @staticmethod
        def generate_subparser(subparsers):
            run_parser = subparsers.add_parser(
                "run", help="run a flatbuffer binary"
            )
            run_parser.set_defaults(api=API.Run)

            for name, attributes in API.Run.registered_args.items():
                if attributes["type"] == bool:
                    run_parser.add_argument(
                      f"{name}",
                        action="store_true", 
                        help=attributes["help"],
                    )
                else:
                    run_parser.add_argument(
                        f"{name}",
                        type=attributes["type"],
                        default=attributes["default"],
                        choices=attributes["choices"],
                        help=attributes["help"],
                    )

            return run_parser

        class TorchInitilizer():
            init_fns = sorted(["randn","arange","zeros"])

            @staticmethod
            def get_initilizer(name):
                return TorchInitilizer[name]
            
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

    class Perf:
        registered_args = {}
        api_only_arg = []

        def __init__(self, args, logging=None, globals=None, file_manager=None, artifacts=None):
            self.logger = logging if logging != None else Logger("ttrt.log")
            self.logging = self.logger.get_logger()
            self.globals = globals if globals != None else Globals(self.logging)
            self.file_manager = file_manager if file_manager != None else FileManager(self.logging)
            self.artifacts = artifacts if artifacts != None else Artifacts(self.logging, self.file_manager)
            self.query = API.Query()
            self.ttnn_binaries = None
            self.ttmetal_binaries = None
            self.tracy_capture_tool_path = f"{artifacts.get_ttmlir_home_path()}/third_party/tt-metal/src/tt-metal-build/tools/profiler/bin/capture-release"
            self.tracy_csvexport_tool_path = f"{artifacts.get_ttmlir_home_path()}/third_party/tt-metal/src/tt-metal-build/tools/profiler/bin/csvexport-release"
            self.tracy_capture_tool_process = None

            for name, _ in API.Run.registered_args.items():
                name = name if not name.startswith('-') else name.lstrip('-')
                name = name.replace('-', '_')
                self[name] = getattr(args, name)
          
        def preprocess(self):
            self.query()

            if self["clean_artifacts"]:
                self.artifacts.clean_artifacts()

            if self["save_artifacts"]:
                self.artifacts.create_artifacts()

            ttnn_binary_paths = BinaryTTNN.find_ttnn_binary_paths(self["binary"])
            ttmetal_binary_paths = BinaryTTMetal.find_ttmetal_binary_paths(self["binary"])

            FileManager.remove_directory(PROFILER_LOGS_DIR)
            FileManager.create_directory(PROFILER_LOGS_DIR)

        def check_constraints(self):
            for path in ttnn_binary_paths:
                bin = BinaryTTNN.prepare_binary(path)
                if not bin.check_version():
                    continue

                if not bin.check_system_desc(self.query):
                    continue

                if self["program_index"] != "all":
                    if not bin.check_program_index_exists(int(self["program_index"])):
                        continue
                
                self.ttnn_binaries.append(bin)

            for path in ttmetal_binary_paths:
                bin = BinaryTTMetal.prepare_binary(path)
                if not bin.check_version():
                    continue

                if not bin.check_system_desc(self.query):
                    continue

                if self["program_index"] != "all":
                    if not bin.check_program_index_exists(int(self["program_index"])):
                        continue

                self.ttmetal_binaries.append(bin)

            FileManager.check_file_exists(self.tracy_capture_tool_path)
            FileManager.check_file_exists(self.tracy_csvexport_tool_path)

        def execute(self):
            import tracy
            import tracy_state
            from tt_metal.tools.profiler.process_ops_logs import process_ops
            from tt_metal.tools.profiler.common import PROFILER_LOGS_DIR, TRACY_OPS_TIMES_FILE_NAME, TRACY_OPS_DATA_FILE_NAME, TRACY_FILE_NAME

            def get_available_port():
                ip = socket.gethostbyname(socket.gethostname())

                for port in range(8086, 8500):
                    try:
                        serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        serv.bind((ip, port))
                        return str(port)
                    except PermissionError as e:
                        pass
                    except OSError as e:
                        pass
                return None

            def generate_report(binary_perf_folder):
                child_calls = ["CompileProgram", "HWCommandQueue_write_buffer"]
                time_out = 15
                time_count = 0
                while not os.path.exists(PROFILER_LOGS_DIR / TRACY_FILE_NAME):
                    if time_count > time_out:
                        raise Exception(f"tracy capture output file {PROFILER_LOGS_DIR / TRACY_FILE_NAME} was not generated")
                    time_count += 1
                    time.sleep(1)

                with open(PROFILER_LOGS_DIR / TRACY_OPS_TIMES_FILE_NAME, "w") as csv_file:
                    child_call_str = f"-x {','.join(child_calls)}"
                    subprocess.run(
                        f"{self.tracy_csvexport_tool_path} -u -p TT_DNN {child_call_str} {PROFILER_LOGS_DIR / TRACY_FILE_NAME}",
                        shell=True,
                        check=True,
                        stdout=csv_file,
                        stderr=subprocess.DEVNULL,
                    )

                with open(PROFILER_LOGS_DIR / TRACY_OPS_DATA_FILE_NAME, "w") as csv_file:
                    subprocess.run(
                        f'{self.tracy_csvexport_tool_path} -m -s ";" {PROFILER_LOGS_DIR / TRACY_FILE_NAME}',
                        shell=True,
                        check=True,
                        stdout=csv_file,
                        stderr=subprocess.DEVNULL,
                    )
                process_ops(binary_perf_folder, "", True)

            def save_perf_artifacts(binary_perf_folder):
                tracy_ops_times_file = (f"{self.artifacts.get_ttmetal_home_path()}/generated/profiler/.logs/{TRACY_OPS_TIMES_FILE_NAME}")
                tracy_ops_data_file = (f"{self.artifacts.get_ttmetal_home_path()}/generated/profiler/.logs/{TRACY_OPS_DATA_FILE_NAME}")
                tracy_file = f"{self.artifacts.get_ttmetal_home_path()}/generated/profiler/.logs/{TRACY_FILE_NAME}"

                try:
                    FileManager.check_file_exists(tracy_ops_times_file)
                    FileManager.check_file_exists(tracy_ops_data_file)
                    FileManager.check_file_exists(tracy_file)
                except Exception as e:
                    raise Exception(f"an unexpected error occurred: {e}")

                for root, dirs, files in os.walk(binary_perf_folder, topdown=False):
                    if root == binary_perf_folder:
                        continue

                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        dest_path = os.path.join(binary_perf_folder, file_name)
                        FileManager.copy_file(dest_path, file_path)

                    if not os.listdir(root):
                        FileManager.remove_directory(root)

            def _execute(binaries):
                for bin in binaries:
                    port = get_available_port()

                    if not port:
                        raise Exception("No available port found")

                    env_vars = dict(os.environ)
                    env_vars["TRACY_PORT"] = port
                    env_vars["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"

                    if self["device"]:
                        env_vars["TT_METAL_DEVICE_PROFILER"] = "1"

                    tracy_capture_tool_command = (f"{self.tracy_capture_tool_path} -o {PROFILER_LOGS_DIR / TRACY_FILE_NAME} -f -p {port}")
                    self.tracy_capture_tool_process = subprocess.Popen(tracy_capture_tool_command, shell=True)

                    test_command = f"python -m tracy -p {artifacts.get_ttmlir_venv_path()}/bin/ttrt run {bin.file_path} --save-artifacts"
                    testProcess = subprocess.Popen(
                        [test_command], shell=True, env=env_vars, preexec_fn=os.setsid
                    )

                    def signal_handler(sig, frame):
                        os.killpg(os.getpgid(testProcess.pid), signal.SIGTERM)
                        captureProcess.terminate()
                        captureProcess.communicate()
                        sys.exit(3)

                    signal.signal(signal.SIGINT, signal_handler)
                    signal.signal(signal.SIGTERM, signal_handler)
                    testProcess.communicate()

                    try:
                        captureProcess.communicate(timeout=15)
                        perf_trace.generate_report(self.artifacts.get_binary_perf_folder_path(bin))
                    except subprocess.TimeoutExpired as e:
                        captureProcess.terminate()
                        captureProcess.communicate()
                        raise Exception(f"No profiling data could be captured. Please make sure you are on the correct build")
            
                    save_perf_artifacts(self.artifacts.get_binary_perf_folder_path(bin))

            if len(self.ttnn_binaries) != 0:
                _execute(self.ttnn_binaries)
            
            if len(self.ttmetal_binaries) != 0:
                _execute(self.ttnn_binaries)

        def postprocess(self):
            pass

        def __str__(self):
            pass

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, value):
            setattr(self, key, value)

        def __call__(self):
            self.preprocess()
            self.check_constraints()
            self.execute()
            self.postprocess()

        @staticmethod
        def register_arg(name, type, default, choices, help, api_only=True):
            API.Perf.registered_args[name] = {"type": type, "default": default, "choices": choices, "help": help}

            if api_only:
                API.Perf.api_only_arg.append(name)

        @staticmethod
        def get_upstream_apis():
            upstream_apis = []
            for arg_name, arg_value in API.Perf.registered_args.items():
                if arg_name not in API.Perf.api_only_arg:
                    upstream_apis.append({"name": arg_name, "type": arg_value["type"], "default": arg_value["default"], "choices": arg_value["choices"], "help": arg_value["help"]})

            return upstream_apis

        @staticmethod
        def generate_subparser(subparsers):
            perf_parser = subparsers.add_parser(
                "run", help="run performance trace and collect performance data"
            )
            perf_parser.set_defaults(api=API.Perf)

            for name, attributes in API.Perf.registered_args.items():
                if attributes["type"] == bool:
                    perf_parser.add_argument(
                          f"{name}",
                        action="store_true", 
                        help=attributes["help"],
                    )
                else:
                    perf_parser.add_argument(
                        f"{name}",
                        type=attributes["type"],
                        default=attributes["default"],
                        choices=attributes["choices"],
                        help=attributes["help"],
                    )

            return perf_parser
