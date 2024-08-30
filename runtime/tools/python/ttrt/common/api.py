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
import shutil
import atexit

from ttrt.common.util import *


class API:
    registered_apis = {}
    registered_functions = [
        "__init__",
        "preprocess",
        "check_constraints",
        "execute",
        "postprocess",
        "__str__",
        "__getitem__",
        "__setitem__",
        "__call__",
        "register_arg",
        "generate_subparser",
    ]

    @staticmethod
    def initialize_apis():
        # register all query arguments
        API.Query.register_arg(
            name="--clean-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="clean all artifacts from previous runs",
        )
        API.Query.register_arg(
            name="--save-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="save all artifacts during run",
        )
        API.Query.register_arg(
            name="--log-file",
            type=str,
            default="",
            choices=None,
            help="log file to dump ttrt output to",
        )

        # register all read arguments
        API.Read.register_arg(
            name="--section",
            type=str,
            default="all",
            choices=sorted(API.Read.read_actions),
            help="output sections of the fb",
        )
        API.Read.register_arg(
            name="--clean-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="clean all artifacts from previous runs",
        )
        API.Read.register_arg(
            name="--save-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="save all artifacts during run",
        )
        API.Read.register_arg(
            name="binary",
            type=str,
            default="",
            choices=None,
            help="flatbuffer binary file",
        )
        API.Read.register_arg(
            name="--log-file",
            type=str,
            default="",
            choices=None,
            help="log file to dump ttrt output to",
        )

        # register all run arguments
        API.Run.register_arg(
            name="--clean-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="clean all artifacts from previous runs",
        )
        API.Run.register_arg(
            name="--save-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="save all artifacts during run",
        )
        API.Run.register_arg(
            name="--program-index",
            type=str,
            default="all",
            choices=["all"] + [str(i) for i in range(0, 5)],
            help="the program inside the fbb to run",
            api_only=False,
        )
        API.Run.register_arg(
            name="--loops",
            type=int,
            default=1,
            choices=None,
            help="number of loops",
            api_only=False,
        )
        API.Run.register_arg(
            name="--init",
            type=str,
            default="randn",
            choices=API.Run.TorchInitilizer.init_fns,
            help="function to initialize tensors with",
            api_only=False,
        )
        API.Run.register_arg(
            name="--identity",
            type=bool,
            default=False,
            choices=[True, False],
            help="do a golden identity test on the output tensors",
            api_only=False,
        )
        API.Run.register_arg(
            name="--rtol",
            type=float,
            default=1e-05,
            choices=None,
            help="rtol for golden test",
            api_only=False,
        )
        API.Run.register_arg(
            name="--atol",
            type=float,
            default=1e-08,
            choices=None,
            help="atol for golden test",
            api_only=False,
        )
        API.Run.register_arg(
            name="--seed",
            type=int,
            default=0,
            choices=None,
            help="seed for random number generator",
            api_only=False,
        )
        API.Run.register_arg(
            name="binary",
            type=str,
            default="",
            choices=None,
            help="flatbuffer binary file",
        )
        API.Run.register_arg(
            name="--log-file",
            type=str,
            default="",
            choices=None,
            help="log file to dump ttrt output to",
        )

        # register all perf arguments
        API.Perf.register_arg(
            name="--clean-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="clean all artifacts from previous runs",
        )
        API.Perf.register_arg(
            name="--save-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="save all artifacts during run",
        )
        API.Perf.register_arg(
            name="--device",
            type=bool,
            default=False,
            choices=[True, False],
            help="collect performance trace on both host and device",
        )
        API.Perf.register_arg(
            name="binary",
            type=str,
            default="",
            choices=None,
            help="flatbuffer binary file",
        )
        API.Perf.register_arg(
            name="--log-file",
            type=str,
            default="",
            choices=None,
            help="log file to dump ttrt output to",
        )
        up_stream_apis = API.Run.get_upstream_apis()
        for api in up_stream_apis:
            API.Perf.register_arg(
                name=api["name"],
                type=api["type"],
                default=api["default"],
                choices=api["choices"],
                help=api["help"],
            )

        # register all check arguments
        API.Check.register_arg(
            name="--clean-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="clean all artifacts from previous runs",
        )
        API.Check.register_arg(
            name="--save-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="save all artifacts during run",
        )
        API.Check.register_arg(
            name="binary",
            type=str,
            default="",
            choices=None,
            help="flatbuffer binary file",
        )
        API.Check.register_arg(
            name="--log-file",
            type=str,
            default="",
            choices=None,
            help="log file to dump ttrt output to",
        )
        API.Check.register_arg(
            name="--system-desc",
            type=str,
            default="",
            choices=None,
            help="system desc to check against",
        )

        # register apis
        API.register_api(API.Query)
        API.register_api(API.Read)
        API.register_api(API.Run)
        API.register_api(API.Perf)
        API.register_api(API.Check)

    @staticmethod
    def register_api(api_class):
        missing_methods = [
            func for func in API.registered_functions if not hasattr(api_class, func)
        ]

        if missing_methods:
            raise TypeError(
                f"API class={api_class.__name__} is missing methods={missing_methods}"
            )

        API.registered_apis[api_class.__name__] = api_class

    class Query:
        registered_args = {}
        api_only_arg = []

        def __init__(self, args={}, logging=None, artifacts=None):
            for name, attributes in API.Query.registered_args.items():
                name = name if not name.startswith("-") else name.lstrip("-")
                name = name.replace("-", "_")

                if type(args) == dict:
                    if name in args.keys():
                        self[name] = args[name]
                    else:
                        self[name] = attributes["default"]
                else:
                    self[name] = getattr(args, name)

            self.logger = logging if logging != None else Logger(self["log_file"])
            self.logging = self.logger.get_logger()
            self.globals = Globals(self.logger)
            self.file_manager = FileManager(self.logger)
            self.artifacts = (
                artifacts
                if artifacts != None
                else Artifacts(self.logger, self.file_manager)
            )
            self.system_desc = None
            self.device_ids = None
            self.report = {}
            self.results = Results(self.logger, self.file_manager)

        def preprocess(self):
            self.logging.debug(f"preprocessing query API")

            if self["clean_artifacts"]:
                self.artifacts.clean_artifacts()

            if self["save_artifacts"]:
                self.artifacts.create_artifacts()

            self.logging.debug(f"finished preprocessing query API")

        def check_constraints(self):
            self.logging.debug(f"checking constraints for query API")
            self.logging.debug(f"finished constraints for query API")

        def execute(self):
            self.logging.debug(f"executing query API")

            import ttrt.runtime

            try:
                self.logging.debug(f"getting system descriptor")
                (
                    self.system_desc,
                    self.device_ids,
                ) = ttrt.runtime.get_current_system_desc()
                self.logging.info(self.system_desc.as_json())
            except Exception as e:
                test_result = {
                    "result": "error",
                    "exception": str(e),
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                }
                self.results.add_result(test_result)
                self.results.save_results()
                raise Exception(f"an unexpected error occurred: {e}")

            self.logging.debug(f"finished executing query API")

        def postprocess(self):
            self.logging.debug(f"postprocessing query API")

            if self["save_artifacts"]:
                self.artifacts.save_system_desc(self.system_desc)

            test_result = {
                "result": "pass",
                "exception": "",
                "log_file": self.logger.file_name,
                "artifacts": self.artifacts.artifacts_folder_path,
            }
            self.results.add_result(test_result)
            self.results.save_results()

            self.logging.debug(f"finished postprocessing query API")

        def __str__(self):
            pass

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, value):
            setattr(self, key, value)

        def __call__(self):
            self.logging.debug(f"starting query API")

            self.preprocess()
            self.check_constraints()
            self.execute()
            self.postprocess()

            self.logging.debug(f"finished query API")

        def get_system_desc_as_dict(self):
            return json.loads(self.system_desc.as_json())

        @staticmethod
        def register_arg(name, type, default, choices, help, api_only=True):
            API.Query.registered_args[name] = {
                "type": type,
                "default": default,
                "choices": choices,
                "help": help,
            }

            if api_only:
                API.Query.api_only_arg.append(name)

        @staticmethod
        def get_upstream_apis():
            upstream_apis = []
            for arg_name, arg_value in API.Query.registered_args.items():
                if arg_name not in API.Query.api_only_arg:
                    upstream_apis.append(
                        {
                            "name": arg_name,
                            "type": arg_value["type"],
                            "default": arg_value["default"],
                            "choices": arg_value["choices"],
                            "help": arg_value["help"],
                        }
                    )

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
        read_actions = [
            "all",
            "version",
            "system_desc",
            "mlir",
            "cpp",
            "inputs",
            "outputs",
        ]

        def __init__(self, args={}, logging=None, artifacts=None):
            for name, attributes in API.Read.registered_args.items():
                name = name if not name.startswith("-") else name.lstrip("-")
                name = name.replace("-", "_")

                if type(args) == dict:
                    if name in args.keys():
                        self[name] = args[name]
                    else:
                        self[name] = attributes["default"]
                else:
                    self[name] = getattr(args, name)

            self.logger = logging if logging != None else Logger(self["log_file"])
            self.logging = self.logger.get_logger()
            self.globals = Globals(self.logger)
            self.file_manager = FileManager(self.logger)
            self.artifacts = (
                artifacts
                if artifacts != None
                else Artifacts(self.logger, self.file_manager)
            )
            self.read_action_functions = {}
            self.ttnn_binaries = []
            self.ttmetal_binaries = []
            self.system_desc_binaries = []
            self.results = Results(self.logger, self.file_manager)

        def preprocess(self):
            self.logging.debug(f"preprocessing read API")

            if self["clean_artifacts"]:
                self.artifacts.clean_artifacts()

            if self["save_artifacts"]:
                self.artifacts.create_artifacts()

            for action in self.read_actions:
                self.read_action_functions[action] = self[action]

            self.logging.debug(f"finished preprocessing read API")

        def check_constraints(self):
            self.logging.debug(f"checking constraints for read API")

            ttsys_binary_paths = self.file_manager.find_ttsys_binary_paths(
                self["binary"]
            )
            ttnn_binary_paths = self.file_manager.find_ttnn_binary_paths(self["binary"])
            ttmetal_binary_paths = self.file_manager.find_ttmetal_binary_paths(
                self["binary"]
            )

            self.logging.debug(f"ttsys_binary_paths={ttsys_binary_paths}")
            self.logging.debug(f"ttnn_binary_paths={ttnn_binary_paths}")
            self.logging.debug(f"ttmetal_binary_paths={ttmetal_binary_paths}")

            for path in ttsys_binary_paths:
                try:
                    bin = SystemDesc(self.logger, self.file_manager, path)
                    if bin.check_version():
                        self.system_desc_binaries.append(bin)
                except Exception as e:
                    test_result = {
                        "file_path": path, 
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                    }
                    self.results.add_result(test_result)

            for path in ttnn_binary_paths:
                try:
                    bin = Binary(self.logger, self.file_manager, path)
                    if bin.check_version():
                        self.ttnn_binaries.append(bin)
                except Exception as e:
                    test_result = {
                        "file_path": path, 
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                    }
                    self.results.add_result(test_result)

            for path in ttmetal_binary_paths:
                try:
                    bin = Binary(self.logger, self.file_manager, path)
                    if bin.check_version():
                        self.ttmetal_binaries.append(bin)
                except Exception as e:
                    test_result = {
                        "file_path": path, 
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                    }
                    self.results.add_result(test_result)

            self.logging.debug(f"finished checking constraints for read API")

        def execute(self):
            self.logging.debug(f"executing read API")

            for bin in self.system_desc_binaries:
                try:
                    self.logging.info(
                        f"reading section={self['section']} from binary={bin.file_path}"
                    )
                    self.read_action_functions[self["section"]](bin)
                except Exception as e:
                    test_result = {
                        "file_path": bin.file_path, 
                        "result": "error",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                    }
                    self.results.add_result(test_result)

            for bin in self.ttnn_binaries:
                try:
                    self.logging.info(
                        f"reading section={self['section']} from binary={bin.file_path}"
                    )
                    self.read_action_functions[self["section"]](bin)
                except Exception as e:
                    test_result = {
                        "file_path": bin.file_path, 
                        "result": "error",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                    }
                    self.results.add_result(test_result)

            for bin in self.ttmetal_binaries:
                try:
                    self.logging.info(
                        f"reading section={self['section']} from binary={bin.file_path}"
                    )
                    self.read_action_functions[self["section"]](bin)
                except Exception as e:
                    test_result = {
                        "file_path": bin.file_path, 
                        "result": "error",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                    }
                    self.results.add_result(test_result)

            self.logging.debug(f"finished executing read API")

        def postprocess(self):
            self.logging.debug(f"postprocessing read API")

            if self["save_artifacts"]:
                for bin in self.ttnn_binaries:
                    self.artifacts.save_binary(bin)

                for bin in self.ttmetal_binaries:
                    self.artifacts.save_binary(bin)


            for bin in self.ttnn_binaries:
                test_result = {
                    "file_path": bin.file_path, 
                    "result": "pass",
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                }
                self.results.add_result(test_result)

            for bin in self.ttmetal_binaries:
                test_result = {
                    "file_path": bin.file_path, 
                    "result": "pass",
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                }
                self.results.add_result(test_result)

            self.results.save_results()

            self.logging.debug(f"finished postprocessing read API")

        def __str__(self):
            pass

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, value):
            setattr(self, key, value)

        def __call__(self):
            self.logging.debug(f"starting read API")

            self.preprocess()
            self.check_constraints()
            self.execute()
            self.postprocess()

            self.logging.debug(f"finished read API")

        def all(self, binary):
            try:
                self.logging.info(binary.fbb.as_json())
            except Exception as e:
                raise Exception(f"failed to read all for binary={binary.file_path}")

        def version(self, binary):
            try:
                self.logging.info(
                    f"\nversion: {binary.fbb.version}\ntt-mlir git hash: {binary.fbb.ttmlir_git_hash}"
                )
            except Exception as e:
                raise Exception(f"failed to read version for binary={binary.file_path}")

        def system_desc(self, binary):
            try:
                import ttrt.binary

                bin_dict = ttrt.binary.as_dict(binary.fbb)
                return self.logging.info(json.dumps(bin_dict["system_desc"], indent=2))
            except Exception as e:
                raise Exception(f"failed to read system_desc for binary={binary.file_path}")

        def mlir(self, binary):
            try:
                import ttrt.binary

                bin_dict = ttrt.binary.as_dict(binary.fbb)

                for i, program in enumerate(bin_dict["programs"]):
                    if "debug_info" not in program:
                        self.logging.info(
                            f"no debug info found for program:{program['name']}"
                        )
                        continue
                    self.logging.info(
                        f"program[{i}]:{program['name']}-{program['debug_info']['mlir']['name']}"
                    )
                    self.logging.info(f"\n{program['debug_info']['mlir']['source']}")
            except Exception as e:
                raise Exception(f"failed to read mlir for binary={binary.file_path}")

        def cpp(self, binary):
            try:
                import ttrt.binary

                bin_dict = ttrt.binary.as_dict(binary.fbb)

                for i, program in enumerate(bin_dict["programs"]):
                    if "debug_info" not in program:
                        self.logging.info(
                            f"no debug info found for program:{program['name']}"
                        )
                        continue
                    self.logging.info(f"program[{i}]:{program['name']}")
                    self.logging.info(f"\n{program['debug_info']['cpp']}")
            except Exception as e:
                raise Exception(f"failed to read cpp for binary={binary.file_path}")

        def inputs(self, binary):
            try:
                import ttrt.binary

                bin_dict = ttrt.binary.as_dict(binary.fbb)

                for program in bin_dict["programs"]:
                    self.logging.info(f"program:{program['name']}")
                    self.logging.info(f"\n{json.dumps(program['inputs'], indent=2)}")
            except Exception as e:
                raise Exception(f"failed to read inputs for binary={binary.file_path}")

        def outputs(self, binary):
            try:
                import ttrt.binary

                bin_dict = ttrt.binary.as_dict(binary.fbb)

                for program in bin_dict["programs"]:
                    self.logging.info(f"program:{program['name']}")
                    self.logging.info(f"\n{json.dumps(program['outputs'], indent=2)}")
            except Exception as e:
                raise Exception(f"failed to read outputs for binary={binary.file_path}")

        @staticmethod
        def register_arg(name, type, default, choices, help, api_only=True):
            API.Read.registered_args[name] = {
                "type": type,
                "default": default,
                "choices": choices,
                "help": help,
            }

            if api_only:
                API.Read.api_only_arg.append(name)

        @staticmethod
        def get_upstream_apis():
            upstream_apis = []
            for arg_name, arg_value in API.Read.registered_args.items():
                if arg_name not in API.Read.api_only_arg:
                    upstream_apis.append(
                        {
                            "name": arg_name,
                            "type": arg_value["type"],
                            "default": arg_value["default"],
                            "choices": arg_value["choices"],
                            "help": arg_value["help"],
                        }
                    )

            return upstream_apis

        @staticmethod
        def generate_subparser(subparsers):
            read_parser = subparsers.add_parser(
                "read", help="read information from flatbuffer binary"
            )
            read_parser.set_defaults(api=API.Read)

            for name, attributes in API.Read.registered_args.items():
                if name == "binary":
                    read_parser.add_argument(f"{name}", help=attributes["help"])
                elif attributes["type"] == bool:
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

    class Run:
        registered_args = {}
        api_only_arg = []

        def __init__(self, args={}, logging=None, artifacts=None):
            for name, attributes in API.Run.registered_args.items():
                name = name if not name.startswith("-") else name.lstrip("-")
                name = name.replace("-", "_")

                if type(args) == dict:
                    if name in args.keys():
                        self[name] = args[name]
                    else:
                        self[name] = attributes["default"]
                else:
                    self[name] = getattr(args, name)

            self.logger = logging if logging != None else Logger(self["log_file"])
            self.logging = self.logger.get_logger()
            self.globals = Globals(self.logger)
            self.file_manager = FileManager(self.logger)
            self.artifacts = (
                artifacts
                if artifacts != None
                else Artifacts(self.logger, self.file_manager)
            )
            self.query = API.Query({}, self.logger, self.artifacts)
            self.ttnn_binaries = []
            self.ttmetal_binaries = []
            self.results = Results(self.logger, self.file_manager)

        def preprocess(self):
            self.logging.debug(f"preprocessing run API")
            self.query()

            if self["clean_artifacts"]:
                self.artifacts.clean_artifacts()

            if self["save_artifacts"]:
                self.artifacts.create_artifacts()

            self.logging.debug(f"finished preprocessing read API")

        def check_constraints(self):
            self.logging.debug(f"checking constraints for run API")

            ttnn_binary_paths = self.file_manager.find_ttnn_binary_paths(self["binary"])
            ttmetal_binary_paths = self.file_manager.find_ttmetal_binary_paths(
                self["binary"]
            )

            self.logging.debug(f"ttnn_binary_paths={ttnn_binary_paths}")
            self.logging.debug(f"ttmetal_binary_paths={ttmetal_binary_paths}")

            for path in ttnn_binary_paths:
                bin = Binary(self.logger, self.file_manager, path)
                try:
                    bin.check_version()
                except Exception as e:
                    test_result = {
                        "file_path": path,
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "seed": self["seed"],
                        "init": self["init"],
                        "rtol": self["rtol"],
                        "atol": self["atol"],
                        "loops": self["loops"],
                        "program_index": self["program_index"],
                    }
                    self.results.add_result(test_result)
                    continue

                try:
                    bin.check_system_desc(self.query)
                except Exception as e:
                    test_result = {
                        "file_path": path,
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "seed": self["seed"],
                        "init": self["init"],
                        "rtol": self["rtol"],
                        "atol": self["atol"],
                        "loops": self["loops"],
                        "program_index": self["program_index"],
                    }
                    self.results.add_result(test_result)
                    continue

                if self["program_index"] != "all":
                    if not bin.check_program_index_exists(int(self["program_index"])):
                        message = f"program index={int(self['program_index'])} is greater than number of programs in: {bin.file_path} - skipping this test"
                        self.logging.warning(message)
                        test_result = {
                            "file_path": path,
                            "result": "skip",
                            "exception": message,
                            "log_file": self.logger.file_name,
                            "artifacts": self.artifacts.artifacts_folder_path,
                            "seed": self["seed"],
                            "init": self["init"],
                            "rtol": self["rtol"],
                            "atol": self["atol"],
                            "loops": self["loops"],
                            "program_index": self["program_index"],
                        }
                        self.results.add_result(test_result)
                        continue

                self.ttnn_binaries.append(bin)

            for path in ttmetal_binary_paths:
                bin = Binary(self.logger, self.file_manager, path)
                try:
                    bin.check_version()
                except Exception as e:
                    test_result = {
                        "file_path": path,
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "seed": self["seed"],
                        "init": self["init"],
                        "rtol": self["rtol"],
                        "atol": self["atol"],
                        "loops": self["loops"],
                        "program_index": self["program_index"],
                    }
                    self.results.add_result(test_result)
                    continue

                try:
                    bin.check_system_desc(self.query)
                except Exception as e:
                    test_result = {
                        "file_path": path,
                        "result": "skip",
                        "exception": str(e),
                        "log_file": self.logger.file_name,
                        "artifacts": self.artifacts.artifacts_folder_path,
                        "seed": self["seed"],
                        "init": self["init"],
                        "rtol": self["rtol"],
                        "atol": self["atol"],
                        "loops": self["loops"],
                        "program_index": self["program_index"],
                    }
                    self.results.add_result(test_result)
                    continue

                if self["program_index"] != "all":
                    if not bin.check_program_index_exists(int(self["program_index"])):
                        message = f"program index={int(self['program_index'])} is greater than number of programs in: {bin.file_path} - skipping this test"
                        self.logging.warning(message)
                        test_result = {
                            "file_path": path,
                            "result": "skip",
                            "exception": message,
                            "log_file": self.logger.file_name,
                            "artifacts": self.artifacts.artifacts_folder_path,
                            "seed": self["seed"],
                            "init": self["init"],
                            "rtol": self["rtol"],
                            "atol": self["atol"],
                            "loops": self["loops"],
                            "program_index": self["program_index"],
                        }
                        self.results.add_result(test_result)
                        continue

                self.ttmetal_binaries.append(bin)

            self.logging.debug(f"finished checking constraints for run API")

        def execute(self):
            self.logging.debug(f"executing run API")

            def _execute(binaries):
                import ttrt.runtime
                import torch

                if len(binaries) == 0:
                    self.logging.warning(f"no binaries found to run - returning early")
                    return

                self.logging.debug(f"setting torch manual seed={self['seed']}")
                torch.manual_seed(self["seed"])
                ttrt.runtime.set_compatible_runtime(binaries[0].fbb)

                self.logging.debug(f"opening device id={self.query.device_ids[0]}")
                device = ttrt.runtime.open_device([self.query.device_ids[0]])
                atexit.register(lambda: ttrt.runtime.close_device(device))

                for bin in binaries:
                    try:
                        self.logging.info(f"evaluating binary={bin.file_path}")

                        program_indices = []
                        if self["program_index"] == "all":
                            program_indices.extend(range(bin.get_num_programs()))
                        else:
                            program_indices.append(int(self["program_index"]))

                        for program_index in program_indices:
                            self.logging.debug(
                                f"evaluating program={program_index} for binary={bin.file_path}"
                            )

                            program = bin.get_program(program_index)
                            program.populate_inputs(
                                API.Run.TorchInitilizer.get_initilizer(self["init"])
                            )
                            program.populate_outputs(
                                API.Run.TorchInitilizer.get_initilizer("zeros")
                            )

                            total_inputs = []
                            total_outputs = []
                            for loop in range(self["loops"]):
                                self.logging.debug(
                                    f"generating inputs/outputs for loop={loop+1}/{self['loops']} for binary={bin.file_path}"
                                )

                                inputs = []
                                outputs = []
                                for i in program.input_tensors:
                                    inputs.append(
                                        ttrt.runtime.create_tensor(
                                            i.data_ptr(),
                                            list(i.shape),
                                            list(i.stride()),
                                            i.element_size(),
                                            Binary.Program.to_data_type(i.dtype),
                                        )
                                    )

                                for i in program.output_tensors:
                                    outputs.append(
                                        ttrt.runtime.create_tensor(
                                            i.data_ptr(),
                                            list(i.shape),
                                            list(i.stride()),
                                            i.element_size(),
                                            Binary.Program.to_data_type(i.dtype),
                                        )
                                    )

                                total_inputs.append(inputs)
                                total_outputs.append(outputs)

                            event = None
                            for loop in range(self["loops"]):
                                self.logging.debug(
                                    f"starting loop={loop+1}/{self['loops']} for binary={bin.file_path}"
                                )

                                event = ttrt.runtime.submit(
                                    device,
                                    bin.fbb,
                                    program_index,
                                    total_inputs[loop],
                                    total_outputs[loop],
                                )

                                self.logging.debug(
                                    f"finished loop={loop+1}/{self['loops']} for binary={bin.file_path}"
                                )

                            ttrt.runtime.wait(event)

                            if self["identity"]:
                                self.logging.debug(
                                    f"checking identity with rtol={self['rtol']} and atol={self['atol']}"
                                )

                                for i, o in zip(
                                    program.input_tensors, program.output_tensors
                                ):
                                    if not torch.allclose(
                                        i, o, rtol=self["rtol"], atol=self["atol"]
                                    ):
                                        self.logging.error(
                                            f"Failed: inputs and outputs do not match in binary"
                                        )
                                        self.logging.error(i - o)

                            self.logging.debug(f"input tensors for program={program_index}")
                            for tensor in program.input_tensors:
                                self.logging.debug(f"{tensor}\n")

                            self.logging.debug(
                                f"output tensors for program={program_index}"
                            )
                            for tensor in program.output_tensors:
                                self.logging.debug(f"{tensor}\n")
                    except Exception as e:
                        test_result = {
                            "file_path": bin.file_path,
                            "result": "error",
                            "exception": str(e),
                            "log_file": self.logger.file_name,
                            "artifacts": self.artifacts.artifacts_folder_path,
                            "seed": self["seed"],
                            "init": self["init"],
                            "rtol": self["rtol"],
                            "atol": self["atol"],
                            "loops": self["loops"],
                            "program_index": self["program_index"],
                        }
                        self.results.add_result(test_result)
                        continue

            self.logging.debug(f"executing ttnn binaries")
            _execute(self.ttnn_binaries)
            self.logging.debug(f"finished executing ttnn binaries")

            self.logging.debug(f"executing ttmetal binaries")
            _execute(self.ttmetal_binaries)
            self.logging.debug(f"finished executing ttmetal binaries")

            self.logging.debug(f"finished executing run API")

        def postprocess(self):
            self.logging.debug(f"postprocessing run API")

            if self["save_artifacts"]:
                for bin in self.ttnn_binaries:
                    self.artifacts.save_binary(bin, self.query)

                for bin in self.ttmetal_binaries:
                    self.artifacts.save_binary(bin, self.query)

            for bin in self.ttnn_binaries:
                test_result = {
                    "file_path": bin.file_path,
                    "result": "pass",
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                    "seed": self["seed"],
                    "init": self["init"],
                    "rtol": self["rtol"],
                    "atol": self["atol"],
                    "loops": self["loops"],
                    "program_index": self["program_index"],
                }
                self.results.add_result(test_result)

            for bin in self.ttmetal_binaries:
                test_result = {
                    "file_path": bin.file_path,
                    "result": "pass",
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                    "seed": self["seed"],
                    "init": self["init"],
                    "rtol": self["rtol"],
                    "atol": self["atol"],
                    "loops": self["loops"],
                    "program_index": self["program_index"],
                }
                self.results.add_result(test_result)

            self.results.save_results()

            self.logging.debug(f"finished postprocessing run API")

        def __str__(self):
            pass

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, value):
            setattr(self, key, value)

        def __call__(self):
            self.logging.debug(f"starting run API")

            self.preprocess()
            self.check_constraints()
            self.execute()
            self.postprocess()

            self.logging.debug(f"finished run API")

        @staticmethod
        def register_arg(name, type, default, choices, help, api_only=True):
            API.Run.registered_args[name] = {
                "type": type,
                "default": default,
                "choices": choices,
                "help": help,
            }

            if api_only:
                API.Run.api_only_arg.append(name)

        @staticmethod
        def get_upstream_apis():
            upstream_apis = []
            for arg_name, arg_value in API.Run.registered_args.items():
                if arg_name not in API.Run.api_only_arg:
                    upstream_apis.append(
                        {
                            "name": arg_name,
                            "type": arg_value["type"],
                            "default": arg_value["default"],
                            "choices": arg_value["choices"],
                            "help": arg_value["help"],
                        }
                    )

            return upstream_apis

        @staticmethod
        def generate_subparser(subparsers):
            run_parser = subparsers.add_parser("run", help="run a flatbuffer binary")
            run_parser.set_defaults(api=API.Run)

            for name, attributes in API.Run.registered_args.items():
                if name == "binary":
                    run_parser.add_argument(f"{name}", help=attributes["help"])
                elif attributes["type"] == bool:
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

        class TorchInitilizer:
            init_fns = sorted(["randn", "arange", "zeros"])

            @staticmethod
            def get_initilizer(name):
                for attr, value in API.Run.TorchInitilizer.__dict__.items():
                    if attr == name:
                        return value

                raise Exception(f"could not find specified init function={name}")

            @staticmethod
            def get_init_fns():
                return API.Run.TorchInitilizer.init_fns

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

        def __init__(self, args={}, logging=None, artifacts=None):
            for name, attributes in API.Perf.registered_args.items():
                name = name if not name.startswith("-") else name.lstrip("-")
                name = name.replace("-", "_")

                if type(args) == dict:
                    if name in args.keys():
                        self[name] = args[name]
                    else:
                        self[name] = attributes["default"]
                else:
                    self[name] = getattr(args, name)

            self.logger = logging if logging != None else Logger(self["log_file"])
            self.logging = self.logger.get_logger()
            self.globals = Globals(self.logger)
            self.file_manager = FileManager(self.logger)
            self.artifacts = (
                artifacts
                if artifacts != None
                else Artifacts(self.logger, self.file_manager)
            )
            self.query = API.Query({}, self.logger, self.artifacts)
            self.ttnn_binaries = []
            self.ttmetal_binaries = []
            self.tracy_capture_tool_path = (
                f"{self.globals.get_ttmetal_home_path()}/capture-release"
            )
            self.tracy_csvexport_tool_path = (
                f"{self.globals.get_ttmetal_home_path()}/csvexport-release"
            )
            self.tracy_capture_tool_process = None

        def preprocess(self):
            self.logging.debug(f"preprocessing perf API")
            self.query()

            if self["clean_artifacts"]:
                self.artifacts.clean_artifacts()

            self.artifacts.create_artifacts()

            self.logging.debug(f"finished preprocessing perf API")

        def check_constraints(self):
            self.logging.debug(f"checking constraints for perf API")

            assert self.file_manager.check_file_exists(
                self.tracy_capture_tool_path
            ), f"perf tool={self.tracy_capture_tool_path} does not exist - rebuild using perf mode"
            assert self.file_manager.check_file_exists(
                self.tracy_csvexport_tool_path
            ), f"perf tool={self.tracy_csvexport_tool_path} does not exist - rebuild using perf mode"

            if not hasattr(self, "binary"):
                # Load from Capsule instead. only TTNN Path is supported for now
                bin = Binary(self.logger, self.file_manager, "", self["capsule"])
                if not bin.check_version():
                    self.logger.warning(
                        "Flatbuffer version not present, are you sure that the binary is valid? - Skipped"
                    )
                    return

                if not bin.check_system_desc(self.query):
                    self.logger.warning(
                        "System desc does not match, are you sure that the binary is valid? - Skipped"
                    )
                    return

                if self["program_index"] != "all":
                    if not bin.check_program_index_exists(int(self["program_index"])):
                        self.logging.warning(
                            f"program index={int(self['program_index'])} is greater than number of programs in: {bin.file_path} - skipping this test"
                        )
                        return
                self.ttnn_binaries.append(bin)
            else:
                ttnn_binary_paths = self.file_manager.find_ttnn_binary_paths(
                    self["binary"]
                )
                ttmetal_binary_paths = self.file_manager.find_ttmetal_binary_paths(
                    self["binary"]
                )

                self.logging.debug(f"ttnn_binary_paths={ttnn_binary_paths}")
                self.logging.debug(f"ttmetal_binary_paths={ttmetal_binary_paths}")

                for path in ttnn_binary_paths:
                    bin = Binary(self.logger, self.file_manager, path)
                    if not bin.check_version():
                        continue

                    if not bin.check_system_desc(self.query):
                        continue

                    if self["program_index"] != "all":
                        if not bin.check_program_index_exists(
                            int(self["program_index"])
                        ):
                            self.logging.warning(
                                f"program index={int(self['program_index'])} is greater than number of programs in: {bin.file_path} - skipping this test"
                            )
                            continue

                    self.ttnn_binaries.append(bin)

                self.logging.debug(f"finished checking constraints for run API")

                for path in ttmetal_binary_paths:
                    bin = Binary(self.logger, self.file_manager, path)
                    if not bin.check_version():
                        continue

                    if not bin.check_system_desc(self.query):
                        continue

                    if self["program_index"] != "all":
                        if not bin.check_program_index_exists(
                            int(self["program_index"])
                        ):
                            self.logging.warning(
                                f"program index={int(self['program_index'])} is greater than number of programs in: {bin.file_path} - skipping this test"
                            )
                            continue

                    self.ttmetal_binaries.append(bin)

            self.logging.debug(f"finished checking constraints for perf API")

        def execute(self):
            self.logging.debug(f"executing perf API")

            sys.path.append(f"{get_ttrt_metal_home_path()}")
            sys.path.append(f"{get_ttrt_metal_home_path()}/ttnn")

            import tracy
            import tracy.tracy_state
            from tt_metal.tools.profiler.process_ops_logs import process_ops
            from tt_metal.tools.profiler.common import (
                PROFILER_LOGS_DIR,
                TRACY_OPS_TIMES_FILE_NAME,
                TRACY_OPS_DATA_FILE_NAME,
                TRACY_FILE_NAME,
            )

            self.file_manager.remove_directory(PROFILER_LOGS_DIR)
            self.file_manager.create_directory(PROFILER_LOGS_DIR)

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
                        raise Exception(
                            f"tracy capture output file {PROFILER_LOGS_DIR / TRACY_FILE_NAME} was not generated"
                        )
                    time_count += 1
                    time.sleep(1)

                with open(
                    PROFILER_LOGS_DIR / TRACY_OPS_TIMES_FILE_NAME, "w"
                ) as csv_file:
                    child_call_str = f"-x {','.join(child_calls)}"
                    subprocess.run(
                        f"{self.tracy_csvexport_tool_path} -u -p TT_DNN {child_call_str} {PROFILER_LOGS_DIR / TRACY_FILE_NAME}",
                        shell=True,
                        check=True,
                        stdout=csv_file,
                        stderr=subprocess.DEVNULL,
                    )

                with open(
                    PROFILER_LOGS_DIR / TRACY_OPS_DATA_FILE_NAME, "w"
                ) as csv_file:
                    subprocess.run(
                        f'{self.tracy_csvexport_tool_path} -m -s ";" {PROFILER_LOGS_DIR / TRACY_FILE_NAME}',
                        shell=True,
                        check=True,
                        stdout=csv_file,
                        stderr=subprocess.DEVNULL,
                    )
                process_ops(binary_perf_folder, "", True)

            def save_perf_artifacts(binary_perf_folder):
                tracy_ops_times_file = f"{self.globals.get_ttmetal_home_path()}/generated/profiler/.logs/{TRACY_OPS_TIMES_FILE_NAME}"
                tracy_ops_data_file = f"{self.globals.get_ttmetal_home_path()}/generated/profiler/.logs/{TRACY_OPS_DATA_FILE_NAME}"
                tracy_file = f"{self.globals.get_ttmetal_home_path()}/generated/profiler/.logs/{TRACY_FILE_NAME}"

                try:
                    self.file_manager.check_file_exists(tracy_ops_times_file)
                    self.file_manager.check_file_exists(tracy_ops_data_file)
                    self.file_manager.check_file_exists(tracy_file)
                except Exception as e:
                    raise Exception(f"an unexpected error occurred: {e}")

                for root, dirs, files in os.walk(binary_perf_folder, topdown=False):
                    if root == binary_perf_folder:
                        continue

                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        dest_path = os.path.join(binary_perf_folder, file_name)
                        self.file_manager.copy_file(dest_path, file_path)

                    if os.listdir(root):
                        self.file_manager.remove_directory(root)

            def _execute(binaries):
                if len(binaries) == 0:
                    self.logging.warning(f"no binaries found to run - returning early")
                    return

                for bin in binaries:
                    port = get_available_port()

                    if not port:
                        raise Exception("No available port found")

                    env_vars = dict(os.environ)
                    env_vars["TRACY_PORT"] = port
                    env_vars["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"

                    if self["device"]:
                        env_vars["TT_METAL_DEVICE_PROFILER"] = "1"

                    current_pythonpath = env_vars.get("PYTHONPATH", "")
                    path1 = f"{get_ttrt_metal_home_path()}"
                    path2 = f"{get_ttrt_metal_home_path()}/ttnn"
                    new_pythonpath = (
                        f"{current_pythonpath}{os.pathsep}{path1}{os.pathsep}{path2}"
                    )
                    env_vars["PYTHONPATH"] = new_pythonpath

                    tracy_capture_tool_command = f"{self.tracy_capture_tool_path} -o {PROFILER_LOGS_DIR / TRACY_FILE_NAME} -f -p {port}"
                    self.tracy_capture_tool_process = subprocess.Popen(
                        tracy_capture_tool_command, shell=True
                    )

                    upstream_apis = API.Run.get_upstream_apis()
                    command_options = ""

                    for api in upstream_apis:
                        name = (
                            api["name"]
                            if not api["name"].startswith("-")
                            else api["name"].lstrip("-")
                        )
                        name = name.replace("-", "_")

                        if api["type"] == bool:
                            if self[name]:
                                command_options += f" {api['name']} "
                        else:
                            command_options += f" {api['name']} {self[name]} "

                    library_link_path = self.globals.get_ld_path(
                        f"{self.globals.get_ttmetal_home_path()}"
                    )
                    test_env_flags = f"LD_LIBRARY_PATH={library_link_path}"

                    ttrt_executable_path = shutil.which("ttrt")
                    test_command = f"{test_env_flags} python -m tracy -p {ttrt_executable_path} run {bin.file_path} --save-artifacts {command_options}"
                    print(f"test command for binary={bin.file_path} is: {test_command}")
                    testProcess = subprocess.Popen(
                        [test_command], shell=True, env=env_vars, preexec_fn=os.setsid
                    )

                    def signal_handler(sig, frame):
                        os.killpg(os.getpgid(testProcess.pid), signal.SIGTERM)
                        self.tracy_capture_tool_process.terminate()
                        self.tracy_capture_tool_process.communicate()
                        sys.exit(3)

                    signal.signal(signal.SIGINT, signal_handler)
                    signal.signal(signal.SIGTERM, signal_handler)
                    testProcess.communicate()

                    try:
                        self.tracy_capture_tool_process.communicate(timeout=15)
                        generate_report(self.artifacts.get_binary_perf_folder_path(bin))
                    except subprocess.TimeoutExpired as e:
                        self.tracy_capture_tool_process.terminate()
                        self.tracy_capture_tool_process.communicate()
                        raise Exception(
                            f"No profiling data could be captured. Please make sure you are on the correct build"
                        )

                    save_perf_artifacts(self.artifacts.get_binary_perf_folder_path(bin))

            if len(self.ttnn_binaries) != 0:
                _execute(self.ttnn_binaries)

            if len(self.ttmetal_binaries) != 0:
                _execute(self.ttnn_binaries)

            self.logging.debug(f"finished executing perf API")

        def postprocess(self):
            self.logging.debug(f"postprocessing perf API")

            if self["save_artifacts"]:
                for bin in self.ttnn_binaries:
                    self.artifacts.save_binary(bin, self.query)

                for bin in self.ttmetal_binaries:
                    self.artifacts.save_binary(bin, self.query)

            self.logging.debug(f"finished postprocessing perf API")

        def __str__(self):
            pass

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, value):
            setattr(self, key, value)

        def __call__(self):
            self.logging.debug(f"starting perf API")

            self.preprocess()
            self.check_constraints()
            self.execute()
            self.postprocess()

            self.logging.debug(f"finished perf API")

        @staticmethod
        def register_arg(name, type, default, choices, help, api_only=True):
            API.Perf.registered_args[name] = {
                "type": type,
                "default": default,
                "choices": choices,
                "help": help,
            }

            if api_only:
                API.Perf.api_only_arg.append(name)

        @staticmethod
        def get_upstream_apis():
            upstream_apis = []
            for arg_name, arg_value in API.Perf.registered_args.items():
                if arg_name not in API.Perf.api_only_arg:
                    upstream_apis.append(
                        {
                            "name": arg_name,
                            "type": arg_value["type"],
                            "default": arg_value["default"],
                            "choices": arg_value["choices"],
                            "help": arg_value["help"],
                        }
                    )

            return upstream_apis

        @staticmethod
        def generate_subparser(subparsers):
            perf_parser = subparsers.add_parser(
                "perf", help="run performance trace and collect performance data"
            )
            perf_parser.set_defaults(api=API.Perf)

            for name, attributes in API.Perf.registered_args.items():
                if name == "binary":
                    perf_parser.add_argument(f"{name}", help=attributes["help"])
                elif attributes["type"] == bool:
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

    class Check:
        registered_args = {}
        api_only_arg = []

        def __init__(self, args={}, logging=None, artifacts=None):
            for name, attributes in API.Check.registered_args.items():
                name = name if not name.startswith("-") else name.lstrip("-")
                name = name.replace("-", "_")

                if type(args) == dict:
                    if name in args.keys():
                        self[name] = args[name]
                    else:
                        self[name] = attributes["default"]
                else:
                    self[name] = getattr(args, name)

            self.logger = logging if logging != None else Logger(self["log_file"])
            self.logging = self.logger.get_logger()
            self.globals = Globals(self.logger)
            self.file_manager = FileManager(self.logger)
            self.artifacts = (
                artifacts
                if artifacts != None
                else Artifacts(self.logger, self.file_manager)
            )
            self.query = API.Query({}, self.logger, self.artifacts)
            self.ttnn_binaries = []
            self.ttmetal_binaries = []
            self.system_desc_binaries = []

        def preprocess(self):
            self.logging.debug(f"preprocessing check API")

            if self["clean_artifacts"]:
                self.artifacts.clean_artifacts()

            if self["save_artifacts"]:
                self.artifacts.create_artifacts()

            self.logging.debug(f"finished preprocessing check API")

        def check_constraints(self):
            self.logging.debug(f"checking constraints for check API")

            ttsys_binary_paths = self.file_manager.find_ttsys_binary_paths(
                self["system_desc"]
            )
            ttnn_binary_paths = self.file_manager.find_ttnn_binary_paths(self["binary"])
            ttmetal_binary_paths = self.file_manager.find_ttmetal_binary_paths(
                self["binary"]
            )

            self.logging.debug(f"ttsys_binary_paths={ttsys_binary_paths}")
            self.logging.debug(f"ttnn_binary_paths={ttnn_binary_paths}")
            self.logging.debug(f"ttmetal_binary_paths={ttmetal_binary_paths}")

            for path in ttsys_binary_paths:
                bin = SystemDesc(self.logger, self.file_manager, path)
                if bin.check_version():
                    self.system_desc_binaries.append(bin)

            for path in ttnn_binary_paths:
                bin = Binary(self.logger, self.file_manager, path)
                if not bin.check_version():
                    continue

                self.ttnn_binaries.append(bin)

            for path in ttmetal_binary_paths:
                bin = Binary(self.logger, self.file_manager, path)
                if not bin.check_version():
                    continue

                self.ttmetal_binaries.append(bin)

            self.logging.debug(f"finished checking constraints for check API")

        def execute(self):
            self.logging.debug(f"executing check API")

            def _execute(binaries, system_desc_to_check):
                if len(binaries) == 0:
                    self.logging.warning(f"no binaries found to run - returning early")
                    return

                for bin in binaries:
                    if system_desc_to_check != None:
                        if (
                            bin.fbb_dict["system_desc"]
                            != system_desc_to_check["system_desc"]
                        ):
                            self.logging.info(
                                f"system desc for device did not match flatbuffer: {bin.file_path}"
                            )
                        else:
                            self.logging.info(
                                f"system desc for device matched flatbuffer: {bin.file_path}"
                            )
                    else:
                        for desc in self.system_desc_binaries:
                            if (
                                bin.fbb_dict["system_desc"]
                                != desc.fbb_dict["system_desc"]
                            ):
                                self.logging.info(
                                    f"system desc for: {desc.file_path} did not match flatbuffer: {bin.file_path}"
                                )
                            else:
                                self.logging.info(
                                    f"system desc for: {desc.file_path} matched flatbuffer: {bin.file_path}"
                                )

            system_desc_to_check = None
            if self["system_desc"] == "" or len(self.system_desc_binaries) == 0:
                self.logging.warning(
                    "no system descriptor file provided - querying from host machine"
                )
                self.query()
                system_desc_to_check = self.query.get_system_desc_as_dict()

            self.logging.debug(f"executing ttnn binaries")
            _execute(self.ttnn_binaries, system_desc_to_check)
            self.logging.debug(f"finished executing ttnn binaries")

            self.logging.debug(f"executing ttmetal binaries")
            _execute(self.ttmetal_binaries, system_desc_to_check)
            self.logging.debug(f"finished executing ttmetal binaries")

            self.logging.debug(f"finished executing check API")

        def postprocess(self):
            self.logging.debug(f"postprocessing check API")

            if self["save_artifacts"]:
                for bin in self.ttnn_binaries:
                    self.artifacts.save_binary(bin)

                for bin in self.ttmetal_binaries:
                    self.artifacts.save_binary(bin)

            self.logging.debug(f"finished postprocessing check API")

        def __str__(self):
            pass

        def __getitem__(self, key):
            return getattr(self, key)

        def __setitem__(self, key, value):
            setattr(self, key, value)

        def __call__(self):
            self.logging.debug(f"starting check API")

            self.preprocess()
            self.check_constraints()
            self.execute()
            self.postprocess()

            self.logging.debug(f"finished check API")

        @staticmethod
        def register_arg(name, type, default, choices, help, api_only=True):
            API.Check.registered_args[name] = {
                "type": type,
                "default": default,
                "choices": choices,
                "help": help,
            }

            if api_only:
                API.Check.api_only_arg.append(name)

        @staticmethod
        def get_upstream_apis():
            upstream_apis = []
            for arg_name, arg_value in API.Check.registered_args.items():
                if arg_name not in API.Check.api_only_arg:
                    upstream_apis.append(
                        {
                            "name": arg_name,
                            "type": arg_value["type"],
                            "default": arg_value["default"],
                            "choices": arg_value["choices"],
                            "help": arg_value["help"],
                        }
                    )

            return upstream_apis

        @staticmethod
        def generate_subparser(subparsers):
            check_parser = subparsers.add_parser(
                "check", help="check a flatbuffer binary against a system desc file"
            )
            check_parser.set_defaults(api=API.Check)

            for name, attributes in API.Check.registered_args.items():
                if name == "binary":
                    check_parser.add_argument(f"{name}", help=attributes["help"])
                elif attributes["type"] == bool:
                    check_parser.add_argument(
                        f"{name}",
                        action="store_true",
                        help=attributes["help"],
                    )
                else:
                    check_parser.add_argument(
                        f"{name}",
                        type=attributes["type"],
                        default=attributes["default"],
                        choices=attributes["choices"],
                        help=attributes["help"],
                    )

            return check_parser
