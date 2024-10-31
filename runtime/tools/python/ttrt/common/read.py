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


class Read:
    registered_args = {}
    read_actions = [
        "all",
        "version",
        "system_desc",
        "mlir",
        "cpp",
        "inputs",
        "outputs",
    ]

    @staticmethod
    def initialize_api():
        Read.register_arg(
            name="--section",
            type=str,
            default="all",
            choices=sorted(Read.read_actions),
            help="output sections of the fb",
        )
        Read.register_arg(
            name="--clean-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="clean all artifacts from previous runs",
        )
        Read.register_arg(
            name="--save-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="save all artifacts during run",
        )
        Read.register_arg(
            name="--log-file",
            type=str,
            default="",
            choices=None,
            help="log file to dump ttrt output to",
        )
        Read.register_arg(
            name="--artifact-dir",
            type=str,
            default=f"{os.getcwd()}/ttrt-artifacts",
            choices=None,
            help="provides a directory path to save artifacts to",
        )
        Read.register_arg(
            name="--result-file",
            type=str,
            default="read_results.json",
            choices=None,
            help="test file to save results to",
        )
        Read.register_arg(
            name="binary",
            type=str,
            default="",
            choices=None,
            help="flatbuffer binary file",
        )

    def __init__(self, args={}, logger=None, artifacts=None):
        print(args)
        for name, attributes in Read.registered_args.items():
            if type(args) == dict:
                if name in args.keys():
                    self[name] = args[name]
                else:
                    self[name] = attributes["default"]
            else:
                # argument got parsed to hyphen's for underscrolls and leading hyphen's removed - need to put back
                converted_name = name
                if name != "binary":
                    converted_name = converted_name.lstrip("-")
                    converted_name = converted_name.replace("-", "_")
                self[name] = getattr(args, converted_name)

        self.logger = logger if logger != None else Logger(self["--log-file"])
        self.logging = self.logger.get_logger()
        self.globals = Globals(self.logger)
        self.file_manager = FileManager(self.logger)
        self.artifacts = (
            artifacts
            if artifacts != None
            else Artifacts(
                self.logger,
                self.file_manager,
                artifacts_folder_path=self["--artifact-dir"],
            )
        )
        self.read_action_functions = {}
        self.ttnn_binaries = []
        self.ttmetal_binaries = []
        self.system_desc_binaries = []
        self.results = Results(self.logger, self.file_manager)

    def preprocess(self):
        self.logging.debug(f"------preprocessing read API")

        if self["--clean-artifacts"]:
            self.artifacts.clean_artifacts()

        if self["--save-artifacts"]:
            self.artifacts.create_artifacts()

        for action in self.read_actions:
            self.read_action_functions[action] = self[action]

        self.logging.debug(f"------finished preprocessing read API")

    def check_constraints(self):
        self.logging.debug(f"------checking constraints for read API")

        ttsys_binary_paths = self.file_manager.find_ttsys_binary_paths(self["binary"])
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
                self.logging.warning(
                    f"SKIP: test={path} was skipped with exception={str(e)}"
                )
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
                self.logging.warning(
                    f"SKIP: test={path} was skipped with exception={str(e)}"
                )
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
                self.logging.warning(
                    f"SKIP: test={path} was skipped with exception={str(e)}"
                )
                self.results.add_result(test_result)

        self.logging.debug(f"------finished checking constraints for read API")

    def execute(self):
        self.logging.debug(f"------executing read API")

        for bin in self.system_desc_binaries:
            try:
                self.logging.info(
                    f"reading section={self['--section']} from binary={bin.file_path}"
                )
                self.read_action_functions[self["--section"]](bin)
            except Exception as e:
                test_result = {
                    "file_path": bin.file_path,
                    "result": "error",
                    "exception": str(e),
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                }
                self.logging.error(
                    f"ERROR: test={bin.file_path} experienced an error with exception={str(e)}"
                )
                self.results.add_result(test_result)
                bin.test_result = "error"

        for bin in self.ttnn_binaries:
            try:
                self.logging.info(
                    f"reading section={self['--section']} from binary={bin.file_path}"
                )
                self.read_action_functions[self["--section"]](bin)
            except Exception as e:
                test_result = {
                    "file_path": bin.file_path,
                    "result": "error",
                    "exception": str(e),
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                }
                self.logging.error(
                    f"ERROR: test={bin.file_path} experienced an error with exception={str(e)}"
                )
                self.results.add_result(test_result)
                bin.test_result = "error"

        for bin in self.ttmetal_binaries:
            try:
                self.logging.info(
                    f"reading section={self['--section']} from binary={bin.file_path}"
                )
                self.read_action_functions[self["--section"]](bin)
            except Exception as e:
                test_result = {
                    "file_path": bin.file_path,
                    "result": "error",
                    "exception": str(e),
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                }
                self.logging.error(
                    f"ERROR: test={bin.file_path} experienced an error with exception={str(e)}"
                )
                self.results.add_result(test_result)
                bin.test_result = "error"

        self.logging.debug(f"------finished executing read API")

    def postprocess(self):
        self.logging.debug(f"------postprocessing read API")

        if self["--save-artifacts"]:
            for bin in self.ttnn_binaries:
                self.artifacts.save_binary(bin)

            for bin in self.ttmetal_binaries:
                self.artifacts.save_binary(bin)

        for bin in self.system_desc_binaries:
            if bin.test_result == "pass":
                test_result = {
                    "file_path": bin.file_path,
                    "result": "pass",
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                }

                self.results.add_result(test_result)
                self.logging.info(f"PASS: test case={bin.file_path}")
            else:
                self.logging.error(f"ERROR: test case={bin.file_path}")

        for bin in self.ttnn_binaries:
            if bin.test_result == "pass":
                test_result = {
                    "file_path": bin.file_path,
                    "result": "pass",
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                }

                self.results.add_result(test_result)
                self.logging.info(f"PASS: test case={bin.file_path}")
            else:
                self.logging.error(f"ERROR: test case={bin.file_path}")

        for bin in self.ttmetal_binaries:
            if bin.test_result == "pass":
                test_result = {
                    "file_path": bin.file_path,
                    "result": "pass",
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                }

                self.results.add_result(test_result)
                self.logging.info(f"PASS: test case={bin.file_path}")
            else:
                self.logging.error(f"ERROR: test case={bin.file_path}")

        self.results.save_results(self["--result-file"])

        self.logging.debug(f"------finished postprocessing read API")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __call__(self):
        self.logging.debug(
            f"----------------------------starting read API----------------------------"
        )

        self.preprocess()
        self.check_constraints()
        self.execute()
        self.postprocess()

        self.logging.debug(
            f"----------------------------finished read API----------------------------"
        )

        return self.results.get_result_code(), self.results.get_results()

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
    def register_arg(name, type, default, choices, help):
        Read.registered_args[name] = {
            "type": type,
            "default": default,
            "choices": choices,
            "help": help,
        }

    @staticmethod
    def generate_subparser(subparsers):
        read_parser = subparsers.add_parser(
            "read", help="read information from flatbuffer binary"
        )
        read_parser.set_defaults(api=Read)

        for name, attributes in Read.registered_args.items():
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
