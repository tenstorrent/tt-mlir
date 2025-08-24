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
import difflib

from ttrt.common.util import *
from ttrt.common.query import Query


class Check:
    registered_args = {}

    @staticmethod
    def initialize_api():
        Check.register_arg(
            name="--clean-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="clean all artifacts from previous runs",
        )
        Check.register_arg(
            name="--save-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="save all artifacts during run",
        )
        Check.register_arg(
            name="--log-file",
            type=str,
            default="",
            choices=None,
            help="log file to dump ttrt output to",
        )
        Check.register_arg(
            name="--artifact-dir",
            type=str,
            default=f"{os.getcwd()}/ttrt-artifacts",
            choices=None,
            help="provides a directory path to save artifacts to",
        )
        Check.register_arg(
            name="--system-desc",
            type=str,
            default="",
            choices=None,
            help="system desc to check against",
        )
        Check.register_arg(
            name="--result-file",
            type=str,
            default="check_results.json",
            choices=None,
            help="test file to save results to",
        )
        Check.register_arg(
            name="binary",
            type=str,
            default="",
            choices=None,
            help="flatbuffer binary file",
        )

    def __init__(self, args={}, logger=None, artifacts=None):
        for name, attributes in Check.registered_args.items():
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
        self.query = Query({"--quiet": True}, self.logger, self.artifacts)
        self.ttnn_binaries = []
        self.ttmetal_binaries = []
        self.system_desc_binaries = []
        self.results = Results(self.logger, self.file_manager)

    def preprocess(self):
        self.logging.debug(f"------preprocessing check API")

        if self["--clean-artifacts"]:
            self.artifacts.clean_artifacts()

        if self["--save-artifacts"]:
            self.artifacts.create_artifacts()

        self.logging.debug(f"------finished preprocessing check API")

    def check_constraints(self):
        self.logging.debug(f"------checking constraints for check API")

        ttsys_binary_paths = self.file_manager.find_ttsys_binary_paths(
            self["--system-desc"]
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
                self.logging.warning(f"test={path} was skipped with exception={str(e)}")
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
                self.logging.warning(f"test={path} was skipped with exception={str(e)}")
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
                self.logging.warning(f"test={path} was skipped with exception={str(e)}")
                self.results.add_result(test_result)

        self.logging.debug(f"------finished checking constraints for check API")

    def execute(self):
        self.logging.debug(f"------executing check API")

        def _execute(binaries):
            if len(binaries) == 0:
                self.logging.warning(f"no binaries found to run - returning early")
                return

            for bin in binaries:
                test_result = {
                    "file_path": bin.file_path,
                    "exception": "",
                    "log_file": self.logger.file_name,
                    "artifacts": self.artifacts.artifacts_folder_path,
                }

                if not self.system_desc_binaries:
                    self.logging.warning(
                        "no system descriptor file provided - querying from host machine"
                    )
                    try:
                        self.query()
                        bin.check_system_desc(self.query)
                    except Exception as e:
                        self.logging.info(str(e))
                        test_result["result"] = "error"
                        test_result["exception"] = str(e)
                        test_result["system_desc"] = "system_desc queried from device"
                    else:
                        self.logging.info(
                            f"system desc for device matched flatbuffer: {bin.file_path}"
                        )
                        test_result["result"] = "pass"
                        test_result["system_desc"] = "system_desc queried from device"
                else:
                    for desc in self.system_desc_binaries:
                        if bin.system_desc_dict != desc.system_desc_dict["system_desc"]:
                            self.logging.info(
                                f"system desc for: {desc.file_path} did not match flatbuffer: {bin.file_path}"
                            )
                            test_result["result"] = "error"
                            test_result[
                                "exception"
                            ] = f"system desc for: {desc.file_path} did not match flatbuffer: {bin.file_path}"
                            test_result["system_desc"] = f"{desc.file_path}"
                        else:
                            self.logging.info(
                                f"system desc for: {desc.file_path} matched flatbuffer: {bin.file_path}"
                            )
                            test_result["result"] = "pass"
                            test_result["system_desc"] = f"{desc.file_path}"

                self.results.add_result(test_result)

        self.logging.debug(f"executing ttnn binaries")
        _execute(self.ttnn_binaries)
        self.logging.debug(f"finished executing ttnn binaries")

        self.logging.debug(f"executing ttmetal binaries")
        _execute(self.ttmetal_binaries)
        self.logging.debug(f"finished executing ttmetal binaries")

        self.logging.debug(f"------finished executing check API")

    def postprocess(self):
        self.logging.debug(f"------postprocessing check API")

        if self["--save-artifacts"]:
            for bin in self.ttnn_binaries:
                self.artifacts.save_binary(bin)

            for bin in self.ttmetal_binaries:
                self.artifacts.save_binary(bin)

        self.results.save_results(self["--result-file"])

        self.logging.debug(f"------finished postprocessing check API")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __call__(self):
        self.logging.debug(
            f"----------------------------starting check API----------------------------"
        )

        self.preprocess()
        self.check_constraints()
        self.execute()
        self.postprocess()

        self.logging.debug(
            f"----------------------------finished check API----------------------------"
        )

        return self.results.get_result_code(), self.results.get_results()

    @staticmethod
    def register_arg(name, type, default, choices, help):
        Check.registered_args[name] = {
            "type": type,
            "default": default,
            "choices": choices,
            "help": help,
        }

    @staticmethod
    def generate_subparser(subparsers):
        check_parser = subparsers.add_parser(
            "check", help="check a flatbuffer binary against a system desc file"
        )
        check_parser.set_defaults(api=Check)

        for name, attributes in Check.registered_args.items():
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
