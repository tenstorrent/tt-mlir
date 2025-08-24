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


class Query:
    registered_args = {}

    @staticmethod
    def initialize_api():
        Query.register_arg(
            name="--clean-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="clean all artifacts from previous runs",
        )
        Query.register_arg(
            name="--save-artifacts",
            type=bool,
            default=False,
            choices=[True, False],
            help="save all artifacts during run",
        )
        Query.register_arg(
            name="--log-file",
            type=str,
            default="",
            choices=None,
            help="log file to dump ttrt output to",
        )
        Query.register_arg(
            name="--artifact-dir",
            type=str,
            default=f"{os.getcwd()}/ttrt-artifacts",
            choices=None,
            help="provides a directory path to save artifacts to",
        )
        Query.register_arg(
            name="--quiet",
            type=bool,
            default=False,
            choices=[True, False],
            help="suppress system desc from being printed",
        )
        Query.register_arg(
            name="--disable-eth-dispatch",
            type=bool,
            default=False,
            choices=[True, False],
            help="disable putting dispatch on ethernet cores - place it on worker cores instead",
        )
        Query.register_arg(
            name="--result-file",
            type=str,
            default="query_results.json",
            choices=None,
            help="test file to save results to",
        )

    def __init__(self, args={}, logger=None, artifacts=None):
        for name, attributes in Query.registered_args.items():
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
        self.system_desc = None
        self.device_ids = None
        self.results = Results(self.logger, self.file_manager)
        self.test_result = "pass"

    def preprocess(self):
        self.logging.debug(f"------preprocessing query API")

        if self["--clean-artifacts"]:
            self.artifacts.clean_artifacts()

        if self["--save-artifacts"]:
            self.artifacts.create_artifacts()

        self.logging.debug(f"------finished preprocessing query API")

    def check_constraints(self):
        self.logging.debug(f"------checking constraints for query API")
        self.logging.debug(f"------finished constraints for query API")

    def execute(self):
        self.logging.debug(f"------executing query API")

        import ttrt.runtime

        try:
            dispatch_core_type = ttrt.runtime.DispatchCoreType.ETH

            if self["--disable-eth-dispatch"]:
                dispatch_core_type = ttrt.runtime.DispatchCoreType.WORKER

            self.logging.debug(f"getting system descriptor")
            self.system_desc = ttrt.runtime.get_current_system_desc(dispatch_core_type)
            self.device_ids = list(range(ttrt.runtime.get_num_available_devices()))

            if not self["--quiet"]:
                self.logging.info(self.system_desc.as_json())
        except Exception as e:
            test_result = {
                "result": "error",
                "exception": str(e),
                "log_file": self.logger.file_name,
                "artifacts": self.artifacts.artifacts_folder_path,
            }
            issue_msg = "See issue https://github.com/tenstorrent/tt-metal/issues/23600"
            self.logging.error(
                f"ERROR: getting system_desc failed\n{issue_msg}\nException: {str(e)}"
            )
            self.results.add_result(test_result)
            self.test_result = "error"

        self.logging.debug(f"------finished executing query API")

    def postprocess(self):
        self.logging.debug(f"------postprocessing query API")

        if self["--save-artifacts"]:
            self.artifacts.save_system_desc(self.system_desc)

        if self.test_result == "pass":
            test_result = {
                "result": "pass",
                "exception": "",
                "log_file": self.logger.file_name,
                "artifacts": self.artifacts.artifacts_folder_path,
            }
            self.results.add_result(test_result)
            self.logging.info(f"PASS: getting system_desc passed")
        else:
            self.logging.error(f"FAIL: getting system_desc failed")

        self.results.save_results(self["--result-file"])

        self.logging.debug(f"------finished postprocessing query API")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __call__(self):
        self.logging.debug(
            f"----------------------------starting query API----------------------------"
        )

        self.preprocess()
        self.check_constraints()
        self.execute()
        self.postprocess()

        self.logging.debug(
            f"----------------------------finished query API----------------------------"
        )

        return self.results.get_result_code(), self.results.get_results()

    def get_system_desc_as_dict(self):
        return json.loads(self.system_desc.as_json())

    @staticmethod
    def register_arg(name, type, default, choices, help):
        Query.registered_args[name] = {
            "type": type,
            "default": default,
            "choices": choices,
            "help": help,
        }

    @staticmethod
    def generate_subparser(subparsers):
        query_parser = subparsers.add_parser(
            "query", help="query information about the current system"
        )
        query_parser.set_defaults(api=Query)

        for name, attributes in Query.registered_args.items():
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
