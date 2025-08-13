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
import pytest
import inspect

import ttrt
from ttrt.common.util import *
from ttrt.common.api import API

from util import *


def test_flatbuffer_read():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    read_instance = API.Read(args=custom_args)
    read_instance()


def test_flatbuffer_cmd_read():
    command = f"ttrt read {BINARY_FILE_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_dir_flatbuffer_read():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = DIRECTORY_PATH
    read_instance = API.Read(args=custom_args)
    read_instance()


def test_dir_flatbuffer_cmd_read():
    command = f"ttrt read {DIRECTORY_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_logger_read():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    read_instance = API.Read(args=custom_args, logger=custom_logger)
    read_instance()


def test_logger_cmd_read():
    command = f"ttrt read {BINARY_FILE_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_artifacts_read():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    artifacts_folder_path = f"{os.getcwd()}/test-artifacts"
    custom_artifacts = Artifacts(
        logger=custom_logger, artifacts_folder_path=artifacts_folder_path
    )
    read_instance = API.Read(args=custom_args, artifacts=custom_artifacts)
    read_instance()


def test_artifacts_cmd_read():
    command = f"ttrt read {BINARY_FILE_PATH} --artifact-dir {os.getcwd()}/test-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_clean_artifacts_read():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    read_instance = API.Read(args=custom_args)
    read_instance()


def test_clean_artifacts_cmd_read():
    command = f"ttrt read {BINARY_FILE_PATH} --clean-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_save_artifacts_read():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    read_instance = API.Read(args=custom_args)
    read_instance()


def test_save_artifacts_cmd_read():
    command = f"ttrt read {BINARY_FILE_PATH} --clean-artifacts --save-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_log_file_read():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--log-file"] = "test.log"
    read_instance = API.Read(args=custom_args)
    read_instance()


def test_artifact_dir_read():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    custom_args["--artifact-dir"] = f"{os.getcwd()}/test-artifacts"
    read_instance = API.Read(args=custom_args)
    read_instance()


def test_section_read():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    sections = [
        "all",
        "version",
        "system_desc",
        "mlir",
        "inputs",
        "outputs",
        "op_stats",
    ]
    for section in sections:
        custom_args["--section"] = section
        read_instance = API.Read(args=custom_args)
        read_instance()


def test_section_cmd_read():
    command = f"ttrt read {BINARY_FILE_PATH} --section all --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)
