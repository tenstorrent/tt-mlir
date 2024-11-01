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


def test_clean_artifacts_check():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    check_instance = API.Check(args=custom_args)
    check_instance()


def test_clean_artifacts_cmd_check():
    command = f"ttrt check {BINARY_FILE_PATH} --clean-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_save_artifacts_check():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    check_instance = API.Check(args=custom_args)
    check_instance()


def test_save_artifacts_cmd_check():
    command = f"ttrt check {BINARY_FILE_PATH} --clean-artifacts --save-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_flatbuffer_check():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    check_instance = API.Check(args=custom_args)
    check_instance()


def test_flatbuffer_cmd_check():
    command = f"ttrt check {BINARY_FILE_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_dir_flatbuffer_check():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = DIRECTORY_PATH
    check_instance = API.Check(args=custom_args)
    check_instance()


def test_dir_flatbuffer_cmd_check():
    command = f"ttrt check {DIRECTORY_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_system_desc_check():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--system-desc"] = SYSTEM_DESC_FILE_PATH
    check_instance = API.Check(args=custom_args)
    check_instance()


def test_system_desc_cmd_check():
    command = f"ttrt check {BINARY_FILE_PATH} --system-desc {SYSTEM_DESC_FILE_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_dir_system_descs_check():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--system-desc"] = SYSTEM_DESC_DIRECTORY_PATH
    check_instance = API.Check(args=custom_args)
    check_instance()


def test_dir_system_descs_cmd_check():
    command = f"ttrt check {BINARY_FILE_PATH} --system-desc {SYSTEM_DESC_DIRECTORY_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_logger_check():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    check_instance = API.Check(args=custom_args, logger=custom_logger)
    check_instance()


def test_artifacts_check():
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
    check_instance = API.Check(args=custom_args, artifacts=custom_artifacts)
    check_instance()


def test_log_file_check():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--log-file"] = "test.log"
    check_instance = API.Check(args=custom_args)
    check_instance()


def test_log_file_cmd_check():
    command = f"ttrt check {BINARY_FILE_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_artifact_dir_check():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    custom_args["--artifact-dir"] = f"{os.getcwd()}/test-artifacts"
    check_instance = API.Check(args=custom_args)
    check_instance()


def test_artifact_dir_cmd_check():
    command = f"ttrt check {BINARY_FILE_PATH} --clean-artifacts --save-artifacts --artifact-dir {os.getcwd()}/test-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)
