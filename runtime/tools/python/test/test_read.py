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

import ttrt
from ttrt.common.util import *
from ttrt.common.api import API

from util import *


def test_flatbuffer():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    read_instance = API.Read(args=custom_args)
    read_instance()

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_flatbuffer.__name__}"


def test_flatbuffer_cmd():
    command = f"ttrt read {BINARY_FILE_PATH} --log-file {test_flatbuffer_cmd.__name__}_read.log"
    sub_process_command(command)

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_flatbuffer_cmd.__name__}"


def test_dir_flatbuffer():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = DIRECTORY_PATH
    read_instance = API.Read(args=custom_args)
    read_instance()

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_dir_flatbuffer.__name__}"


def test_dir_flatbuffer_cmd():
    command = f"ttrt read {DIRECTORY_PATH} --log-file {test_dir_flatbuffer_cmd.__name__}_read.log"
    sub_process_command(command)

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_dir_flatbuffer_cmd.__name__}"


def test_logger():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    read_instance = API.Read(args=custom_args, logger=custom_logger)
    read_instance()

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_logger.__name__}"


def test_logger_cmd():
    command = (
        f"ttrt read {BINARY_FILE_PATH} --log-file {test_logger_cmd.__name__}_read.log"
    )
    sub_process_command(command)

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_logger_cmd.__name__}"


def test_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    artifacts_folder_path = f"{os.getcwd()}/test-artifacts"
    custom_artifacts = Artifacts(
        logger=custom_logger, artifacts_folder_path=artifacts_folder_path
    )
    read_instance = API.Read(args=custom_args, artifacts=custom_artifacts)
    read_instance()

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_artifacts.__name__}"


def test_artifacts_cmd():
    command = f"ttrt read {BINARY_FILE_PATH} --artifact-dir {os.getcwd()}/test-artifacts --log-file {test_artifacts_cmd.__name__}_read.log"
    sub_process_command(command)

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_artifacts_cmd.__name__}"


def test_clean_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    read_instance = API.Read(args=custom_args)
    read_instance()

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_clean_artifacts.__name__}"


def test_clean_artifacts_cmd():
    command = f"ttrt read {BINARY_FILE_PATH} --clean-artifacts --log-file {test_clean_artifacts_cmd.__name__}_read.log"
    sub_process_command(command)

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_clean_artifacts_cmd.__name__}"


def test_save_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    read_instance = API.Read(args=custom_args)
    read_instance()

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_save_artifacts.__name__}"


def test_save_artifacts_cmd():
    command = f"ttrt read {BINARY_FILE_PATH} --clean-artifacts --save-artifacts --log-file {test_save_artifacts_cmd.__name__}_read.log"
    sub_process_command(command)

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_save_artifacts_cmd.__name__}"


def test_log_file():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--log-file"] = "test.log"
    read_instance = API.Read(args=custom_args)
    read_instance()

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_log_file.__name__}"


def test_artifact_dir():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    custom_args["--artifact-dir"] = f"{os.getcwd()}/test-artifacts"
    read_instance = API.Read(args=custom_args)
    read_instance()

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_artifact_dir.__name__}"


def test_section():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--section"] = "all"
    read_instance = API.Read(args=custom_args)
    read_instance()

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_section.__name__}"


def test_section_cmd():
    command = f"ttrt read {BINARY_FILE_PATH} --section mlir --log-file {test_section_cmd.__name__}_read.log"
    sub_process_command(command)

    assert (
        check_results("read_results.json") == 0
    ), f"one of more tests failed in={test_section_cmd.__name__}"
