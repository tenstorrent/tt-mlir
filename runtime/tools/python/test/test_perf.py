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


def test_flatbuffer():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--host-only"] = True
    perf_instance = API.Perf(args=custom_args)
    perf_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_flatbuffer_cmd():
    command = f"ttrt perf {BINARY_FILE_PATH} --host-only --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_dir_flatbuffer():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    custom_args["binary"] = DIRECTORY_PATH
    custom_args["--host-only"] = True
    perf_instance = API.Perf(args=custom_args)
    perf_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_dir_flatbuffer_cmd():
    command = f"ttrt perf {DIRECTORY_PATH} --host-only --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_logger():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--host-only"] = True
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    perf_instance = API.Perf(args=custom_args, logger=custom_logger)
    perf_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--host-only"] = True
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    artifacts_folder_path = f"{os.getcwd()}/test-artifacts"
    custom_artifacts = Artifacts(
        logger=custom_logger, artifacts_folder_path=artifacts_folder_path
    )
    perf_instance = API.Perf(args=custom_args, artifacts=custom_artifacts)
    perf_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_clean_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--host-only"] = True
    custom_args["--clean-artifacts"] = True
    perf_instance = API.Perf(args=custom_args)
    perf_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_clean_artifacts_cmd():
    command = f"ttrt perf {BINARY_FILE_PATH} --host-only --clean-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_save_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--host-only"] = True
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    perf_instance = API.Perf(args=custom_args)
    perf_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_save_artifacts_cmd():
    command = f"ttrt perf {BINARY_FILE_PATH} --host-only --clean-artifacts --save-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_log_file():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--host-only"] = True
    custom_args["--log-file"] = "test.log"
    perf_instance = API.Perf(args=custom_args)
    perf_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_log_file_cmd():
    command = f"ttrt perf {BINARY_FILE_PATH} --host-only --log-file ttrt-results/test.log --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_artifact_dir():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--host-only"] = True
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    custom_args["--artifact-dir"] = f"{os.getcwd()}/test-artifacts"
    perf_instance = API.Perf(args=custom_args)
    perf_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_artifact_dir_cmd():
    command = f"ttrt perf {BINARY_FILE_PATH} --host-only --clean-artifacts --save-artifacts --artifact-dir {os.getcwd()}/test-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_program_index():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--host-only"] = True
    custom_args["--program-index"] = "0"
    perf_instance = API.Perf(args=custom_args)
    perf_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_program_index_cmd():
    command = f"ttrt perf {BINARY_FILE_PATH} --host-only --program-index 0 --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_loops():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--host-only"] = True
    custom_args["--loops"] = 1
    perf_instance = API.Perf(args=custom_args)
    perf_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_loops_cmd():
    command = f"ttrt perf {BINARY_FILE_PATH} --host-only --loops 1 --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_device():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    custom_args["binary"] = BINARY_FILE_PATH
    perf_instance = API.Perf(args=custom_args)
    perf_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_device_cmd():
    command = f"ttrt perf {BINARY_FILE_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_perf.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"
