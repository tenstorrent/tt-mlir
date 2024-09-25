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
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_flatbuffer_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_dir_flatbuffer():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = DIRECTORY_PATH
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_dir_flatbuffer_cmd():
    command = f"ttrt run {DIRECTORY_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_logger():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    run_instance = API.Run(args=custom_args, logger=custom_logger)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    artifacts_folder_path = f"{os.getcwd()}/test-artifacts"
    custom_artifacts = Artifacts(
        logger=custom_logger, artifacts_folder_path=artifacts_folder_path
    )
    run_instance = API.Run(args=custom_args, artifacts=custom_artifacts)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_clean_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_clean_artifacts_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --clean-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_save_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_save_artifacts_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --clean-artifacts --save-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_log_file():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--log-file"] = "test.log"
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_log_file_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_artifact_dir():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    custom_args["--artifact-dir"] = f"{os.getcwd()}/test-artifacts"
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_artifact_dir_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --clean-artifacts --save-artifacts --artifact-dir {os.getcwd()}/test-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_program_index():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--program-index"] = "0"
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_program_index_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --program-index 0 --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_loops():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--loops"] = 1
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_loops_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --loops 1 --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_init():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--init"] = "randn"
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_init_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --init randn --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


@pytest.mark.skip
def test_identity():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--identity"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


@pytest.mark.skip
def test_identity_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --identity --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_non_zero():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--non-zero"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_non_zero_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --non-zero --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_rtol():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--rtol"] = 1e-05
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_rtol_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --rtol 1e-05 --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_atol():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--atol"] = 1e-08
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_atol_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --atol 1e-08 --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_seed():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--seed"] = 1
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_seed_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --seed 1 --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_load_kernels_from_disk():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--load-kernels-from-disk"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_load_kernels_from_disk_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --load-kernels-from-disk --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_enable_async_ttnn():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--enable-async-ttnn"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_enable_async_ttnn_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --enable-async-ttnn --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_run.json"
    sub_process_command(command)

    assert (
        check_results(f"ttrt-results/{inspect.currentframe().f_code.co_name}_run.json")
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"
