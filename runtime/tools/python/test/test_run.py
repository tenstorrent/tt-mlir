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
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_flatbuffer.__name__}"


def test_flatbuffer_cmd():
    command = (
        f"ttrt run {BINARY_FILE_PATH} --log-file {test_flatbuffer_cmd.__name__}_run.log"
    )
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_flatbuffer_cmd.__name__}"


def test_dir_flatbuffer():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = DIRECTORY_PATH
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_dir_flatbuffer.__name__}"


def test_dir_flatbuffer_cmd():
    command = f"ttrt run {DIRECTORY_PATH} --log-file {test_dir_flatbuffer_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_dir_flatbuffer_cmd.__name__}"


def test_logger():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    run_instance = API.Run(args=custom_args, logger=custom_logger)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_logger.__name__}"


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
    run_instance = API.Run(args=custom_args, artifacts=custom_artifacts)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_artifacts.__name__}"


def test_clean_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_clean_artifacts.__name__}"


def test_clean_artifacts_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --clean-artifacts --log-file {test_clean_artifacts_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_clean_artifacts_cmd.__name__}"


def test_save_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_save_artifacts.__name__}"


def test_save_artifacts_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --clean-artifacts --save-artifacts --log-file {test_save_artifacts_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_save_artifacts_cmd.__name__}"


def test_log_file():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--log-file"] = "test.log"
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_log_file.__name__}"


def test_log_file_cmd():
    command = (
        f"ttrt run {BINARY_FILE_PATH} --log-file {test_log_file_cmd.__name__}_run.log"
    )
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_log_file_cmd.__name__}"


def test_artifact_dir():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    custom_args["--artifact-dir"] = f"{os.getcwd()}/test-artifacts"
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_artifact_dir.__name__}"


def test_artifact_dir_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --clean-artifacts --save-artifacts --artifact-dir {os.getcwd()}/test-artifacts --log-file {test_artifact_dir_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_artifact_dir_cmd.__name__}"


def test_program_index():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--program-index"] = "0"
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_program_index.__name__}"


def test_program_index_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --program-index 0 --log-file {test_program_index_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_program_index_cmd.__name__}"


def test_loops():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--loops"] = 1
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_loops.__name__}"


def test_loops_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --loops 1 --log-file {test_loops_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_loops_cmd.__name__}"


def test_init():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--init"] = "randn"
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_init.__name__}"


def test_init_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --init randn --log-file {test_init_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_init_cmd.__name__}"


@pytest.mark.skip
def test_identity():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--identity"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_identity.__name__}"


@pytest.mark.skip
def test_identity_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --identity --log-file {test_identity_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_identity_cmd.__name__}"


def test_non_zero():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--non-zero"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_non_zero.__name__}"


def test_non_zero_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --non-zero --log-file {test_non_zero_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_non_zero_cmd.__name__}"


def test_rtol():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--rtol"] = 1e-05
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_rtol.__name__}"


def test_rtol_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --rtol 1e-05 --log-file {test_rtol_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_rtol_cmd.__name__}"


def test_atol():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--atol"] = 1e-08
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_atol.__name__}"


def test_atol_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --atol 1e-08 --log-file {test_atol_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_atol_cmd.__name__}"


def test_seed():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--seed"] = 1
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_seed.__name__}"


def test_seed_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --seed 1 --log-file {test_seed_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_seed_cmd.__name__}"


def test_load_kernels_from_disk():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--load-kernels-from-disk"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_load_kernels_from_disk.__name__}"


def test_load_kernels_from_disk_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --load-kernels-from-disk --log-file {test_load_kernels_from_disk_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_load_kernels_from_disk_cmd.__name__}"


def test_enable_async_ttnn():
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--enable-async-ttnn"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_enable_async_ttnn.__name__}"


def test_enable_async_ttnn_cmd():
    command = f"ttrt run {BINARY_FILE_PATH} --enable-async-ttnn --log-file {test_enable_async_ttnn_cmd.__name__}_run.log"
    sub_process_command(command)

    assert (
        check_results("run_results.json") == 0
    ), f"one of more tests failed in={test_enable_async_ttnn_cmd.__name__}"
