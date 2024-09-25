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


def test_clean_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
    custom_args["--clean-artifacts"] = True
    query_instance = API.Query(args=custom_args)
    query_instance()

    assert (
        check_results(
            f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
        )
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_clean_artifacts_cmd():
    command = f"ttrt query --clean-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_query.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
    sub_process_command(command)

    assert (
        check_results(
            f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
        )
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_save_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    query_instance = API.Query(args=custom_args)
    query_instance()

    assert (
        check_results(
            f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
        )
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_save_artifacts_cmd():
    command = f"ttrt query --clean-artifacts --save-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_query.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
    sub_process_command(command)

    assert (
        check_results(
            f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
        )
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_log_file():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
    custom_args["--log-file"] = "test.log"
    query_instance = API.Query(args=custom_args)
    query_instance()

    assert (
        check_results(
            f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
        )
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_log_file_cmd():
    command = f"ttrt query --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_query.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
    sub_process_command(command)

    assert (
        check_results(
            f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
        )
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_artifact_dir():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    custom_args["--artifact-dir"] = f"{os.getcwd()}/test-artifacts"
    query_instance = API.Query(args=custom_args)
    query_instance()

    assert (
        check_results(
            f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
        )
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_artifact_dir_cmd():
    command = f"ttrt query --clean-artifacts --save-artifacts --artifact-dir {os.getcwd()}/test-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_query.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
    sub_process_command(command)

    assert (
        check_results(
            f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
        )
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_logger():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    query_instance = API.Query(args=custom_args, logger=custom_logger)
    query_instance()

    assert (
        check_results(
            f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
        )
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_artifacts():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    artifacts_folder_path = f"{os.getcwd()}/test-artifacts"
    custom_artifacts = Artifacts(
        logger=custom_logger, artifacts_folder_path=artifacts_folder_path
    )
    query_instance = API.Query(args=custom_args, artifacts=custom_artifacts)
    query_instance()

    assert (
        check_results(
            f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
        )
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_quiet():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
    custom_args["--quiet"] = True
    query_instance = API.Query(args=custom_args)
    query_instance()

    assert (
        check_results(
            f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
        )
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"


def test_quiet_cmd():
    command = f"ttrt query --quiet --log-file ttrt-results/{inspect.currentframe().f_code.co_name}_query.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
    sub_process_command(command)

    assert (
        check_results(
            f"ttrt-results/{inspect.currentframe().f_code.co_name}_query.json"
        )
        == 0
    ), f"one of more tests failed in={inspect.currentframe().f_code.co_name}"
