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


def test_flatbuffer_perf():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = PERF_BINARY_FILE_PATH
    custom_args["--host-only"] = True
    perf_instance = API.Perf(args=custom_args)
    perf_instance()


def test_flatbuffer_cmd_perf():
    command = f"ttrt perf {PERF_BINARY_FILE_PATH} --host-only --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_logger_perf():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = PERF_BINARY_FILE_PATH
    custom_args["--host-only"] = True
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    perf_instance = API.Perf(args=custom_args, logger=custom_logger)
    perf_instance()


def test_clean_artifacts_perf():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = PERF_BINARY_FILE_PATH
    custom_args["--host-only"] = True
    custom_args["--clean-artifacts"] = True
    perf_instance = API.Perf(args=custom_args)
    perf_instance()


def test_clean_artifacts_cmd_perf():
    command = f"ttrt perf {PERF_BINARY_FILE_PATH} --clean-artifacts --host-only --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_log_file_perf():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = PERF_BINARY_FILE_PATH
    custom_args["--host-only"] = True
    custom_args["--log-file"] = "test.log"
    perf_instance = API.Perf(args=custom_args)
    perf_instance()


def test_log_file_cmd_perf():
    command = f"ttrt perf {PERF_BINARY_FILE_PATH} --host-only --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_program_index_perf():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = PERF_BINARY_FILE_PATH
    custom_args["--host-only"] = True
    custom_args["--program-index"] = "0"
    perf_instance = API.Perf(args=custom_args)
    perf_instance()


def test_program_index_cmd_perf():
    command = f"ttrt perf {PERF_BINARY_FILE_PATH} --program-index 0 --host-only --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_loops_perf():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = PERF_BINARY_FILE_PATH
    custom_args["--host-only"] = True
    custom_args["--loops"] = 1
    perf_instance = API.Perf(args=custom_args)
    perf_instance()


def test_loops_cmd_perf():
    command = f"ttrt perf {PERF_BINARY_FILE_PATH} --loops 1 --host-only --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_device_perf():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = PERF_BINARY_FILE_PATH
    perf_instance = API.Perf(args=custom_args)
    perf_instance()


def test_device_cmd_perf():
    command = f"ttrt perf {PERF_BINARY_FILE_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)
