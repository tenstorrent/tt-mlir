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


def test_location_present_perf():
    # Is there any way to prove if an Op is relevant? Well for now we know that it's MNIST.
    # Since it's an MNIST module, ignore and all of the locations from the `deallocate` op calls.

    # Need to get MLIR module from flatbuffer using ttrt read
    API.initialize_apis()
    reader = API.Read()
    # Load Perf Binary
    reader.load_binaries_from_path(PERF_BINARY_FILE_PATH)
    # Unwrap MLIR from read result
    module_mlir = reader.mlir()[0][0]["ttnn"]

    # Parse MLIR Module and populate locs_to_find with all relevant locations

    from ttmlir.dialects import ttnn, tt
    from ttmlir import ir, util

    with ir.Context() as ctx:
        tt.register_dialect(ctx)
        ttnn.register_dialect(ctx)
        module = ir.Module.parse(module_mlir, ctx)

    locs_to_find = {}

    for op in module.body.operations:
        for region in op.regions:
            for block in region.blocks:
                for op in block.operations:
                    if op.operation.name == "ttnn.deallocate":
                        # Don't register the locations of the deallocate ops
                        continue
                    locs_to_find[util.get_loc_name(op.location)] = True
                    print(op, op.operation.name)
