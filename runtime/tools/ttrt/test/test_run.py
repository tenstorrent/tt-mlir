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


def test_flatbuffer_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_flatbuffer_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


# This test causes a failure in pytest_runtest_teardown when skipped() so I will comment it out
# def test_dir_flatbuffer_run():
#     API.initialize_apis()
#     custom_args = {}
#     custom_args[
#         "--result-file"
#     ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
#     custom_args["binary"] = DIRECTORY_PATH
#     run_instance = API.Run(args=custom_args)
#     run_instance()

# This test causes a failure in pytest_runtest_teardown when skipped() so I will comment it out
# def test_dir_flatbuffer_cmd_run():
#     command = f"ttrt run {DIRECTORY_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
#     sub_process_command(command)


def test_logger_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    log_file_name = "test.log"
    custom_logger = Logger(log_file_name)
    run_instance = API.Run(args=custom_args, logger=custom_logger)
    run_instance()


def test_artifacts_run():
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
    run_instance = API.Run(args=custom_args, artifacts=custom_artifacts)
    run_instance()


def test_clean_artifacts_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_clean_artifacts_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --clean-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_save_artifacts_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_save_artifacts_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --clean-artifacts --save-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_log_file_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--log-file"] = "test.log"
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_log_file_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_artifact_dir_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--clean-artifacts"] = True
    custom_args["--save-artifacts"] = True
    custom_args["--artifact-dir"] = f"{os.getcwd()}/test-artifacts"
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_artifact_dir_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --clean-artifacts --save-artifacts --artifact-dir {os.getcwd()}/test-artifacts --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_program_index_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--program-index"] = "0"
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_program_index_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --program-index 0 --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_loops_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--loops"] = 1
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_loops_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --loops 1 --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_init_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--init"] = "randn"
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_init_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --init randn --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_non_zero_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--non-zero"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_non_zero_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --non-zero --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_rtol_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--rtol"] = 1e-05
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_rtol_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --rtol 1e-05 --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_atol_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--atol"] = 1e-08
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_atol_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --atol 1e-08 --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_seed_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--seed"] = 1
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_seed_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --seed 1 --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_load_kernels_from_disk_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--load-kernels-from-disk"] = True
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_load_kernels_from_disk_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --load-kernels-from-disk --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_enable_async_ttnn_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_enable_async_ttnn_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_benchmark_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--benchmark"] = 0  # Enable benchmark mode without threshold
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_benchmark_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --benchmark --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_benchmark_with_threshold_run():
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args[
        "--benchmark"
    ] = 10000.0  # High threshold in milliseconds that should always pass
    run_instance = API.Run(args=custom_args)
    run_instance()


def test_benchmark_with_threshold_cmd_run():
    command = f"ttrt run {BINARY_FILE_PATH} --benchmark 10000.0 --log-file ttrt-results/{inspect.currentframe().f_code.co_name}.log --result-file ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    sub_process_command(command)


def test_benchmark_threshold_failure():
    """Test that benchmark threshold failure raises an exception"""
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args[
        "--benchmark"
    ] = 0.001  # Very low threshold in milliseconds that should always fail
    run_instance = API.Run(args=custom_args)

    # Expect this to raise an exception due to threshold failure
    with pytest.raises(Exception) as exc_info:
        run_instance()

    # Verify the exception message contains threshold information
    assert "Benchmark threshold check failed" in str(exc_info.value)
    assert "above threshold" in str(exc_info.value)


def test_benchmark_defaults_configuration():
    """Test that benchmark mode sets the correct default values"""
    API.initialize_apis()
    custom_args = {}
    custom_args["binary"] = BINARY_FILE_PATH
    custom_args["--benchmark"] = 0  # Enable benchmark mode without threshold

    run_instance = API.Run(args=custom_args)

    # Verify benchmark mode sets the expected defaults
    assert run_instance["--loops"] == 2
    assert run_instance["--enable-program-cache"] == True
    assert run_instance["--disable-golden"] == True
    assert run_instance["--program-index"] == 0


def test_benchmark_argument_parsing():
    """Test that benchmark argument parsing works correctly for different input types"""
    from ttrt.common.run import Run
    import argparse

    # Clear and re-initialize to ensure clean state
    Run.registered_args = {}
    Run.initialize_api()

    # Create parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    run_parser = Run.generate_subparser(subparsers)

    # Test cases for argument parsing
    test_cases = [
        # (args, expected_benchmark_value, description)
        (["run", "dummy.ttnn"], None, "No benchmark flag should be None"),
        (
            ["run", "--benchmark", "dummy.ttnn"],
            0,
            "Benchmark flag without value should be 0",
        ),
        (
            ["run", "--benchmark", "1000.5", "dummy.ttnn"],
            1000.5,
            "Benchmark flag with float value",
        ),
        (
            ["run", "--benchmark", "1000", "dummy.ttnn"],
            1000.0,
            "Benchmark flag with int value converted to float",
        ),
    ]

    for args, expected, description in test_cases:
        try:
            parsed_args = parser.parse_args(args)
            actual = getattr(parsed_args, "benchmark", None)
            assert (
                actual == expected
            ), f"{description}: expected {expected}, got {actual}"
        except Exception as e:
            pytest.fail(f"{description}: Failed with error: {e}")


def test_benchmark_backward_compatibility():
    """Test that existing benchmark usage still works"""
    API.initialize_apis()
    custom_args = {}
    custom_args[
        "--result-file"
    ] = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    custom_args["binary"] = BINARY_FILE_PATH
    # Test with None (disabled), 0 (enabled without threshold), and threshold values

    for benchmark_value in [None, 0, 10000.0]:
        if benchmark_value is not None:
            custom_args["--benchmark"] = benchmark_value
        else:
            custom_args.pop("--benchmark", None)

        run_instance = API.Run(args=custom_args)

        # Should not raise an exception for any of these values
        if benchmark_value is None:
            # Benchmark mode should be disabled
            assert run_instance["--benchmark"] is None
        else:
            # Benchmark mode should be enabled
            assert run_instance["--benchmark"] == benchmark_value
            # Should have benchmark defaults set
            assert run_instance["--loops"] == 2
            assert run_instance["--enable-program-cache"] == True
            assert run_instance["--disable-golden"] == True
            assert run_instance["--program-index"] == 0
