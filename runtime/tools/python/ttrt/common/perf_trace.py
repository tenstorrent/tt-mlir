# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttrt.binary
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
import sys
import shutil

from ttrt.common.util import *
import tracy
import tracy_state
from tt_metal.tools.profiler.process_ops_logs import process_ops

from tt_metal.tools.profiler.common import (
    PROFILER_LOGS_DIR,
    TRACY_FILE_NAME,
    TRACY_OPS_TIMES_FILE_NAME,
    TRACY_OPS_DATA_FILE_NAME,
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_HOST_DEVICE_SYNC_INFO,
)

#######################################################################################
######################################**GLOBALS**######################################
#######################################################################################
TRACY_CAPTURE_TOOL = f"{TT_MLIR_HOME}/third_party/tt-metal/src/tt-metal-build/tools/profiler/bin/capture-release"
TRACY_CSVEXPROT_TOOL = f"{TT_MLIR_HOME}/third_party/tt-metal/src/tt-metal-build/tools/profiler/bin/csvexport-release"

#######################################################################################
#####################################**PERF-UTILS**####################################
#######################################################################################
def get_available_port():
    ip = socket.gethostbyname(socket.gethostname())

    for port in range(8086, 8500):
        try:
            serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serv.bind((ip, port))
            return str(port)
        except PermissionError as e:
            pass
        except OSError as e:
            pass
    return None


def run_perf_artifact_setup(port):
    subprocess.run(
        f"rm -rf {PROFILER_LOGS_DIR}; mkdir -p {PROFILER_LOGS_DIR}",
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("Verifying tracy profiling tools")
    check_file_exists(TRACY_CAPTURE_TOOL)
    check_file_exists(TRACY_CSVEXPROT_TOOL)

    captureProcess = None
    captureCommand = (
        f"{TRACY_CAPTURE_TOOL} -o {PROFILER_LOGS_DIR / TRACY_FILE_NAME} -f -p {port}"
    )
    print(f"Capture command: {captureCommand}")
    captureProcess = subprocess.Popen(captureCommand, shell=True)

    return captureProcess


def generate_report(binary_perf_folder):
    child_calls = ["CompileProgram", "HWCommandQueue_write_buffer"]
    timeOut = 15
    timeCount = 0
    while not os.path.exists(PROFILER_LOGS_DIR / TRACY_FILE_NAME):
        print(f"tracy capture out not found, will try again in 1 second")
        if timeCount > timeOut:
            print(
                f"tracy capture output file {PROFILER_LOGS_DIR / TRACY_FILE_NAME} was not generated"
            )
            sys.exit(1)
        timeCount += 1
        time.sleep(1)

    with open(PROFILER_LOGS_DIR / TRACY_OPS_TIMES_FILE_NAME, "w") as csvFile:
        child_call_str = f"-x {','.join(child_calls)}"
        subprocess.run(
            f"{TRACY_CSVEXPROT_TOOL} -u -p TT_DNN {child_call_str} {PROFILER_LOGS_DIR / TRACY_FILE_NAME}",
            shell=True,
            check=True,
            stdout=csvFile,
            stderr=subprocess.DEVNULL,
        )

    print(
        f"Host side ops time report generated at {PROFILER_LOGS_DIR / TRACY_OPS_TIMES_FILE_NAME}"
    )

    with open(PROFILER_LOGS_DIR / TRACY_OPS_DATA_FILE_NAME, "w") as csvFile:
        subprocess.run(
            f'{TRACY_CSVEXPROT_TOOL} -m -s ";" {PROFILER_LOGS_DIR / TRACY_FILE_NAME}',
            shell=True,
            check=True,
            stdout=csvFile,
            stderr=subprocess.DEVNULL,
        )

    print(
        f"Host side ops data report generated at {PROFILER_LOGS_DIR / TRACY_OPS_DATA_FILE_NAME}"
    )
    process_ops(binary_perf_folder, "", True)


def generate_params_dict(perf_csv_file):
    # TAPS: TODO - need this support for UIDs
    import pandas as pd
    import json

    # Load the CSV file into a DataFrame
    df = pd.read_csv(perf_csv_file)

    # Convert each row to a JSON string
    json_rows = df.apply(lambda row: row.to_json(), axis=1).tolist()

    # Print or process the JSON strings
    for json_str in json_rows:
        print(json_str)


def save_perf_artifacts(perf_folder):
    profiler_device_side_log_file = (
        f"{TT_METAL_HOME}/generated/profiler/.logs/{PROFILER_DEVICE_SIDE_LOG}"
    )
    profiler_host_device_sync_info_file = (
        f"{TT_METAL_HOME}/generated/profiler/.logs/{PROFILER_HOST_DEVICE_SYNC_INFO}"
    )
    profiler_log_location_record_file = (
        f"{TT_METAL_HOME}/generated/profiler/.logs/.locations.log"
    )
    tracy_ops_times_file = (
        f"{TT_METAL_HOME}/generated/profiler/.logs/{TRACY_OPS_TIMES_FILE_NAME}"
    )
    tracy_ops_data_file = (
        f"{TT_METAL_HOME}/generated/profiler/.logs/{TRACY_OPS_DATA_FILE_NAME}"
    )
    tracy_file = f"{TT_METAL_HOME}/generated/profiler/.logs/{TRACY_FILE_NAME}"

    try:
        check_file_exists(tracy_ops_times_file)
        check_file_exists(tracy_ops_data_file)
        check_file_exists(tracy_file)

        shutil.copy(
            tracy_ops_times_file, os.path.join(perf_folder, "tracy_ops_times_file.csv")
        )
        print(f"File '{tracy_ops_times_file}' copied to '{perf_folder}' successfully.")

        shutil.copy(
            tracy_ops_data_file, os.path.join(perf_folder, "tracy_ops_times_file.csv")
        )
        print(f"File '{tracy_ops_data_file}' copied to '{perf_folder}' successfully.")

        shutil.copy(tracy_file, os.path.join(perf_folder, "tracy_file.tracy"))
        print(f"File '{tracy_file}' copied to '{perf_folder}' successfully.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Copy datetime folder files into root perf directory
    for root, dirs, files in os.walk(perf_folder, topdown=False):
        if root == perf_folder:
            continue

        for file_name in files:
            # Full path of the file
            file_path = os.path.join(root, file_name)
            # Destination path in the parent folder
            dest_path = os.path.join(perf_folder, file_name)

            # Move the file
            shutil.move(file_path, dest_path)
            print(f"Moved {file_path} to {dest_path}")

        # Remove the subfolder after moving the files
        if not os.listdir(root):  # Check if the directory is empty
            os.rmdir(root)
