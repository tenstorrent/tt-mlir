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
    print("Verifying tracy profiling tools")
    check_file_exists(TRACY_CAPTURE_TOOL)
    check_file_exists(TRACY_CSVEXPROT_TOOL)

    captureProcess = None
    captureCommand = f"{TRACY_CAPTURE_TOOL} -o {TRACY_OUT_FILE} -f -p {port}"
    print(f"Capture command: {captureCommand}")
    captureProcess = subprocess.Popen(captureCommand, shell=True)

    return captureProcess


def generate_report():
    child_calls = ["CompileProgram", "HWCommandQueue_write_buffer"]
    timeOut = 15
    timeCount = 0
    while not os.path.exists(TRACY_OUT_FILE):
        print(f"tracy capture out not found, will try again in 1 second")
        if timeCount > timeOut:
            print(f"tracy capture output file {TRACY_OUT_FILE} was not generated")
            sys.exit(1)
        timeCount += 1
        time.sleep(1)

    with open(TRACY_OPS_FILE, "w") as csvFile:
        child_call_str = f"-x {','.join(child_calls)}"
        subprocess.run(
            f"{TRACY_CSVEXPROT_TOOL} -u -p TT_DNN {child_call_str} {TRACY_OUT_FILE}",
            shell=True,
            check=True,
            stdout=csvFile,
            stderr=subprocess.DEVNULL,
        )

    print(f"Host side ops time report generated at {TRACY_OPS_FILE}")

    with open(TRACY_DATA_FILE, "w") as csvFile:
        subprocess.run(
            f'{TRACY_CSVEXPROT_TOOL} -m -s ";" {TRACY_OUT_FILE}',
            shell=True,
            check=True,
            stdout=csvFile,
            stderr=subprocess.DEVNULL,
        )

    print(f"Host side ops data report generated at {TRACY_DATA_FILE}")
    process_ops(TTRT_PERF_ARTIFACTS, "", True)


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
