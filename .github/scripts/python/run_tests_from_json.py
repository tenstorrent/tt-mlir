#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import os
import hashlib
import time


def main(machine, image, jobid):
    json_file = "_test_to_run.json"
    work_dir = os.getcwd()

    try:
        with open(json_file, "r") as f:
            tests = json.load(f)
    except FileNotFoundError:
        print(f"Error: {json_file} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {json_file}")
        sys.exit(1)

    test_no = 1
    for test in tests:
        test_type = test.get("type", "")
        args = test.get("args", "")
        flags = test.get("flags", "")
        hash_string = f"{machine}-{image}-{test_type}-{args}-{flags}"
        hash = hashlib.md5(hash_string.encode()).hexdigest()
        test["hash"] = hash

        script_path = f".github/test_scripts/{test_type}.sh"
        cmd = [script_path, args, flags]

        start_time = time.time()
        try:
            perf_report_path = f"{work_dir}/perf_reports/perf_{machine}_{image}_{test_no}_{hash}_{jobid}"
            test_report_path = f"{work_dir}/test_reports/report_{machine}_{image}_{test_no}_{hash}_{jobid}.xml"
            env = os.environ.copy()
            env["PERF_REPORT_PATH"] = perf_report_path
            env["TEST_REPORT_PATH"] = test_report_path
            print(
                f"\033[1;96m====================================\nRunning test {test_no}-{hash}: {cmd}\n====================================\n\n\n\n\033[0m"
            )
            result = subprocess.run(cmd, check=True, env=env)
            print(f"\033[92m SUCCESS running {script_path} \033[0m")
            test["result"] = "SUCCESS"
        except subprocess.CalledProcessError as e:
            print(f"\033[91m FAILED running {script_path}: {e}\033[0m")
            test["result"] = "FAIL"
        except FileNotFoundError:
            print(f"\033[91m ERROR: Script {script_path} not found\033[0m")
            test["result"] = "ERROR"

        end_time = time.time()
        test["duration"] = end_time - start_time
        print("\n\n\n\n\n")
        test_no = test_no + 1

    print(
        f"\033[1;96m====================================\ TEST SUMMARY \n====================================\n\n\033[0m"
    )
    # Create _test_duration file with test results summary
    duration_file = "_test_duration"
    with open(duration_file, "w") as f:
        for test in tests:
            result = test.get("result", "UNKNOWN")
            hash_val = test.get("hash", "")
            duration = test.get("duration", 0)
            f.write(f"{result} {hash_val} {duration:.2f}\n")
            print(f"{result} {hash_val} {duration:.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_tests_from_json.py <machine> <image> <jobid>")
        sys.exit(1)

    machine = sys.argv[1]
    image = sys.argv[2]
    jobid = sys.argv[3]

    main(machine, image, jobid)
