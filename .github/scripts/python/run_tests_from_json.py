#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import os
import time
import test_common


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
        test_type = test.get("script", "")
        args = test.get("args", "")
        args = [args] if isinstance(args, str) else args
        reqs = test.get("reqs", "")
        reqs = [reqs] if isinstance(reqs, str) else reqs
        hash, hash_string = test_common.compute_hash(test, machine, image)
        test["hash"] = hash
        test["hash_string"] = hash_string

        cmd = [test_type] + args
        start_time = time.time()
        try:
            ttrt_report_path = f"{work_dir}/ttrt_results/{machine}_{image}_{test_no}_{hash}_{jobid}.json"
            test_report_path = f"{work_dir}/test_reports/report_{machine}_{image}_{test_no}_{hash}_{jobid}.xml"
            env = os.environ.copy()
            env["TTRT_REPORT_PATH"] = ttrt_report_path
            env["TEST_REPORT_PATH"] = test_report_path
            env["REQUIREMENTS"] = " ".join(reqs)
            print(
                f"\033[1;96m====================================\n\033[1;96mRunning test {test_no}-{hash}:\n\033[1;96m{hash_string}\n\033[1;96m{cmd}\n\033[1;96m====================================\n\n\n\n\033[0m"
            )
            sys.stdout.flush()
            sys.stderr.flush()
            result = subprocess.run(cmd, check=True, env=env)
            print(f"\n\033[92m SUCCESS running {test_type} \033[0m")
            test["result"] = "SUCCESS"
            test["returncode"] = 0
        except subprocess.CalledProcessError as e:
            print(f"\n\033[91m FAILURE running {test_type}: {e}\033[0m")
            test["result"] = "FAILURE"
            test["returncode"] = e.returncode
        except FileNotFoundError:
            print(f"\n\033[91m ERROR: Script {test_type} not found\033[0m")
            test["result"] = "ERROR"
            test["returncode"] = 1

        end_time = time.time()
        duration = end_time - start_time
        test["duration"] = duration
        print(f" Test duration {duration:.2f}s\n\n\n\n\n")
        test_no = test_no + 1

    print(
        "\033[1;96m====================================\n\033[1;96m TEST SUMMARY \n\033[1;96m====================================\033[0m"
    )
    # Create _test_duration file with test results summary
    allpassed = True
    duration_file = "_test_duration"
    with open(duration_file, "w") as f:
        for test in tests:
            result = test.get("result", "UNKNOWN")
            hash_val = test.get("hash", "")
            hash_string = test.get("hash_string", "")
            duration = test.get("duration", 0)
            f.write(f"{hash_val} {duration:.2f}\n")
            if test.get("returncode", 1) != 0:
                allpassed = False
                color = "\033[91m"
            else:
                color = "\033[92m"
            print(f"{color}{result} {hash_string} done in {duration:.2f}s\033[0m")

    print("\033[1;96m====================================")
    if allpassed:
        print("\033[92m ALL TESTS PASSED \033[0m")
        sys.exit(0)
    else:
        print("\033[91m SOME TESTS FAILED \033[0m")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_tests_from_json.py <machine> <image> <jobid>")
        sys.exit(1)

    machine = sys.argv[1]
    image = sys.argv[2]
    jobid = sys.argv[3]

    main(machine, image, jobid)
