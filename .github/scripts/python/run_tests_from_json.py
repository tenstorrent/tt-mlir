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
        iterations = test.get("iterations", 1)
        hash, hash_string = test_common.compute_hash(test, machine, image)
        test["hash"] = hash
        test["hash_string"] = hash_string
        test["iterations_results"] = []

        cmd = [test_type] + args

        # Run the test for the specified number of iterations
        for iteration in range(1, iterations + 1):
            iteration_suffix = f"_iter{iteration}" if iterations > 1 else ""
            start_time = time.time()
            try:
                ttrt_report_path = f"{work_dir}/ttrt_results/{machine}_{image}_{test_no}_{hash}{iteration_suffix}_{jobid}.json"
                test_report_path = f"{work_dir}/test_reports/report_{machine}_{image}_{test_no}_{hash}{iteration_suffix}_{jobid}.xml"
                env = os.environ.copy()
                env["TTRT_REPORT_PATH"] = ttrt_report_path
                env["TEST_REPORT_PATH"] = test_report_path
                env["REQUIREMENTS"] = " ".join(reqs)
                iteration_msg = (
                    f" (iteration {iteration}/{iterations})" if iterations > 1 else ""
                )
                print(
                    f"\033[1;96m====================================\n\033[1;96mRunning test {test_no}-{hash}{iteration_msg}:\n\033[1;96m{hash_string}\n\033[1;96m{cmd}\n\033[1;96m====================================\n\n\n\n\033[0m"
                )
                sys.stdout.flush()
                sys.stderr.flush()
                result = subprocess.run(cmd, check=True, env=env)
                print(f"\n\033[92m SUCCESS running {test_type}{iteration_msg} \033[0m")
                iteration_result = {"result": "SUCCESS", "returncode": 0}
            except subprocess.CalledProcessError as e:
                print(
                    f"\n\033[91m FAILURE running {test_type}{iteration_msg}: {e}\033[0m"
                )
                iteration_result = {"result": "FAILURE", "returncode": e.returncode}
            except FileNotFoundError:
                print(f"\n\033[91m ERROR: Script {test_type} not found\033[0m")
                iteration_result = {"result": "ERROR", "returncode": 1}

            end_time = time.time()
            duration = end_time - start_time
            iteration_result["duration"] = duration
            test["iterations_results"].append(iteration_result)
            print(f" Test iteration duration {duration:.2f}s\n\n\n\n\n")

        # Aggregate results across iterations
        total_duration = sum(r["duration"] for r in test["iterations_results"])
        all_passed = all(r["returncode"] == 0 for r in test["iterations_results"])
        test["duration"] = total_duration
        test["result"] = "SUCCESS" if all_passed else "FAILURE"
        test["returncode"] = 0 if all_passed else 1
        test_no = test_no + 1

    print(
        "\033[1;96m====================================\n\033[1;96m TEST SUMMARY \n\033[1;96m====================================\033[0m"
    )
    # Create _test_duration file with test results summary
    allpassed = True
    duration_file = "_test_duration"
    summary_file = "_test_summary"
    no = 1
    with open(summary_file, "w") as sf:
        with open(duration_file, "w") as f:
            for test in tests:
                result = test.get("result", "UNKNOWN")
                hash_val = test.get("hash", "")
                hash_string = test.get("hash_string", "")
                duration = test.get("duration", 0)
                script = test.get("script", "")
                args = test.get("args", "")
                iterations = test.get("iterations", 1)
                iterations_results = test.get("iterations_results", [])

                f.write(f"{hash_val} {duration:.2f}\n")

                if iterations > 1:
                    sf.write(
                        f"Test {no}: {script} {args} ({iterations} iterations) Result: {result} Total: {duration:.2f}s\n"
                    )
                    for idx, iter_result in enumerate(iterations_results, 1):
                        iter_dur = iter_result.get("duration", 0)
                        iter_res = iter_result.get("result", "UNKNOWN")
                        sf.write(f"  Iteration {idx}: {iter_res} in {iter_dur:.2f}s\n")
                else:
                    sf.write(
                        f"Test {no}: {script} {args} Result: {result} in: {duration:.2f}s\n"
                    )

                if test.get("returncode", 1) != 0:
                    allpassed = False
                    color = "\033[91m"
                else:
                    color = "\033[92m"

                iteration_info = f" ({iterations} iterations)" if iterations > 1 else ""
                print(
                    f"{color}{result} {hash_string}{iteration_info} done in {duration:.2f}s\033[0m"
                )
                no = no + 1

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
