#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import test_common
from itertools import product

default_duration = 150.0  # default duration in seconds if not found in _test_durations
do_array_unroll_for = [
    "runs-on",
    "image",
]  # only unroll arrays for these fields


def unroll_arrays(tests):
    # Unroll tests with array fields into multiple tests
    unrolled_tests = []
    for test in tests:
        # Find all fields that are arrays and support unroll only for predefined fields
        array_fields = {
            k: v
            for k, v in test.items()
            if isinstance(v, list) and k in do_array_unroll_for
        }
        if not array_fields:
            unrolled_tests.append(test)
            continue

        # Create a list of all combinations of array fields
        keys, values = zip(*array_fields.items())
        for combination in product(*values):
            new_test = test.copy()
            for k, v in zip(keys, combination):
                new_test[k] = v
            unrolled_tests.append(new_test)

    return unrolled_tests


def main(input_filename, target_duration, component_filter):
    # Load the input JSON file
    with open(input_filename, "r") as f:
        tests = json.load(f)

    tests = unroll_arrays(tests)
    print(f"Unrolled to {len(tests)} tests.")

    if component_filter != "-all":
        # Filter tests based on the component filter
        filtered_tests = []
        for test in tests:
            if "if" in test:
                if test["if"] in component_filter:
                    filtered_tests.append(test)
                else:
                    print(
                        f"Excluding test '{test.get('name', '')}' due to component filter '{component_filter}'"
                    )
            else:
                # If no "if" field, include the test by default
                filtered_tests.append(test)
        tests = filtered_tests

    print(f"Filtered to {len(tests)} tests with opt components '{component_filter}'.")

    # load saved durations
    durations = {}
    try:
        with open("_test_durations", "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        hash_str, duration_str = parts
                        durations[hash_str] = float(duration_str)
    except FileNotFoundError:
        print("Warning: _test_durations file not found, using default durations")

    # Group tests by runs-on and image
    test_matrix = {}
    for test in tests:
        runs_on = test.get("runs-on", "")
        image = test.get("image", "")
        key = f"{runs_on}_{image}"

        if key not in test_matrix:
            test_matrix[key] = {
                "runs-on": runs_on,
                "image": image,
                "total_duration": 0.0,
                "tests": [],
            }

        # Add all other fields to the tests array
        test_copy = {k: v for k, v in test.items() if k not in ["runs-on", "image"]}
        hash, hash_string = test_common.compute_hash(test_copy, runs_on, image)
        duration = durations.get(hash, default_duration)
        test_copy["duration"] = duration
        test_matrix[key]["tests"].append(test_copy)
        test_matrix[key]["total_duration"] += duration

    # Convert to list format
    test_matrix = list(test_matrix.values())

    # Apply bin packing algorithm to split groups that exceed target duration
    final_test_matrix = []

    for group in test_matrix:
        if group["total_duration"] <= target_duration:
            # Group is within target duration, keep as is
            final_test_matrix.append(group)
        else:
            # Group exceeds target duration, apply bin packing
            tests = group["tests"]
            bins = []

            # Sort tests by duration in descending order for better bin packing
            sorted_tests = sorted(tests, key=lambda x: x["duration"], reverse=True)

            for test in sorted_tests:
                # Find the first bin that can fit this test
                placed = False
                for bin_group in bins:
                    if (
                        bin_group["total_duration"] + test["duration"]
                        <= target_duration
                    ):
                        bin_group["tests"].append(test)
                        bin_group["total_duration"] += test["duration"]
                        placed = True
                        break

                # If no existing bin can fit this test, create a new bin
                if not placed:
                    new_bin = {
                        "runs-on": group["runs-on"],
                        "image": group["image"],
                        "total_duration": test["duration"],
                        "tests": [test],
                    }
                    bins.append(new_bin)

            # Optimize bins by trying to move tests between bins to better balance durations
            improved = True
            while improved:
                improved = False

                # Try to move tests between bins to better balance durations
                for i in range(len(bins)):
                    for j in range(len(bins)):
                        if i == j:
                            continue

                        bin_i = bins[i]
                        bin_j = bins[j]

                        # Try moving each test from bin_i to bin_j
                        for test_idx, test in enumerate(bin_i["tests"][:]):
                            # Check if moving this test would improve balance
                            if (
                                bin_j["total_duration"] + test["duration"]
                                <= target_duration
                                and bin_i["total_duration"]
                                > bin_j["total_duration"] + test["duration"]
                            ):

                                # Move the test
                                bin_i["tests"].remove(test)
                                bin_i["total_duration"] -= test["duration"]
                                bin_j["tests"].append(test)
                                bin_j["total_duration"] += test["duration"]
                                improved = True
                                break

                        if improved:
                            break
                    if improved:
                        break

            # Add all bins to the final matrix
            final_test_matrix.extend(bins)

    test_matrix = final_test_matrix

    # Add "sh-run": true to tests where runs-on == "n300-llmbox"
    for test in test_matrix:
        if test.get("runs-on") == "n300-llmbox":
            test["sh-run"] = True

    # Save the test matrix to a file
    with open("_test_matrix.json", "w") as f:
        json.dump(test_matrix, f, indent=2)

    # Print the test matrix
    print(f"Generated job matrix, {len(test_matrix)} jobs:")
    print(json.dumps(test_matrix, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python generate_test_matrix.py <input_filename> <target_duration_minutes> <component_filter>"
        )
        sys.exit(1)

    main(sys.argv[1], float(sys.argv[2]) * 60.0, sys.argv[3])
