# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
import sys
import xml.etree.ElementTree as ET

DURATION_THRESHOLD = (
    0.2  # Threshold in seconds for updating test durations in JSON files
)


def extract_test_case_info(xml_file):
    """
    Extract test case names and their execution times from a JUnit XML report.

    Args:
        xml_file (str): Path to the XML file.

    Returns:
        dict: A dictionary with test case names as keys and their execution time in seconds as values.
    """
    test_cases_info = {}
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for testsuite in root.findall("testsuite"):
            # Iterate over all <testcase> elements within the current <testsuite>
            for testcase in testsuite.findall("testcase"):
                try:
                    path = testcase.get("classname").replace(".", "/")
                    name = testcase.get("name")
                    test_cases_info[f"{path}.py::{name}"] = float(
                        testcase.get("time", 0)
                    )
                except ValueError:
                    print(
                        f"Warning: Non-numeric time value encountered in {xml_file} for test case '{name}'"
                    )

    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}")

    return test_cases_info


def process_directory(directory):
    """
    Process all JUnit XML files in the directory and subdirectories.

    Args:
        directory (str): Path to the root directory.

    Returns:
        dict: A dictionary with test case names as keys and their execution times as values across all files.
    """
    all_test_cases = {}

    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml") and "_builder_" in file:
                arch = file.split("_")[1]
                xml_file_path = os.path.join(subdir, file)
                # Check if it's a JUnit XML report by looking for <testsuite> tag
                try:
                    test_cases_info = extract_test_case_info(xml_file_path)
                    if arch not in all_test_cases:
                        all_test_cases[arch] = {}
                    all_test_cases[arch].update(test_cases_info)
                except Exception as e:
                    print(f"Error reading file {xml_file_path}: {e}")

    return all_test_cases


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python .github/scripts/python/get_test_duration_from_junit_xmls.py <reports_directory> <json_output_directory>"
        )
        sys.exit(1)

    root_directory = sys.argv[1]
    output_dir = sys.argv[2]
    print(f"Dir to process {root_directory}, output file {output_dir}")

    test_case_data = process_directory(root_directory)

    # Assign a small default duration (0.01 seconds) to tests with 0 duration
    # This helps pytest's least_duration splitting algorithm distribute tests properly
    default_duration = 0.01

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for arch, tests in test_case_data.items():
        # Assign default duration to tests with 0 duration
        for test_name, duration in tests.items():
            if duration == 0.0:
                tests[test_name] = default_duration

        # Write arch-specific JSON file
        output_file = os.path.join(output_dir, f"{arch}.json")

        # Load existing data if file exists
        existing_tests = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as f:
                    existing_tests = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        # Merge with existing data: only update if duration change is >= DURATION_THRESHOLD
        merged_tests = existing_tests.copy()
        for test_name, new_duration in tests.items():
            if test_name in merged_tests:
                old_duration = merged_tests[test_name]
                if abs(new_duration - old_duration) >= DURATION_THRESHOLD:
                    merged_tests[test_name] = new_duration
            else:
                merged_tests[test_name] = new_duration

        # Write sorted JSON to ensure consistent ordering
        json_output = json.dumps(merged_tests, indent=4, sort_keys=True)

        with open(output_file, "w") as f:
            f.write(json_output)

        print(f"Test case data for {arch} has been written to {output_file}")


if __name__ == "__main__":
    main()
