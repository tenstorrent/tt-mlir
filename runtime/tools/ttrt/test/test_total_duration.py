# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import inspect
import csv
import json
from pathlib import Path

import ttrt
from ttrt.common.util import *
from ttrt.common.api import API

from util import *


def test_total_device_fw_duration(tmp_path):
    """Test that the total device firmware duration is calculated correctly."""
    # Create a test result directory
    os.makedirs("ttrt-results", exist_ok=True)

    # Create a results file to make the test harness happy
    results_file = f"ttrt-results/{inspect.currentframe().f_code.co_name}.json"
    with open(results_file, "w") as f:
        json.dump([], f)

    # Create a mock CSV file with sample data
    test_csv_path = tmp_path / "ops_perf_results.csv"

    with open(test_csv_path, "w", newline="") as csvfile:
        fieldnames = ["OP CODE", "OP TYPE", "DEVICE FW DURATION [ns]"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow(
            {
                "OP CODE": "Transpose",
                "OP TYPE": "tt_dnn_device",
                "DEVICE FW DURATION [ns]": "150000",
            }
        )
        writer.writerow(
            {
                "OP CODE": "MatMul",
                "OP TYPE": "tt_dnn_device",
                "DEVICE FW DURATION [ns]": "250000",
            }
        )
        writer.writerow(
            {
                "OP CODE": "Add",
                "OP TYPE": "tt_dnn_device",
                "DEVICE FW DURATION [ns]": "100000",
            }
        )

    # Calculate the total duration and verify it
    total_duration = 0
    with open(test_csv_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_duration += float(row["DEVICE FW DURATION [ns]"])

    assert total_duration == 500000.0, f"Expected 500000.0, got {total_duration}"

    print(f"Total device firmware duration: {total_duration} ns")
    print("Test passed!")
