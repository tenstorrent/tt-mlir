#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import csv
import sys
import argparse


def process_csv_with_batching(input_file, batch_size=16):
    """
    Process CSV file and add batch totals for DEVICE KERNEL DURATION every N operations.

    Args:
        input_file: Path to input CSV file
        batch_size: Number of operations to group together (default: 16)
    """

    rows = []

    # Read all rows
    with open(input_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Process rows and add batch totals
    processed_rows = []
    current_batch = []
    batch_count = 0

    for row in rows:
        # Skip TOTAL and AVERAGE rows
        if row["OP CODE"] in ["TOTAL", "AVERAGE"]:
            processed_rows.append(row)
            continue

        current_batch.append(row)

        # When we reach batch_size, calculate total and add summary row
        if len(current_batch) == batch_size:
            batch_count += 1

            # Calculate total device kernel duration for this batch
            total_device_duration = 0
            for batch_row in current_batch:
                device_duration_str = batch_row["DEVICE KERNEL DURATION [ns]"].strip()
                if device_duration_str:  # Only sum non-empty values
                    total_device_duration += int(device_duration_str)

            # Add all rows from current batch
            processed_rows.extend(current_batch)

            # Add batch summary row
            batch_summary = {
                "OP CODE": f"BATCH_{batch_count}_TOTAL",
                "HOST DURATION [ns]": "",
                "DEVICE KERNEL DURATION [ns]": str(total_device_duration),
            }
            processed_rows.append(batch_summary)

            # Reset for next batch
            current_batch = []

    # Handle remaining rows that don't fill a complete batch
    if current_batch:
        batch_count += 1
        total_device_duration = 0
        for batch_row in current_batch:
            device_duration_str = batch_row["DEVICE KERNEL DURATION [ns]"].strip()
            if device_duration_str:
                total_device_duration += int(device_duration_str)

        processed_rows.extend(current_batch)

        batch_summary = {
            "OP CODE": f"BATCH_{batch_count}_TOTAL",
            "HOST DURATION [ns]": "",
            "DEVICE KERNEL DURATION [ns]": str(total_device_duration),
        }
        processed_rows.append(batch_summary)

    # Write processed rows back to file
    with open(input_file, "w", newline="") as f:
        fieldnames = ["OP CODE", "HOST DURATION [ns]", "DEVICE KERNEL DURATION [ns]"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)

    print(f"Processed {batch_count} batches of {batch_size} operations each")
    print(f"Updated file: {input_file}")


def calculate_last_n_batch_average(csv_file, n=25):
    """
    Calculate the average of the last N batch totals from the CSV file and append to CSV.

    Args:
        csv_file: Path to the CSV file
        n: Number of last batches to average (default: 25)
    """

    batch_totals = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        for row in rows:
            op_code = row["OP CODE"]
            if op_code.startswith("BATCH_") and op_code.endswith("_TOTAL"):
                device_duration_str = row["DEVICE KERNEL DURATION [ns]"].strip()
                if device_duration_str:
                    batch_totals.append(int(device_duration_str))

    if len(batch_totals) < n:
        print(f"Warning: Only {len(batch_totals)} batches found, but requested {n}")
        n = len(batch_totals)

    # Get the last n batch totals
    last_n_totals = batch_totals[-n:]

    # Calculate average
    average = sum(last_n_totals) / len(last_n_totals)

    print(f"Last {n} batch totals: {last_n_totals}")
    print(f"Average of last {n} batch totals: {average:.2f} ns")

    # Append the analysis to the CSV file
    with open(csv_file, "a", newline="") as f:
        fieldnames = ["OP CODE", "HOST DURATION [ns]", "DEVICE KERNEL DURATION [ns]"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Add separator row
        writer.writerow(
            {"OP CODE": "", "HOST DURATION [ns]": "", "DEVICE KERNEL DURATION [ns]": ""}
        )

        # Add analysis rows
        writer.writerow(
            {
                "OP CODE": f"LAST_{n}_BATCHES_AVERAGE",
                "HOST DURATION [ns]": "",
                "DEVICE KERNEL DURATION [ns]": f"{average:.2f}",
            }
        )

        writer.writerow(
            {
                "OP CODE": f"LAST_{n}_BATCHES_COUNT",
                "HOST DURATION [ns]": "",
                "DEVICE KERNEL DURATION [ns]": str(n),
            }
        )

    print(f"Analysis appended to {csv_file}")

    return average


def main():
    parser = argparse.ArgumentParser(description="Add batch totals to performance CSV")
    parser.add_argument("input_file", help="Input CSV file path")
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=16,
        help="Number of operations per batch (default: 16)",
    )
    parser.add_argument(
        "-n",
        "--last-n-batches",
        type=int,
        default=25,
        help="Number of last batches to average (default: 25)",
    )

    args = parser.parse_args()

    process_csv_with_batching(args.input_file, args.batch_size)

    # Calculate average of last N batch totals
    print("\n" + "=" * 50)
    print("BATCH AVERAGE ANALYSIS")
    print("=" * 50)
    calculate_last_n_batch_average(args.input_file, args.last_n_batches)


if __name__ == "__main__":
    main()
