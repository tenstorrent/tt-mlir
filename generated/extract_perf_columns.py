#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import csv
import sys
import argparse
from pathlib import Path

# Columns to extract
COLUMNS_TO_EXTRACT = ["OP CODE", "HOST DURATION [ns]", "DEVICE KERNEL DURATION [ns]"]


def find_latest_csv(reports_dir):
    """Find the most recent ops_perf_results CSV file."""
    csv_files = list(reports_dir.glob("**/ops_perf_results_*.csv"))

    if not csv_files:
        print(f"No ops_perf_results CSV files found in {reports_dir}")
        sys.exit(1)

    # Get the most recent file by modification time
    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
    return latest_csv


def extract_columns(input_csv, output_csv=None):
    """Extract specific columns from CSV."""
    input_path = Path(input_csv)

    if output_csv is None:
        output_csv = input_path.parent / f"{input_path.stem}_extracted.csv"
    else:
        output_csv = Path(output_csv)

    print(f"Input:  {input_path}")
    print(f"Output: {output_csv}")

    with open(input_path, "r") as infile, open(output_csv, "w", newline="") as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Read header row and find column indices
        header = next(reader)
        column_indices = []
        found_columns = []

        for col_name in COLUMNS_TO_EXTRACT:
            try:
                idx = header.index(col_name)
                column_indices.append(idx)
                found_columns.append(col_name)
                print(f"Found '{col_name}' at index {idx}")
            except ValueError:
                print(f"Warning: Column '{col_name}' not found in header!")

        if not column_indices:
            print("Error: No requested columns found!")
            sys.exit(1)

        # Find indices of duration columns in extracted columns
        try:
            host_duration_col_idx = found_columns.index("HOST DURATION [ns]")
        except ValueError:
            host_duration_col_idx = None

        try:
            device_duration_col_idx = found_columns.index("DEVICE KERNEL DURATION [ns]")
        except ValueError:
            device_duration_col_idx = None

        # Write header with only selected columns
        writer.writerow(found_columns)

        # Collect data rows and calculate totals
        data_rows = []
        total_host_duration = 0
        total_device_duration = 0
        row_count = 0

        for row in reader:
            extracted_row = [
                row[idx] if idx < len(row) else "" for idx in column_indices
            ]
            data_rows.append(extracted_row)

            # Sum up host duration if present
            if (
                host_duration_col_idx is not None
                and len(extracted_row) > host_duration_col_idx
            ):
                try:
                    duration_val = extracted_row[host_duration_col_idx]
                    if duration_val and duration_val.strip():
                        total_host_duration += float(duration_val)
                except (ValueError, TypeError):
                    pass  # Skip non-numeric values

            # Sum up device kernel duration if present
            if (
                device_duration_col_idx is not None
                and len(extracted_row) > device_duration_col_idx
            ):
                try:
                    duration_val = extracted_row[device_duration_col_idx]
                    if duration_val and duration_val.strip():
                        total_device_duration += float(duration_val)
                        row_count += 1
                except (ValueError, TypeError):
                    pass  # Skip non-numeric values

        # Write all data rows
        for row in data_rows:
            writer.writerow(row)

        # Add summary rows
        if (
            host_duration_col_idx is not None or device_duration_col_idx is not None
        ) and row_count > 0:
            writer.writerow([])  # Blank line

            # Total row
            total_row = [""] * len(found_columns)
            total_row[0] = "TOTAL"
            if host_duration_col_idx is not None:
                total_row[host_duration_col_idx] = str(int(total_host_duration))
            if device_duration_col_idx is not None:
                total_row[device_duration_col_idx] = str(int(total_device_duration))
            writer.writerow(total_row)

            # Average row
            avg_row = [""] * len(found_columns)
            avg_row[0] = "AVERAGE"
            if host_duration_col_idx is not None:
                avg_host_duration = total_host_duration / row_count
                avg_row[host_duration_col_idx] = str(int(avg_host_duration))
            if device_duration_col_idx is not None:
                avg_device_duration = total_device_duration / row_count
                avg_row[device_duration_col_idx] = str(int(avg_device_duration))
            writer.writerow(avg_row)

            # Print summary
            if host_duration_col_idx is not None:
                print(f"\nTotal Host Duration: {int(total_host_duration):,} ns")
                print(
                    f"Average Host Duration: {int(avg_host_duration):,} ns ({row_count} rows)"
                )
            if device_duration_col_idx is not None:
                print(
                    f"Total Device Kernel Duration: {int(total_device_duration):,} ns"
                )
                print(
                    f"Average Device Kernel Duration: {int(avg_device_duration):,} ns ({row_count} rows)"
                )

    print(f"\nDone! Created {output_csv.name}")
    print(f"Extracted {len(found_columns)} columns from {len(header)} total columns")


def main():
    parser = argparse.ArgumentParser(
        description="Extract performance columns from ops_perf_results CSV files"
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        help="Input CSV file (if not provided, uses the most recent ops_perf_results file)",
    )
    parser.add_argument(
        "-o", "--output", help="Output CSV file (default: <input>_extracted.csv)"
    )
    parser.add_argument(
        "-d",
        "--dir",
        default="/localdev/vtang/tt-mlir/generated/profiler_cosh/reports",
        help="Directory to search for latest CSV (default: %(default)s)",
    )

    args = parser.parse_args()

    # Determine input CSV
    if args.input_csv:
        input_csv = Path(args.input_csv)
        if not input_csv.exists():
            print(f"Error: File not found: {input_csv}")
            sys.exit(1)
    else:
        reports_dir = Path(args.dir)
        if not reports_dir.exists():
            print(f"Error: Directory not found: {reports_dir}")
            sys.exit(1)
        input_csv = find_latest_csv(reports_dir)
        print(f"Using latest CSV: {input_csv.name}\n")

    # Extract columns
    extract_columns(input_csv, args.output)


if __name__ == "__main__":
    main()
