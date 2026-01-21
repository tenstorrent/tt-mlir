#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Generate depth-specific heatmap CSVs from JIT performance data.
Each CSV has batch sizes as columns and layer sizes as rows.
Values are PERCENT_OF_TTNN_PERF (JIT speedup percentage).
"""
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict


def generate_heatmap_csvs(jit_csv_file, output_dir="heatmap_csvs", name_suffix=""):
    """
    Generate one CSV per depth with:
    - Rows: Layer sizes (512, 1024, 2048, 4096)
    - Columns: Batch sizes (1, 32, 64, 128, 256, 512, 1024, 2048)
    - Values: PERCENT_OF_TTNN_PERF
    """

    jit_path = Path(jit_csv_file)
    if not jit_path.exists():
        print(f"Error: File {jit_csv_file} not found")
        sys.exit(1)

    # Create output directory with optional suffix
    if name_suffix:
        output_path = Path(f"{output_dir}_{name_suffix}")
    else:
        output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Reading JIT data from: {jit_csv_file}")
    print(f"Output directory: {output_path}")
    print()

    # Read JIT data and organize by depth
    # Structure: data[depth][layer_size][batch_size] = percent_of_ttnn_perf
    data = defaultdict(lambda: defaultdict(dict))

    with open(jit_csv_file, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                depth = int(row["DEPTH"])
                batch_size = int(row["BATCH_SIZE"])
                layer_size = int(row["LAYER_SIZE"])

                # Get speedup percentage, handle N/A and FAILED
                speedup = row["PERCENT_OF_TTNN_PERF"].strip()
                if speedup in ["N/A", "FAILED", ""]:
                    speedup_value = None
                else:
                    speedup_value = float(speedup)

                data[depth][layer_size][batch_size] = speedup_value

            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue

    # Get all unique batch sizes and layer sizes (sorted)
    all_batch_sizes = set()
    all_layer_sizes = set()

    for depth_data in data.values():
        for layer_size, batch_dict in depth_data.items():
            all_layer_sizes.add(layer_size)
            all_batch_sizes.update(batch_dict.keys())

    batch_sizes = sorted(all_batch_sizes)
    layer_sizes = sorted(all_layer_sizes)

    print(f"Found depths: {sorted(data.keys())}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Layer sizes: {layer_sizes}")
    print()

    # Generate one CSV per depth
    files_created = []

    for depth in sorted(data.keys()):
        output_file = output_path / f"depth_{depth:02d}_heatmap.csv"

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Header row: LAYER_SIZE, then all batch sizes
            header = ["LAYER_SIZE"] + [str(bs) for bs in batch_sizes]
            writer.writerow(header)

            # Data rows: one per layer size
            for layer_size in layer_sizes:
                row = [str(layer_size)]

                for batch_size in batch_sizes:
                    value = data[depth][layer_size].get(batch_size)

                    if value is None:
                        row.append("")  # Empty cell for missing/failed data
                    else:
                        row.append(f"{value:.2f}")

                writer.writerow(row)

        files_created.append(output_file)
        print(f"Created: {output_file.name}")

    print()
    print(
        f"Successfully created {len(files_created)} heatmap CSV files in '{output_dir}/'"
    )
    print()
    print("Each CSV has:")
    print("  - Rows: Layer sizes (512, 1024, 2048, 4096)")
    print("  - Columns: Batch sizes (1, 32, 64, 128, 256, 512, 1024, 2048)")
    print("  - Values: JIT speedup % (PERCENT_OF_TTNN_PERF)")
    print()
    print("Usage tips:")
    print("  - Import into spreadsheet software for heatmap visualization")
    print("  - Values > 100% mean JIT is faster than TTNN")
    print("  - Empty cells indicate failed configurations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate depth-specific heatmap CSVs from JIT performance data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate_heatmap_csvs.py sweep_jit_only.csv
  python3 generate_heatmap_csvs.py sweep_jit_only.csv my_heatmaps
  python3 generate_heatmap_csvs.py sweep_jit_only.csv heatmaps --name experiment1
  python3 generate_heatmap_csvs.py sweep_jit_only.csv -n v2
        """,
    )

    parser.add_argument("jit_csv", help="JIT CSV file to generate heatmaps from")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="heatmap_csvs",
        help="Output directory for heatmap CSVs (default: heatmap_csvs)",
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="name_suffix",
        default="",
        help="Optional suffix to append to heatmap filenames",
    )

    args = parser.parse_args()

    generate_heatmap_csvs(args.jit_csv, args.output_dir, args.name_suffix)
