#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import csv
import re
import sys


def extract_operands(ttnn_op):
    # Extract all %<number> operands from the TTNN Op string
    return re.findall(r"%\d+", ttnn_op)


def main(input_csv, output_csv):
    with open(input_csv, newline="") as f:
        reader = list(csv.reader(f))
    header = reader[0]
    rows = reader[1:]

    # Dynamically find column indices
    col_idx = {name: i for i, name in enumerate(header)}
    ttnn_op_idx = col_idx.get("TTNN Op")
    pcc_idx = col_idx.get("PCC")
    abs_err_idx = col_idx.get("Abs Err")
    rel_err_idx = col_idx.get("Rel Err")
    lig_idx = col_idx.get("LIG")

    if None in (ttnn_op_idx, pcc_idx, abs_err_idx, rel_err_idx, lig_idx):
        raise Exception("Could not find all required columns in header")

    # Map: operand -> (ATOL, RTOL, PCC)
    op_metrics = {}

    # First pass: collect output operand metrics only if LIG is True
    for row in rows:
        ttnn_op = row[ttnn_op_idx]
        output_operands = extract_operands(ttnn_op)
        lig = row[lig_idx].strip() == "True"
        if output_operands and lig:
            output_operand = output_operands[0]
            abs_err, rel_err, pcc = row[abs_err_idx], row[rel_err_idx], row[pcc_idx]
            op_metrics[output_operand] = (abs_err, rel_err, pcc)

    # Second pass: attach input operand metrics
    new_header = header + ["InputOps"]
    max_inputs = 0
    # First, determine the max number of input operands
    for row in rows:
        ttnn_op = row[ttnn_op_idx]
        operands = extract_operands(ttnn_op)
        input_operands = operands[1:]  # skip output
        max_inputs = max(max_inputs, len(input_operands))

    # Prepare new columns (one per input operand)
    input_metrics_cols = [f"Input{i}_Metrics" for i in range(max_inputs)]
    new_header += input_metrics_cols

    # Now, process each row
    new_rows = []
    for row in rows:
        ttnn_op = row[ttnn_op_idx]
        operands = extract_operands(ttnn_op)
        input_operands = operands[1:]  # skip output
        input_metrics = []
        for op in input_operands:
            m = op_metrics.get(op, ("", "", ""))
            # Format as fixed-point with 8 decimals if possible
            formatted = []
            for x in m:
                try:
                    formatted.append("{:.8f}".format(float(x)))
                except Exception:
                    formatted.append(str(x))
            input_metrics.append("\n".join(formatted))
        # Pad if fewer than max_inputs
        while len(input_metrics) < max_inputs:
            input_metrics.append("")
        new_rows.append(row + [",".join(input_operands)] + input_metrics)

    # Write output
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        writer.writerows(new_rows)


if __name__ == "__main__":
    # Usage: python attach_input_metrics.py input.csv output.csv
    if len(sys.argv) not in (2, 3):
        print("Usage: python attach_input_metrics.py input.csv [output.csv]")
        sys.exit(1)
    if len(sys.argv) == 2:
        inf = sys.argv[1]
        outf = inf.replace(".csv", "_processed.csv")
    else:
        inf, outf = sys.argv[1], sys.argv[2]
    main(inf, outf)
