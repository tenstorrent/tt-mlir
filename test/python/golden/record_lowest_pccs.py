# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Record the lowest PCCs and the TT-MLIR ops that caused them.

Reads golden_report.json from each program_* directory (e.g. produced by
run_mlir_load_split_execute.py) and writes a sorted report of the lowest
PCCs with the op location and debug string (op name/type).

Usage (from repo root after source env/activate):

  # After running run_mlir_load_split_execute.py:
  python test/python/golden/record_lowest_pccs.py --artifact-dir ./builder-artifacts/load_split_execute

  # Optional: limit to top N lowest, custom output file
  python test/python/golden/record_lowest_pccs.py --artifact-dir ./builder-artifacts/load_split_execute --top-n 50 --output lowest_pccs.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_all_reports(artifact_dir: Path):
    """Load golden_report.json from every program_* subdir. Returns list of (program_idx, report)."""
    reports = []
    for sub in sorted(artifact_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith("program_"):
            continue
        try:
            idx = int(sub.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        report_path = sub / "golden_report.json"
        if not report_path.exists():
            continue
        with open(report_path) as f:
            raw = json.load(f)
        # Saved format is {"program_N": { "output_0": {"0": {...}}, "loc": {"0": {...}} }}
        if (
            isinstance(raw, dict)
            and len(raw) == 1
            and next(iter(raw.keys())).startswith("program_")
        ):
            report = next(iter(raw.values()))
        else:
            report = raw
        reports.append((idx, report))
    return reports


def extract_pcc_entries(program_idx: int, report: dict):
    """From one program's golden_report, yield (program_idx, loc, op_debug_str, actual_pcc, result)."""
    for loc, device_results in report.items():
        if not isinstance(device_results, dict):
            continue
        pccs = []
        debug_str = None
        result = "pass"
        for device_id, data in device_results.items():
            if not isinstance(data, dict):
                continue
            pcc = data.get("actual_pcc")
            if pcc is not None and isinstance(pcc, (int, float)):
                pccs.append(float(pcc))
            if data.get("debug_info") is not None:
                debug_str = data["debug_info"]
            r = data.get("result", "pass")
            if r == "fail":
                result = "fail"
        if not pccs:
            continue
        min_pcc = min(pccs)
        yield (program_idx, loc, debug_str or "", min_pcc, result)


def main():
    ap = argparse.ArgumentParser(
        description="Record lowest PCCs and the TT-MLIR ops that caused them from golden reports."
    )
    ap.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("./builder-artifacts/load_split_execute"),
        help="Root directory containing program_0, program_1, ... with golden_report.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("lowest_pcc_report.json"),
        help="Output JSON file for the lowest-PCC report",
    )
    ap.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of lowest-PCC entries to record (default 50)",
    )
    args = ap.parse_args()

    if not args.artifact_dir.exists():
        print("Error: artifact dir not found:", args.artifact_dir, file=sys.stderr)
        sys.exit(1)

    reports = load_all_reports(args.artifact_dir)
    if not reports:
        print("No program golden_report.json found under", args.artifact_dir, file=sys.stderr)
        sys.exit(1)

    entries = []
    for program_idx, report in reports:
        for item in extract_pcc_entries(program_idx, report):
            entries.append(item)

    # Sort by PCC ascending (lowest first)
    entries.sort(key=lambda x: x[3])

    # Keep lowest top_n
    lowest = entries[: args.top_n]

    # Build output structure
    out_list = []
    for program_idx, loc, op_debug_str, actual_pcc, result in lowest:
        out_list.append(
            {
                "program_index": program_idx,
                "loc": loc,
                "op_debug_str": op_debug_str,
                "actual_pcc": actual_pcc,
                "result": result,
            }
        )

    out_dict = {
        "artifact_dir": str(args.artifact_dir.resolve()),
        "total_ops_checked": len(entries),
        "lowest_pcc_count": len(lowest),
        "lowest_pccs": out_list,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_dict, f, indent=2)

    print("Wrote", args.output.resolve())
    print("Total ops with PCC:", len(entries))
    print("Recorded lowest", len(lowest), "PCCs.")
    if lowest:
        print("\nLowest PCCs (op / loc / pcc):")
        for program_idx, loc, op_debug_str, actual_pcc, result in lowest[:20]:
            op_short = (op_debug_str or loc)[:60]
            print(f"  pcc={actual_pcc:.6f}  program_{program_idx}  {op_short}  [{loc}]")


if __name__ == "__main__":
    main()
