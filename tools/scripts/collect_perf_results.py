#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Compare device performance of two sharding configurations by lowering
each StableHLO MLIR graph through the full pipeline and running ttrt perf.

Pipeline per graph:
    1. (auto only) ttmlir-opt --stablehlo-pipeline with enable-auto-sharding
    2. ttmlir-opt --stablehlo-pipeline        -> stablehlo_with_ccls.mlir
    3. ttmlir-opt --stablehlo-to-ttir-pipeline -> ttir.mlir
    4. ttmlir-opt --ttir-to-ttnn-backend-pipeline -> ttnn.mlir
    5. ttmlir-translate --ttnn-to-flatbuffer   -> graph.ttnn
    6. ttrt perf                               -> ops_perf_results.csv

Usage:
    python3 collect_perf_results.py --manual-input <manual.mlir> --auto-input <clean.mlir> [options]

    # Run auto-sharding search + perf comparison (all output goes to generated/):
    python3 collect_perf_results.py --manual-input /path/to/manual_sharding.mlir \\
                                    --auto-input  /path/to/clean.mlir

    # Skip auto-sharding search, use a pre-computed winner:
    python3 collect_perf_results.py --manual-input /path/to/manual_sharding.mlir \\
                                    --auto-input  /path/to/winner.mlir \\
                                    --skip-auto-sharding

All generated artifacts are stored under <tt-mlir>/generated/ by default.
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

# Derive TTMLIR_ROOT from this script's location: tools/scripts/ -> repo root
SCRIPT_DIR = Path(__file__).resolve().parent
TTMLIR_ROOT = SCRIPT_DIR.parent.parent
GENERATED_DIR = TTMLIR_ROOT / "generated"
TTMLIR_OPT = TTMLIR_ROOT / "build" / "bin" / "ttmlir-opt"
TTMLIR_TRANSLATE = TTMLIR_ROOT / "build" / "bin" / "ttmlir-translate"
SYSTEM_DESC = TTMLIR_ROOT / "ttrt-artifacts" / "system_desc.ttsys"
TTRT = "ttrt"


TTCORE_ATTR_RE = re.compile(
    r'ttcore\.runtime_tensor_sharding = #ttcore<[^>]*<[^>]*>[^>]*<[^>]*>>'
)
TTCORE_SHARD_STATUS_RE = re.compile(
    r'ttcore\.shard_status = #ttcore\.shard_status<[^>]*>'
)
MARK_ARG_RE = re.compile(
    r'\s+(%\S+) = stablehlo\.custom_call @tt\.mark_argument\((%\S+)\)'
)

FUNC_START_RE = re.compile(r'^\s*func\.func\b')
FUNC_END_RE = re.compile(r'^\s*\} loc\(')


def _strip_mark_args_in_block(lines: list[str]) -> list[str]:
    """Remove tt.mark_argument calls within a single function body,
    replacing SSA uses only within the same function scope."""
    replacements: dict[str, str] = {}
    kept: list[str] = []
    for line in lines:
        m = MARK_ARG_RE.match(line)
        if m:
            replacements[m.group(1)] = m.group(2)
            continue
        kept.append(line)

    if not replacements:
        return kept

    block = '\n'.join(kept)
    for result_ssa, input_ssa in sorted(
        replacements.items(), key=lambda x: -len(x[0])
    ):
        block = re.sub(
            re.escape(result_ssa) + r'(?![0-9a-zA-Z_])', input_ssa, block
        )
    return block.split('\n')


def preprocess_mlir(input_path: str, output_path: Path) -> None:
    """Strip tt-xla-specific annotations that the standalone pipeline can't handle.

    Removes:
      - ttcore.runtime_tensor_sharding attributes (prevents duplicate dict key crash
        in ApplyArgumentShardStatusPass)
      - ttcore.shard_status attributes (prevents duplicate dict key crash when
        re-running the pipeline on winner MLIR that already has shard_status)
      - stablehlo.custom_call @tt.mark_argument ops (can't be legalized to TTNN);
        each is replaced by forwarding its input SSA value to all users.
        Replacements are scoped per function to avoid cross-function SSA conflicts.
    """
    with open(input_path) as f:
        content = f.read()

    # --- ttcore.runtime_tensor_sharding ---
    content = re.sub(r',\s*' + TTCORE_ATTR_RE.pattern, '', content)
    content = re.sub(r'\s*\{' + TTCORE_ATTR_RE.pattern + r'\}', '', content)
    content = re.sub(TTCORE_ATTR_RE.pattern + r'\s*,\s*', '', content)

    # --- ttcore.shard_status ---
    content = re.sub(r',\s*' + TTCORE_SHARD_STATUS_RE.pattern, '', content)
    content = re.sub(r'\s*\{' + TTCORE_SHARD_STATUS_RE.pattern + r'\}', '', content)
    content = re.sub(TTCORE_SHARD_STATUS_RE.pattern + r'\s*,\s*', '', content)

    # --- tt.mark_argument custom calls (scoped per function) ---
    lines = content.split('\n')
    out_lines: list[str] = []
    func_buf: list[str] | None = None

    for line in lines:
        if FUNC_START_RE.match(line):
            func_buf = [line]
            continue
        if func_buf is not None:
            func_buf.append(line)
            if FUNC_END_RE.match(line):
                out_lines.extend(_strip_mark_args_in_block(func_buf))
                func_buf = None
            continue
        out_lines.append(line)

    if func_buf is not None:
        out_lines.extend(_strip_mark_args_in_block(func_buf))

    with open(output_path, 'w') as f:
        f.write('\n'.join(out_lines))


def run_cmd(cmd: list[str], label: str, log_path: Path | None = None) -> bool:
    print(f"  [{label}] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if log_path:
        with open(log_path, "w") as f:
            f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
            f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
            f.write(f"=== STDERR ===\n{result.stderr}\n\n")
            f.write(f"=== EXIT CODE: {result.returncode} ===\n")
    if result.returncode != 0:
        print(f"  ERROR [{label}]: exit code {result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    {line}")
        return False
    return True


def run_auto_sharding(
    clean_input: str, mesh_shape: str, dump_dir: Path, dump_variants: bool
) -> str | None:
    """Run the auto-sharding pass and return path to the winner MLIR."""
    dump_dir.mkdir(parents=True, exist_ok=True)

    pipeline_opts = (
        f"mesh-shape={mesh_shape} "
        f"enable-auto-sharding=true "
        f"dump-dir={dump_dir}"
    )
    if dump_variants:
        pipeline_opts += " dump-variants=true"

    cmd = [
        str(TTMLIR_OPT),
        f'--stablehlo-pipeline={pipeline_opts}',
        clean_input,
    ]

    log_path = dump_dir / "auto_sharding_search.log"
    print(f"\n{'='*60}")
    print("Step 1: Running auto-sharding search")
    print(f"  Input: {clean_input}")
    print(f"  Dump dir: {dump_dir}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    with open(log_path, "w") as f:
        f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
        f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
        f.write(f"=== STDERR ===\n{result.stderr}\n\n")
        f.write(f"=== EXIT CODE: {result.returncode} ===\n")

    if result.returncode != 0:
        print(f"  ERROR: auto-sharding search failed (exit {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    {line}")
        print(f"  See log: {log_path}")
        return None

    winners = sorted(glob.glob(str(dump_dir / "**/winner_stablehlo_with_hints.mlir"), recursive=True))
    if not winners:
        print("  ERROR: no winner_stablehlo_with_hints.mlir found after auto-sharding")
        return None

    winner = winners[-1]
    print(f"  Winner: {winner}")
    return winner


def lower_to_flatbuffer(
    input_mlir: str, output_dir: Path, label: str, mesh_shape: str
) -> Path | None:
    """Lower a StableHLO MLIR graph through the full pipeline to a flatbuffer."""
    d = output_dir / label
    d.mkdir(parents=True, exist_ok=True)

    preprocessed = d / "00_preprocessed.mlir"
    stablehlo_ccl = d / "01_stablehlo_with_ccls.mlir"
    ttir = d / "02_ttir.mlir"
    ttnn = d / "03_ttnn.mlir"
    flatbuffer = d / f"04_graph_{label}.ttnn"

    print(f"  Preprocessing {input_mlir} ...")
    preprocess_mlir(input_mlir, preprocessed)

    steps = [
        (
            "stablehlo-pipeline",
            [str(TTMLIR_OPT), f'--stablehlo-pipeline=mesh-shape={mesh_shape}',
             str(preprocessed), "-o", str(stablehlo_ccl)],
        ),
        (
            "stablehlo-to-ttir",
            [str(TTMLIR_OPT), "--stablehlo-to-ttir-pipeline",
             str(stablehlo_ccl), "-o", str(ttir)],
        ),
        (
            "ttir-to-ttnn",
            [str(TTMLIR_OPT),
             f"--ttir-to-ttnn-backend-pipeline=system-desc-path={SYSTEM_DESC}",
             str(ttir), "-o", str(ttnn)],
        ),
        (
            "ttnn-to-flatbuffer",
            [str(TTMLIR_TRANSLATE), "--ttnn-to-flatbuffer",
             str(ttnn), "-o", str(flatbuffer)],
        ),
    ]

    for step_label, cmd in steps:
        log = d / f"{step_label}.log"
        if not run_cmd(cmd, f"{label}/{step_label}", log):
            print(f"  FAILED at {step_label} for {label}. See {log}")
            return None

    return flatbuffer


def run_ttrt_perf(flatbuffer: Path, artifact_dir: Path) -> Path | None:
    """Run ttrt perf and return path to ops_perf_results.csv."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        TTRT, "perf", str(flatbuffer),
        "--artifact-dir", str(artifact_dir),
    ]
    label = flatbuffer.stem
    log = flatbuffer.parent / "ttrt_perf.log"
    if not run_cmd(cmd, f"{label}/ttrt-perf", log):
        print(f"  FAILED: ttrt perf for {label}. See {log}")
        return None

    perf_csv = artifact_dir / flatbuffer.name / "perf" / "ops_perf_results.csv"
    if not perf_csv.exists():
        alt_candidates = list(artifact_dir.rglob("ops_perf_results.csv"))
        if alt_candidates:
            perf_csv = alt_candidates[0]
        else:
            print(f"  WARNING: ops_perf_results.csv not found for {label}")
            return None

    return perf_csv


def sum_device_kernel_duration(csv_path: Path) -> int | None:
    """Sum the 'DEVICE KERNEL DURATION [ns]' column from an ops_perf_results.csv."""
    if not csv_path.exists():
        return None

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]

        try:
            col_idx = header.index("DEVICE KERNEL DURATION [ns]")
        except ValueError:
            return None

        total = 0
        for row in reader:
            val = row[col_idx].strip()
            if val and val != "N/A":
                try:
                    total += int(val)
                except ValueError:
                    try:
                        total += int(float(val))
                    except ValueError:
                        pass
        return total


def op_count_from_csv(csv_path: Path) -> int:
    """Count total ops in the perf CSV."""
    if not csv_path.exists():
        return 0
    with open(csv_path) as f:
        return max(0, sum(1 for _ in f) - 1)


def main():
    parser = argparse.ArgumentParser(
        description="Compare perf of manual vs auto-sharding configurations"
    )
    parser.add_argument(
        "--manual-input", required=True,
        help="Path to manually-sharded StableHLO MLIR input"
    )
    parser.add_argument(
        "--auto-input", required=True,
        help="Path to clean StableHLO MLIR input (for auto-sharding search), "
             "or path to a pre-computed winner (with --skip-auto-sharding)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=f"Directory to store all outputs (default: <tt-mlir>/generated/perf_comparison)"
    )
    parser.add_argument(
        "--mesh-shape", default="1,2",
        help="Mesh shape for stablehlo-pipeline (default: 1,2)"
    )
    parser.add_argument(
        "--skip-auto-sharding", action="store_true",
        help="Skip auto-sharding search; --auto-input is treated as a "
             "pre-computed winner MLIR"
    )
    parser.add_argument(
        "--dump-variants", action="store_true",
        help="Dump each auto-sharding variant IR to disk for inspection"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else GENERATED_DIR / "perf_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    auto_input = args.auto_input

    if not args.skip_auto_sharding:
        input_stem = Path(args.auto_input).stem
        auto_sharding_dir = GENERATED_DIR / "auto_sharding" / input_stem
        winner = run_auto_sharding(
            args.auto_input, args.mesh_shape, auto_sharding_dir, args.dump_variants
        )
        if winner is None:
            print("\nAuto-sharding search failed. Aborting.")
            sys.exit(1)
        auto_input = winner

    graphs = {
        "manual_sharding": {
            "input": args.manual_input,
            "description": "Manual sharding",
        },
        "auto_sharding": {
            "input": auto_input,
            "description": "Auto-sharding winner",
        },
    }

    results = {}

    for label, info in graphs.items():
        print(f"\n{'='*60}")
        print(f"Step 2: Compiling + running on device: {label}")
        print(f"  Input:       {info['input']}")
        print(f"  Description: {info['description']}")
        print(f"{'='*60}")

        if not Path(info["input"]).exists():
            print(f"  ERROR: Input file not found: {info['input']}")
            results[label] = {"status": "MISSING_INPUT"}
            continue

        flatbuffer = lower_to_flatbuffer(
            info["input"], output_dir, label, args.mesh_shape
        )
        if flatbuffer is None:
            results[label] = {"status": "LOWER_FAILED"}
            continue

        artifact_dir = output_dir / label / "ttrt_artifacts"
        perf_csv = run_ttrt_perf(flatbuffer, artifact_dir)
        if perf_csv is None:
            results[label] = {"status": "PERF_FAILED", "flatbuffer": str(flatbuffer)}
            continue

        duration = sum_device_kernel_duration(perf_csv)
        op_count = op_count_from_csv(perf_csv)
        results[label] = {
            "status": "OK",
            "duration_ns": duration,
            "op_count": op_count,
            "perf_csv": str(perf_csv),
            "flatbuffer": str(flatbuffer),
        }

    print(f"\n\n{'='*70}")
    print("PERFORMANCE COMPARISON: Manual vs Auto-Sharding")
    print(f"{'='*70}\n")

    summary_lines = []
    summary_lines.append("Performance Comparison: Manual vs Auto-Sharding")
    summary_lines.append("=" * 55)
    summary_lines.append("")

    for label, info in graphs.items():
        r = results.get(label, {})
        summary_lines.append(f"  {label}:")
        summary_lines.append(f"    Input: {info['input']}")
        summary_lines.append(f"    Description: {info['description']}")
        if r.get("status") == "OK":
            dur = r["duration_ns"]
            dur_str = f"{dur:,} ns" if dur is not None else "N/A"
            summary_lines.append(f"    Total Device Kernel Duration: {dur_str}")
            summary_lines.append(f"    Op Count: {r['op_count']}")
        else:
            summary_lines.append(f"    Status: {r.get('status', 'UNKNOWN')}")
        summary_lines.append("")

    ok_results = {k: v for k, v in results.items() if v.get("status") == "OK" and v.get("duration_ns") is not None}
    if len(ok_results) == 2:
        manual = results["manual_sharding"]
        auto = results["auto_sharding"]
        m_dur = manual["duration_ns"]
        a_dur = auto["duration_ns"]
        if m_dur > 0:
            speedup = m_dur / a_dur
            diff_pct = ((m_dur - a_dur) / m_dur) * 100
        else:
            speedup = float("inf")
            diff_pct = 0.0

        summary_lines.append("Comparison:")
        summary_lines.append(f"  Manual:  {m_dur:>12,} ns")
        summary_lines.append(f"  Auto:    {a_dur:>12,} ns")
        summary_lines.append(f"  Speedup: {speedup:.3f}x (auto vs manual)")
        summary_lines.append(f"  Diff:    {diff_pct:+.1f}% {'faster' if a_dur < m_dur else 'slower'} (auto)")
        summary_lines.append("")
        if a_dur < m_dur:
            summary_lines.append("WINNER: auto_sharding is faster")
        elif a_dur > m_dur:
            summary_lines.append("WINNER: manual_sharding is faster")
        else:
            summary_lines.append("RESULT: Both configurations have equal performance")
    elif ok_results:
        label = next(iter(ok_results))
        summary_lines.append(f"Only {label} completed successfully.")
    else:
        summary_lines.append("Neither configuration completed successfully.")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    summary_path = output_dir / "perf_comparison.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
