#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Run independent, counterbalanced D2M and TTNN full-block wall trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _schedule(trials_per_order: int) -> list[str]:
    schedule = []
    for pair_index in range(trials_per_order):
        schedule.extend(
            ("d2m-first", "ttnn-first")
            if pair_index % 2 == 0
            else ("ttnn-first", "d2m-first")
        )
    return schedule


def _command(
    harness: Path,
    d2m_binary: Path,
    ttnn_binary: Path,
    manifest: Path,
    order: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(harness),
        "--workload",
        "decoder",
        "--backend",
        "both",
        "--sequence",
        str(args.sequence),
        "--warmup",
        str(args.warmup),
        "--loops",
        str(args.loops),
        "--seed",
        str(args.seed),
        "--input-profile",
        args.input_profile,
        "--pcc-threshold",
        str(args.pcc_threshold),
        "--backend-order",
        order,
        "--d2m-binary",
        str(d2m_binary),
        "--compiler-ttnn-binary",
        str(ttnn_binary),
        "--output",
        str(manifest),
    ]
    if args.phase_breakdown:
        command.append("--phase-breakdown")
    return command


def main() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--d2m-binary", type=Path, required=True)
    parser.add_argument("--compiler-ttnn-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--harness",
        type=Path,
        default=repo_root / "tools/scripts/d2m_pipeclean/llama_prefill_layer_perf.py",
    )
    parser.add_argument("--sequence", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--trials-per-order", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-profile",
        choices=("generic", "llama-decoder-structured-v1"),
        default="llama-decoder-structured-v1",
    )
    parser.add_argument("--pcc-threshold", type=float, default=0.95)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--expected-runtime-sha256")
    parser.add_argument("--expected-input-corpus-sha256")
    parser.add_argument(
        "--phase-breakdown",
        action="store_true",
        help="Collect diagnostic phase timers inside the same wall envelope.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.trials_per_order < 2:
        parser.error("--trials-per-order must be at least 2")
    if args.warmup < 2:
        parser.error("--warmup must be at least 2")
    if args.loops < 5:
        parser.error("--loops must be at least 5")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    harness = args.harness.resolve()
    d2m_binary = args.d2m_binary.resolve()
    ttnn_binary = args.compiler_ttnn_binary.resolve()
    for path in (harness, d2m_binary, ttnn_binary):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")

    output_dir = args.output_dir.resolve()
    if not args.dry_run and output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"--output-dir must be empty or absent: {output_dir}")
    schedule = _schedule(args.trials_per_order)
    planned_trials = []
    for index, order in enumerate(schedule, start=1):
        stem = f"trial_{index:02d}_{order.replace('-', '_')}"
        manifest = output_dir / f"{stem}.json"
        planned_trials.append(
            {
                "index": index,
                "backend_order": order,
                "manifest": str(manifest),
                "log": str(output_dir / f"{stem}.log"),
                "command": _command(
                    harness, d2m_binary, ttnn_binary, manifest, order, args
                ),
            }
        )

    index = {
        "schema_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
        "harness": {"path": str(harness), "sha256": _sha256(harness)},
        "binaries": {
            "d2m": {"path": str(d2m_binary), "sha256": _sha256(d2m_binary)},
            "ttnn": {"path": str(ttnn_binary), "sha256": _sha256(ttnn_binary)},
        },
        "parameters": {
            "sequence": args.sequence,
            "warmup": args.warmup,
            "loops": args.loops,
            "trials_per_order": args.trials_per_order,
            "seed": args.seed,
            "input_profile": args.input_profile,
            "pcc_threshold": args.pcc_threshold,
            "phase_breakdown": args.phase_breakdown,
            "timeout_seconds": args.timeout_seconds,
        },
        "trials": planned_trials,
    }
    if args.dry_run:
        print(json.dumps(index, indent=2, sort_keys=True))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "wall_trial_index.json"
    _write_json(index_path, index)

    for trial in planned_trials:
        begin = time.perf_counter()
        try:
            completed = subprocess.run(
                trial["command"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            captured = error.stdout or ""
            if isinstance(captured, bytes):
                captured = captured.decode(errors="replace")
            Path(trial["log"]).write_text(captured)
            trial["status"] = "timeout"
            trial["elapsed_seconds"] = time.perf_counter() - begin
            _write_json(index_path, index)
            raise SystemExit(
                f"trial {trial['index']} exceeded {args.timeout_seconds} seconds"
            ) from error

        Path(trial["log"]).write_text(completed.stdout)
        trial["elapsed_seconds"] = time.perf_counter() - begin
        trial["returncode"] = completed.returncode
        if completed.returncode != 0:
            trial["status"] = "failed"
            _write_json(index_path, index)
            raise SystemExit(f"trial {trial['index']} failed; see {trial['log']}")

        def fail_trial(message: str) -> None:
            trial["status"] = "failed"
            trial["error"] = message
            _write_json(index_path, index)
            raise RuntimeError(message)

        manifest = json.loads(Path(trial["manifest"]).read_text())
        if manifest.get("backend_order") != trial["backend_order"]:
            fail_trial(f"trial {trial['index']} recorded the wrong order")
        if manifest.get("output_validation", {}).get("status") != "pass":
            fail_trial(f"trial {trial['index']} failed output validation")
        runtime = manifest.get("runtime", {})
        runtime_hash = runtime.get("sha256")
        corpus_hash = manifest.get("input_corpus", {}).get("sha256")
        if not runtime_hash or not corpus_hash:
            fail_trial(
                f"trial {trial['index']} did not record runtime and corpus hashes"
            )
        if (
            args.expected_runtime_sha256
            and runtime_hash != args.expected_runtime_sha256
        ):
            fail_trial(
                f"trial {trial['index']} runtime SHA-256 {runtime_hash} "
                f"!= {args.expected_runtime_sha256}"
            )
        if (
            args.expected_input_corpus_sha256
            and corpus_hash != args.expected_input_corpus_sha256
        ):
            fail_trial(
                f"trial {trial['index']} input corpus SHA-256 {corpus_hash} "
                f"!= {args.expected_input_corpus_sha256}"
            )
        trial["status"] = "pass"
        trial["runtime"] = runtime
        trial["input_corpus_sha256"] = corpus_hash
        _write_json(index_path, index)

    runtime_hashes = {trial["runtime"]["sha256"] for trial in planned_trials}
    corpus_hashes = {trial["input_corpus_sha256"] for trial in planned_trials}
    if len(runtime_hashes) != 1:
        message = f"trials loaded different runtimes: {runtime_hashes}"
        index["status"] = "failed"
        index["error"] = message
        _write_json(index_path, index)
        raise RuntimeError(message)
    if len(corpus_hashes) != 1:
        message = f"trials generated different inputs: {corpus_hashes}"
        index["status"] = "failed"
        index["error"] = message
        _write_json(index_path, index)
        raise RuntimeError(message)
    runtime_hash = next(iter(runtime_hashes))
    corpus_hash = next(iter(corpus_hashes))
    index["completed_utc"] = datetime.now(timezone.utc).isoformat()
    index["runtime_sha256"] = runtime_hash
    index["input_corpus_sha256"] = corpus_hash
    index["status"] = "pass"
    _write_json(index_path, index)
    print(index_path)


if __name__ == "__main__":
    main()
