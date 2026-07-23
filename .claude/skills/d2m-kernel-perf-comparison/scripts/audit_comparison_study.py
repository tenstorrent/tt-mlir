#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Report which D2M-versus-TTNN performance claims have required controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


def _load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return value


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _check(name: str, passed: bool, detail: str) -> dict[str, str]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "detail": detail,
    }


def _wall_result(wall: dict, backend: str) -> dict:
    matches = [
        result
        for result in wall.get("results", [])
        if (backend == "d2m" and result.get("backend") == "D2M-JIT")
        or (backend == "ttnn" and "TTNN" in result.get("backend", ""))
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {backend} result, found {len(matches)}")
    return matches[0]


def _runtime_identity(wall: dict) -> tuple[str, str] | None:
    runtime = wall.get("runtime")
    if not isinstance(runtime, dict) or not runtime.get("sha256"):
        return None
    return runtime.get("path", ""), runtime["sha256"]


def _control_ready(control: dict, base: Path) -> tuple[bool, str]:
    if control.get("status") != "complete":
        return False, f"status={control.get('status', 'missing')}"
    evidence = control.get("evidence", [])
    if not evidence:
        return False, "complete control has no evidence artifacts"
    missing = [value for value in evidence if not _resolve(base, value).is_file()]
    if missing:
        return False, f"missing evidence: {missing}"
    return True, f"evidence artifacts={len(evidence)}"


def _ordered_invocations(invocations: dict) -> list[dict]:
    def trailing_index(name: str) -> int:
        return int(name.rsplit("_", maxsplit=1)[1])

    ordered = [
        value
        for _, value in sorted(
            invocations.items(), key=lambda item: trailing_index(item[0])
        )
    ]
    return ordered


def _target_matches_contract(target: dict, contract: dict) -> bool:
    mesh = target.get("main_mesh_shape", {})
    device_count = mesh.get("x", 0) * mesh.get("y", 0)
    expected_grid = contract.get("physical_grid", contract.get("worker_grid"))
    grids = target.get("physical_grid_sizes", [])
    return (
        device_count == contract.get("device_count")
        and bool(grids)
        and all(grid == expected_grid for grid in grids)
    )


def _relative_difference_percent(lhs: float, rhs: float) -> float:
    midpoint = statistics.median((lhs, rhs))
    return 100.0 * abs(lhs - rhs) / midpoint if midpoint else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit unsuccessfully unless every claim is supported.",
    )
    args = parser.parse_args()

    study = _load(args.study)
    if study.get("schema_version") != 1:
        raise ValueError("unsupported or missing study schema_version")
    base = args.study.resolve().parent
    artifacts = study["artifacts"]
    requirements = study.get("requirements", {})
    binary_manifest = _load(_resolve(base, artifacts["binary_manifest"]))
    walls = [_load(_resolve(base, path)) for path in artifacts["wall_manifests"]]
    d2m_device = _load(_resolve(base, artifacts["d2m_device_manifest"]))

    checks = []
    contract = study["semantic_contract"]
    source_ir = _resolve(base, contract["source_ir"])
    checks.append(
        _check(
            "source_ir_exists",
            source_ir.is_file(),
            str(source_ir),
        )
    )
    expected_source_hash = contract.get("source_ir_sha256")
    actual_source_hash = (
        hashlib.sha256(source_ir.read_bytes()).hexdigest()
        if source_ir.is_file()
        else None
    )
    checks.append(
        _check(
            "source_ir_sha256_matches",
            bool(expected_source_hash) and actual_source_hash == expected_source_hash,
            f"expected={expected_source_hash}, actual={actual_source_hash}",
        )
    )

    binary_failures = [
        check
        for check in binary_manifest.get("checks", [])
        if check["status"] != "pass"
    ]
    checks.append(
        _check(
            "binary_logical_io_parity",
            not binary_failures,
            f"non-passing binary checks={len(binary_failures)}",
        )
    )
    targets = [binary.get("target", {}) for binary in binary_manifest["binaries"]]
    checks.append(
        _check(
            "binary_target_parity",
            bool(targets) and all(target == targets[0] for target in targets[1:]),
            f"targets={targets}",
        )
    )
    checks.append(
        _check(
            "binary_target_matches_contract",
            bool(targets)
            and all(_target_matches_contract(target, contract) for target in targets),
            (
                f"expected_device_count={contract.get('device_count')}, "
                "expected_physical_grid="
                f"{contract.get('physical_grid', contract.get('worker_grid'))}"
            ),
        )
    )

    build_manifest_value = artifacts.get("build_manifest")
    build_manifest_path = (
        _resolve(base, build_manifest_value) if build_manifest_value else None
    )
    build_manifest = (
        _load(build_manifest_path)
        if build_manifest_path is not None and build_manifest_path.is_file()
        else None
    )
    checks.append(
        _check(
            "build_manifest_exists",
            build_manifest is not None,
            str(build_manifest_path) if build_manifest_path else "not declared",
        )
    )
    binary_hashes_by_backend = {
        "d2m" if Path(binary["path"]).suffix == ".ttm" else "ttnn": binary["sha256"]
        for binary in binary_manifest["binaries"]
    }
    build_binary_hashes = {
        backend: record.get("sha256")
        for backend, record in (build_manifest or {}).get("binaries", {}).items()
        if isinstance(record, dict)
    }
    build_commands = (build_manifest or {}).get("commands", {})
    provenance_ready = bool(build_manifest) and all(
        (
            build_manifest.get("schema_version") == 1,
            bool(actual_source_hash),
            build_manifest.get("source_ir_sha256") == actual_source_hash,
            bool(build_manifest.get("compiler_git_sha")),
            bool(build_manifest.get("system_desc_sha256")),
            bool(build_commands.get("d2m")),
            bool(build_commands.get("ttnn")),
            build_binary_hashes == binary_hashes_by_backend,
        )
    )
    checks.append(
        _check(
            "build_manifest_proves_binary_provenance",
            provenance_ready,
            (
                f"source_hash={build_manifest.get('source_ir_sha256') if build_manifest else None}, "
                f"compiler_git_sha={build_manifest.get('compiler_git_sha') if build_manifest else None}, "
                f"binary_hashes={build_binary_hashes}"
            ),
        )
    )

    runtime_identities = [_runtime_identity(wall) for wall in walls]
    checks.append(
        _check(
            "one_hashed_runtime",
            bool(runtime_identities)
            and None not in runtime_identities
            and len(set(runtime_identities)) == 1,
            f"identities={sorted({str(identity) for identity in runtime_identities})}",
        )
    )

    corpus_hashes = [wall.get("input_corpus", {}).get("sha256") for wall in walls]
    checks.append(
        _check(
            "one_input_corpus",
            bool(corpus_hashes)
            and None not in corpus_hashes
            and len(set(corpus_hashes)) == 1,
            f"hashes={sorted({str(value) for value in corpus_hashes})}",
        )
    )
    validation_failures = [
        index
        for index, wall in enumerate(walls)
        if wall.get("output_validation", {}).get("status") != "pass"
    ]
    checks.append(
        _check(
            "all_trials_validate_outputs",
            not validation_failures,
            f"failing trial indexes={validation_failures}",
        )
    )

    orders = Counter(wall.get("backend_order") for wall in walls)
    minimum_trials = int(requirements.get("minimum_trials_per_order", 2))
    checks.append(
        _check(
            "counterbalanced_trial_count",
            all(
                orders[order] >= minimum_trials for order in ("d2m-first", "ttnn-first")
            ),
            f"orders={dict(orders)}, required_per_order={minimum_trials}",
        )
    )

    minimum_warmups = int(requirements.get("minimum_warmups", 2))
    minimum_samples = int(requirements.get("minimum_samples_per_trial", 5))
    bad_samples = []
    binary_hashes = {
        Path(binary["path"]).suffix: binary["sha256"]
        for binary in binary_manifest["binaries"]
    }
    for trial_index, wall in enumerate(walls):
        for backend, suffix in (("d2m", ".ttm"), ("ttnn", ".ttnn")):
            result = _wall_result(wall, backend)
            if (
                result.get("warmup", 0) < minimum_warmups
                or len(result.get("samples_ms", [])) < minimum_samples
                or not result.get("program_cache_enabled")
                or result.get("binary", {}).get("sha256") != binary_hashes[suffix]
            ):
                bad_samples.append(f"trial={trial_index},backend={backend}")
    checks.append(
        _check(
            "warm_cached_exact_binary_trials",
            not bad_samples,
            f"nonconforming={bad_samples}",
        )
    )

    order_medians = {}
    for order in ("d2m-first", "ttnn-first"):
        selected = [wall for wall in walls if wall.get("backend_order") == order]
        order_medians[order] = {
            backend: (
                statistics.median(
                    _wall_result(wall, backend)["median_ms"] for wall in selected
                )
                if selected
                else None
            )
            for backend in ("d2m", "ttnn")
        }

    order_effects = {}
    for backend in ("d2m", "ttnn"):
        d2m_first = order_medians["d2m-first"][backend]
        ttnn_first = order_medians["ttnn-first"][backend]
        order_effects[backend] = (
            _relative_difference_percent(d2m_first, ttnn_first)
            if d2m_first is not None and ttnn_first is not None
            else None
        )
    maximum_order_effect = float(requirements.get("maximum_order_effect_percent", 10.0))
    checks.append(
        _check(
            "backend_order_effect_within_limit",
            all(
                effect is not None and effect <= maximum_order_effect
                for effect in order_effects.values()
            ),
            (
                f"effects_percent={order_effects}, "
                f"maximum_percent={maximum_order_effect}"
            ),
        )
    )
    aggregate_medians = {
        backend: statistics.median(
            _wall_result(wall, backend)["median_ms"] for wall in walls
        )
        for backend in ("d2m", "ttnn")
    }
    aggregate_ratio = aggregate_medians["d2m"] / aggregate_medians["ttnn"]

    discard_initial = int(requirements.get("discard_initial_device_invocations", 1))
    minimum_steady = int(requirements.get("minimum_steady_device_invocations", 2))
    steady = _ordered_invocations(d2m_device["invocations"])[discard_initial:]
    transfer_fractions = [
        invocation.get("location_correlated_transfer_gap_fraction", 0.0)
        for invocation in steady
    ]
    minimum_fraction = float(requirements.get("minimum_d2m_transfer_gap_fraction", 0.8))
    transfer_internal_ready = (
        len(transfer_fractions) >= minimum_steady
        and statistics.median(transfer_fractions) >= minimum_fraction
    )

    api_ready = all(check["status"] == "pass" for check in checks)
    controls = study["controls"]
    transfer_ready, transfer_detail = _control_ready(
        controls["transfer_capsules"], base
    )
    dispatch_ready, dispatch_detail = _control_ready(controls["dispatch_slope"], base)

    claims = [
        {
            "claim": "full_block_api_latency",
            "status": "supported" if api_ready else "blocked",
            "detail": (
                "all semantic/full-block gates pass"
                if api_ready
                else "see failed checks"
            ),
        },
        {
            "claim": "d2m_internal_transfer_attribution",
            "status": "supported" if transfer_internal_ready else "blocked",
            "detail": f"steady location-correlated fractions={transfer_fractions}",
        },
        {
            "claim": "cross_backend_transfer_attribution",
            "status": "supported" if transfer_ready else "blocked",
            "detail": transfer_detail,
        },
        {
            "claim": "dispatch_attribution",
            "status": "supported" if dispatch_ready else "blocked",
            "detail": dispatch_detail,
        },
    ]
    for name, control in controls["semantic_capsules"].items():
        ready, detail = _control_ready(control, base)
        claims.append(
            {
                "claim": f"semantic_capsule:{name}",
                "status": "supported" if ready else "blocked",
                "detail": detail,
            }
        )

    report = {
        "study_id": study["study_id"],
        "checks": checks,
        "aggregate_medians_ms": aggregate_medians,
        "d2m_over_ttnn_aggregate_median": aggregate_ratio,
        "order_medians_ms": order_medians,
        "order_effect_percent": order_effects,
        "claims": claims,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")

    if args.require_ready and any(claim["status"] != "supported" for claim in claims):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
