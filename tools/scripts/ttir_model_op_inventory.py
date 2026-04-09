# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Build ttir-op-report.txt and golden-style ops.mlir from preprocessed TTIR MLIR.

Expects one generic-form TTIR op per line (as emitted by ttmlir-opt). Scans for
``"ttir.<name>"`` operations, counts mnemonics, and deduplicates op lines after
normalizing SSA names.

Operand provenance is tracked: if an operand is defined by ``ttir.full`` or
``ttir.constant``, it is treated as a *const* operand. This affects both the
uniqueness key (same op + types but const-vs-dynamic differs = different config)
and the emitted ``ops.mlir`` (const operands are inlined as ``ttir.full`` /
``ttir.constant`` instead of becoming func args, matching the golden style in
``test/python/golden/mlir_snippets/``).

``ttir.full`` and ``ttir.constant`` are **excluded** from ``ops.mlir`` unless
their result is directly returned by the module. They are still counted in the
report.

Used by the ttir-model-op-analysis skill; run after the frontend normalization
pipeline that writes ``<basename>.preprocessed.mlir``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_CONST_OPS = frozenset({"full", "constant"})

_DEF_RE = re.compile(r"^\s*(%\d+)\s*=\s*")
_OPERAND_RE = re.compile(r"%(?:arg)?\d+")


def extract_types_segment(line: str, from_idx: int = 0) -> tuple[str, str] | None:
    idx = line.find(" : (", from_idx)
    if idx < 0:
        return None
    i = idx + len(" : (")
    depth = 1
    start = i
    j = i
    while j < len(line) and depth:
        c = line[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                inp = line[start:j]
                rest = line[j + 1 :].strip()
                if not rest.startswith("->"):
                    return None
                out = rest[2:].strip()
                return inp, out
        j += 1
    return None


def _parse_attrs(rest: str) -> tuple[str, str]:
    """Parse optional ``<{...}>`` attribute block, return (attr_str, remainder)."""
    if not rest.startswith("<"):
        return "", rest
    depth = 1
    k = 1
    while k < len(rest) and depth:
        if rest[k] == "<":
            depth += 1
        elif rest[k] == ">":
            depth -= 1
        k += 1
    return rest[:k], rest[k:].lstrip()


def parse_ttir_op(line: str) -> dict | None:
    m = re.search(r'"ttir\.(\w+)"\(', line)
    if not m:
        m2 = re.search(r'"ttir\.(\w+)"\(\)', line)
        if not m2:
            return None
        name = m2.group(1)
        rest = line[m2.end() :].lstrip()
        attr_str, rest = _parse_attrs(rest)
        ts = extract_types_segment(rest, 0) or extract_types_segment(line, 0)
        if not ts:
            return None
        return {
            "name": name,
            "operands": [],
            "attr_str": attr_str,
            "inp_types_s": ts[0],
            "out_types_s": ts[1],
            "raw": line.strip(),
        }
    name = m.group(1)
    j = m.end() - 1
    if line[j] != "(":
        return None
    depth = 0
    k = j
    operands_inner = ""
    rest = ""
    while k < len(line):
        c = line[k]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                operands_inner = line[j + 1 : k]
                rest = line[k + 1 :].lstrip()
                break
        k += 1
    else:
        return None
    operands = [s.strip() for s in operands_inner.split(",") if s.strip()]
    attr_str, rest = _parse_attrs(rest)
    ts = extract_types_segment(rest, 0) or extract_types_segment(line, 0)
    if not ts:
        return None
    return {
        "name": name,
        "operands": operands,
        "attr_str": attr_str,
        "inp_types_s": ts[0],
        "out_types_s": ts[1],
        "raw": line.strip(),
    }


def split_input_types(inp: str) -> list[str]:
    inp = inp.strip()
    if not inp:
        return []
    depth = 0
    parts: list[str] = []
    cur: list[str] = []
    for c in inp:
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
            continue
        cur.append(c)
    if cur:
        parts.append("".join(cur).strip())
    return parts


# ---------------------------------------------------------------------------
# Pass 1: build SSA def-map, identify constant producers, find returned SSAs
# ---------------------------------------------------------------------------

_RETURN_RE = re.compile(r"^\s*return\s+(.+?)\s*:")


def _find_returned_ssas(path: Path) -> set[str]:
    """Collect all SSA values that appear directly in ``return`` statements."""
    returned: set[str] = set()
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _RETURN_RE.match(line)
            if not m:
                continue
            for ssa in _OPERAND_RE.findall(m.group(1)):
                returned.add(ssa)
    return returned


def build_def_map(
    preprocessed_path: Path,
) -> tuple[dict[str, str], dict[str, dict], set[str]]:
    """Return (const_defs, all_parsed, returned_ssas).

    *const_defs*: ``%N`` -> raw line of the defining ``ttir.full`` / ``ttir.constant``.
    *all_parsed*: ``%N`` -> parsed dict for *every* ``ttir.*`` op that defines a value.
    *returned_ssas*: set of SSA values that appear in ``return`` statements.
    """
    const_defs: dict[str, str] = {}
    all_parsed: dict[str, dict] = {}
    with preprocessed_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"ttir.' not in line:
                continue
            dm = _DEF_RE.match(line)
            if not dm:
                continue
            ssa = dm.group(1)
            p = parse_ttir_op(line)
            if not p:
                continue
            all_parsed[ssa] = p
            if p["name"] in _CONST_OPS:
                const_defs[ssa] = line.strip()
    returned_ssas = _find_returned_ssas(preprocessed_path)
    return const_defs, all_parsed, returned_ssas


def _normalize_const_line(raw: str) -> str:
    """Strip the leading ``%N = `` from a const-producer line for embedding in a key."""
    return re.sub(r"^%\d+\s*=\s*", "", raw.strip())


# ---------------------------------------------------------------------------
# Uniqueness key: op name + attrs + types + per-operand provenance
# ---------------------------------------------------------------------------


def make_config_key(parsed: dict, const_defs: dict[str, str]) -> tuple[str, list[bool]]:
    """Return (key_string, is_const_mask).

    ``is_const_mask[i]`` is True when operand *i* is produced by a const op.
    The key includes provenance so that e.g.  ``pow(dyn, dyn)`` and
    ``pow(dyn, const<full 2.0 ...>)`` are separate configurations.
    """
    name = parsed["name"]
    attr = parsed["attr_str"]
    types_in = parsed["inp_types_s"]
    types_out = parsed["out_types_s"]
    operands = parsed.get("operands", [])

    provenance_parts: list[str] = []
    is_const: list[bool] = []
    for op_str in operands:
        if op_str in const_defs:
            provenance_parts.append(
                "const:" + _normalize_const_line(const_defs[op_str])
            )
            is_const.append(True)
        else:
            provenance_parts.append("dyn")
            is_const.append(False)

    key = (
        f'"ttir.{name}"'
        f" {attr}"
        f" : ({types_in}) -> {types_out}"
        f" |prov| {','.join(provenance_parts)}"
    )
    return key, is_const


# ---------------------------------------------------------------------------
# Emit a single func.func for ops.mlir
# ---------------------------------------------------------------------------


def emit_func(
    mnemonic: str,
    idx: int,
    parsed: dict,
    is_const: list[bool],
    const_defs: dict[str, str],
) -> str:
    types_list = split_input_types(parsed["inp_types_s"])
    n = len(types_list)
    attr = (" " + parsed["attr_str"]) if parsed["attr_str"] else ""
    out_t = parsed["out_types_s"]
    operands = parsed.get("operands", [])

    if n == 0:
        sig = f"func.func @{mnemonic}_{idx}() -> {out_t}"
        op_line = f'%0 = "ttir.{mnemonic}"(){attr} : () -> {out_t}'
        return f"  {sig} {{\n    {op_line}\n    return %0 : {out_t}\n  }}"

    # Separate dynamic args from const-inlined values.
    body_lines: list[str] = []
    const_ssa_counter = 0
    arg_counter = 0
    op_operand_names: list[str] = []
    func_arg_parts: list[str] = []

    for i in range(n):
        if i < len(is_const) and is_const[i] and i < len(operands):
            raw_const = const_defs.get(operands[i], "")
            if raw_const:
                local = f"%c{const_ssa_counter}"
                const_ssa_counter += 1
                normalized = _normalize_const_line(raw_const)
                body_lines.append(f"    {local} = {normalized}")
                op_operand_names.append(local)
                continue
        arg_name = f"%arg{arg_counter}"
        func_arg_parts.append(f"{arg_name}: {types_list[i]}")
        op_operand_names.append(arg_name)
        arg_counter += 1

    args_sig = ", ".join(func_arg_parts)
    sig = f"func.func @{mnemonic}_{idx}({args_sig}) -> {out_t}"

    op_line = (
        f'%0 = "ttir.{mnemonic}"('
        + ", ".join(op_operand_names)
        + f"){attr} : ({parsed['inp_types_s']}) -> {out_t}"
    )
    body_lines.append(f"    {op_line}")
    body_lines.append(f"    return %0 : {out_t}")

    body = "\n".join(body_lines)
    return f"  {sig} {{\n{body}\n  }}"


# ---------------------------------------------------------------------------
# Main inventory logic
# ---------------------------------------------------------------------------


def write_inventory(
    preprocessed_path: Path,
    report_path: Path,
    ops_path: Path,
    *,
    verbose: bool,
) -> tuple[int, Counter, int]:
    """Run inventory on one file.

    Returns ``(exit_code, mnemonic_counts, n_unique_configs)``.
    """
    const_defs, _, returned_ssas = build_def_map(preprocessed_path)
    if verbose:
        print(f"found {len(const_defs)} constant-producing SSA values", file=sys.stderr)
        print(f"found {len(returned_ssas)} returned SSA values", file=sys.stderr)

    all_mnemonics: list[str] = []
    unparsed = 0
    seen: dict[str, tuple[dict, list[bool]]] = {}
    skipped_const = 0

    with preprocessed_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"ttir.' not in line:
                continue
            p = parse_ttir_op(line)
            if not p:
                unparsed += 1
                continue
            all_mnemonics.append(p["name"])

            # Skip full/constant from ops.mlir unless their result is returned.
            if p["name"] in _CONST_OPS:
                dm = _DEF_RE.match(line)
                if not dm or dm.group(1) not in returned_ssas:
                    skipped_const += 1
                    continue

            key, is_const = make_config_key(p, const_defs)
            if key not in seen:
                seen[key] = (p, is_const)

    if verbose and unparsed:
        print(
            f"warning: {unparsed} lines mention ttir. but were not parsed",
            file=sys.stderr,
        )
    if verbose and skipped_const:
        print(
            f"skipped {skipped_const} full/constant instances from ops.mlir "
            f"(not returned by module)",
            file=sys.stderr,
        )

    counts = Counter(all_mnemonics)
    total = sum(counts.values())
    distinct = len(counts)

    items = sorted(seen.items(), key=lambda kv: (kv[1][0]["name"], kv[0]))
    mnemonic_next: dict[str, int] = defaultdict(int)
    funcs: list[str] = []
    for _key, (p, is_const) in items:
        m = p["name"]
        i = mnemonic_next[m]
        mnemonic_next[m] += 1
        funcs.append(emit_func(m, i, p, is_const, const_defs))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    ops_path.parent.mkdir(parents=True, exist_ok=True)

    lines_r = [
        f"Source: {preprocessed_path.name}",
        f"Total ttir op instances: {total}",
        f"Distinct mnemonics: {distinct}",
        f"Distinct op configurations (normalized SSA + provenance): {len(seen)}",
        "",
        "Per-mnemonic counts (descending):",
    ]
    for name, c in counts.most_common():
        pct = 100.0 * c / total if total else 0.0
        lines_r.append(f"  {name:40s} {c:6d}  ({pct:5.2f}%)")

    report_path.write_text("\n".join(lines_r) + "\n", encoding="utf-8")
    ops_path.write_text("module {\n" + "\n\n".join(funcs) + "\n}\n", encoding="utf-8")

    if verbose:
        print(
            f"wrote {report_path} and {ops_path} "
            f"({total} instances, {distinct} mnemonics, {len(seen)} unique configs)"
        )
    return 0, counts, len(seen)


def _write_combined_report(
    dir_path: Path,
    per_file: list[tuple[str, int, int, int]],
    combined_counts: Counter,
) -> None:
    """Write a directory-level ``ttir-op-report.txt`` aggregating all files."""
    grand_total = sum(combined_counts.values())
    distinct = len(combined_counts)
    total_configs = sum(n for _, _, _, n in per_file)

    name_w = max(len(name) for name, *_ in per_file)
    sep = "\u2500" * 72

    lines = [
        f"Directory: {dir_path.name}/",
        f"Files: {len(per_file)}",
        f"Total ttir op instances: {grand_total}",
        f"Distinct mnemonics (union): {distinct}",
        f"Distinct op configurations (sum): {total_configs}",
        "",
        sep,
        "",
        "Per-file summary:",
    ]
    for name, total, n_mnem, n_cfg in per_file:
        lines.append(
            f"  {name:{name_w}s}  {total:6d} instances"
            f"  {n_mnem:3d} mnemonics"
            f"  {n_cfg:4d} configs"
        )
    lines.append("")
    lines.append(sep)
    lines.append("")
    lines.append("Combined per-mnemonic counts (descending):")
    for name, c in combined_counts.most_common():
        pct = 100.0 * c / grand_total if grand_total else 0.0
        lines.append(f"  {name:40s} {c:6d}  ({pct:5.2f}%)")

    report = dir_path / "ttir-op-report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Combined report written to {report}")


def _resolve_inputs(paths: list[Path]) -> list[Path]:
    """Expand directories to their ``*.mlir`` contents, keep files as-is."""
    files: list[Path] = []
    for p in paths:
        p = p.resolve()
        if p.is_dir():
            expanded = sorted(p.glob("*.mlir"))
            if not expanded:
                print(f"warning: no .mlir files in {p}", file=sys.stderr)
            files.extend(expanded)
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: skipping non-existent path: {p}", file=sys.stderr)
    return files


def main() -> int:
    p = argparse.ArgumentParser(
        description="Emit ttir-op-report.txt and ops.mlir from preprocessed TTIR MLIR."
    )
    p.add_argument(
        "preprocessed",
        type=Path,
        nargs="+",
        help="One or more .mlir files (or directories of .mlir files)",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output report path (default: alongside preprocessed as ttir-op-report.txt). "
        "Only used when a single file is given.",
    )
    p.add_argument(
        "--ops",
        type=Path,
        default=None,
        help="Output ops.mlir path (default: alongside preprocessed as ops.mlir). "
        "Only used when a single file is given.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    mlir_files = _resolve_inputs(args.preprocessed)
    if not mlir_files:
        print("error: no .mlir files found", file=sys.stderr)
        return 1

    if len(mlir_files) == 1:
        f = mlir_files[0]
        out_dir = f.parent
        report = args.report if args.report else out_dir / "ttir-op-report.txt"
        ops = args.ops if args.ops else out_dir / "ops.mlir"
        rc, _counts, _n_configs = write_inventory(f, report, ops, verbose=args.verbose)
        return rc

    if args.report or args.ops:
        print(
            "warning: --report and --ops are ignored with multiple inputs",
            file=sys.stderr,
        )

    rc = 0
    combined_counts: Counter = Counter()
    per_file: list[tuple[str, int, int, int]] = []
    for f in mlir_files:
        out_dir = f.parent
        report = out_dir / "ttir-op-report.txt"
        ops = out_dir / "ops.mlir"
        print(f"Processing {f} ...")
        file_rc, counts, n_configs = write_inventory(
            f, report, ops, verbose=args.verbose
        )
        if file_rc:
            rc = 1
        total = sum(counts.values())
        rel = str(f.relative_to(Path.cwd())) if f.is_relative_to(Path.cwd()) else str(f)
        per_file.append((rel, total, len(counts), n_configs))
        combined_counts += counts

    common = Path(os.path.commonpath([f.parent for f in mlir_files]))
    _write_combined_report(common, per_file, combined_counts)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
