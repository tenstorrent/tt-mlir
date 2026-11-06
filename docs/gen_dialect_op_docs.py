# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate per-dialect op reference markdown from TableGen (.td) op files.

This is a lightweight, dependency-free generator so the docs publish pipeline
(pip + sphinx-build, no compiler build) can include a per-dialect op reference.

For each dialect it reads the `*Ops.td` file(s), extracts each op's mnemonic,
summary, and description, and writes a markdown page to
`docs/src/autogen/md/Dialect/<Dialect>Op.md`.

Run from the repo root:  python docs/gen_dialect_op_docs.py
"""

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "src", "autogen", "md", "Dialect")

# dialect display name -> (mnemonic prefix, [td files relative to repo root])
DIALECTS = {
    "TTIR": ("ttir", ["include/ttmlir/Dialect/TTIR/IR/TTIROps.td"]),
    "TTNN": ("ttnn", ["include/ttmlir/Dialect/TTNN/IR/TTNNOps.td"]),
    "TTCore": ("ttcore", ["include/ttmlir/Dialect/TTCore/IR/TTCoreOps.td"]),
    "TTKernel": ("ttkernel", ["include/ttmlir/Dialect/TTKernel/IR/TTKernelOps.td"]),
    "TTMetal": ("ttmetal", ["include/ttmlir/Dialect/TTMetal/IR/TTMetalOps.td"]),
    "D2M": (
        "d2m",
        [
            "include/ttmlir/Dialect/D2M/IR/D2MOps.td",
            "include/ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.td",
        ],
    ),
}

# Matches:  def TTIR_AbsOp: TTIR_ElementwiseUnaryOp<"abs", ...> {
#           def TTIR_AllocOp : TTIR_Op<"alloc"> {
DEF_RE = re.compile(r'^def\s+(\w+)\s*:\s*\w+<\s*"([^"]+)"')
SUMMARY_RE = re.compile(r'let\s+summary\s*=\s*"([^"]*)"')


def parse_ops(td_path):
    """Return a list of (mnemonic, summary, description) for ops in one .td file."""
    with open(td_path, "r") as fd:
        lines = fd.readlines()

    ops = []
    i = 0
    n = len(lines)
    while i < n:
        m = DEF_RE.match(lines[i])
        if not m:
            i += 1
            continue
        mnemonic = m.group(2)
        summary = ""
        description = ""
        j = i + 1
        # Body ends at a line that starts with '}' in column 0 (top-level close).
        while j < n and not lines[j].startswith("}"):
            sm = SUMMARY_RE.search(lines[j])
            if sm and not summary:
                summary = sm.group(1).strip()
            if "let description = [{" in lines[j]:
                desc_lines = []
                j += 1
                while j < n and "}]" not in lines[j]:
                    desc_lines.append(lines[j].rstrip("\n"))
                    j += 1
                description = _dedent(desc_lines)
            j += 1
        ops.append((mnemonic, summary, description))
        i = j + 1
    return ops


def _dedent(desc_lines):
    """Strip common leading indentation from a description block."""
    stripped = [ln for ln in desc_lines if ln.strip()]
    if not stripped:
        return ""
    indent = min(len(ln) - len(ln.lstrip()) for ln in stripped)
    return "\n".join(ln[indent:] if len(ln) >= indent else ln for ln in desc_lines).strip()


def render(dialect, prefix, ops):
    out = [f"# {dialect} Dialect Ops\n"]
    out.append(
        f"Auto-generated reference of operations in the `{dialect}` dialect "
        f"({len(ops)} ops).\n"
    )
    for mnemonic, summary, description in sorted(ops, key=lambda o: o[0]):
        out.append(f"## `{prefix}.{mnemonic}`\n")
        if summary:
            out.append(f"{summary}\n")
        if description:
            out.append(f"{description}\n")
    return "\n".join(out) + "\n"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for dialect, (prefix, td_files) in DIALECTS.items():
        ops = []
        for rel in td_files:
            path = os.path.join(REPO_ROOT, rel)
            if os.path.exists(path):
                ops.extend(parse_ops(path))
        if not ops:
            continue
        out_path = os.path.join(OUT_DIR, f"{dialect}Op.md")
        with open(out_path, "w") as fd:
            fd.write(render(dialect, prefix, ops))
        total += len(ops)
        print(f"{dialect}: {len(ops)} ops -> {os.path.relpath(out_path, REPO_ROOT)}")
    print(f"Done. {total} ops across {len(DIALECTS)} dialects.")


if __name__ == "__main__":
    main()
