# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Derive an authoritative layout + reshard summary from the final TTNN IR.

The greedy optimizer's decision trace only records the reshards *it* decided.
Passes that run afterwards (notably OperationValidationAndFallback: input
relayouts and output-layout reverts) insert additional reshards that never
reach the trace. The final TTNN IR is the ground truth: every reshard is an
explicit ``ttnn.to_memory_config`` / ``ttnn.to_layout`` op, and every tensor
carries its chosen ``#ttnn.ttnn_layout`` encoding. This module parses that IR
string into a compact, human-readable summary.

The parser is line-oriented and matches the shape the TTNN printer emits (one
op per line, ``%res = "ttnn.op"(%operands...) ... : (inTypes) -> outType``). It
is deliberately tolerant: anything it cannot parse is skipped rather than
raising, so a printer change degrades the summary instead of breaking the tool.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Ops that move/relayout a tensor (a "reshard" in the L1-sharding sense).
RESHARD_OPS = {"to_memory_config", "to_layout"}
# Ops that are not interesting as producers/consumers in the summary.
_NOISE_OPS = {"deallocate", "get_device", "empty"}

_LAYOUT_DEF = re.compile(r"#(ttnn_layout\d*)\s*=\s*#ttnn\.ttnn_layout<(.+)>\s*$")
_OP_LINE = re.compile(r'%(\S+)\s*=\s*"ttnn\.(\w+)"\((.*?)\)')
_RETURN_LINE = re.compile(r"\breturn\s+%(\S+)\b")
_FUNC_ARG = re.compile(r"%(arg\d+):\s*tensor<[^>]*#(ttnn_layout\d*)")
_SSA = re.compile(r"%(\w+)")
_LAYOUT_REF = re.compile(r"#(ttnn_layout\d*)")


@dataclass
class LayoutDesc:
    name: str
    buffer: str = "?"  # l1 / dram / system_memory
    mem_layout: str = "?"  # interleaved / width_sharded / block_sharded / ...
    grid: str = "?"  # e.g. 1x64
    cores: str = ""  # e.g. (0,0)-(7,6)

    def short(self) -> str:
        s = f"{self.buffer}/{self.mem_layout}/{self.grid}"
        if self.cores:
            s += f" cores={self.cores}"
        return s


@dataclass
class OpLayout:
    index: int
    op_name: str  # e.g. ttnn.matmul
    result_layout: str  # short descriptor
    program_config: str = ""  # e.g. matmul_multi_core_reuse_multi_cast_1d @8x8


@dataclass
class IRReshard:
    kind: str  # to_memory_config / to_layout
    producer: str  # producing op name or "input"
    consumer: str  # consuming op name or "return" / "?"
    from_layout: str  # short descriptor
    to_layout: str  # short descriptor
    is_output_revert: bool = False


@dataclass
class IRSummary:
    ops: List[OpLayout] = field(default_factory=list)
    reshards: List[IRReshard] = field(default_factory=list)


def _parse_layout_defs(text: str) -> Dict[str, LayoutDesc]:
    descs: Dict[str, LayoutDesc] = {}
    for line in text.splitlines():
        m = _LAYOUT_DEF.match(line.strip())
        if not m:
            continue
        name, body = m.group(1), m.group(2)
        d = LayoutDesc(name=name)
        # grid: the <RxC> tuple that sits right before "memref"
        gm = re.search(r"<(\d+x\d+)>,\s*memref", body)
        if gm:
            d.grid = gm.group(1)
        # buffer type: the #l1 / #dram inside the memref
        bm = re.search(r"memref<.*?,\s*#(\w+)>", body)
        if bm:
            d.buffer = bm.group(1)
        # memory layout: the <token> immediately after the memref closes
        mm = re.search(r"#(?:l1|dram|system_memory)>,\s*<(\w+)>", body)
        if mm:
            d.mem_layout = mm.group(1)
        # optional core ranges
        cm = re.search(r"core_range<\((\d+,\d+)\),\s*\((\d+,\d+)\)>", body)
        if cm:
            d.cores = f"({cm.group(1)})-({cm.group(2)})"
        descs[name] = d
    return descs


def _last_layout_ref(s: str) -> Optional[str]:
    refs = _LAYOUT_REF.findall(s)
    return refs[-1] if refs else None


_PCFG = re.compile(r"#ttnn\.(matmul_\w+?)_program_config<")
_PCFG_GRID = re.compile(
    r"compute_with_storage_grid_size = #ttnn\.core_coord<(\d+),\s*(\d+)>"
)


def _program_config(line: str) -> str:
    """Compact label for a matmul program config the optimizer chose, if any.

    e.g. 'matmul_multi_core_reuse_multi_cast_1d @8x8'. Empty when the op carries
    no program config (the optimizer only attaches one for its sharded picks)."""
    m = _PCFG.search(line)
    if not m:
        return ""
    name = m.group(1)
    g = _PCFG_GRID.search(line)
    return f"{name} @{g.group(1)}x{g.group(2)}" if g else name


def parse_ir_summary(ttnn_mlir: str) -> IRSummary:
    descs = _parse_layout_defs(ttnn_mlir)

    def short(name: Optional[str]) -> str:
        if name is None:
            return "?"
        d = descs.get(name)
        return d.short() if d else name

    # First pass: map each SSA value to the op that defines it and the layout
    # of its result, and record use order.
    def_op: Dict[str, str] = {}
    def_layout: Dict[str, Optional[str]] = {}
    consumers: Dict[str, List[str]] = {}
    ordered = []  # (result_ssa, op_name, operand_ssas, result_layout)

    for am in _FUNC_ARG.finditer(ttnn_mlir):
        def_op[am.group(1)] = "input"
        def_layout[am.group(1)] = am.group(2)

    for line in ttnn_mlir.splitlines():
        rm = _RETURN_LINE.search(line)
        if rm:
            consumers.setdefault(rm.group(1), []).append("return")
            continue
        m = _OP_LINE.search(line)
        if not m:
            continue
        res, op, operands = m.group(1), m.group(2), m.group(3)
        # result layout = last layout ref on the line (the -> outType part)
        result_layout = _last_layout_ref(line)
        operand_ssas = _SSA.findall(operands)
        def_op[res] = op
        def_layout[res] = result_layout
        for o in operand_ssas:
            consumers.setdefault(o, []).append(op)
        ordered.append((res, op, operand_ssas, result_layout, _program_config(line)))

    summary = IRSummary()
    idx = 0
    for res, op, operand_ssas, result_layout, pcfg in ordered:
        if op in RESHARD_OPS:
            src = operand_ssas[0] if operand_ssas else None
            # consumer = first non-noise user of this reshard's result
            users = [u for u in consumers.get(res, []) if u not in _NOISE_OPS]
            consumer = users[0] if users else "?"
            summary.reshards.append(
                IRReshard(
                    kind=op,
                    producer=def_op.get(src, "input") if src else "?",
                    consumer=consumer,
                    from_layout=short(def_layout.get(src) if src else None),
                    to_layout=short(result_layout),
                    is_output_revert=(consumer == "return"),
                )
            )
            continue
        if op in _NOISE_OPS:
            continue
        summary.ops.append(
            OpLayout(
                index=idx,
                op_name=f"ttnn.{op}",
                result_layout=short(result_layout),
                program_config=pcfg,
            )
        )
        idx += 1

    return summary


def render_ir_summary(summary: IRSummary) -> str:
    lines = []
    lines.append("=== Final-IR layout & reshard summary (authoritative) ===")
    lines.append("")
    lines.append("-- Op layouts (from final TTNN IR) --")
    for op in summary.ops:
        cfg = f"   [{op.program_config}]" if op.program_config else ""
        lines.append(f"[{op.index}] {op.op_name}  ->  {op.result_layout}{cfg}")
    lines.append("")

    lines.append(f"-- Reshards ({len(summary.reshards)}) [from final IR] --")
    if not summary.reshards:
        lines.append("    no reshards in final IR")
    for r in summary.reshards:
        tag = "  (output revert)" if r.is_output_revert else ""
        lines.append(
            f"    {r.producer} -> {r.consumer}: "
            f"{r.from_layout} -> {r.to_layout}  [{r.kind}]{tag}"
        )
    lines.append("")
    return "\n".join(lines)


def render_ir_summary_from_mlir(ttnn_mlir: str) -> str:
    return render_ir_summary(parse_ir_summary(ttnn_mlir))
