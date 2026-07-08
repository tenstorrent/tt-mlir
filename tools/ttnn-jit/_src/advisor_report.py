# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Render a DecisionTraceReport as a human-readable text report."""

from ttnn_jit._src.decision_trace_parser import (
    DecisionTraceReport,
    PRESSURE_ACTIONS,
)


def _fmt_bytes(n: int) -> str:
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


def _render_reshards(report: DecisionTraceReport, lines: list) -> None:
    # Reshards live on the edges between ops - a consumer whose chosen input
    # layout differs from the producer's output layout forces a reshard. These
    # are invisible in the per-op layout table (two ops can even share the same
    # layout *label* yet differ in shard shape/grid and still reshard), so
    # surface them explicitly.
    name_by_index = {op.op_index: op.op_name for op in report.ops}
    for fc in report.final_choices:
        name_by_index.setdefault(fc.op_index, fc.op_name)

    def label(idx: int) -> str:
        return f"op{idx} {name_by_index.get(idx, '?')}"

    resharded = [e for e in report.edges if e.has_reshard]
    # NOTE: this is the greedy pass's own view. Passes that run after the trace
    # is written (input relayouts, output-layout reverts) add more reshards; the
    # authoritative count is in the final-IR summary. Kept here as rationale.
    lines.append(f"-- Reshards decided by greedy pass ({len(resharded)}) --")
    lines.append("   (rationale only; authoritative reshards are in the IR summary)")
    if not report.edges:
        lines.append("    (no edge data in trace)")
    elif not resharded:
        lines.append("    no reshards inserted")
    else:
        for e in sorted(
            resharded, key=lambda x: (x.consumer_op_index, x.operand_index)
        ):
            lines.append(
                f"    {label(e.producer_op_index)} -> "
                f"{label(e.consumer_op_index)} (operand {e.operand_index}): "
                f"reshard into {e.reshard_layout}"
            )
    lines.append("")


def render_text_report(report: DecisionTraceReport) -> str:
    lines = []
    lines.append(f"=== L1 Sharding Advisor: {report.function_name} ===")
    lines.append(f"ops: {report.total_ops}  beam_width: {report.beam_width}")
    lines.append("")

    final_by_index = {fc.op_index: fc.chosen_layout for fc in report.final_choices}

    lines.append("-- Per-op layouts (validated survivors -> FINAL) --")
    for op in report.ops:
        chosen = final_by_index.get(op.op_index, "?")
        lines.append(f"[{op.op_index}] {op.op_name}  @ {op.op_location}")
        for vl in op.validated_layouts:
            marker = "*" if vl.layout == chosen else " "
            mem = "L1" if vl.is_l1 else "DRAM"
            lines.append(
                f"    {marker} rank{vl.rank}: {vl.layout}  "
                f"[{mem} cores={vl.core_count} l1={_fmt_bytes(vl.output_l1_usage)}]"
            )
        lines.append(f"    => FINAL: {chosen}")
    lines.append("")

    _render_reshards(report, lines)

    s = report.spill
    lines.append("-- L1 spill accounting --")
    if not s.ran:
        lines.append("    spill management not run (memory-layout-analysis disabled)")
    else:
        pct = (100.0 * s.final_occupied / s.budget) if s.budget else 0.0
        lines.append(
            f"    budget={_fmt_bytes(s.budget)}  "
            f"final_occupied={_fmt_bytes(s.final_occupied)} ({pct:.1f}%)  "
            f"live_tensors={s.final_live_tensors}  total_spills={s.total_spills}"
        )
        pressure = [e for e in s.events if e.action in PRESSURE_ACTIONS]
        if not pressure:
            lines.append("    no pressure events")
        for e in pressure:
            victim = f" victim={e.victim_name}" if e.victim_name else ""
            detail = f" ({e.details})" if e.details else ""
            lines.append(
                f"    @{e.position} {e.op_name}: {e.action}  "
                f"{_fmt_bytes(e.occupied_before)} -> {_fmt_bytes(e.occupied_after)}"
                f"{victim}{detail}"
            )
    return "\n".join(lines)
