# tools/ttnn-jit/_src/decision_trace_parser.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Parse greedy-optimizer decision-trace JSON into typed records.

JSON schema is produced by lib/Dialect/TTNN/Diagnostics/DecisionTrace.cpp.
Keys are camelCase; beam scores are nested under "score"; "spillManagement"
is omitted entirely when the spill pass did not run.
"""

import json
from dataclasses import dataclass, field
from typing import List

# Spill actions that indicate real L1 pressure (vs. bookkeeping).
PRESSURE_ACTIONS = {
    "oom",
    "eviction",
    "demotion_success",
    "demotion_failed",
    "fragmentation_demote",
    "self_spill",
    "revalidation",
}


@dataclass
class ValidatedLayout:
    rank: int
    layout: str
    is_l1: bool
    is_sharded: bool
    core_count: int
    output_l1_usage: int


@dataclass
class OpDecision:
    op_index: int
    op_name: str
    op_location: str
    used_dram_fallback: bool
    validated_layouts: List[ValidatedLayout]


@dataclass
class FinalChoice:
    op_index: int
    op_name: str
    chosen_layout: str


@dataclass
class Edge:
    producer_op_index: int
    consumer_op_index: int
    operand_index: int
    producer_result_index: int
    has_reshard: bool
    # Layout the producer's output is resharded into for this consumer operand;
    # empty string when has_reshard is False.
    reshard_layout: str = ""


@dataclass
class SpillEvent:
    position: int
    op_name: str
    action: str
    occupied_before: int
    occupied_after: int
    op_l1_usage: int
    victim_name: str
    details: str


@dataclass
class SpillSummary:
    ran: bool
    budget: int = 0
    final_occupied: int = 0
    final_live_tensors: int = 0
    total_spills: int = 0
    events: List[SpillEvent] = field(default_factory=list)


@dataclass
class DecisionTraceReport:
    function_name: str
    beam_width: int
    total_ops: int
    ops: List[OpDecision]
    final_choices: List[FinalChoice]
    spill: SpillSummary
    edges: List[Edge] = field(default_factory=list)


def _parse_beam_entry(entry: dict) -> ValidatedLayout:
    score = entry.get("score", {})
    return ValidatedLayout(
        rank=entry.get("rank", 0),
        layout=entry.get("outputLayout", "null"),
        is_l1=score.get("isL1", False),
        is_sharded=score.get("isSharded", False),
        core_count=score.get("coreCount", 0),
        output_l1_usage=score.get("outputL1Usage", 0),
    )


def _parse_op(op: dict) -> OpDecision:
    return OpDecision(
        op_index=op.get("opIndex", 0),
        op_name=op.get("opName", ""),
        op_location=op.get("opLocation", ""),
        used_dram_fallback=op.get("usedDramFallback", False),
        validated_layouts=[_parse_beam_entry(b) for b in op.get("beam", [])],
    )


def _parse_final_choice(fc: dict) -> FinalChoice:
    return FinalChoice(
        op_index=fc.get("opIndex", 0),
        op_name=fc.get("opName", ""),
        chosen_layout=fc.get("chosenLayout", "null"),
    )


def _parse_edge(e: dict) -> Edge:
    return Edge(
        producer_op_index=e.get("producerOpIndex", -1),
        consumer_op_index=e.get("consumerOpIndex", -1),
        operand_index=e.get("operandIndex", 0),
        producer_result_index=e.get("producerResultIndex", 0),
        has_reshard=e.get("hasReshard", False),
        reshard_layout=e.get("reshardLayout", ""),
    )


def _parse_spill_event(e: dict) -> SpillEvent:
    return SpillEvent(
        position=e.get("position", 0),
        op_name=e.get("opName", ""),
        action=e.get("action", ""),
        occupied_before=e.get("occupiedL1Before", 0),
        occupied_after=e.get("occupiedL1After", 0),
        op_l1_usage=e.get("opL1Usage", 0),
        victim_name=e.get("victimName", ""),
        details=e.get("details", ""),
    )


def _parse_spill(data: dict) -> SpillSummary:
    spill = data.get("spillManagement")
    if spill is None:
        return SpillSummary(ran=False)
    return SpillSummary(
        ran=True,
        budget=spill.get("budget", 0),
        final_occupied=spill.get("finalOccupied", 0),
        final_live_tensors=spill.get("finalLiveTensors", 0),
        total_spills=spill.get("totalSpills", 0),
        events=[_parse_spill_event(e) for e in spill.get("events", [])],
    )


def parse_decision_trace(data: dict) -> DecisionTraceReport:
    return DecisionTraceReport(
        function_name=data.get("functionName", ""),
        beam_width=data.get("beamWidth", 0),
        total_ops=data.get("totalOps", 0),
        ops=[_parse_op(op) for op in data.get("forwardPass", [])],
        final_choices=[_parse_final_choice(fc) for fc in data.get("finalChoices", [])],
        spill=_parse_spill(data),
        edges=[_parse_edge(e) for e in data.get("edges", [])],
    )


def load_decision_trace(path: str) -> DecisionTraceReport:
    with open(path) as f:
        return parse_decision_trace(json.load(f))
