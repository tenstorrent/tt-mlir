# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from ttnn_jit._src.decision_trace_parser import (
    DecisionTraceReport,
    OpDecision,
    ValidatedLayout,
    FinalChoice,
    SpillSummary,
    SpillEvent,
    Edge,
)
from ttnn_jit._src.advisor_report import render_text_report


def _report(spill, edges=None):
    return DecisionTraceReport(
        function_name="my_model",
        beam_width=4,
        total_ops=1,
        ops=[
            OpDecision(
                op_index=0,
                op_name="ttnn.matmul",
                op_location="loc",
                used_dram_fallback=False,
                validated_layouts=[
                    ValidatedLayout(0, "L1/block_sharded/8x4", True, True, 32, 65536),
                    ValidatedLayout(1, "DRAM/interleaved", False, False, 0, 0),
                ],
            )
        ],
        final_choices=[
            FinalChoice(0, "ttnn.matmul", "L1/block_sharded/8x4"),
            FinalChoice(1, "ttnn.reshape", "L1/interleaved"),
            FinalChoice(2, "ttnn.matmul", "L1/width_sharded/1x64"),
            FinalChoice(3, "ttnn.matmul", "L1/width_sharded/1x64"),
        ],
        spill=spill,
        edges=edges or [],
    )


def test_report_marks_final_choice_and_layouts():
    text = render_text_report(_report(SpillSummary(ran=False)))
    assert "my_model" in text
    assert "ttnn.matmul" in text
    assert "L1/block_sharded/8x4" in text
    # The chosen layout line is marked with '*'.
    assert "* rank0: L1/block_sharded/8x4" in text
    assert "=> FINAL: L1/block_sharded/8x4" in text


def test_report_notes_spill_not_run():
    text = render_text_report(_report(SpillSummary(ran=False)))
    assert "spill management not run" in text


def test_report_renders_pressure_events_only():
    spill = SpillSummary(
        ran=True,
        budget=1000,
        final_occupied=500,
        final_live_tensors=3,
        total_spills=1,
        events=[
            SpillEvent(
                5, "ttnn.add", "eviction", 900, 700, 200, "ttnn.matmul", "freed"
            ),
            SpillEvent(6, "ttnn.add", "live_added", 700, 800, 100, "", ""),
        ],
    )
    text = render_text_report(_report(spill))
    assert "total_spills=1" in text
    assert "eviction" in text
    assert "victim=ttnn.matmul" in text
    # Bookkeeping action is filtered out of the timeline.
    assert "live_added" not in text


def test_report_no_edge_data():
    text = render_text_report(_report(SpillSummary(ran=False)))
    assert "-- Reshards decided by greedy pass (0) --" in text
    assert "(no edge data in trace)" in text


def test_report_no_reshards():
    edges = [Edge(0, 2, 0, 0, has_reshard=False)]
    text = render_text_report(_report(SpillSummary(ran=False), edges))
    assert "-- Reshards decided by greedy pass (0) --" in text
    assert "no reshards inserted" in text


def test_report_lists_reshards_with_names_and_target_layout():
    edges = [
        Edge(0, 1, 0, 0, has_reshard=True, reshard_layout="L1/interleaved"),
        Edge(2, 3, 0, 0, has_reshard=False),
    ]
    text = render_text_report(_report(SpillSummary(ran=False), edges))
    assert "-- Reshards decided by greedy pass (1) --" in text
    # Producer/consumer op names + operand + target layout are all surfaced.
    assert (
        "op0 ttnn.matmul -> op1 ttnn.reshape (operand 0): "
        "reshard into L1/interleaved" in text
    )


def test_report_reshard_between_same_layout_label():
    # Two ops with the IDENTICAL layout string can still reshard (different
    # shard shape/grid). This is the case the per-op table cannot reveal, so the
    # Reshards section must still list it.
    edges = [
        Edge(2, 3, 0, 0, has_reshard=True, reshard_layout="L1/width_sharded/1x64"),
    ]
    text = render_text_report(_report(SpillSummary(ran=False), edges))
    assert "-- Reshards decided by greedy pass (1) --" in text
    assert "op2 ttnn.matmul -> op3 ttnn.matmul (operand 0)" in text
