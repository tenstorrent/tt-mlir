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
)
from ttnn_jit._src.advisor_report import render_text_report


def _report(spill):
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
        final_choices=[FinalChoice(0, "ttnn.matmul", "L1/block_sharded/8x4")],
        spill=spill,
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
            SpillEvent(5, "ttnn.add", "eviction", 900, 700, 200, "ttnn.matmul", "freed"),
            SpillEvent(6, "ttnn.add", "live_added", 700, 800, 100, "", ""),
        ],
    )
    text = render_text_report(_report(spill))
    assert "total_spills=1" in text
    assert "eviction" in text
    assert "victim=ttnn.matmul" in text
    # Bookkeeping action is filtered out of the timeline.
    assert "live_added" not in text
