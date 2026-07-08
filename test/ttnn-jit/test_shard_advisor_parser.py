# test/ttnn-jit/test_shard_advisor_parser.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from ttnn_jit._src.decision_trace_parser import (
    parse_decision_trace,
    PRESSURE_ACTIONS,
)

SAMPLE = {
    "version": 3,
    "functionName": "my_model",
    "beamWidth": 4,
    "totalOps": 2,
    "forwardPass": [
        {
            "opIndex": 0,
            "opName": "ttnn.matmul",
            "opLocation": 'loc("m":1:2)',
            "usedDramFallback": False,
            "beam": [
                {
                    "rank": 0,
                    "outputLayout": "L1/block_sharded/8x4",
                    "score": {
                        "isL1": True,
                        "isSharded": True,
                        "inputDramBytes": 0,
                        "requiresReshard": False,
                        "coreCount": 32,
                        "outputL1Usage": 65536,
                    },
                },
                {
                    "rank": 1,
                    "outputLayout": "DRAM/interleaved",
                    "score": {
                        "isL1": False,
                        "isSharded": False,
                        "inputDramBytes": 4096,
                        "requiresReshard": True,
                        "coreCount": 0,
                        "outputL1Usage": 0,
                    },
                },
            ],
        }
    ],
    "finalChoices": [
        {"opIndex": 0, "opName": "ttnn.matmul", "chosenLayout": "L1/block_sharded/8x4"},
    ],
    "edges": [
        {
            "producerOpIndex": 0,
            "consumerOpIndex": 1,
            "operandIndex": 1,
            "producerResultIndex": 0,
            "hasReshard": True,
            "reshardLayout": "L1/width_sharded/1x64",
        },
        {
            "producerOpIndex": 0,
            "consumerOpIndex": 1,
            "operandIndex": 0,
            "producerResultIndex": 0,
            "hasReshard": False,
        },
    ],
    "spillManagement": {
        "budget": 1048576,
        "scheduleSize": 2,
        "totalSpills": 1,
        "finalOccupied": 524288,
        "finalLiveTensors": 3,
        "events": [
            {
                "position": 5,
                "opName": "ttnn.add",
                "action": "eviction",
                "occupiedL1Before": 900000,
                "occupiedL1After": 700000,
                "opL1Usage": 200000,
                "victimName": "ttnn.matmul",
                "details": "freed 200000",
            },
            {
                "position": 6,
                "opName": "ttnn.add",
                "action": "live_added",
                "occupiedL1Before": 700000,
                "occupiedL1After": 800000,
                "opL1Usage": 100000,
            },
        ],
    },
}


def test_parses_top_level_and_beam():
    r = parse_decision_trace(SAMPLE)
    assert r.function_name == "my_model"
    assert r.beam_width == 4
    assert r.total_ops == 2
    assert len(r.ops) == 1
    op = r.ops[0]
    assert op.op_name == "ttnn.matmul"
    assert len(op.validated_layouts) == 2
    assert op.validated_layouts[0].layout == "L1/block_sharded/8x4"
    assert op.validated_layouts[0].is_l1 is True
    assert op.validated_layouts[0].core_count == 32
    assert op.validated_layouts[0].output_l1_usage == 65536


def test_parses_final_choices():
    r = parse_decision_trace(SAMPLE)
    assert len(r.final_choices) == 1
    assert r.final_choices[0].chosen_layout == "L1/block_sharded/8x4"


def test_parses_spill_and_pressure_filter():
    r = parse_decision_trace(SAMPLE)
    assert r.spill.ran is True
    assert r.spill.budget == 1048576
    assert r.spill.total_spills == 1
    assert len(r.spill.events) == 2
    pressure = [e for e in r.spill.events if e.action in PRESSURE_ACTIONS]
    assert len(pressure) == 1
    assert pressure[0].action == "eviction"
    assert pressure[0].victim_name == "ttnn.matmul"


def test_missing_spill_management_marks_not_run():
    data = dict(SAMPLE)
    data.pop("spillManagement")
    r = parse_decision_trace(data)
    assert r.spill.ran is False
    assert r.spill.events == []


def test_parses_edges_and_reshards():
    r = parse_decision_trace(SAMPLE)
    assert len(r.edges) == 2
    resharded = [e for e in r.edges if e.has_reshard]
    assert len(resharded) == 1
    e = resharded[0]
    assert e.producer_op_index == 0
    assert e.consumer_op_index == 1
    assert e.operand_index == 1
    assert e.reshard_layout == "L1/width_sharded/1x64"
    # Non-resharded edge carries no target layout.
    non = [e for e in r.edges if not e.has_reshard][0]
    assert non.reshard_layout == ""


def test_missing_edges_defaults_empty():
    data = dict(SAMPLE)
    data.pop("edges")
    r = parse_decision_trace(data)
    assert r.edges == []
