# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shard-advisor test over a longer op sequence (FFN-style block).

Exercises the full trace -> greedy optimizer -> decision-trace -> report path
on a multi-op graph (2 matmuls + bias + activation + gate + unary + residual),
rather than a single op. Requires a TT device and SYSTEM_DESC_PATH.
"""
import pytest
import torch
import ttnn

import ttnn_jit
from ttnn_jit._src.shard_advisor import AdvisorReport
from utils import create_dram_tensor


def ffn_block(x, w1, b1, w2):
    h = ttnn.matmul(x, w1)  # up-projection   [256,512]x[512,1024] -> [256,1024]
    h = ttnn.add(h, b1)  # bias            [256,1024]
    h = ttnn.relu(h)  # activation      [256,1024]
    h = ttnn.multiply(h, h)  # gate            [256,1024]
    y = ttnn.matmul(h, w2)  # down-projection [256,1024]x[1024,512] -> [256,512]
    z = ttnn.exp(y)  # unary           [256,512]
    out = ttnn.add(z, y)  # residual-ish    [256,512]
    return out


@pytest.mark.forked
def test_shard_advisor_ffn_block(device):
    advised = ttnn_jit.shard_advisor(optimization_level=2)(ffn_block)

    x = create_dram_tensor(device, (256, 512), torch.bfloat16)
    w1 = create_dram_tensor(device, (512, 1024), torch.bfloat16)
    b1 = create_dram_tensor(device, (256, 1024), torch.bfloat16)
    w2 = create_dram_tensor(device, (1024, 512), torch.bfloat16)

    report = advised(x, w1, b1, w2)

    assert isinstance(report, AdvisorReport)
    assert report.trace.function_name == "ffn_block"

    # 7 real ops (+ a func.return pseudo-op); each real op gets a final layout.
    assert report.trace.total_ops >= 7
    assert len(report.trace.final_choices) >= 7

    # This workload is tiny relative to L1, so the greedy optimizer shards the
    # whole chain into L1 with no spills.
    assert report.trace.spill.ran is True
    assert report.trace.spill.total_spills == 0

    # Both matmuls (at least) land in an L1-sharded layout.
    l1_finals = [
        fc for fc in report.trace.final_choices if "l1" in fc.chosen_layout.lower()
    ]
    assert len(l1_finals) >= 2

    # Report renders the header, the spill section, and the op names.
    assert "L1 Sharding Advisor: ffn_block" in report.text
    assert "L1 spill accounting" in report.text
    assert "ttnn.matmul" in report.text
    assert "=> FINAL:" in report.text
