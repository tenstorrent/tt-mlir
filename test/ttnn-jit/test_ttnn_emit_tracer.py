# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Direct-TTNN emission tracer: trace a ttnn fn to TTNN IR (no TTIR) and advise.

Exercises the trace_ttnn -> ttnn-to-ttnn-l1-advisor path: the tracer emits the
TTNN dialect directly (default DRAM-interleaved layouts) and the greedy L1
optimizer runs with no lowering.
"""
import pytest
import torch
import ttnn

from ttnn_jit._src.shard_advisor import ShardAdvisor, AdvisorReport
from ttnn_jit._src.ttnn_emit_tracer import trace_ttnn


def _mk(dev, shape):
    return ttnn.from_torch(
        torch.randn(*shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def test_ttnn_tracer_forces_ttnn_pipeline():
    # The direct-TTNN tracer emits TTNN, so the scoped TTIR pipeline can't
    # consume it; the advisor must switch to the ttnn-input pipeline.
    adv = ShardAdvisor(lambda x: x, tracer="ttnn")
    assert adv.pipeline == "ttnn"


@pytest.mark.forked
def test_ttnn_tracer_emits_ttnn_directly(device):
    # A tiny FFN: the traced module must be pure TTNN (no ttir.* ops) with
    # layout encodings assigned by the tracer, not by ttnn-layout.
    w1 = _mk(device, (1, 1, 128, 256))
    w2 = _mk(device, (1, 1, 256, 128))

    def ffn(x):
        h = ttnn.matmul(x, w1)
        h = ttnn.silu(h)
        return ttnn.matmul(h, w2)

    module, _out_type = trace_ttnn(ffn, _mk(device, (1, 1, 128, 128)))
    text = str(module)
    # Emitted directly as TTNN: matmuls/silu present, no TTIR survives.
    assert text.count('"ttnn.matmul"') == 2
    assert '"ttnn.silu"' in text
    assert "ttir." not in text
    # Every tensor carries a ttnn_layout encoding (synthesized by the tracer).
    assert "#ttnn.ttnn_layout" in text
    # Weights lifted to parameter args, activation tagged input.
    assert "#ttcore.argument_type<parameter>" in text
    assert "#ttcore.argument_type<input>" in text


@pytest.mark.forked
def test_ttnn_tracer_advises_l1_sharding(device):
    # End-to-end: trace to TTNN + run the ttnn-to-ttnn-l1-advisor. The optimizer
    # must shard the matmuls into L1 and the report must stay 1:1.
    w1 = _mk(device, (1, 1, 128, 256))
    w2 = _mk(device, (1, 1, 256, 128))

    def ffn(x):
        h = ttnn.matmul(x, w1)
        return ttnn.matmul(h, w2)

    report = ShardAdvisor(ffn, optimization_level=2, tracer="ttnn").run(
        _mk(device, (1, 1, 128, 128))
    )
    assert isinstance(report, AdvisorReport)
    assert "=> FINAL:" in report.text
    # 1:1: exactly two matmuls, no lowering-introduced fusion/decomposition.
    assert report.ttnn_mlir.count('"ttnn.matmul"') == 2
    # The optimizer reached an L1 layout (the advisor's whole purpose).
    assert "l1" in report.text
