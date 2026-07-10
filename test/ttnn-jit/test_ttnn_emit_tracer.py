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


# The full-decoder sweeps run the greedy optimizer over a graph that touches
# RotaryEmbeddingLlamaOp; keep them @pytest.mark.forked and one pipeline run per
# process (see test_interception_decoder.py's RoPE-churn note).
@pytest.mark.forked
def test_ttnn_tracer_sweeps_decoder_prefill(device):
    from test_interception_decoder import (
        _dummy_decoder_state_dict,
        _stub_llama_config,
        _mk as _mkd,
        _HIDDEN,
        _HD,
    )
    from _autoport.functional_decoder import FunctionalDecoder

    seq = 128
    dec = FunctionalDecoder.from_state_dict(
        _dummy_decoder_state_dict(),
        hf_config=_stub_llama_config(),
        layer_idx=0,
        mesh_device=device,
        max_batch_size=1,
        max_seq_len=seq,
        page_block_size=64,
    )
    cos, sin = _mkd(device, (1, 1, seq, _HD)), _mkd(device, (1, 1, seq, _HD))
    page_table = ttnn.from_torch(
        torch.zeros(1, 2, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    hidden = _mkd(device, (1, 1, seq, _HIDDEN))

    def traced(hs):
        return dec.prefill_forward(hs, rot_mats=(cos, sin), page_table=page_table)

    report = ShardAdvisor(traced, optimization_level=2, tracer="ttnn").run(hidden)
    # The whole prefill decoder is emitted + optimized as pure TTNN.
    assert "ttir." not in report.ttnn_mlir
    assert '"ttnn.split_query_key_value_and_split_heads"' in report.ttnn_mlir
    assert '"ttnn.scaled_dot_product_attention"' in report.ttnn_mlir
    assert report.ttnn_mlir.count('"ttnn.paged_fill_cache"') == 2
    assert "=> FINAL:" in report.text


@pytest.mark.forked
def test_ttnn_tracer_sweeps_decoder_decode(device):
    from test_interception_decoder import (
        _dummy_decoder_state_dict,
        _stub_llama_config,
        _mk as _mkd,
        _HIDDEN,
        _HD,
    )
    from _autoport.functional_decoder import FunctionalDecoder

    dec = FunctionalDecoder.from_state_dict(
        _dummy_decoder_state_dict(),
        hf_config=_stub_llama_config(),
        layer_idx=0,
        mesh_device=device,
        max_batch_size=1,
        max_seq_len=128,
        page_block_size=64,
    )
    cos, sin = _mkd(device, (1, 1, _HD, _HD)), _mkd(device, (1, 1, _HD, _HD))
    current_pos = ttnn.from_torch(
        torch.zeros(1, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    page_table = ttnn.from_torch(
        torch.zeros(1, 2, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x = _mkd(device, (1, 1, 1, _HIDDEN))

    def traced(hs):
        return dec.decode_forward(
            hs, current_pos=current_pos, rot_mats=(cos, sin), page_table=page_table
        )

    report = ShardAdvisor(traced, optimization_level=2, tracer="ttnn").run(x)
    # Decode op vocabulary emitted natively + optimized as pure TTNN.
    assert "ttir." not in report.ttnn_mlir
    assert '"ttnn.nlp_create_qkv_heads_decode"' in report.ttnn_mlir
    assert '"ttnn.paged_scaled_dot_product_attention_decode"' in report.ttnn_mlir
    assert report.ttnn_mlir.count('"ttnn.paged_update_cache"') == 2
    assert "=> FINAL:" in report.text
