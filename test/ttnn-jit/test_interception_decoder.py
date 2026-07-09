# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
import ttnn

from ttnn_jit._src.shard_advisor import ShardAdvisor, AdvisorReport
from ttnn_jit._src.interception_tracer import trace_intercepted
from _autoport.functional_decoder import FunctionalDecoder, _FunctionalMLP
from models.common.modules.lazy_weight import LazyWeight


def _lazy(dev, shape):
    return LazyWeight(
        source=torch.randn(*shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=dev,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _mk(dev, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        torch.randn(*shape, dtype=torch.bfloat16),
        dtype=dtype,
        layout=layout,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# Llama-3.1-8B decoder shapes: hidden=4096, intermediate=14336,
# q=32*128=4096, kv=8*128=1024. HF stores proj weights as [out, in].
_HIDDEN, _INTER, _NH, _NKV, _HD = 4096, 14336, 32, 8, 128


def _stub_llama_config():
    return SimpleNamespace(
        model_type="llama",
        hidden_size=_HIDDEN,
        intermediate_size=_INTER,
        num_attention_heads=_NH,
        num_key_value_heads=_NKV,
        head_dim=_HD,
        rms_norm_eps=1e-5,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        max_position_embeddings=128,
    )


def _dummy_decoder_state_dict():
    p = "model.layers.0."

    def w(*shape):
        return torch.randn(*shape, dtype=torch.bfloat16)

    q, kv = _NH * _HD, _NKV * _HD
    return {
        p + "self_attn.q_proj.weight": w(q, _HIDDEN),
        p + "self_attn.k_proj.weight": w(kv, _HIDDEN),
        p + "self_attn.v_proj.weight": w(kv, _HIDDEN),
        p + "self_attn.o_proj.weight": w(_HIDDEN, q),
        p + "mlp.gate_proj.weight": w(_INTER, _HIDDEN),
        p + "mlp.up_proj.weight": w(_INTER, _HIDDEN),
        p + "mlp.down_proj.weight": w(_HIDDEN, _INTER),
        p + "input_layernorm.weight": w(_HIDDEN),
        p + "post_attention_layernorm.weight": w(_HIDDEN),
    }


# NOTE on process isolation (@pytest.mark.forked): every test here runs a
# ttir_to_ttnn_runtime_pipeline() whose OpModel constraint queries drive
# tt-metal's ttnn::graph::GraphProcessor capture. That capture carries
# process-global state that is NOT reset between pipeline invocations, and it
# accumulates: once a pipeline call has processed a large graph (especially one
# touching RotaryEmbeddingLlamaOp, as the full decoder does), any *subsequent*
# pipeline call in the same process becomes catastrophically slow - measured
# directly here as a first full-decoder run finishing in ~13s while a second
# pipeline call in the same process churned >500k lines of graph capture without
# finishing (effectively a hang). Forking each test into its own subprocess
# sidesteps this: the parent process only performs collection (it never runs a
# pipeline, so it is never poisoned), and each test body + its function-scoped
# `device` fixture run in a fresh fork with clean graph-capture state. This is
# why the marker works here even though an earlier iteration found forking
# ineffective - that earlier file mixed forked and non-forked tests, so a
# non-forked test poisoned the shared parent before the fork happened. Keep ALL
# tests in this file forked. The underlying accumulation is a pre-existing
# tt-metal/OpModel limitation, not a bug in the tracer or the RoPE TTIR op.
@pytest.mark.forked
def test_interception_traces_full_decoder(device):
    # End-to-end capstone: instantiate the real autoport FunctionalDecoder (one
    # Llama-3.1-8B decoder layer) with dummy weights, trace its whole
    # prefill_forward through the interception tracer, and run the greedy L1
    # optimizer over the resulting TTIR. Exercises the full transformer op
    # vocabulary in one graph: rms_norm x2, linear x5, qkv-split, RoPE x2, SDPA,
    # concat-heads, paged_fill_cache x2, typecast, residual adds, SiLU, multiply.
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

    cos, sin = _mk(device, (1, 1, seq, _HD)), _mk(device, (1, 1, seq, _HD))
    page_table = ttnn.from_torch(
        torch.zeros(1, 2, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    hidden = _mk(device, (1, 1, seq, _HIDDEN))

    def traced(hs):
        return dec.prefill_forward(hs, rot_mats=(cos, sin), page_table=page_table)

    advisor = ShardAdvisor(traced, optimization_level=2, tracer="interception")
    report = advisor.run(hidden)

    assert isinstance(report, AdvisorReport)
    # A full decoder layer is ~18 ops; every op should reach a FINAL choice.
    assert report.trace.total_ops >= 15
    assert len(report.trace.final_choices) >= 15
    assert "=> FINAL:" in report.text
    # Signature ops of a decoder layer must appear in the advice. ShardAdvisor
    # now defaults to the scoped 1:1 pipeline (no fusing/decomposition), so the
    # 5 projections stay ttnn.linear (as traced) rather than being decomposed
    # into ttnn.matmul the way the full runtime pipeline would.
    for op in (
        "ttnn.rms_norm",
        "ttnn.linear",
        "ttnn.rotary_embedding_llama",
        "ttnn.scaled_dot_product_attention",
    ):
        assert op in report.text, f"expected {op} in report"
    # L1 spill accounting must have run.
    assert report.trace.spill.ran


@pytest.mark.forked
@pytest.mark.forked
def test_interception_traces_decode_layer(device):
    # Decode-phase decoder traces to TTIR with the decode-variant op vocabulary
    # (nlp_create_qkv_heads_decode, paged_update_cache, RoPE decode,
    # paged_scaled_dot_product_attention_decode, nlp_concat_heads_decode), and
    # the in-place KV-cache updates thread into attention so each cache has a
    # single user. Trace-only: the decode path exercises fixed-layout ops
    # (HeightSharded RoPE/SDPA-decode inputs) whose layout handling the
    # analysis pipeline does not yet apply, so this asserts the trace, not an
    # optimizer run.
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
    cos, sin = _mk(device, (1, 1, _HD, _HD)), _mk(device, (1, 1, _HD, _HD))
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
    x = _mk(device, (1, 1, 1, _HIDDEN))

    def traced(hs):
        return dec.decode_forward(
            hs, current_pos=current_pos, rot_mats=(cos, sin), page_table=page_table
        )

    module, _ = trace_intercepted(traced, x)
    ir = str(module)

    assert ir.count('"ttir.paged_update_cache"') == 2  # k + v cache updates
    assert ir.count('"ttir.paged_scaled_dot_product_attention_decode"') == 1
    assert ir.count('"ttir.rotary_embedding_llama"') == 2  # q + k, decode mode

    # Cache threading: each in-place update's result must feed the decode SDPA
    # (attention reads the UPDATED cache), not the original cache tensor -- so
    # every paged_update_cache result appears as an SDPA operand.
    import re

    update_results = re.findall(r'(%\S+) = "ttir\.paged_update_cache"', ir)
    assert len(update_results) == 2
    sdpa_line = next(
        line
        for line in ir.splitlines()
        if "paged_scaled_dot_product_attention_decode" in line
    )
    for result in update_results:
        assert (
            result in sdpa_line
        ), f"cache update {result} not threaded into decode SDPA"


def test_interception_attention_chain(device):
    # Attention block covering the nlp_create_qkv_heads /
    # scaled_dot_product_attention / nlp_concat_heads tracer handlers through the
    # greedy optimizer on device. (The decoder capstone exercises the
    # split_query_key_value_and_split_heads / concatenate_heads path plus RoPE.)
    def attn(xqkv):
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=32, num_kv_heads=32, transpose_k_heads=False
        )
        out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)
        return ttnn.experimental.nlp_concat_heads(out)

    advisor = ShardAdvisor(attn, optimization_level=2, tracer="interception")
    # 12288 = 128*(32+2*32), all-32-heads for shape simplicity.
    xqkv = _mk(device, (1, 1, 32, 12288))
    report = advisor.run(xqkv)

    assert len(report.trace.final_choices) >= 1
    assert "=> FINAL:" in report.text


@pytest.mark.forked
def test_interception_traces_functional_mlp(device):
    # Llama-3.1-8B MLP shapes: hidden=4096, intermediate=14336. Weights are
    # stored transposed: gate/up [1,1,4096,14336], down [1,1,14336,4096].
    mlp = _FunctionalMLP(
        gate=_lazy(device, (1, 1, 4096, 14336)),
        up=_lazy(device, (1, 1, 4096, 14336)),
        down=_lazy(device, (1, 1, 14336, 4096)),
        activation_dtype=ttnn.bfloat16,
    )

    def traced(x):
        return mlp.forward(x)

    advisor = ShardAdvisor(traced, optimization_level=2, tracer="interception")
    x = _mk(device, (1, 1, 32, 4096))
    report = advisor.run(x)

    assert isinstance(report, AdvisorReport)
    assert len(report.trace.final_choices) >= 1
    assert "=> FINAL:" in report.text
    # The MLP has 3 linears -> at least 3 layout decisions in the report.
    assert report.trace.total_ops >= 3
