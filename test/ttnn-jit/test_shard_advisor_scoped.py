# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re

import pytest
import torch
import ttnn

import ttnn_jit
from ttnn_jit._src.shard_advisor import ShardAdvisor, AdvisorReport


def _mk(dev, shape):
    return ttnn.from_torch(
        torch.randn(*shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# Pipeline-infra ttnn ops that never show up as their own decision-trace
# entries: destination-tensor allocations, the device handle, and reshards
# (which the greedy pass reports separately via `trace.edges`).
_TTNN_INFRA_OPS = {
    "ttnn.empty",
    "ttnn.get_device",
    "ttnn.to_memory_config",
    "ttnn.to_layout",
}


def _ttnn_compute_op_names(mlir_text):
    names = re.findall(r'"(ttnn\.[a-z_0-9]+)"', mlir_text)
    return [n for n in names if n not in _TTNN_INFRA_OPS]


def test_default_tracer_is_direct_ttnn():
    # A ttnn-framework model is the TTNN dialect, so the direct-TTNN tracer is
    # the default and it selects the no-lowering ttnn-input pipeline.
    adv = ShardAdvisor(lambda x: x)
    assert adv.tracer == "ttnn"
    assert adv.pipeline == "ttnn"


def test_explicit_scoped_pipeline_preserved():
    # The TTIR path stays available for ops not yet in the direct-TTNN tracer.
    adv = ShardAdvisor(lambda x: x, tracer="interception", pipeline="scoped")
    assert adv.pipeline == "scoped"


@pytest.mark.forked
def test_scoped_ffn_reports_and_is_one_to_one(device):
    # Two chained matmuls: the traced graph has exactly 2 ttnn.matmul ops; the
    # scoped pipeline must not fuse/decompose them away.
    w1 = _mk(device, (1, 1, 128, 256))
    w2 = _mk(device, (1, 1, 256, 128))

    def ffn(x):
        h = ttnn.matmul(x, w1)
        return ttnn.matmul(h, w2)

    report = ShardAdvisor(ffn, optimization_level=2, tracer="interception").run(
        _mk(device, (1, 1, 128, 128))
    )
    assert isinstance(report, AdvisorReport)
    assert "=> FINAL:" in report.text
    # 1:1: exactly two matmuls in the final IR (no fusion, no extra decomposed ops).
    assert report.ttnn_mlir.count('"ttnn.matmul"') == 2


# NOTE on process isolation: this file mirrors test_interception_decoder.py's
# one-RoPE-pipeline-per-process constraint (see that file's module docstring
# for the measured accumulation issue in tt-metal's ttnn::graph::GraphProcessor
# capture). test_scoped_decoder_one_to_one below traces + runs the greedy
# optimizer over the full Llama decoder (which touches RotaryEmbeddingLlamaOp)
# exactly once and must stay @pytest.mark.forked and the only pipeline run in
# its process. Do NOT add a second full-decoder pipeline run (e.g. a
# pipeline="full" comparison) to that test body -- empirically, a second
# pipeline call after one that touched RoPE in the same process churns
# unboundedly (see test_interception_decoder.py). The scoped-vs-full
# characterization below instead uses a small non-RoPE graph, which is safe to
# run through both pipelines in one process.
@pytest.mark.forked
def test_scoped_decoder_one_to_one(device):
    # End-to-end validation of the scoped pipeline on the real Llama decoder:
    # trace the whole prefill_forward through the interception tracer and run
    # the greedy L1 optimizer over the resulting TTIR via the scoped
    # ttir-to-ttnn-l1-advisor pipeline (no fusion/decomposition).
    from test_interception_decoder import (
        _dummy_decoder_state_dict,
        _stub_llama_config,
        _mk as _mkd,
        _HIDDEN,
        _HD,
    )
    from _autoport.functional_decoder import FunctionalDecoder
    from ttnn_jit._src.interception_tracer import trace_intercepted

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

    # Ground truth: raw TTIR straight out of the interception tracer, before
    # any pipeline runs. trace_intercepted only builds IR via monkeypatched
    # python calls -- no device kernels, no OpModel/GraphProcessor capture --
    # so calling it here ahead of the real (pipeline-running) advisor call
    # below is cheap and does not contribute to the RoPE graph-capture
    # accumulation the module note above warns about.
    raw_module, _ = trace_intercepted(traced, hidden)
    raw_ir = str(raw_module)

    report = ShardAdvisor(
        traced, optimization_level=2, tracer="interception", pipeline="scoped"
    ).run(hidden)

    assert isinstance(report, AdvisorReport)
    assert "=> FINAL:" in report.text
    assert report.trace.spill.ran

    # 1:1 mapping: signature ops of a decoder layer survive as TRACED, with
    # counts unchanged between the raw trace and the final (post-optimizer)
    # IR -- i.e. the scoped pipeline neither fuses nor decomposes them.
    # paged_fill_cache is included: it is a zero-result in-place op whose result
    # nothing consumes, so it used to be DCE'd out of the graph before the
    # optimizer saw it. The tracer's keep-alive anchor (returning otherwise-dead
    # results as extra outputs) now preserves it, so it survives 1:1 like the
    # rest.
    for op in (
        "rms_norm",
        "linear",
        "rotary_embedding_llama",
        "scaled_dot_product_attention",
        "paged_fill_cache",
    ):
        raw_count = raw_ir.count(f'"ttir.{op}"')
        final_count = report.ttnn_mlir.count(f'"ttnn.{op}"')
        assert raw_count > 0, f"fixture assumption broken: no ttir.{op} traced"
        assert final_count == raw_count, (
            f"ttnn.{op}: traced {raw_count} but final IR has {final_count} "
            f"(scoped pipeline dropped, fused, or decomposed this op)"
        )
    # linear must stay ttnn.linear -- not decomposed into ttnn.matmul.
    assert '"ttnn.matmul"' not in report.ttnn_mlir

    # The optimizer sees every traced op (nothing silently dropped): its
    # op-decision count covers at least the distinct compute ops present in the
    # final IR it produced.
    compute_ops = _ttnn_compute_op_names(report.ttnn_mlir)
    assert report.trace.total_ops >= len(compute_ops)


@pytest.mark.forked
def test_scoped_vs_full_characterization(device, tmp_path):
    # Characterization, not a strict-equality test: run a small non-RoPE graph
    # (two chained matmuls) through both the scoped and full pipelines in the
    # same process -- safe here because there's no RoPE op involved (see the
    # module note above for why the decoder test above must stay isolated to
    # a single pipeline run). Record how scoped and full differ in op count
    # and reshard count; the divergence itself is the point, not a bug.
    w1 = _mk(device, (1, 1, 128, 256))
    w2 = _mk(device, (1, 1, 256, 128))

    def ffn(x):
        h = ttnn.matmul(x, w1)
        return ttnn.matmul(h, w2)

    x = _mk(device, (1, 1, 128, 128))

    scoped = ShardAdvisor(
        ffn, optimization_level=2, tracer="interception", pipeline="scoped"
    ).run(x)
    full = ShardAdvisor(
        ffn, optimization_level=2, tracer="interception", pipeline="full"
    ).run(x)

    assert "=> FINAL:" in scoped.text
    assert "=> FINAL:" in full.text

    diff_path = tmp_path / "scoped_vs_full.txt"
    diff_path.write_text(
        f"scoped ops={scoped.trace.total_ops} full ops={full.trace.total_ops}\n"
        f"scoped reshards={sum(e.has_reshard for e in scoped.trace.edges)} "
        f"full reshards={sum(e.has_reshard for e in full.trace.edges)}\n"
    )
    print("scoped-vs-full diff written to", diff_path)


def test_shard_advisor_decorator_pipeline_passthrough():
    # Device-free: ShardAdvisor construction only sets attributes and creates
    # out_dir, so the decorator's pipeline= kwarg can be checked without
    # opening a device.
    @ttnn_jit.shard_advisor(pipeline="full")
    def model(a, b):
        return a

    assert model.advisor.pipeline == "full"

    @ttnn_jit.shard_advisor()
    def default_model(a, b):
        return a

    assert default_model.advisor.pipeline == "scoped"
