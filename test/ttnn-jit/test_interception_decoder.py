# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn

from ttnn_jit._src.shard_advisor import ShardAdvisor, AdvisorReport
from _autoport.functional_decoder import _FunctionalMLP
from models.common.modules.lazy_weight import LazyWeight


def _lazy(dev, shape):
    return LazyWeight(
        source=torch.randn(*shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        device=dev,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


# NOTE on ordering: this test is intentionally placed *before*
# test_interception_traces_functional_mlp in this file. RotaryEmbeddingLlamaOp's
# OpModel constraint query drives tt-metal's real ttnn::graph::GraphProcessor
# capture (unlike most ops, whose OpModel uses an analytical estimate), and
# that capture mechanism carries process-global state that is not reset
# between pipeline invocations. A *second* ttir_to_ttnn_runtime_pipeline() call
# in the same process that touches RoPE becomes catastrophically slow -
# confirmed via gdb: the main thread ends up stuck building/destroying an
# ever-growing nlohmann::json graph inside ScopedGraphCapture (not a true
# deadlock, just unusably slow, effectively a hang). Running RoPE only once
# per process - by keeping this the first ShardAdvisor.run() call that
# touches ttnn.experimental.rotary_embedding_llama in the file - avoids the
# issue without touching any product code. `@pytest.mark.forked` does NOT fix
# this (verified): pytest-forked uses os.fork(), which copies the parent's
# already-populated process state via COW rather than starting a clean
# process image, so the fork happening *after* an earlier test already ran
# does not help. This is a pre-existing tt-metal/OpModel limitation, not a
# bug in the RoPE TTIR op, its TTIRToTTNN lowering, or its tracer handler -
# each of those was verified correct via the passing report (rope layouts hit
# "=> FINAL:") when run once. If a later test needs to add a *second*
# RoPE-touching ShardAdvisor call to this file, it will need to run in a
# genuinely separate process (e.g. a dedicated pytest invocation), not just a
# fork.
def test_interception_attention_chain_with_rope(device):
    import torch as _torch

    def mk(shape):
        return ttnn.from_torch(
            _torch.randn(*shape, dtype=_torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    cos = mk((1, 1, 32, 128))
    sin = mk((1, 1, 32, 128))
    tm = mk((1, 1, 32, 32))

    def attn(xqkv):
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv, num_heads=32, num_kv_heads=32, transpose_k_heads=False
        )
        q = ttnn.experimental.rotary_embedding_llama(
            q, cos, sin, tm, is_decode_mode=False
        )
        k = ttnn.experimental.rotary_embedding_llama(
            k, cos, sin, tm, is_decode_mode=False
        )
        out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True)
        return ttnn.experimental.nlp_concat_heads(out)

    advisor = ShardAdvisor(attn, optimization_level=2, tracer="interception")
    xqkv = mk(
        (1, 1, 32, 12288)
    )  # 12288 = 128*(32+2*32), all-32-heads for shape simplicity
    report = advisor.run(xqkv)

    assert len(report.trace.final_choices) >= 1
    assert "=> FINAL:" in report.text


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
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    report = advisor.run(x)

    assert isinstance(report, AdvisorReport)
    assert len(report.trace.final_choices) >= 1
    assert "=> FINAL:" in report.text
    # The MLP has 3 linears -> at least 3 layout decisions in the report.
    assert report.trace.total_ops >= 3
