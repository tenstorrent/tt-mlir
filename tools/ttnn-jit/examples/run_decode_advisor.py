# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run the L1 shard advisor over one real Llama-3.1-8B decoder layer, DECODE phase.

Companion to run_decoder_advisor.py (prefill). Traces decode_forward -- a single
token (seq=1) with in-place paged KV-cache updates and paged SDPA-decode --
through the interception tracer, runs the greedy L1 optimizer at opt-2, and
prints the sharding report. Exercises the decode op vocabulary (RoPE decode,
paged_update_cache, paged_scaled_dot_product_attention_decode) whose greedy
layout handling the optimizer resolves without a workaround pass at opt-2.

Setup (once):
    source env/activate
    export LIBRARY_PATH="$(pwd)/.local/libnsl-shim:$LIBRARY_PATH"
    export PYTHONPATH="$TT_METAL_HOME/tools:$PYTHONPATH"
    export SYSTEM_DESC_PATH="$(pwd)/ttrt-artifacts/system_desc.ttsys"

Run:
    python tools/ttnn-jit/examples/run_decode_advisor.py
"""
import sys
import time
from types import SimpleNamespace

import torch
import ttnn

sys.path.insert(0, "test/ttnn-jit")  # for the _autoport package
from _autoport.functional_decoder import FunctionalDecoder
from ttnn_jit._src.shard_advisor import ShardAdvisor

# Llama-3.1-8B shapes.
H, I, NH, NKV, HD = 4096, 14336, 32, 8, 128
Q, KV = NH * HD, NKV * HD
SEQ = 128


def stub_config():
    return SimpleNamespace(
        model_type="llama", hidden_size=H, intermediate_size=I,
        num_attention_heads=NH, num_key_value_heads=NKV, head_dim=HD,
        rms_norm_eps=1e-5, attention_bias=False, mlp_bias=False,
        hidden_act="silu", max_position_embeddings=SEQ,
    )


def dummy_state_dict():
    p = "model.layers.0."

    def w(*s):
        return torch.randn(*s, dtype=torch.bfloat16)

    return {
        p + "self_attn.q_proj.weight": w(Q, H),
        p + "self_attn.k_proj.weight": w(KV, H),
        p + "self_attn.v_proj.weight": w(KV, H),
        p + "self_attn.o_proj.weight": w(H, Q),
        p + "mlp.gate_proj.weight": w(I, H),
        p + "mlp.up_proj.weight": w(I, H),
        p + "mlp.down_proj.weight": w(H, I),
        p + "input_layernorm.weight": w(H),
        p + "post_attention_layernorm.weight": w(H),
    }


def mk(dev, shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        torch.randn(*shape, dtype=torch.bfloat16),
        dtype=dtype, layout=layout, device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def main():
    dev = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        dec = FunctionalDecoder.from_state_dict(
            dummy_state_dict(), hf_config=stub_config(), layer_idx=0,
            mesh_device=dev, max_batch_size=1, max_seq_len=SEQ, page_block_size=64,
        )
        cos, sin = mk(dev, (1, 1, HD, HD)), mk(dev, (1, 1, HD, HD))
        current_pos = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        page_table = ttnn.from_torch(
            torch.zeros(1, 2, dtype=torch.int32), dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden = mk(dev, (1, 1, 1, H))  # decode: a single token

        def traced(hs):
            return dec.decode_forward(
                hs, current_pos=current_pos, rot_mats=(cos, sin),
                page_table=page_table,
            )

        t0 = time.time()
        report = ShardAdvisor(
            traced, optimization_level=2, tracer="interception"
        ).run(hidden)
        print(f"\n=== decode advisor finished in {time.time() - t0:.1f}s ===")
        print(f"ops={report.trace.total_ops}  "
              f"final_choices={len(report.trace.final_choices)}  "
              f"spill.ran={report.trace.spill.ran}  "
              f"total_spills={report.trace.spill.total_spills}\n")
        print(report.text)
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
