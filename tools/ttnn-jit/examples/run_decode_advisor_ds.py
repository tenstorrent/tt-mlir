# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shard-advisor run over the autoport FunctionalDecoder, tuned for the
DRAM-sharded (DS) matmul path.

Same decoder as run_decode_advisor.py, but set up so the optimizer's DS
eligibility gate can actually fire:

  * BFP4 weights          -- isDRAMShardEligible() requires a bfp4/bfp8
                             DRAM-interleaved weight.
  * decode at batch 32    -- the gate wants M % 32 == 0 and M / 32 == 1, i.e.
                             exactly one activation tile row. Batch 1 decode
                             (M = 1) is rejected outright.
  * K % 256 == 0          -- Llama-3.1-8B satisfies this for all 5 projections
                             (K = 4096 / 14336; K/32 divisible by the 8 in0
                             cores).

Env knobs: SA_BATCH (32), SA_SEQ (128), SA_MODE (decode|prefill),
SA_WEIGHT_DTYPE (bfloat4_b|bfloat8_b|bfloat16), SA_TRACER (ttnn|interception),
SA_OUT (output dir).

Run (inside the tt-mlir env, with SYSTEM_DESC_PATH set):
    python tools/ttnn-jit/examples/run_decode_advisor_ds.py
"""
import json
import os
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

SEQ = int(os.environ.get("SA_SEQ", "128"))
BATCH = int(os.environ.get("SA_BATCH", "32"))
MODE = os.environ.get("SA_MODE", "decode")
PAGE_BLOCK = 64
TRACER = os.environ.get("SA_TRACER", "ttnn")
OUT = os.environ.get("SA_OUT", "/home/mvasiljevic/shard-advice-analysis/ds-runs/llama31-8b-decode")

_DTYPES = {
    "bfloat4_b": ttnn.bfloat4_b,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat16": ttnn.bfloat16,
}
WEIGHT_DTYPE = _DTYPES[os.environ.get("SA_WEIGHT_DTYPE", "bfloat4_b")]


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
            mesh_device=dev, max_batch_size=BATCH, max_seq_len=SEQ,
            page_block_size=PAGE_BLOCK, weight_dtype=WEIGHT_DTYPE,
        )
        blocks_per_user = SEQ // PAGE_BLOCK
        page_table = ttnn.from_torch(
            torch.arange(BATCH * blocks_per_user, dtype=torch.int32).reshape(
                BATCH, blocks_per_user
            ),
            dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if MODE == "decode":
            cos, sin = mk(dev, (1, 1, HD, HD)), mk(dev, (1, 1, HD, HD))
            current_pos = ttnn.from_torch(
                torch.zeros(BATCH, dtype=torch.int32), dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            hidden = mk(dev, (1, 1, BATCH, H))

            def traced(hs):
                return dec.decode_forward(
                    hs, current_pos=current_pos, rot_mats=(cos, sin),
                    page_table=page_table,
                )
        else:
            cos, sin = mk(dev, (1, 1, SEQ, HD)), mk(dev, (1, 1, SEQ, HD))
            hidden = mk(dev, (1, 1, SEQ, H))

            def traced(hs):
                return dec.prefill_forward(
                    hs, rot_mats=(cos, sin), page_table=page_table
                )

        print(f"[ds-advisor] mode={MODE} batch={BATCH} seq={SEQ} "
              f"weight_dtype={WEIGHT_DTYPE} tracer={TRACER} out={OUT}")
        t0 = time.time()
        report = ShardAdvisor(
            traced, optimization_level=2, tracer=TRACER, out_dir=OUT
        ).run(hidden)
        print(f"\n=== advisor finished in {time.time() - t0:.1f}s ===")
        print(f"ops={report.trace.total_ops}  "
              f"final_choices={len(report.trace.final_choices)}  "
              f"spill.ran={report.trace.spill.ran}  "
              f"total_spills={report.trace.spill.total_spills}\n")
        print(report.text)

        # DS summary: the whole point of the run.
        ir = report.ttnn_mlir or ""
        n_ds = ir.count("matmul_multi_core_reuse_multi_cast_dram_sharded")
        print(f"\n=== DRAM-sharded program configs in final IR: {n_ds} ===")
        for line in ir.splitlines():
            if "dram_sharded" in line:
                print("  " + line.strip()[:200])
        with open(os.path.join(OUT, "ds_summary.json"), "w") as f:
            json.dump(
                {"mode": MODE, "batch": BATCH, "seq": SEQ,
                 "weight_dtype": str(WEIGHT_DTYPE), "tracer": TRACER,
                 "total_ops": report.trace.total_ops,
                 "final_choices": len(report.trace.final_choices),
                 "dram_sharded_matmuls": n_ds},
                f, indent=2,
            )
    finally:
        ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
