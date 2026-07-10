# L1 Shard Advisor — Llama-3.1-8B decode layer

Findings from running the ttnn-jit L1 shard advisor over one real
`FunctionalDecoder.decode_forward` (single token, seq=1) through the scoped
`ttir-to-ttnn-l1-advisor` pipeline at optimization-level 2.

## How to reproduce

```bash
source env/activate
export LIBRARY_PATH="$(pwd)/.local/libnsl-shim:$LIBRARY_PATH"
export PYTHONPATH="$TT_METAL_HOME/tools:$PYTHONPATH"
export SYSTEM_DESC_PATH="$(pwd)/ttrt-artifacts/system_desc.ttsys"
python tools/ttnn-jit/examples/run_decode_advisor.py
```

Programmatic:

```python
from ttnn_jit._src.shard_advisor import ShardAdvisor
report = ShardAdvisor(traced_fn, optimization_level=2, tracer="interception").run(x)
print(report.text)          # op-layout + reshard summary (+ decision-trace rationale)
report.ttnn_mlir            # authoritative final TTNN IR
```

Run stats: `ops=25  final_choices=22  spill.ran=True  total_spills=1` in ~8.5s.

## Advisor output — per-op L1 layout (from final TTNN IR)

```
[1]  ttnn.rms_norm                                 -> l1/width_sharded/1x64  cores=(0,0)-(7,7)
[2]  ttnn.linear (qkv)                             -> l1/width_sharded/1x64  cores=(0,0)-(7,7)
[4]  ttnn.split_query_key_value_and_split_heads    -> l1/interleaved/8x8
[8]  ttnn.rotary_embedding_llama (q, decode)       -> l1/height_sharded/1x1 cores=(0,0)-(0,0)
[9]  ttnn.rotary_embedding_llama (k, decode)       -> l1/height_sharded/1x1 cores=(0,0)-(0,0)
     ttnn.paged_update_cache (k, v)                -> in-place; DRAM cache, value input HeightSharded
[10] ttnn.paged_scaled_dot_product_attention_decode-> dram/interleaved/1x1
[12] ttnn.linear (o_proj)                          -> l1/width_sharded/1x64  cores=(0,0)-(7,7)
[13] ttnn.add (attn residual)                      -> l1/width_sharded/1x64  cores=(0,0)-(7,7)
[15] ttnn.rms_norm                                 -> l1/width_sharded/1x64  cores=(0,0)-(7,7)
[16] ttnn.linear (gate) / [17] linear (up)         -> l1/width_sharded/1x64  cores=(0,0)-(7,7)
[18] ttnn.silu / [19] ttnn.multiply                -> l1/width_sharded/1x64  cores=(0,0)-(7,7)
[20] ttnn.linear (down)                            -> l1/width_sharded/1x64  cores=(0,0)-(7,7)
[21] ttnn.add (mlp residual)                       -> l1/width_sharded/1x64  cores=(0,0)-(7,7)
```

Key reshards the advisor inserts: RoPE activation input `l1/interleaved/8x8 ->
l1/height_sharded/1x1`; the const cos/sin/trans_mat tables reshard to
`height_sharded`; the paged index tensors (`update_idxs`, `page_table`) get
`to_layout` RowMajor conversions.

## Comparison vs hand-tuned expert (`optimized_decoder.py`, `_attention_decode` / `_mlp_decode`)

**Agrees on the entire memory-layout skeleton:**

| op | expert | advisor | match |
|---|---|---|---|
| rms_norm | L1 width-sharded | L1 width_sharded 1×64 | yes |
| qkv / o_proj linears | L1 width-sharded | L1 width_sharded 1×64 | yes |
| RoPE (q,k) decode | height-sharded | l1 height_sharded 1×1 | yes |
| paged_update_cache | DRAM cache, in place | DRAM, threaded | yes |
| paged SDPA-decode | DRAM out | DRAM interleaved | yes |
| MLP gate/up/down | L1 width-sharded | L1 width_sharded 1×64 | yes |

**What the advisor also reports:** for the sharding strategy it picks, the
optimizer generates and backend-validates the matmul program config, and the
advisor surfaces it per op (`report.json` `program_config`) — every width-sharded
linear shows `matmul_multi_core_reuse_multi_cast_1d @8x8`.

**Expert techniques the advisor does NOT capture:**

1. **Low precision** — expert casts the MLP `silu·up` intermediate to `bfloat8_b`; advisor stays faithful to the source dtype (traces bf16/bfp8/bfp4 as written, but doesn't *recommend* a change).
2. **DRAM-sharded weight matmuls** — every expert decode linear uses `_dram_matmul_config` (weights sharded across DRAM banks). The advisor picks a valid alternative (width-sharded activation + 1d-multicast program config) and reports it; the DRAM-sharded-weight strategy is a distinct optimizer feature landing soon, at which point the advisor reports it through the same path.
3. **Compute-kernel configs** — expert pins hifi2/hifi4; the scoped advisor doesn't run SetComputeKernelConfig.

**Genuine divergences:**
- Residual adds: expert keeps them in **DRAM**; advisor keeps them **L1 width-sharded** (better if it fits L1; expert likely chose DRAM for L1 headroom).
- QKV-head split: the traced functional decoder uses `split_query_key_value_and_split_heads` (→ interleaved) where the expert uses `nlp_create_qkv_heads_decode` (→ height-sharded). This is a model-source difference, not an optimizer one.

**Bottom line:** on the axes the advisor reasons about — L1 layout / sharding and
the matmul program config for that strategy — it matches expert intent op-for-op.
The remaining gaps to a hand-tuned decode are dtype precision (faithful-to-source
by design) and the DRAM-sharded-weight matmul strategy (a coming optimizer
feature).

## Compiler fixes that made decode lower e2e

Before these, the decode graph DRAM-fell-back on RoPE and failed
`OperationValidationAndFallback`:

- `7c13b6485b` — `shouldReshardConstantOperand`: let an op reshard constant-derived
  operands (RoPE cos/sin/trans_mat `<parameter>` args → HeightSharded).
- `440532078d` — RuleBook-driven RowMajor input siblings + `PagedUpdateCacheRuleBook`:
  synthesize RowMajor candidates for the page_table / update_idxs index tensors
  (tiled-only pool otherwise has none) and drive the value input to HeightSharded.
- `1fa28bcdb0` — e2e decode advisor regression test + this example.

**Build note:** the advisor runs the pipeline through the Python bindings
(`libTTMLIRCompiler.so`), a separate artifact from `ttmlir-opt`. After changing
optimizer/pass code, a full `cmake --build build` (relink + reinstall) is
required for the fixes to reach the advisor — building only `ttmlir-opt` is not
enough.
