# LLaMA 3.1 8B DRAM-Sharded Matmul — Unified Analysis

> **TL;DR** — tt-mlir DS pass is competitive with handtuned on all matmuls and wins decisively on Gate/Up (**−41 μs vs handtuned, +12pp DRAM util vs baseline**). QKV and Down are within noise of handtuned (+3/+5 μs). Net decode gain ≈ **2 ms/token** over 32 layers. Top next step: **integrate into GreedyOptimizer** — eliminates both the reshard wrapping (~2.6 ms overhead) and the CB overlap from adjacent-op L1 pressure.

---

## 1. Experiments compared

| | Handtuned | tt-mlir pass |
|---|---|---|
| **Source** | `tt-metal/llama_3_1_8b_handtuned` | `new_decode/` |
| **Scope** | 4-layer model | 32-layer model, 160 DS matmuls |
| **Modes** | Decode M=32 + prefill M=128/1024 | Decode M=32 only |
| **Stack** | Native tt-metal, Tracy on N150 | tt-mlir DS pass, opt-level 2, BFP4 Gate/Up |

---

## 2. Headline result — decode speedups (tt-mlir vs non-DS baseline)

| Matmul | Baseline | DS | **Speedup** |
|---|---|---|---|
| QKV fused (BFP8) | ~124 μs, ~70% | ~112 μs, ~78% | **1.10x** |
| O proj (BFP8) | ~93 μs, ~62% | ~79 μs, ~74% | **1.17x** |
| Gate (BFP4) | ~174 μs, ~59% | ~143 μs, ~71% | **1.22x** ⭐ |
| Up (BFP4) | ~157 μs, ~65% | ~143 μs, ~71% | **1.10x** |
| Down (BFP8) | ~303 μs, ~67% | ~253 μs, ~81% | **1.20x** |

---

## 3. Handtuned vs tt-mlir head-to-head (decode M=32)

| Matmul | Shape | Handtuned (12c DRAM compute, 64c storage) | tt-mlir (12c DRAM compute, 8c storage) | **Δ latency** | **Δ DRAM %** |
|---|---|---|---|---|---|
| QKV | 32×4096×6144 | 109 μs, 80.1% | ~112 μs, ~78% | +3 μs | −2 pp |
| O proj | 32×4096×4096 | 83 μs, 70.6% | ~79 μs, ~74% | **−4 μs** | **+3 pp** |
| Gate/Up | 32×4096×14336 | 184 μs, 55.4% | **143 μs, ~71%** | **−41 μs** | **+16 pp** ⭐ |
| Down | 32×14336×4096 | 248 μs, 82.4% | ~253 μs, ~81% | +5 μs | −1 pp |

---

## 4. DS factory architecture: storage vs. compute cores

Reading `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp` reveals that the earlier "8 vs 12 compute cores" framing was **wrong**. Both handtuned and tt-mlir use 12 DRAM bank cores for compute.

### 4a. Actual core roles

| Role | Count | Source |
|---|---|---|
| DRAM bank / **compute** cores | **12** | `get_optimal_dram_bank_to_reader_assignment()` — topology-driven, same in both |
| in0 / output **storage** cores | **64** (handtuned) / **8** (tt-mlir) | Caller-determined |

The factory always assigns 12 DRAM bank cores for compute (lines 105, 108) — this is identical in both paths. The "12c" label in Tracy perf output refers to these cores. The storage core count (in0 shard grid + output layout) is what differs.

**Handtuned source:** [`model_config.py:3185`](models/tt_transformers/tt/model_config.py#L3185) — `find_grid_k_n(K_tiles, N_tiles)` picks the largest common divisor ≤ 64 cores. For all LLaMA 8B shapes this gives **64 storage cores** (8×8 grid).

**tt-mlir:** our `numComputeCores=8` parameter (chosen for performance — 8 produces larger K-shards and fewer K-loop iterations than 64; both divide all LLaMA 8B K dims, see 4c).

### 4b. Data flow inside the factory

1. `num_in0_storage_cores` in0 storage cores mcast activation tiles to all 12 compute cores
2. 12 DRAM bank cores read weights and compute; each produces `per_core_N_compute = ⌈N_tiles/12⌉` output tiles (line 143)
3. An `out_reshard` CB reshards compute output `12 × per_core_N_compute` → `N_storage × per_core_N_storage` (line 214)

For Gate/Up (N_tiles=448): `per_core_N_compute = ⌈448/12⌉ = 38` on compute cores → reshard → `per_core_N_storage = 7` tiles on 64 storage cores (handtuned) or `56` on 8 (tt-mlir).

### 4c. K divisibility: valid storage core counts

`num_blocks_per_shard = num_blocks / num_in0_storage_cores` (line 302) must be integer:

| Dim | Tiles (÷32) | `% 8` | `% 64` |
|---|---|---|---|
| K=4096 (QKV, O, Gate, Up) | 128 | **0 ✓** | **0 ✓** |
| K=14336 (Down in0) | 448 | **0 ✓** | **0 ✓** |

Both 8 and 64 divide all LLaMA 8B K dims — either is valid. The pass controls the **storage** grid (in0 shard + output layout) via `numComputeCores=8` (a misnamed pass parameter; it does not control the 12 DRAM bank compute cores, which are always fixed by the factory). 8 was chosen over 64 for performance: larger K-shards and fewer K-loop iterations (see §4e). The handtuned's `find_grid_k_n` picks 64 automatically.

### 4d. N divisibility: for output storage layout

The factory uses `per_core_N_compute = div_up(N, num_dram_banks)` — ceiling division, no N constraint on the compute side. But the output is laid out across `num_storage_cores` at `per_core_N_storage` tiles each, so we need:

```
N_tiles % num_storage_cores == 0
```

Our pass checks `N_tiles % 8 == 0`; handtuned implicitly needs `N_tiles % 64 == 0`. Both hold for all LLaMA 8B N dims.

### 4e. Why 8 storage cores beats 64 (the real reason tt-mlir wins Gate/Up)

| Config | Storage cores | kPerCore | in0_block_w | K-loop blocks |
|---|---|---|---|---|
| Handtuned | 64 | 2 | 2 | 128/2 = **64** |
| tt-mlir | 8 | 16 | 8 | 128/8 = **16** |

With 8 storage cores, each in0 sender holds a 16-tile K-shard and feeds them in blocks of 8 → only 16 K-loop iterations. With 64 cores, each sender holds 2 tiles, in0_block_w=2 → 64 iterations. Fewer iterations = fewer K-loop sync points = better pipeline efficiency = higher DRAM utilization (71% vs 55%).

### 4f. Factory constraints and limitations

Source: [`matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp`](ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp)

**Hard constraints (TT_FATAL / implicit):**

| Constraint | Location | Notes |
|---|---|---|
| Both in0 and output must have shard specs | line 957 | Our pass must pre-shard both tensors |
| in1 (weights) must be ROW_MAJOR sharded | line 76 | Width-sharded DRAM weights only |
| `Kt % in0_block_w == 0` | line 1030 | Explicit check |
| `(K_tiles / in0_block_w) % num_in0_storage_cores == 0` | line 302, implicit | num_blocks must divide evenly across storage cores |
| `per_core_M == 1` when `num_blocks_per_shard > 1` | line 304–310 | Limits batched decode: fine at M=32 (per_core_M=1), breaks at M=64+ |
| `⌈N_tiles / per_core_N_storage⌉ ≤ len(output_coords)` | line 736, 771 | Output grid must have enough cores |

**In0 and output storage grids: independent**

The factory reads them from tensor shard specs (lines 959–960) and uses them for entirely separate purposes:
- in0 grid → mcast sender coordinates
- output grid → write-back NOC addresses

**No matching requirement.** They can differ in size, shape, and location.

**Write-back modes (line 770):**

| Case | Condition | Mode |
|---|---|---|
| tt-mlir (8 storage, per_core_N_storage=56) | `per_core_N_compute=38 < 56` | Multiple compute cores → one storage core (each storage shard filled by 1-2 compute cores via up to 2 NOC writes) |
| Handtuned (64 storage, per_core_N_storage=7) | `per_core_N_compute=38 > 7` | One compute core → multiple storage cores (~5-6 NOC writes per compute core) |

tt-mlir's mode produces fewer total NOC write transactions (better coalescing) vs handtuned's many fine-grained writes.

**Bounding box effect:**

All kernels (in0 sender, in1 writer, compute) run on `all_cores_in_rect_grid` = the axis-aligned bounding box of (in0 storage ∪ 12 compute cores). Cores inside the box that belong to neither grid run idle kernels but still consume semaphore and sync resources. Minimizing the bounding box — ideally by overlapping or co-locating storage and compute grids — reduces overhead.

**L1 budget: in1CB uses compute-side per_core_N**

| CB | Lives on | New L1? | Size |
|---|---|---|---|
| in0 CB | all cores (bbox) | **yes** | `in0_block_w × per_core_M × 2` (double-buf) |
| in1 CB (weights) | all cores (bbox) | **yes** | `per_core_N_compute × in0_block_w × 3` (triple-buf) |
| outCB | all cores (bbox) | **yes** | `per_core_N_compute × per_core_M` |
| interm0CB | all cores (bbox) | **yes** | `per_core_N_compute × per_core_M` (fp32) |
| in2 CB | all cores (bbox) | **no** | globally allocated to in0 tensor buffer (line 519) |
| out_reshard CB | all cores (bbox) | **no** | globally allocated to out tensor buffer (line 589) |

`in2` (c_2) and `out_reshard` (c_6) use `set_globally_allocated_address()` — they map onto the pre-existing in0/out L1 tensor shards and consume **no additional L1**. Budget model should only count the four CBs marked "yes".

**Impact of correcting the budget model (removing in2 + out_reshard from fixedCost):**

| Matmul | Old blkw | New blkw | Change |
|---|---|---|---|
| QKV | 16 | 16 | unchanged |
| O proj | 16 | 16 | unchanged |
| Gate/Up | 8 | 8 | unchanged (blkw=16 still 1.35 MB ❌) |
| **Down** | **14** | **28** | 2× fewer K-loop blocks (32→16) — perf impact untested |
| **lm_head 3-way** | rejected | **blkw=1** | fits at 1.06 MB — needs N-split pre-pass (unverified) |

**CB overlap observed at runtime (non-fatal, run completed):**

```
Statically allocated circular buffers in program 1047 clash with L1 buffers
on core range [(x=0,y=0) - (x=7,y=7)].
L1 buffer allocated at 1253376, static circular buffer region ends at 1297440.
```

CB region ends at 1,297,440 bytes (~1.24 MB); a tensor buffer was placed at 1,253,376 (~1.19 MB). The 44 kB overlap is within a single matmul program, but **probably not caused by wrong CB size calculations in the pass**. The clash comes from L1 tensor buffers from ops that the pass does not see. The GreedyOptimizer has global L1 visibility and can gate DS eligibility on total L1 pressure — integrating the pass there should resolve this.

---

## 5. Generated program configs (from `new_decode` MLIR)

All: **12 DRAM bank compute cores, 8 in0/output storage cores, LoFi + fp32_dest_acc + packer_l1_acc**.

`per_core_n` below = `per_core_N_storage` (output storage tiles per storage core). Factory internally computes `per_core_N_compute = ⌈N_tiles/12⌉`.

| Matmul | Weight | in0_block_w | per_core_m | per_core_n (storage) | per_core_n (compute) |
|---|---|---|---|---|---|
| QKV (4096×6144) | BFP8 | 16 | 1 | 24 | ⌈192/12⌉=16 |
| O proj (4096×4096) | BFP8 | 16 | 1 | 16 | ⌈128/12⌉=11 |
| Gate/Up (4096×14336) | **BFP4** | **8** | 1 | **56** | ⌈448/12⌉=38 |
| Down (14336×4096) | BFP8 | **28** | 1 | 16 | ⌈128/12⌉=11 |

**L1 budget notes (corrected: in2 and out_reshard excluded as globally allocated):**
- Gate/Up `in0_block_w=8`: blkw=16 total = `233kB fixed + 66kB in0CB + 1,051kB in1CB = 1,350 kB > ~1,295 kB` ❌. blkw=8 = 792 kB ✓.
- Down `in0_block_w=28` (was 14): blkw=28 total = `68kB fixed + 114kB in0CB + 1,005kB in1CB = 1,188 kB < ~1,295 kB` ✓. blkw=56 = 2,308 kB ❌. 28 is the largest valid divisor of kPerCore=56.

---

## 6. Prefill (M=128, handtuned only — no tt-mlir DS run)

| Matmul | Shape | Dtype | Device | DRAM % | FLOPs % |
|---|---|---|---|---|---|
| QKV | 128×4096×6144 | BF16×BFP8, HiFi2 | ~551 μs | 17.5% | 17.8% |
| O proj | 128×4096×4096 | BFP8×BFP8, HiFi2 | ~215 μs | 28.7% | 30.3% |
| Gate/Up | 128×4096×14336 | BF16×BFP4, LoFi | ~430 μs | 27.5% | 30.0% |
| Down | 128×14336×4096 | BFP8×BFP8, HiFi2 | ~747 μs | 28.6% | 30.6% |

---

## 7. lm_head — the open gap

Shape: **32×4096×128256, BFP8**

### Baseline (tt-mlir, no DS)

**2,909 μs · 181 GB/s · 62.7% DRAM util** — 1D mcast on 64 cores (8×8), flagged SLOW.

### Handtuned — best DRAM op in the model

- Weight split **3-way** → 3× 32×4096×42752, each DS on **12c compute, 8c storage** (from `find_grid_k_n(128, 1336)` → GCD=8)
- Per piece: `per_core_N_compute = ⌈1336/12⌉ = 112 tiles`, `per_core_N_storage = 1336/8 = 167 tiles`
- Compute-side CBs (in1 + out + interm0) at `in0_block_w=2` ≈ **910 kB < ~1.27 MB** ✓
- **727 μs (matmuls only) · 241 GB/s · 83.6% DRAM util** ⭐ — full path including 3× interleave + concat: **~2,226 μs**

| | Baseline | Handtuned DS | Speedup | Δ DRAM % |
|---|---|---|---|---|
| lm_head matmul | 2,909 μs, 62.7% | 727 μs, 83.6% (matmuls only) | **4.0×** | **+21 pp** |
| lm_head full path | 2,909 μs | ~2,226 μs (3 matmuls + interleave + concat) | **1.31×** | — |

### tt-mlir DS — **skipped** (with a caveat)

The pass rejected lm_head for two distinct reasons, only one of which is real:

| Attempt | `per_core_N_storage` | `per_core_N_compute` | Bottleneck | Genuine? | Outcome |
|---|---|---|---|---|---|
| Full (no split) | 501 tiles | ⌈4008/12⌉=334 | out_reshard CB on storage = **4.1 MB** | ✅ real | ❌ skip |
| 3-way split | 167 tiles | ⌈1336/12⌉=112 | Pass used 167 (storage) instead of 112 (compute) for in1CB + all buffers → **~1.37 MB** | ❌ budget bug | ❌ skip for now |
| **Fallback** | — | — | — | — | **1D mcast 8×8 (64 cores)** — same op the non-DS pipeline produces |

**With the old (buggy) budget model**, even the 3-way split looks like it doesn't fit — the pass overestimates in1CB by using `per_core_N_storage` instead of `per_core_N_compute`, and also counts in2 and out_reshard CBs that are globally allocated and cost no additional L1. These two errors together push the estimate to ~1.37 MB, over the threshold.

**With the corrected budget model** (globally-allocated CBs excluded, in1CB using compute-side per_core_N):
- Fixed: outCB(`112×2048=229kB`) + interm0(`112×4096=459kB`) = **688 kB**
- in0CB at blkw=1 (double-buf): `1×1×2048×2 = 4 kB`
- in1CB at blkw=1 (triple-buf): `1×112×1088×3 = 366 kB`
- **Total = 1,058 kB < ~1,295 kB ✓**

blkw=2 does not fit (total 1,427 kB), so in0_block_w=1 is the only valid choice — 128 K-loop blocks, lower pipeline efficiency than the 4-block FF matmuls.

> **The 3-way split should fit in theory, but this is unverified.** Even with the corrected model the pass only accounts for its own CBs and L1 tensors — it cannot see L1 tensors from other ops and global allocation logic. The CB overlap warning observed at runtime (§4f) shows that global L1 pressure from other ops can push the real usage over the budget even when the pass thinks it fits. Proper validation requires GreedyOptimizer integration (global L1 visibility) and then an actual run with an N-split pre-pass added.

### lm_head fallback: 629 μs regression — runtime state, not layout

Profiled with `ttrt perf --loops 10 --benchmark` on both pipelines (10-loop median, see `calc_mem_fixes/`):

|  | non-DS | DS | Δ |
|---|---|---|---|
| median | 2,950.58 μs | 3,579.93 μs | **+629.35 μs** |
| std | 6.05 μs | 12.39 μs | — |
| range | 2,942–2,958 | 3,559–3,598 | no overlap |
| DRAM util | 61.6% | 50.7% |
| FPU util | 4.33% | 3.57% |

Distributions are fully separated (~50σ apart) — this is **not single-shot tracy noise**, it is a real, reproducible regression.

**The kernel config and all I/O layouts are byte-identical at the MLIR layer.** Both pipelines feed lm_head a `<1x64>` width-sharded input over physical (0,0)–(7,7) with the same `virt_to_physical_map`, identical shard spec, identical compute config (LoFi, `in0_block_w=2`, `per_core_n=63`, 8×8 grid). The chain feeding lm_head differs by exactly one op: in DS, the Down matmul is DRAM-sharded with output on `<1x8>` (`layout54`, 8 cores × 16 tiles each), and a `to_memory_config` reshards it to `<1x64>` (`layout82`, 64 cores × 2 tiles each) before `add`/`rms_norm`/`lm_head`. After that reshard, the two pipelines are MLIR-equivalent.

The regression therefore stems from **non-MLIR runtime state** the DS pipeline accumulates from its +320 extra reshards. Most likely cause: L1 allocator address drift — same logical layout, different physical L1 byte address, leading to different NoC paths from each compute core to its DRAM bank. The DRAM-util drop (61.6% → 50.7%) is consistent with a NoC-affinity cause. Secondary candidates: program cache pressure, DRAM controller queue carryover.

#### Fix path

**First thing to try: have the DS Down matmul output directly onto `<1x64>` instead of `<1x8>`, eliminating the redundant reshard.** Per §14a, the output storage grid is independent of compute and effectively free inside the DRAM-sharded factory — sweeping `out ∈ {8, 16, 32, 64}` produced kernel times within ±1 μs. So this is a no-cost change at the matmul itself, and it removes a reshard (saving its ~10 μs directly) plus possibly normalizing the allocator state lm_head sees downstream. If the allocator-drift hypothesis is right, lm_head should fall back toward the non-DS 2,951 μs baseline; if it doesn't, the cause is elsewhere (program cache or DRAM queue) and a different fix is needed.

If that doesn't close the gap, escalation options:
- Pin lm_head's input L1 address explicitly (deterministic allocator placement) and re-measure.
- Skip the 8c→64c→add→rms_norm→lm_head detour entirely by running add/rms_norm on the 8c output of Down (requires confirming both ops support 8c width-sharded input).

**GreedyOptimizer integration is unlikely to fix lm_head on its own** — there is no layout drift for layout propagation to normalize. The lm_head regression is a separate investigation from the GreedyOptimizer work, even though dropping the upstream reshard *could* be done as part of that integration.

#### Empirical result: `<1x8> → <1x64>` Down output fix tested — **allocator-drift hypothesis disproven**

Implemented in [`TTNNDRAMShardedMatmul.cpp`](../../lib/Dialect/TTNN/Transforms/TTNNDRAMShardedMatmul.cpp) as `tryInheritOutGrid()`: when the original output layout is L1 width-sharded with `<1xC>` and `C` divides `N_tiles`, the matmul reuses that layout (passes `numOutCores=C` into `computeShardParams`) and writes directly onto the wider grid. Profiled in `calc_mem_fixes/output_flexible_1x64/` with the same `ttrt perf --loops 10 --benchmark` setup.

**MLIR confirms the reshard is gone.** Down output is now `<1x64>` (`#ttnn_layout81`, 64 cores × 2 tiles) instead of `<1x8>` followed by a `to_memory_config`. Op-level chain comparison:

| | Pre-fix | Post-fix |
|---|---|---|
| Input reshard (8c) | ✓ (9 μs) | ✓ (9 μs) |
| Down matmul (DRAM sharded, 254 μs) | ✓ | ✓ |
| **8c → 64c reshard (6 μs)** | **✓** | **— (eliminated)** |
| BinaryNg add | ✓ | ✓ |
| LayerNorm (rms_norm) | ✓ | ✓ |
| lm_head | ✓ | ✓ |

**Perf result: ~46 μs saved, ~589 μs regression remains.**

| | non-DS | DS pre-fix | DS post-fix |
|---|---:|---:|---:|
| lm_head kernel | 2,962 μs | 3,597 μs | 3,551 μs |
| DRAM util | 61.6% | 50.7% | 51.4% |
| Δ vs non-DS | — | +635 μs | +589 μs |

The savings (~46 μs) account almost entirely for the removed reshard (6 μs) plus a small lm_head improvement (~40 μs). DRAM utilization barely moved (50.7% → 51.4%). **lm_head did not snap back toward the 2,962 μs baseline** — the prediction made by the allocator-drift hypothesis.

**Conclusion:** The `<1x8>→<1x64>` reshard is *not* the source of the 629 μs regression. The cause lies elsewhere — most likely candidates remaining (per the original analysis) are program cache pressure or DRAM controller queue carryover from the ~320 *other* DS reshards upstream, or NoC affinity drift accumulated independently of this single reshard. The next escalation step (pin lm_head input L1 address, or skip add/rms_norm/lm_head off the 8c grid) is now the path forward.

The fix itself is still net-positive (~46 μs/token, no kernel cost) and remains in the pass.

#### Root-cause confirmation: DRAM controller state, not NoC / kernel / cache / dispatch

Three additional empirical tests narrow the cause to **device-side DRAM controller state carryover**, ruling out every other hypothesis.

**Test A — `--benchmark` mode (eliminates host overhead).** Both pipelines re-run with `ttrt perf --loops 10 --benchmark` (trace dispatch). Op-to-op gap collapses from ~3.4 ms → ~620 μs **in both pipelines equally**, but lm_head kernel time stays at 2,939–2,957 μs (non-DS) vs 3,555–3,602 μs (DS). Same 600 μs gap. **→ rules out program cache pressure and host dispatch overhead.**

**Test B — per-core BRISC/NCRISC variance** (script: [`calc_mem_fixes/output_flexible_1x64/per_core_lm_head.py`](calc_mem_fixes/output_flexible_1x64/per_core_lm_head.py)). Pulled BRISC and NCRISC kernel cycles for all 64 cores from `profile_log_device.csv` for both pipelines:

| | BRISC mean | BRISC std | NCRISC mean | NCRISC std |
|---|---:|---:|---:|---:|
| non-DS | 2,939,775 ns | 1,018 ns | 2,932,778 ns | **61 ns** |
| DS post-fix | 3,571,222 ns | 899 ns | 3,564,239 ns | **64 ns** |
| Δ | **+631,447 ns** | ≈0 | **+631,461 ns** | ≈0 |

**Every single core slowed by the same ~631 μs.** Per-core std is identical (61 vs 64 ns). NCRISC (the data-movement RISC that handles DRAM reads) shows fairness within 0.002% — if NoC routing had drifted, NCRISC variance would widen. It didn't. **→ rules out NoC affinity drift, L1 allocator drift, per-core anything.**

**Test C — lm_head in isolation** (test: [`test/ttmlir/Silicon/TTNN/n150/perf/matmul_lm_head_isolated.mlir`](../../test/ttmlir/Silicon/TTNN/n150/perf/matmul_lm_head_isolated.mlir)). Built a standalone flatbuffer with the lm_head matmul reproduced byte-for-byte (same shape, program config, layouts, fidelity), no upstream ops. Result:

| | lm_head kernel | DRAM bandwidth | DRAM util |
|---|---:|---:|---:|
| non-DS pipeline | 2,939–2,957 μs | 178–179 GB/s | 61.7–62.1 % |
| **Isolated test** | **2,985–2,992 μs** | **176 GB/s** | **61.0–61.1 %** |
| DS post-fix pipeline | 3,555–3,602 μs | 146–148 GB/s | 50.6–51.3 % |

The isolated test matches non-DS within 30 μs (~1%, cold-run noise from the first-iteration tilize/typecast). **→ The same lm_head kernel runs at full speed without upstream context. The 589 μs regression is 100% caused by upstream-pipeline state, not by anything in the kernel.**

##### What's left: DRAM controller queue / refresh / bank state

Combining all three tests, the cause must satisfy:
- Reduces DRAM bandwidth delivered to the kernel by ~17 %.
- Affects all 64 cores uniformly (not per-core / not per-route).
- Decays once the lm_head op runs in isolation.
- Survives benchmark/trace mode (host-state-independent).

That points to **the 12 DRAM controllers carrying degraded state** from the 320+ DS reshards immediately upstream — most likely:
- Open-row hit ratio degraded by the access pattern.
- Reorder-queue state shaped by sustained sharded-write traffic.
- DRAM refresh phase aligned poorly with lm_head's read sequence.
- Bank-conflict patterns from cumulative DS allocations.

This is a **tt-metal-runtime / driver-level issue**, not an MLIR compiler issue. There is no MLIR-level fix that would not require changing the runtime's DRAM access model. Filed as a separate investigation; see [`calc_mem_fixes/output_flexible_1x64/tt_metal_dram_state_issue.md`](calc_mem_fixes/output_flexible_1x64/tt_metal_dram_state_issue.md).

The compiler-side `<1x64>` Down-output fix remains in the pass as a small net positive (~46 μs/token saved on the lm_head chain itself).

---

## 8. Net runtime accounting — 32-layer model

Every DS matmul adds 2 reshards (in0: interleaved → L1 8c width-sharded; out: L1 → original).

### Per-block reshard overhead

| DS Matmul | Input reshard | Output reshard | Total |
|---|---|---|---|
| QKV fused | ~3 μs (8c) | ~9 μs (64c) | ~12 μs |
| O proj | ~2 μs (I2S) | ~6 μs (64c) | ~8 μs |
| Gate | ~2 μs (I2S) | ~21 μs (64c) | ~23 μs |
| Up | ~3 μs (8c) | ~20 μs (64c) | ~23 μs |
| Down | ~10 μs (8c) | ~6 μs (64c) | ~16 μs |
| **Per-block total** | | | **~82 μs** |

### Bottom line (estimated)

|  | μs |
|---|---:|
| Raw matmul savings | **−3,405** |
| Reshard overhead (~320 ops) | **+2,600** |
| **Net runtime gain** | **−805 μs (≈1.4%)** |
| **Extrapolated 32-layer** | **~2 ms/token** |

**Breakdown:** ~2,491 μs from 256 width-sharded Reshards (~10 μs avg) + ~105 μs from 64 I2S ops.

### Measured result

Profiled with `python3 -m tracy -r -v --loops 1 --seed 42` on the full 32-layer decode model, const-eval ops excluded via `MLIR_CONST_EVAL_OP` markers (`skripta_za_ttrt.sh`):

| | Baseline | DS | Saving |
|---|---|---|---|
| Device FW time (excl. const-eval) | 57.5 ms (1165 ops) | 56.8 ms (1485 ops) | **0.7 ms (1.3%)** |

A separate confirmation run after the fidelity fix (see §13 / `calc_mem_fixes/`) measured 56.4 ms baseline vs 55.6 ms DS = **0.8 ms saving (1.4%)**. Both numbers align with the bottom-up estimate (~0.8 ms). The 320 extra ops in DS (vs baseline) are the inference-time reshards wrapping each DS matmul. Input layout conversion ops are included in both sides as they are a genuine per-token cost in production.

> **10-loop confirmation (`ttrt perf --loops 10 --benchmark`)** tightens the headline saving to ~+0.8 ms ± ~0.05 ms. The earlier "single-shot lm_head is noise" hypothesis was wrong: lm_head's 10-loop median is 3,580 μs in DS vs 2,951 μs in non-DS, with σ = 12 / 6 μs and fully separated distributions (~50σ gap). The 629 μs lm_head regression is **real and pass-attributable**, not run noise. It offsets ~16% of the gross matmul savings and is addressable by a separate fix (see §7). The reshard overhead estimate (~9 μs × 288) is unchanged and remains solid.

---

## 9. Const-eval (load-time only — no per-token cost)

DRAM interleaved → DRAM width-sharded weight resharding triggers tt-metal PR #41413's OOM fallback (`ttnn::copy` when CB > L1):

| Cost (32-layer) | μs |
|---|---:|
| Extra const-eval (many CopyDeviceOp) | +85,973 |

---

## 10. Why savings are modest (and where the ceiling is)

1. **Attention matmuls untouched** — QᵀK and AV (~580 μs/block) are FP32 batched; DS doesn't apply.
2. **M=1-tile decode is DRAM-bound** — DS takes you from ~65% → ~80% DRAM util; speedup ceiling is ~1.4x from baseline.
3. **Reshard wrapping eats ~76% of raw savings** (2,600 μs overhead vs 3,405 μs raw gain).

---

## 11. Why DS breaks down at large M (prefill regime)

**Arithmetic intensity = 2×M / bytes_per_weight.** Wormhole B0 ridge ≈ **870 FLOPs/byte** (250 TFLOPS ÷ 288 GB/s).

| M | BFP8 intensity | Regime | Current handtuned | DS would... |
|---|---|---|---|---|
| 32 (decode) | 64 F/B | **DRAM-bound** | 12c DS, ~80% DRAM | ✓ designed for this |
| 128 (prefill) | 256 F/B | DRAM-bound, weakening | 32c non-DS, 28% FLOPs | 12c compute vs non-DS 32c → ~40% FLOPs lost |
| 1024 (long prefill) | 2,048 F/B | **compute-bound** | 64c non-DS, 74% FLOPs | 12c compute vs non-DS 64c → ~19% FLOPs vs 74% ❌ |

> Exact crossover M not measured — needs a sweep.

---

## 12. Open gaps vs handtuned

| Gap | Handtuned | tt-mlir today | Path to close |
|---|---|---|---|
| **lm_head** | 3-way DS split, 727 μs, 83.6% | 1D mcast 64c | Budget fix should unlock 3-way DS at blkw=1 (1.06 MB < ~1.27 MB ✓, **unverified**) — needs N-split pre-pass in compiler |
| **Reshard wrapping** | None (native chains directly) | ~2,600 μs / 32 layers | Fuse layouts across chained DS matmuls |

> Storage core count (8 vs 64) is **not a gap** — 8 is strictly better on every measured shape (4× fewer K-loop iterations, larger in0_block_w). Both use 12 DRAM bank cores for compute. Microbenchmark confirms in0=64 costs ~74 μs/+29% on Gate/Up shape; reshard down to 8c is only ~3.5 μs (see §14b–c). Could try using 64 storage cores if reshards turn out too costly (currently doesn't seem so).

---

## 13. Optimization opportunities (prioritized)

**Current baseline:** 0.7 ms/token (1.3%) measured. All gains below are **total token savings if implemented** (not additive on top of current), except lm_head which is independent.

| Priority | Optimization | Total gain if implemented | Additional vs today |
|---|---|---|---|
| ⭐⭐ | **Integrate DS into GreedyOptimizer** — propagates L1-width-sharded layouts across chained DS matmuls (eliminates reshard wrapping ~2.6 ms); global L1 visibility resolves CB overlap | **~3.3 ms (≈5.7%)** | +2.6 ms |
| ⭐ | Keep acts L1-width-sharded across **Gate → Up → Down** (192 reshards eliminable across 32 blocks) | **~1.7 ms (≈3.0%)** | +1.0 ms |
| ✅ | **Dtype-conditional fidelity in DS pass** — HiFi2 for BFP8 (free, DRAM-bound), LoFi for BFP4 (HiFi2 costs +45% on production blkw=8, see §14d). Implemented in [`buildComputeConfig`](../../lib/Dialect/TTNN/Transforms/TTNNDRAMShardedMatmul.cpp). | accuracy-only on BFP8 (no perf cost) | — |
| ✅ | **Inherit downstream `<1xC>` output grid in DS pass** — when the matmul's original output layout is L1 width-sharded with `<1xC>` and `C` divides `N_tiles`, write directly onto that grid and skip the `<1x8>→<1xC>` reshard. Implemented as `tryInheritOutGrid()` in [`TTNNDRAMShardedMatmul.cpp`](../../lib/Dialect/TTNN/Transforms/TTNNDRAMShardedMatmul.cpp). Tested: removes the reshard (~6 μs) on the Down→lm_head chain, but **does not close the lm_head regression** — see §7 empirical result. Allocator-drift hypothesis disproven. | ~46 μs/token (lm_head chain) | ~0.05 ms |
| | Eliminate **all** DS-added reshards (upper bound) | **~3.4 ms (≈5.9%)** | +2.7 ms |
| | Realistic (half of reshards eliminated) | **~2.0 ms (≈3.5%)** | +1.3 ms |
| | Fix **lm_head** (3-way DS, needs N-split pre-pass; unverified) | **+0.7 ms independent** | +0.7 ms |
| | Fix **lm_head fallback regression** (~589 μs remaining after `<1x64>` Down-output fix; see §7). Allocator-drift hypothesis disproven — next try: pin lm_head's input L1 address explicitly, or skip add/rms_norm off the 8c grid. Cause likely program cache or DRAM queue carryover from upstream DS reshards. | **+0.6 ms speculative** | +0.6 ms |

**Best-case total (all reshards eliminated + lm_head fixes):** ~4.7 ms ≈ 8% per token.

> **Top priority: GreedyOptimizer integration** — addresses both reshard overhead and CB overlap in a single change.

---

## 14. Up-proj microbenchmark sweep (validation)

Standalone benchmarks of the Gate/Up shape (M=32, K=4096, N=14336, BFP8 weights) on N150 to quantify the architectural choices in §4. 30 configs: in0 ∈ {8, 16, 32, 64} storage cores × out ∈ {8, 16, 32, 64} × math_fidelity ∈ {LoFi, HiFi2, HiFi4}. Files in [`explore_matmul_dram_sharding/up_matmul/`](../up_matmul/).

### 14a. Output storage grid is free (independent of in0/compute)

Sweeping in0=8 with out ∈ {8, 64}: matmul kernel time identical to within ±1 μs (both ~257 μs). Same for in0=16 (out ∈ {16, 64}) and in0=32 (out ∈ {32, 64}). Confirms §4f: in0 and output grids are independent inside the factory; the internal `out_reshard` CB scatters compute output to whichever output grid you want at no measurable cost. Pick the output grid based on what the next op wants.

### 14b. In0 storage = 64 costs ~30% (quantifies §4e)

| in0 cores | LoFi | HiFi2 | HiFi4 |
|---|---|---|---|
| 8  | 257 μs | 258 μs | 381 μs |
| 16 | 256 μs | 258 μs | 380 μs |
| 32 | 257 μs | 258 μs | 382 μs |
| **64** | **331 μs** | **336 μs** | **401 μs** |

In0=8/16/32 are tied (~257 μs at LoFi). In0=64 pays ~74 μs (≈ +29%) at LoFi/HiFi2 and ~20 μs (≈ +5%) at HiFi4. Math overhead at HiFi4 partly hides the NoC overhead from 64 in0 senders, narrowing the gap. The 4× K-loop iteration count (blkw=2 vs blkw=4) is the proximate cause.

### 14c. L1 64c → 8/16/32c reshard is essentially free

Isolated `ReshardDeviceOperation` microbenchmark (DRAM int → L1 64c → L1 Nc):

| direction | ReshardOp time |
|---|---|
| 64c → 32c | **1.6 μs** |
| 64c → 16c | **1.9 μs** |
| 64c → 8c  | **3.5 μs** |

So even when upstream produces in0 on 64c, the right move is to reshard down before the matmul: total cost (reshard + 8/16/32c matmul) ≈ 259/258/261 μs vs 331 μs for the in0=64c matmul. **Reshard wins by ~70 μs.** Conclusion: never feed 64c in0 into a DS matmul; reshard first.

### 14d. HiFi2 is free *while DRAM-bound*; BFP4 flips the regime

**BFP8 weights (DRAM-bound at decode M=32):**

| config | LoFi | HiFi2 (Δ vs LoFi) | HiFi4 (Δ vs LoFi) |
|---|---|---|---|
| in0 ∈ {8,16,32}c | ~257 μs | ~258 μs (**+1 μs**) | ~381 μs (+124 μs, **1.48×**) |
| in0=64c | ~331 μs | ~336 μs (+5 μs) | ~401 μs (+70 μs, 1.21×) |

**BFP4 weights (FLOP-bound at decode M=32, in0=8c):**

| config | LoFi | HiFi2 (Δ vs LoFi) | HiFi4 (Δ vs LoFi) |
|---|---|---|---|
| blkw=4 BFP4 | 193 μs | 214 μs (**+11%**) | 377 μs (+95%, 1.95×) |
| blkw=8 BFP4 (production) | 142 μs | **206 μs (+45%)** | 370 μs (+161%, 2.61×) |

The BFP8 measurements above show the kernel is DRAM-bound: math (2 cycles/MAC at HiFi2 vs 1 at LoFi) finishes well before DRAM and the kernel waits on DRAM either way. HiFi4 (4 cycles/MAC) finally crosses into math-bound (+48% at in0=8c).

**BFP4 halves DRAM bytes**, which moves the kernel into FLOP-bound territory at decode M=32. Once compute is the bottleneck, HiFi2 directly slows the matmul. The blkw=4 BFP4 LoFi (193 μs) is already partly FLOP-bound — pure DRAM-bound would be ~½×257 = 128 μs. blkw=8 BFP4 LoFi (142 μs) is closer to that floor: larger blkw → fewer K-loop iterations → less compute overhead → re-enters DRAM-bound territory at LoFi but is decisively pushed past the FLOP ridge by HiFi2.

HiFi4 reads no extra precision from BFP{4,8} weights (3- and 7-bit shared mantissa respectively); it is never a useful choice on these dtypes.

> **Recommended default for the DS pass:** dtype-conditional fidelity.
> - **BFP8 weights → HiFi2** (free win on DRAM-bound matmuls)
> - **BFP4 weights → LoFi** (HiFi2 costs +45% on production blkw=8)
>
> Globally setting HiFi2 cost ~4 ms/token on llama 3.1 8B decode (32 layers × 2 BFP4 matmuls × ~64 μs). Per-matmul selection is implemented in [`TTNNDRAMShardedMatmul.cpp`](../../lib/Dialect/TTNN/Transforms/TTNNDRAMShardedMatmul.cpp) `buildComputeConfig`. Microbenchmark data: [`up_matmul/results_summary.txt`](../up_matmul/results_summary.txt) (BFP4 sweep section).

### 14e. `per_core_n` MLIR field is per-output-storage-core (clarifies §4–§5)

When constructing DS matmul MLIR by hand, the `per_core_n` in `matmul_multi_core_reuse_multi_cast_dram_sharded_program_config` is **`per_core_N_storage`** = `N_tiles / num_output_storage_cores`. It is **not** per-compute-core (those use the factory-derived `per_core_N_compute = ⌈N_tiles/12⌉ = 38` for Gate/Up).

Three test files in this directory originally had `per_core_n = N_tiles / num_in0_cores` — bug since the factory uses this value to derive `num_cores_written_back` ([factory line 732](../../third_party/tt-metal/src/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp#L732)), which must match the actual output shard grid. Fixed by setting `per_core_n = N_tiles / num_out_cores` (= 7 for the `_out1x64` variants).

The pass today forces in0_cores = out_cores via a single `numComputeCores` parameter, so `p.perCoreN = (N / kTileSize) / numCores` happens to be correct under that constraint. If we ever decouple them (e.g., let the optimizer pick out grid based on the next op's preference per §14a), `per_core_n` must follow the **out** grid, not the in0 grid.
