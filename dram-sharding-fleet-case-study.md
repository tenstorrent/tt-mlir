# Case study: DRAM-sharded matmul across the Blackhole decode fleet

Measured 2026-08-18. Blackhole, single chip (1350 MHz, 10x11 worker grid, 8 DRAM banks).
tt-mlir `ae60283368`, tt-metal `f1f4ff75`. Graphs are the `g1` decode graphs from two
tt-xla Performance Benchmark runs:

- no DRAM sharding: [run 30806889612](https://github.com/tenstorrent/tt-xla/actions/runs/30806889612)
- DRAM sharding: [run 30768002414](https://github.com/tenstorrent/tt-xla/actions/runs/30768002414)

This extends [`dram-sharded-matmul-case-study.md`](dram-sharded-matmul-case-study.md), which
measured Qwen2.5-3B alone. **Its verdict generalizes; its mechanism does not.**

## Verdict: DS is slower on every model where it changes anything

Matmul device time per decode step, traced region. 10 models measured, all fully configured
on both sides except where noted.

| model | DS us | no-DS us | DS / noDS | verdict |
|---|---|---|---|---|
| qwen_2_5_0_5b | 2209.8 | 2227.5 | 0.99x | parity (only 24/121 on DS) |
| llama_3_1_8b | 19537.5 | 17845.9 | 1.09x | DS worse |
| falcon3_3b | 9799.3 | 8309.7 | 1.18x | DS worse |
| qwen_3_8b | 26145.0 | 21523.7 | 1.21x | DS worse |
| qwen_3_0_6b | 2522.2 | 2051.9 | 1.23x | DS worse |
| falcon3_7b | 24828.1 | 20041.1 | 1.24x | DS worse |
| llama_3_2_1b | 4405.3 | 3515.5 | 1.25x | DS worse |
| falcon3_1b | 5034.5 | 3970.4 | 1.27x | DS worse |
| **qwen_2_5_3b** | 16603.1 | 8842.6 | **1.88x** | DS worse (worst case) |
| qwen_2_5_1_5b | 7126.6 | 8551.4 | *0.83x* | **not a valid A/B** — see below |

**Nine of nine fair comparisons put DS behind, by 1.09x to 1.88x. One is at parity. No model
comes out ahead.** The single apparent win is the one model whose baseline was never configured.

At the level of individual *shapes* the picture is not uniform: 4 of 32 DS-path shapes do achieve
higher bandwidth under DS (all qkv, +3.6% to +11.2%), worth -666.5 us against +22750.7 us of
losses. See the dedicated section below — "no model wins" and "no shape ever wins" are different
claims, and only the first is true.

So the original single-model conclusion holds fleet-wide: `DRAMShardedProgramConfig` does not
bring gain, with or without tweaked parameters. What was wrong in that study was the
*explanation* — see the mechanism section.

### The one row that does not count

tt-mlir emitted **no program config at all** for any of Qwen2.5-1.5B's 141 matmuls in the
no-DS compile, so ttnn's runtime fallback chose per op (the report shows 141 "1D mcast",
chosen by ttnn, not tt-mlir). Its DS compile is also incomplete: 112 DS configs and 29
unconfigured, one of them lm_head. That row therefore compares *what the two compiles
produced*, not DS against tt-mlir's multicast heuristic, and cannot support a DS win. An
earlier draft read those runtime labels as tt-mlir's choice and concluded DS "sometimes wins";
that is retracted.

It does expose a separate bug worth its own ticket: **tt-mlir assigned zero program configs to
an entire model**, and ttnn's fallback then delivered only 103.0 and 96.5 GB/s on two shapes.

## Post-patch measurement: the collapse guard, on silicon

A third CI run ([run 32161672283](https://github.com/tenstorrent/tt-xla/actions/runs/32161672283))
carries the `kMinBlockWidthFraction` guard, which declines DS when the fitted `in0_block_w` falls
below half of `kPerCore`. Static check first: DS program configs dropped in exactly four models,
by exactly the counts predicted from the pre-patch measurements.

| model | DS configs before -> after | delta | shapes declined |
|---|---|---|---|
| qwen_2_5_3b | 144 -> 108 | -36 | `11008x2048` x36 |
| qwen_3_8b | 180 -> 72 | -108 | `4096x12288` x72 + `12288x4096` x36 |
| llama_3_1_8b | 160 -> 128 | -32 | `14336x4096` x32 |
| falcon3_7b | 140 -> 56 | -84 | `3072x23040` x56 + `23040x3072` x28 |

The other six models are unchanged, also as predicted. Measured traced-region matmul time:

| model | DS | patched | no-DS | recovered | patched/noDS | was | predicted | err |
|---|---|---|---|---|---|---|---|---|
| qwen_2_5_3b | 16603.1 | 10480.5 | 8842.6 | 6122.6 | **1.19x** | 1.88x | 10468 | +0.1% |
| qwen_3_8b | 26145.0 | 21939.4 | 21523.7 | 4205.6 | **1.02x** | 1.21x | 21981 | -0.2% |
| llama_3_1_8b | 19537.5 | 18126.9 | 17845.9 | 1410.6 | **1.02x** | 1.09x | 18124 | +0.0% |
| falcon3_7b | 24828.1 | 20109.5 | 20041.1 | 4718.6 | **1.00x** | 1.24x | 20114 | -0.0% |
| llama_3_2_1b (control) | 4405.3 | 4404.5 | 3515.5 | 0.8 | 1.25x | 1.25x | 4405 | -0.0% |
| **these 5** | **91519.1** | **75060.8** | **71768.8** | **16458.2** | **1.05x** | 1.28x | 16427 | +0.2% |

**The guard does what it was built to do.** Recovery of 16458 us against 16427 predicted — 0.2%
error — and three of the four changed models land at effective parity with DS-off (1.00x-1.02x).
`llama_3_2_1b` is the control: its configs did not change and it reproduces to 0.8 us (0.02%),
which rules out drift between CI runs as an explanation for the rest.

**What the guard cannot reach.** Qwen2.5-3B remains 1.19x behind because its surviving DS shapes
are ceiling-limited rather than collapsed — gate/up at 1.31x and o_proj at 1.42x, both with `w` at
maximum. And the six untouched models keep their full penalties (1.18x-1.27x for the four that
lose), because nothing in them collapses. Fleet-wide that is the 70/30 split: the guard removes
the collapse-driven loss entirely and leaves the ~320 vs ~390 GB/s ceiling gap untouched.

## Why DS loses: it saturates ~20% below multicast

Achieved weight bandwidth (weight bytes / measured time) on identical shapes:

| | DS range | 1D mcast range |
|---|---|---|
| llama_3_2_1b | 242.8 - 294.3 GB/s | 348.6 - 378.8 GB/s |
| falcon3_7b | 275.7 - 316.3 GB/s | 258.3 - 386.3 GB/s |
| qwen_2_5_3b | 101.4 - 286.5 GB/s | 347.9 - 375.4 GB/s |

lm_head is on 1D multicast in *every* compile and lands at 386.9-389.7 GB/s across all
models, which fixes the practical DRAM ceiling. **The DS kernel tops out at 316.3 GB/s;
multicast reaches ~390.** That gap, not any single misconfiguration, is why DS loses
everywhere — and it is why tuning cannot rescue it.

### The ceiling is real, and it is not DRAM bandwidth

Splitting the 32 on-DS shapes at a 200 KB burst shows a genuine saturation knee:

| | n | GB/s range | corr(burst, GB/s) | corr(bytes, GB/s) |
|---|---|---|---|---|
| burst < 200 KB | 15 | 101.4 - 292.0 | **+0.836** | +0.215 |
| burst >= 200 KB | 17 | 282.7 - 316.3 | **-0.402** | **+0.892** |

Below the knee, burst drives bandwidth and op size is nearly irrelevant. Above it, burst buys
nothing — the correlation turns slightly negative, which is scatter — and what remains is almost
entirely op size (16.71 MB -> 287.0 GB/s, 75.20 MB -> 316.3 GB/s).

So the DS plateau is **283-316 GB/s**, and it is *not* the DRAM ceiling: multicast reaches 346-404
on the same shapes. DS is saturating against something else.

**Hypothesis, consistent with the arithmetic:** DS reads weights through the 8 bank-local cores,
one per DRAM bank. At roughly 43 GB/s of per-core NOC read bandwidth (32 B/cycle at 1350 MHz) that
is ~345 GB/s aggregate, and the observed maximum of 316.3 GB/s is 92% of it. Multicast spreads the
same reads over 63-89 worker cores, so its ceiling is DRAM rather than a handful of NOC links.
This is inference from the bandwidth arithmetic, not a direct per-core NOC measurement.

If it is right, three things follow, all consistent with the fleet data: the DS ceiling cannot be
lifted by any program-config parameter (the 8 readers are structural, one per bank); burst only
governs how close to that ceiling a shape gets; and the fix is exactly what tt-metal does for its
fast path — GCB plus the tensor prefetcher, which moves reading off the bank cores into a
streaming pipeline feeding many receivers.

**The reader count is the device's DRAM bank count, and tt-metal derives it per arch.** The
number 8 is not a tt-metal constant and is unrelated to tt-mlir's `kNumIn0Cores`; the two are
different core sets that happen to coincide on Blackhole. `Device::get_optimal_dram_bank_to_logical_
worker_assignment` (`tt_metal/impl/device/device.cpp:975`) takes `num_dram_banks =
num_dram_channels()` and hands the physical DRAM coordinates to
`get_optimal_dram_to_physical_worker_assignment` (`tt_metal/common/core_assignment.cpp:188`), which
places one worker per bank at `worker_x = dram_core.x + 1` — the core immediately to the right of
that bank's controller, with per-arch row fixups (`max(y, 2)` on Blackhole for its non-tensix rows
0-1; `y+1` at rows 0 and 6 on Wormhole, plus harvested-row and dispatch-column walking). The
program factory then sets `num_dram_banks = all_worker_cores_ordered.size()` and
`num_worker_cores = num_dram_banks`, and the in1 kernel reads only its own bank —
`dram_bank_id = get_arg_val<uint32_t>(3)`, used as `{.bank_id = dram_bank_id}` on every
`noc_async_read`.

From the SoC descriptors that gives **8 readers on Blackhole** (8 channels x 3.984 GiB = 31.9 GiB)
against **12 on Wormhole** (6 physical channels split into 12 one-GiB views = 12 GiB). So the
ceiling arithmetic above is Blackhole-specific: a Wormhole part has 1.5x the bank-local readers for
the same weight bytes, with each bank carrying a narrower shard
(`per_core_N_compute = div_up(N, num_dram_banks)`, so N/12 rather than N/8). **This predicts DS
should fare better on Wormhole than on Blackhole**, which is the direction an n150 measurement would
test. Not measured here — every number in this study is p150.

By contrast the activation side does *not* scale with the arch: `kNumIn0Cores = 8` is a tt-mlir
constant, so `kPerCore = kTiles / 8` and the eligibility gate `(K/32) % 8 == 0` are
arch-independent, and the collapse guard therefore declines the same shapes on either part. Metal's
only in0-side divisibility requirement is against the activation shard itself
(`matmul_device_operation.cpp:1259`, `(shard_shape[1] / tile_width) % in0_block_w == 0`), so the
hardcoded 8 stays legal on a 12-bank device.

The Qwen2.5-3B down_proj at 101.4 GB/s is a *second, additive* problem on top of that ceiling
(the prime-`k/core` trap below), which is why 3B is 1.88x rather than ~1.25x.

Multicast is not uniformly better: on falcon3_7b's `3072x5120` DS reaches 287.3 GB/s against
multicast's 258.3 (0.90x, DS wins that shape by 183 us). Both heuristics have bad spots; DS's
are more frequent and its ceiling is lower.

## Per-matmul comparison

Weight shape `K x N` identifies the projection. At batch 32 these are data-movement ops, so
bandwidth is the metric (metal's own model puts `PM COMPUTE` at 4.7 us against `PM BANDWIDTH`
41.0 us for the 3B MLP shapes).

### qwen_2_5_3b — worst case, 1.88x

| K x N | n | DS us | DS GB/s | DS cfg | noDS us | noDS GB/s | penalty | delta us |
|---|---|---|---|---|---|---|---|---|
| 11008x2048 (down) | 36 | 236.3 | **101.4** | DRAM-sharded | 65.9 | 363.7 | **3.59x** | +6134.6 |
| 2048x11008 (gate/up) | 72 | 83.6 | 286.5 | DRAM-sharded | 63.8 | 375.4 | 1.31x | +1427.0 |
| 2048x2048 (o_proj) | 36 | 18.2 | 245.5 | DRAM-sharded | 12.8 | 347.9 | 1.42x | +192.4 |
| 2048x2560 (qkv) | 36 | 15.9 | 350.1 | 1D mcast | 15.7 | 355.6 | 1.02x | +8.7 |
| 2048x151936 (lm_head) | 1 | 850.9 | 388.5 | 1D mcast | 853.1 | 387.6 | 1.00x | -2.1 |

The two shapes on multicast in *both* compiles return 1.00x and 1.02x, bounding run-to-run
noise under 2%.

### llama_3_2_1b — typical case, 1.25x

| K x N | n | DS us | DS GB/s | noDS us | noDS GB/s | penalty | delta us |
|---|---|---|---|---|---|---|---|
| 2048x8192 (gate/up) | 32 | 62.8 | 283.9 | 47.1 | 378.8 | 1.33x | +503.8 |
| 8192x2048 (down) | 16 | 60.6 | 294.3 | 49.4 | 361.0 | 1.23x | +179.1 |
| 2048x3072 (qkv) | 16 | 25.9 | 257.7 | 18.4 | 363.9 | 1.41x | +121.2 |
| 2048x2048 (o_proj) | 16 | 18.4 | 242.8 | 12.8 | 348.6 | 1.44x | +89.2 |
| 2048x128256 (lm_head) | 1 | 718.0 | 388.7 | 721.4 | 386.9 | 1.00x | -3.3 |

Note down_proj here is *healthy* — `k/core=32`, `in0_block_w=32`, one K-step, 256-tile burst —
and DS still loses 1.23x. This is the observation that kills the `in0_block_w` explanation as
a general account.

### falcon3_7b — largest measured, 1.24x

| K x N | n | DS us | DS GB/s | noDS us | noDS GB/s | penalty | delta us |
|---|---|---|---|---|---|---|---|
| 3072x23040 (gate/up) | 56 | 257.5 | 292.0 | 194.7 | 386.3 | 1.32x | +3518.7 |
| 23040x3072 (down) | 28 | 237.7 | 316.3 | 195.0 | 385.6 | 1.22x | +1195.6 |
| 3072x5120 | 28 | 58.2 | 287.3 | 64.7 | 258.3 | **0.90x** | -182.7 |
| 3072x3072 (o_proj) | 28 | 36.4 | 275.7 | 27.4 | 365.6 | 1.33x | +250.4 |
| 3072x131072 (lm_head) | 1 | 1102.7 | 388.0 | 1097.7 | 389.7 | 1.00x | +5.0 |

### qwen_2_5_0_5b — parity

| K x N | n | DS us | DS GB/s | DS cfg | noDS us | noDS GB/s | penalty |
|---|---|---|---|---|---|---|---|
| 4864x896 (down) | 24 | 20.7 | 223.5 | DRAM-sharded | 20.6 | 224.9 | 1.01x |
| 896x4864 (gate/up) | 48 | 19.9 | 233.0 | 1D mcast | 20.2 | 229.7 | 0.99x |
| 896x1152 (qkv) | 24 | 8.0 | 136.4 | 1D mcast | 8.2 | 134.0 | 0.98x |
| 896x896 (o_proj) | 24 | 8.0 | 107.2 | 1D mcast | 8.0 | 106.5 | 0.99x |
| 896x151936 (lm_head) | 1 | 374.9 | 385.8 | 1D mcast | 376.9 | 383.8 | 0.99x |

Only 24 of 121 matmuls take the DS path, and the one that does matches multicast exactly.
Remaining models (qwen_3_0_6b, falcon3_1b, falcon3_3b, qwen_3_8b, llama_3_1_8b) follow the
llama_3_2_1b pattern; their per-matmul CSVs are in `fleet/<model>__<variant>.matmuls.csv`.

## Attributing the delta: the matmuls, not their side effects

A reasonable hypothesis is that DS makes the matmul itself faster and then loses the gain to
reshards and layout conversions introduced alongside it. **That is not what happens.** Splitting
the traced step into matmul time, layout/reshard time (Reshard, ShardedToInterleaved,
InterleavedToSharded, ToMemoryConfig, ToLayout, Copy) and everything else:

| model | matmul DS | matmul noDS | d matmul | layout DS | layout noDS | d layout | other d | step d |
|---|---|---|---|---|---|---|---|---|
| qwen_2_5_0_5b | 2209.8 | 2227.5 | -17.7 | 232.2 | 287.9 | -55.7 | +2.1 | -71.2 |
| qwen_3_0_6b | 2522.2 | 2051.9 | +470.3 | 402.0 | 441.2 | -39.3 | +29.8 | +460.8 |
| llama_3_2_1b | 4405.3 | 3515.5 | +889.8 | 181.1 | 188.4 | -7.3 | +55.1 | +937.6 |
| falcon3_1b | 5034.5 | 3970.4 | +1064.1 | 349.3 | 363.9 | -14.6 | +37.0 | +1086.5 |
| falcon3_3b | 9799.3 | 8309.7 | +1489.6 | 550.3 | 523.3 | +27.0 | +130.0 | +1646.6 |
| qwen_2_5_3b | 16603.1 | 8842.6 | +7760.5 | 454.8 | 456.3 | -1.4 | +324.9 | +8083.9 |
| falcon3_7b | 24828.1 | 20041.1 | +4787.1 | 698.0 | 648.2 | +49.9 | +262.2 | +5099.1 |
| qwen_3_8b | 26145.0 | 21523.7 | +4621.3 | 639.9 | 552.8 | +87.1 | +213.1 | +4921.5 |
| llama_3_1_8b | 19537.5 | 17845.9 | +1691.6 | 445.3 | 334.9 | +110.5 | +158.8 | +1960.9 |

The matmul delta accounts for essentially the whole step delta in every model. Layout work moves
by at most +110 us (llama_3_1_8b) and is frequently *cheaper* under DS — -55.7 us on
qwen_2_5_0_5b, -39.3 on qwen_3_0_6b, -14.6 on falcon3_1b. So DRAM sharding does not pay for
itself with extra reshuffling; it simply runs the matmuls slower.

Two refinements worth keeping:

- **The side effects that do exist are elementwise, not reshards.** The largest non-matmul
  regressions are `BinaryNgDeviceOperation` (+154.6 us on qwen_2_5_3b, +243.1 on falcon3_7b) and
  `LayerNormDeviceOperation` (+110.4 on qwen_2_5_3b) — neighbours reading DS-shaped layouts. Real,
  but 2-5% of the matmul penalty, never enough to change a verdict.
- **On the one model where DS is not behind, layout savings are the reason.** qwen_2_5_0_5b's
  matmuls are a wash (-17.7 us) and its step is -71.2 us, three quarters of which is less layout
  work. That is the mirror image of the hypothesis: side effects help there, they just cannot
  help enough anywhere else.

## By projection: which matmul role pays the cost

Roles are inferred from shape and instance count (the IR carries no names): `lm_head` is the
single vocab-width matmul, `gate/up` is the group appearing twice per layer, `down` is the
largest-K per-layer group, `o_proj` and `qkv` are the remaining per-layer groups. **gate and up
share the same `K x N` and receive the same config, so they are physically indistinguishable in
this data and are reported as one group of two per layer.**

| projection | on DS path | penalty range | DS GB/s | noDS GB/s | fleet delta |
|---|---|---|---|---|---|
| **down** | 9/9 | 0.98x - **3.59x** | 101 - 316 | 223 - 404 | **+10948 us** |
| **gate/up** | 8/9 | 0.99x - 1.51x | 223 - 306 | 301 - 386 | **+9973 us** |
| o_proj | 8/9 | 1.04x - 1.44x | 197 - 291 | 205 - 389 | +1899 us |
| qkv | 7/9 | **0.90x** - 1.54x | 237 - 298 | 258 - 386 | **-70 us** |
| lm_head | 0/9 | ~1.00x | 385 - 390 | 384 - 390 | ~0 (control) |

Four things fall out of this that the per-model view hid:

1. **down and gate/up are the whole problem.** Together they are +20.9 ms of the ~+22.8 ms
   fleet-wide matmul cost — about 92%. o_proj adds a consistent but small +1.9 ms.
2. **qkv is net *free*, and is where every DS win lives.** All four "DS faster" shapes are qkv:
   falcon3_3b and falcon3_7b `3072x5120` (0.90x), llama_3_1_8b `4096x6144` (0.96x), qwen_3_8b
   `4096x6144` (0.97x). DS runs 287-298 GB/s there — its ordinary range — and wins only because
   multicast drops to 258-286 GB/s on those shapes. Across the fleet qkv nets -70 us.
4. **lm_head is never routed to DS in any of the nine models**, always runs 1D multicast, and
   lands at 385-390 GB/s every time. That is a nine-model control: it fixes the achievable DRAM
   bandwidth and confirms run-to-run agreement within 1%.
5. **The two smallest models look neutral for the wrong reason.** qwen_2_5_0_5b's and
   qwen_3_0_6b's down_proj come out at 1.01x and 0.98x, but their multicast baselines only reach
   222-225 GB/s — DS is not doing well there, multicast is doing badly. On every model whose
   multicast baseline reaches its normal 359-404 GB/s, DS loses down_proj by 1.22x-1.29x.

Read together with the bandwidth envelope: the DS kernel delivers a strikingly consistent
240-316 GB/s regardless of projection or model, while multicast delivers 346-404 GB/s whenever
it is well configured. DS wins exactly where multicast happens to fall below that envelope.

### down

| model | K x N | n | on DS? | w | DS us | DS GB/s | noDS us | noDS GB/s | penalty | total delta us |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen_3_0_6b | 3072x1024 | 28 | yes | 12 | 14.7 | 227.4 | 15.0 | 222.8 | 0.98x | -8.4 |
| qwen_2_5_0_5b | 4864x896 | 24 | yes | 19 | 20.7 | 223.5 | 20.6 | 224.9 | 1.01x | +3.0 |
| llama_3_2_1b | 8192x2048 | 16 | yes | 32 | 60.6 | 294.3 | 49.4 | 361.0 | 1.23x | +179.1 |
| falcon3_1b | 8192x2048 | 18 | yes | 32 | 61.0 | 292.4 | 49.7 | 358.9 | 1.23x | +203.2 |
| falcon3_3b | 9216x3072 | 22 | yes | 18 | 98.6 | 304.9 | 78.9 | 381.2 | 1.25x | +434.0 |
| qwen_3_8b | 12288x4096 | 36 | yes | 16 | 171.3 | 312.3 | 132.6 | 403.4 | 1.29x | +1393.1 |
| llama_3_1_8b | 14336x4096 | 32 | yes | 14 | 198.6 | 314.2 | 154.4 | 404.2 | 1.29x | +1414.0 |
| qwen_2_5_3b | 11008x2048 | 36 | yes | 1 | 236.3 | 101.4 | 65.9 | 363.7 | 3.59x | +6134.6 |
| falcon3_7b | 23040x3072 | 28 | yes | 18 | 237.7 | 316.3 | 195.0 | 385.6 | 1.22x | +1195.6 |

On the DS path: 9/9 shapes, penalty 0.98x-3.59x, DS 101-316 GB/s vs noDS 223-404 GB/s, total +10948.2 us

### gate/up

| model | K x N | n | on DS? | w | DS us | DS GB/s | noDS us | noDS GB/s | penalty | total delta us |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen_3_0_6b | 1024x3072 | 56 | yes | 4 | 15.0 | 222.6 | 9.9 | 336.3 | 1.51x | +284.3 |
| qwen_2_5_0_5b | 896x4864 | 48 | no | 2 | 19.9 | 233.0 | 20.2 | 229.7 | 0.99x | -13.9 |
| llama_3_2_1b | 2048x8192 | 32 | yes | 8 | 62.8 | 283.9 | 47.1 | 378.8 | 1.33x | +503.8 |
| falcon3_1b | 2048x8192 | 36 | yes | 8 | 63.1 | 282.7 | 47.1 | 378.6 | 1.34x | +575.0 |
| qwen_2_5_3b | 2048x11008 | 72 | yes | 4 | 83.6 | 286.5 | 63.8 | 375.4 | 1.31x | +1427.0 |
| falcon3_3b | 3072x9216 | 44 | yes | 6 | 101.0 | 297.9 | 78.5 | 383.3 | 1.29x | +990.8 |
| llama_3_1_8b | 4096x14336 | 64 | yes | 8 | 108.0 | 305.7 | 109.6 | 301.5 | 0.99x | -97.2 |

(That last row is the fleet's only **bfp4** shape — 33.03 MB of weights, not the 62.39 MB a bfp8
reading would imply. Its cheaper 576 B tiles are what let `w=8` fit against a 56-tile shard,
giving the largest burst in the fleet at 448 tiles.)
| qwen_3_8b | 4096x12288 | 72 | yes | 4 | 177.7 | 300.9 | 139.2 | 384.1 | 1.28x | +2770.9 |
| falcon3_7b | 3072x23040 | 56 | yes | 2 | 257.5 | 292.0 | 194.7 | 386.3 | 1.32x | +3518.7 |

On the DS path: 8/9 shapes, penalty 0.99x-1.51x, DS 223-306 GB/s vs noDS 301-386 GB/s, total +9973.4 us

### qkv

| model | K x N | n | on DS? | w | DS us | DS GB/s | noDS us | noDS GB/s | penalty | total delta us |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen_2_5_0_5b | 896x1152 | 24 | no | 4 | 8.0 | 136.4 | 8.2 | 134.0 | 0.98x | -3.5 |
| qwen_2_5_3b | 2048x2560 | 36 | no | 8 | 15.9 | 350.1 | 15.7 | 355.6 | 1.02x | +8.7 |
| qwen_3_0_6b | 1024x4096 | 28 | yes | 4 | 18.8 | 236.9 | 12.2 | 365.9 | 1.54x | +185.6 |
| llama_3_2_1b | 2048x3072 | 16 | yes | 8 | 25.9 | 257.7 | 18.4 | 363.9 | 1.41x | +121.2 |
| falcon3_1b | 2048x4096 | 18 | yes | 8 | 33.3 | 267.5 | 23.1 | 385.8 | 1.44x | +184.0 |
| falcon3_7b | 3072x5120 | 28 | yes | 12 | 58.2 | 287.3 | 64.7 | 258.3 | 0.90x | -182.7 |
| falcon3_3b | 3072x5120 | 22 | yes | 12 | 58.2 | 287.0 | 64.8 | 258.1 | 0.90x | -143.7 |
| llama_3_1_8b | 4096x6144 | 32 | yes | 8 | 89.8 | 297.7 | 93.5 | 286.1 | 0.96x | -116.9 |
| qwen_3_8b | 4096x6144 | 36 | yes | 8 | 90.2 | 296.4 | 93.5 | 286.1 | 0.97x | -117.6 |

On the DS path: 7/9 shapes, penalty 0.90x-1.54x, DS 237-298 GB/s vs noDS 258-386 GB/s, total -70.2 us

### o_proj

| model | K x N | n | on DS? | w | DS us | DS GB/s | noDS us | noDS GB/s | penalty | total delta us |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen_2_5_0_5b | 896x896 | 24 | no | 4 | 8.0 | 107.2 | 8.0 | 106.5 | 0.99x | -1.3 |
| qwen_3_0_6b | 2048x1024 | 28 | yes | 8 | 11.3 | 197.4 | 10.9 | 205.0 | 1.04x | +11.7 |
| qwen_2_5_3b | 2048x2048 | 36 | yes | 8 | 18.2 | 245.5 | 12.8 | 347.9 | 1.42x | +192.4 |
| falcon3_1b | 2048x2048 | 18 | yes | 8 | 18.3 | 243.5 | 12.9 | 346.0 | 1.42x | +97.6 |
| llama_3_2_1b | 2048x2048 | 16 | yes | 8 | 18.4 | 242.8 | 12.8 | 348.6 | 1.44x | +89.2 |
| falcon3_7b | 3072x3072 | 28 | yes | 12 | 36.4 | 275.7 | 27.4 | 365.6 | 1.33x | +250.4 |
| falcon3_3b | 3072x3072 | 22 | yes | 12 | 36.4 | 275.4 | 27.2 | 368.3 | 1.34x | +201.8 |
| llama_3_1_8b | 4096x4096 | 32 | yes | 16 | 61.3 | 290.6 | 45.9 | 388.5 | 1.34x | +494.7 |
| qwen_3_8b | 4096x4096 | 36 | yes | 16 | 61.6 | 289.4 | 46.0 | 387.5 | 1.34x | +561.6 |

On the DS path: 8/9 shapes, penalty 1.04x-1.44x, DS 197-291 GB/s vs noDS 205-389 GB/s, total +1899.4 us

### lm_head

| model | K x N | n | on DS? | w | DS us | DS GB/s | noDS us | noDS GB/s | penalty | total delta us |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen_2_5_0_5b | 896x151936 | 1 | no | 2 | 374.9 | 385.8 | 376.9 | 383.8 | 0.99x | -2.0 |
| qwen_3_0_6b | 1024x151936 | 1 | no | 2 | 426.9 | 387.2 | 429.9 | 384.5 | 0.99x | -3.0 |
| llama_3_2_1b | 2048x128256 | 1 | no | 2 | 718.0 | 388.7 | 721.4 | 386.9 | 1.00x | -3.3 |
| falcon3_1b | 2048x131072 | 1 | no | 2 | 738.2 | 386.4 | 733.8 | 388.7 | 1.01x | +4.3 |
| qwen_2_5_3b | 2048x151936 | 1 | no | 2 | 850.9 | 388.5 | 853.1 | 387.6 | 1.00x | -2.1 |
| falcon3_7b | 3072x131072 | 1 | no | 2 | 1102.7 | 388.0 | 1097.7 | 389.7 | 1.00x | +5.0 |
| falcon3_3b | 3072x131072 | 1 | no | 2 | 1103.5 | 387.7 | 1096.9 | 390.0 | 1.01x | +6.6 |
| llama_3_1_8b | 4096x128256 | 1 | no | 2 | 1432.0 | 389.8 | 1435.0 | 389.0 | 1.00x | -2.9 |
| qwen_3_8b | 4096x151936 | 1 | no | 2 | 1719.2 | 384.6 | 1705.8 | 387.6 | 1.01x | +13.3 |

Never on the DS path in any model (9 shapes), all ~1.00x — these are the controls.

## Did DS ever achieve more GB/s than multicast on the same shape?

Yes — checked exhaustively over all 45 shape groups (32 on the DS path, 13 controls).

### DS achieved higher GB/s, and the shape IS on the DS path: 6 of 32

| model | K x N | n | DS GB/s | noDS GB/s | DS advantage | total delta us |
|---|---|---|---|---|---|---|
| falcon3_7b | 3072x5120 | 28 | 287.3 | 258.3 | +11.2% | -182.7 |
| falcon3_3b | 3072x5120 | 22 | 287.0 | 258.1 | +11.2% | -143.7 |
| llama_3_1_8b | 4096x6144 | 32 | 297.7 | 286.1 | +4.1% | -116.9 |
| qwen_3_8b | 4096x6144 | 36 | 296.4 | 286.1 | +3.6% | -117.6 |
| qwen_3_0_6b | 3072x1024 | 28 | 227.4 | 222.8 | +2.0% | -8.4 |
| llama_3_1_8b | 4096x14336 | 64 | 305.7 | 301.5 | +1.4% | -97.2 |

### DS higher but shape NOT on DS path (same config both sides -> noise): 8

| model | K x N | DS GB/s | noDS GB/s | delta |
|---|---|---|---|---|
| qwen_2_5_0_5b | 896x1152 | 136.4 | 134.0 | +1.8% |
| qwen_2_5_0_5b | 896x4864 | 233.0 | 229.7 | +1.5% |
| qwen_3_0_6b | 1024x151936 | 387.2 | 384.5 | +0.7% |
| qwen_2_5_0_5b | 896x896 | 107.2 | 106.5 | +0.7% |
| qwen_2_5_0_5b | 896x151936 | 385.8 | 383.8 | +0.5% |
| llama_3_2_1b | 2048x128256 | 388.7 | 386.9 | +0.5% |
| qwen_2_5_3b | 2048x151936 | 388.5 | 387.6 | +0.2% |
| llama_3_1_8b | 4096x128256 | 389.8 | 389.0 | +0.2% |

**Calibrating against that noise floor matters.** The 13 control shapes carry the *same* 1D
multicast config in both compiles, so any difference is run-to-run variation — and they spread up
to +1.8%. That means the bottom two "wins" above (llama_3_1_8b `4096x14336` at +1.4% and
qwen_3_0_6b `3072x1024` at +2.0%) are not distinguishable from noise. The four above it are:
+3.6% to +11.2%, all comfortably clear of the floor.

So the precise statement is:

- **4 shapes show a real DS bandwidth advantage** (+3.6% to +11.2%), and all four are qkv:
  falcon3_7b and falcon3_3b `3072x5120`, llama_3_1_8b and qwen_3_8b `4096x6144`.
- **26 of 32 on-DS shapes lose.**
- Those 4-6 wins are worth **-666.5 us** in total; the losses are worth **+22750.7 us**. The wins
  are real but 34x too small to matter.

In every case the win comes from multicast underperforming on that shape (258-286 GB/s, below its
usual 346-404), not from DS exceeding its own 240-316 GB/s envelope. DS never beat a
well-configured multicast on any shape in the fleet.

## Per-matmul classification, by shape and model

`w` is `in0_block_w` from the DS graph's IR. Classes:

- **control (not on DS)** — the shape stayed on 1D multicast in both compiles. These should land
  at ~1.00x and do, which is what validates the rest of the table.
- **neutral** — on the DS path, within 5%.
- **(a) config pathology** — on DS and far below the DS kernel's own ~320 GB/s ceiling
  (<200 GB/s), i.e. mis-parameterised rather than shape-limited.
- **(b) DS ceiling** — on DS, running at 220-320 GB/s, which is what this kernel achieves; the
  loss is multicast reaching 336-404 GB/s on the same shape. Not fixable by tuning.
- **DS faster** — DS wins, invariably because multicast has a bad spot on that shape.

| model | K x N | n | w | DS us | DS GB/s | noDS us | noDS GB/s | penalty | class |
|---|---|---|---|---|---|---|---|---|---|
| qwen_2_5_0_5b | 896x4864 | 48 | 2 | 19.9 | 233.0 | 20.2 | 229.7 | 0.99x | control (not on DS) |
| qwen_2_5_0_5b | 4864x896 | 24 | 19 | 20.7 | 223.5 | 20.6 | 224.9 | 1.01x | neutral |
| qwen_2_5_0_5b | 896x151936 | 1 | 2 | 374.9 | 385.8 | 376.9 | 383.8 | 0.99x | control (not on DS) |
| qwen_2_5_0_5b | 896x1152 | 24 | 4 | 8.0 | 136.4 | 8.2 | 134.0 | 0.98x | control (not on DS) |
| qwen_2_5_0_5b | 896x896 | 24 | 4 | 8.0 | 107.2 | 8.0 | 106.5 | 0.99x | control (not on DS) |
| qwen_3_0_6b | 1024x3072 | 56 | 4 | 15.0 | 222.6 | 9.9 | 336.3 | 1.51x | (b) DS ceiling |
| qwen_3_0_6b | 1024x4096 | 28 | 4 | 18.8 | 236.9 | 12.2 | 365.9 | 1.54x | (b) DS ceiling |
| qwen_3_0_6b | 1024x151936 | 1 | 2 | 426.9 | 387.2 | 429.9 | 384.5 | 0.99x | control (not on DS) |
| qwen_3_0_6b | 3072x1024 | 28 | 12 | 14.7 | 227.4 | 15.0 | 222.8 | 0.98x | neutral |
| qwen_3_0_6b | 2048x1024 | 28 | 8 | 11.3 | 197.4 | 10.9 | 205.0 | 1.04x | neutral |
| llama_3_2_1b | 2048x8192 | 32 | 8 | 62.8 | 283.9 | 47.1 | 378.8 | 1.33x | (b) DS ceiling |
| llama_3_2_1b | 8192x2048 | 16 | 32 | 60.6 | 294.3 | 49.4 | 361.0 | 1.23x | (b) DS ceiling |
| llama_3_2_1b | 2048x128256 | 1 | 2 | 718.0 | 388.7 | 721.4 | 386.9 | 1.00x | control (not on DS) |
| llama_3_2_1b | 2048x3072 | 16 | 8 | 25.9 | 257.7 | 18.4 | 363.9 | 1.41x | (b) DS ceiling |
| llama_3_2_1b | 2048x2048 | 16 | 8 | 18.4 | 242.8 | 12.8 | 348.6 | 1.44x | (b) DS ceiling |
| falcon3_1b | 2048x8192 | 36 | 8 | 63.1 | 282.7 | 47.1 | 378.6 | 1.34x | (b) DS ceiling |
| falcon3_1b | 8192x2048 | 18 | 32 | 61.0 | 292.4 | 49.7 | 358.9 | 1.23x | (b) DS ceiling |
| falcon3_1b | 2048x131072 | 1 | 2 | 738.2 | 386.4 | 733.8 | 388.7 | 1.01x | control (not on DS) |
| falcon3_1b | 2048x4096 | 18 | 8 | 33.3 | 267.5 | 23.1 | 385.8 | 1.44x | (b) DS ceiling |
| falcon3_1b | 2048x2048 | 18 | 8 | 18.3 | 243.5 | 12.9 | 346.0 | 1.42x | (b) DS ceiling |
| falcon3_3b | 3072x9216 | 44 | 6 | 101.0 | 297.9 | 78.5 | 383.3 | 1.29x | (b) DS ceiling |
| falcon3_3b | 9216x3072 | 22 | 18 | 98.6 | 304.9 | 78.9 | 381.2 | 1.25x | (b) DS ceiling |
| falcon3_3b | 3072x5120 | 22 | 12 | 58.2 | 287.0 | 64.8 | 258.1 | 0.90x | **DS faster** |
| falcon3_3b | 3072x131072 | 1 | 2 | 1103.5 | 387.7 | 1096.9 | 390.0 | 1.01x | control (not on DS) |
| falcon3_3b | 3072x3072 | 22 | 12 | 36.4 | 275.4 | 27.2 | 368.3 | 1.34x | (b) DS ceiling |
| qwen_2_5_3b | 11008x2048 | 36 | 1 | 236.3 | 101.4 | 65.9 | 363.7 | 3.59x | **(a) config pathology** |
| qwen_2_5_3b | 2048x11008 | 72 | 4 | 83.6 | 286.5 | 63.8 | 375.4 | 1.31x | (b) DS ceiling |
| qwen_2_5_3b | 2048x151936 | 1 | 2 | 850.9 | 388.5 | 853.1 | 387.6 | 1.00x | control (not on DS) |
| qwen_2_5_3b | 2048x2048 | 36 | 8 | 18.2 | 245.5 | 12.8 | 347.9 | 1.42x | (b) DS ceiling |
| qwen_2_5_3b | 2048x2560 | 36 | 8 | 15.9 | 350.1 | 15.7 | 355.6 | 1.02x | control (not on DS) |
| falcon3_7b | 3072x23040 | 56 | 2 | 257.5 | 292.0 | 194.7 | 386.3 | 1.32x | (b) DS ceiling |
| falcon3_7b | 23040x3072 | 28 | 18 | 237.7 | 316.3 | 195.0 | 385.6 | 1.22x | (b) DS ceiling |
| falcon3_7b | 3072x5120 | 28 | 12 | 58.2 | 287.3 | 64.7 | 258.3 | 0.90x | **DS faster** |
| falcon3_7b | 3072x131072 | 1 | 2 | 1102.7 | 388.0 | 1097.7 | 389.7 | 1.00x | control (not on DS) |
| falcon3_7b | 3072x3072 | 28 | 12 | 36.4 | 275.7 | 27.4 | 365.6 | 1.33x | (b) DS ceiling |
| qwen_3_8b | 4096x12288 | 72 | 4 | 177.7 | 300.9 | 139.2 | 384.1 | 1.28x | (b) DS ceiling |
| qwen_3_8b | 12288x4096 | 36 | 16 | 171.3 | 312.3 | 132.6 | 403.4 | 1.29x | (b) DS ceiling |
| qwen_3_8b | 4096x6144 | 36 | 8 | 90.2 | 296.4 | 93.5 | 286.1 | 0.97x | **DS faster** |
| qwen_3_8b | 4096x4096 | 36 | 16 | 61.6 | 289.4 | 46.0 | 387.5 | 1.34x | (b) DS ceiling |
| qwen_3_8b | 4096x151936 | 1 | 2 | 1719.2 | 384.6 | 1705.8 | 387.6 | 1.01x | control (not on DS) |
| llama_3_1_8b | 4096x14336 | 64 | 8 | 108.0 | 305.7 | 109.6 | 301.5 | 0.99x | neutral |
| llama_3_1_8b | 14336x4096 | 32 | 14 | 198.6 | 314.2 | 154.4 | 404.2 | 1.29x | (b) DS ceiling |
| llama_3_1_8b | 4096x6144 | 32 | 8 | 89.8 | 297.7 | 93.5 | 286.1 | 0.96x | **DS faster** |
| llama_3_1_8b | 4096x4096 | 32 | 16 | 61.3 | 290.6 | 45.9 | 388.5 | 1.34x | (b) DS ceiling |
| llama_3_1_8b | 4096x128256 | 1 | 2 | 1432.0 | 389.8 | 1435.0 | 389.0 | 1.00x | control (not on DS) |

### tally
- (b) DS ceiling: 23
- control (not on DS): 13
- neutral: 4
- **DS faster**: 4
- **(a) config pathology**: 1

The distribution is the headline: **23 of 32 DS-path shapes are ceiling-limited, exactly one is
a config pathology, and 4 are cases where DS wins because multicast is the badly configured
one.** The single pathology is Qwen2.5-3B's `11008x2048` at 101.4 GB/s — the prime-`k/core`
trap. Every other DS loss is the kernel doing the best it can.

Note the four **DS faster** rows: falcon3_3b and falcon3_7b `3072x5120` (0.90x, multicast at
258 GB/s), llama_3_1_8b and qwen_3_8b `4096x6144` (0.96x/0.97x, multicast at 286 GB/s). DS
reaches 287-298 GB/s there — its normal range — and wins only because multicast dropped below it.
Their models still lose overall, because the other shapes lose more. Two further shapes are
nominally faster under DS but within the 1.8% control-noise floor; see the dedicated section
above.

## The `in0_block_w` trap: an additive penalty, not the cause

tt-mlir shards the DS activation over a hardcoded 8 cores (`MatmulRules.cpp:41`,
`kNumIn0Cores = 8`), so `kPerCore = kTiles / 8`. `computeShardParams`
(`MatmulProgramConfig.cpp:370-390`) then starts `in0_block_w` at `kPerCore` and walks
**down the divisors of `kPerCore`** until the circular buffers fit. Metal enforces the same
divisibility at `matmul_device_operation.cpp:1263`.

What predicts bandwidth is the **read burst** = `in0_block_w x per-bank shard width`, i.e. how
much weight data each bank hands over per inner iteration. Validated across all 32 on-DS shapes
in the fleet, taking weight bytes from each shape's own dtype — one shape is bfp4, the rest bfp8
(see "Weight dtypes are not uniform" under Method; assuming bfp8 everywhere yields impossible
>390 GB/s figures):

| predictor | correlation with achieved GB/s |
|---|---|
| **burst in KB** | **0.832** |
| burst in tiles | 0.775 |
| total weight bytes | 0.503 |
| burst vs bytes (the confound) | 0.458 |

The cleanest evidence is two same-size pairs, where burst is the only thing that moves:

- qwen_2_5_3b `11008x2048` and `2048x11008` are **both 23.95 MB**: burst 8.5 KB -> **101.4 GB/s**,
  burst 182.8 KB -> **286.5 GB/s**. Same bytes, 21x the burst, 2.8x the bandwidth.
- falcon3_7b `3072x23040` and `23040x3072` are **both 75.20 MB**: burst 191 KB -> 292.0 GB/s,
  burst 229 KB -> **316.3 GB/s**.

### Why an in0-named parameter governs in1's DRAM reads

`in0_block_w` blocks the **K** dimension, and K is shared: it is in0's *width* and in1's *height*.
The parameter is named from in0's perspective, but because it fixes the K extent of the block it
sizes both operands at once:

    in0 block:  per_core_M   x  in0_block_w      (M-tiles x K-tiles)
    in1 block:  in0_block_w  x  shard_n          (K-tiles x N-tiles)

which is why it appears in both CB terms in `computeShardParams`:

    in0CB = in0BlockW * perCoreM        * kBf16Tile   * (doubleBuf ? 2 : 1)
    in1CB = in0BlockW * perCoreNCompute * kWeightTile * (doubleBuf ? 3 : 1)

In **decode** the in1 term dominates completely. At batch 32, `per_core_M = ceil(32/32) = 1` — the
activation is one tile tall — so in0's block stays tiny while in1's scales with `shard_n`:

| shape | w | shard_n | in0CB | in1CB | ratio |
|---|---|---|---|---|---|
| qwen_2_5_3b down_proj | 1 | 8 | 4 KB | 25.5 KB | 6x |
| qwen_2_5_3b gate/up | 4 | 43 | 16 KB | 548 KB | 34x |
| falcon3_7b gate/up | 2 | 90 | 8 KB | 574 KB | 72x |

The two operands also sit in different places: in0 is already resident in L1 (width-sharded
activation) while in1 **streams from DRAM every iteration**. So the DRAM traffic that sets
bandwidth is entirely an in1 story, even though the knob carries in0's name. The naming is
sensible for prefill, where a large `per_core_M` makes in0's block genuinely expensive; decode is
the degenerate case where M=1 tile collapses in0's contribution to noise.

**Correction to an earlier version of this study.** It claimed saturation above ~68 tiles, taken
from the single-op 3B sweep where a 64-tile (68 KB) burst reached 279.7 GB/s. That does not
transfer: across the fleet, 68 KB bursts reach only **237-246 GB/s**, and ~300 GB/s needs a burst
around 200 KB — roughly 3x higher than claimed. The sweep held its shape at 23.95 MB, and **op
size is a real secondary term**: at comparable burst, moving 4.5 MB instead of 24-53 MB costs
6-14%. Burst dominates, size modulates, and a threshold measured on one large shape should not be
quoted as a general one.

Static survey of the DS down_proj config across the whole fleet (no device needed):

| model | down_proj K x N | k/core | `in0_block_w` | K-steps | burst tiles |
|---|---|---|---|---|---|
| **qwen_2_5_3b** | 11008x2048 | **43 (prime)** | **1** | **43** | **8** |
| qwen_2_5_0_5b | 4864x896 | 19 | 19 | 1 | 76 |
| qwen_3_0_6b | 3072x1024 | 12 | 12 | 1 | 48 |
| llama_3_2_1b | 8192x2048 | 32 | 32 | 1 | 256 |
| falcon3_1b | 8192x2048 | 32 | 32 | 1 | 256 |
| qwen_2_5_1_5b | 8960x1536 | 35 | 35 | 1 | 210 |
| falcon3_3b | 9216x3072 | 36 | 18 | 2 | 216 |
| qwen_3_8b | 12288x4096 | 48 | 16 | 3 | 256 |
| ministral_8b | 12288x4096 | 48 | 16 | 3 | 256 |
| llama_3_1_8b | 14336x4096 | 56 | 14 | 4 | 224 |
| falcon3_7b | 23040x3072 | 90 | 18 | 5 | 216 |

Since `11008 = 32 x 8 x 43`, dividing by 8 in0 cores leaves a **prime**, collapsing the
ladder to `{43, 1}` — and `w=43` needs 1096.5K of `in1CB` alone, so it is rejected (metal
reports a real L1 clash, 41 KB over). Every other model's `k/core` factorizes and gets a
burst 6x-32x larger. **Qwen2.5-3B is the only model in the fleet in this trap**, and it is
the only model at 1.88x rather than ~1.25x. Every other model gets a healthy burst — llama_3_2_1b's
down_proj runs `in0_block_w=32` with a 256-tile burst — and DS still loses 1.09x-1.27x there.
So the prime-`k/core` collapse explains why 3B is *extra* bad; it does not explain why DS loses
at all. That is the ~320 vs ~390 GB/s ceiling gap above.

### Two mechanisms look alike; only one hurts

A small `in0_block_w` is not by itself a problem. `in1CB = w x shard_n x 1088 x 3`, so a
wide per-bank shard forces `w` down while the burst stays large: falcon3_7b's gate/up gets
`w=2` and qwen_3_8b's `w=4`, but with `shard_n` of 90 and 16 tiles their bursts are 180 and
64 tiles — inside the saturated region. A heuristic keyed on `w` alone would flag these as
broken. Key on **burst tiles**, not `w`.

## Second-order finding: unconfigured ops are decided by upstream layout

On qwen_2_5_1_5b, lm_head carries **no program config in either compile**. Yet ttnn's runtime
fallback chose 1D multicast in the no-DS compile (356.0 GB/s) and `default` in the DS compile
(155.5 GB/s) — a 2.29x swing worth **898 us** on a single op that neither compile configured
and that DS does not own.

The likely cause is that enabling DS changes the layouts feeding lm_head, which changes what
ttnn's fallback picks. Two lessons: leaving a matmul unconfigured makes its performance a
function of unrelated upstream decisions, and a "DS on/off" A/B silently moves ops outside the
DS path. Both are arguments for configuring every matmul explicitly rather than for or against
DS.

## Coverage: 8 models were unprofilable until ttrt was fixed

`ttrt` cannot generate a BFP_BFloat8 host tensor:

```
Exception: unsupported dtype: BFP_BFloat8
```

so any decode graph whose `main` takes a bfp8 `<input>` argument cannot be executed at all.
In every affected model those arguments are **KV cache** tensors (`ttcore.kv_cache`,
e.g. `tensor<32x8x128x64x!ttcore.tile<32x32, bfp_bf8>>`). The Qwen 2.5 family uses a bf16
cache — all its bfp8 args are const-eval `<parameter>`s — which is the only reason those
three models are measurable.

| model | bfp8 `<input>` args | runnable |
|---|---|---|
| qwen_2_5_0_5b / 1_5b / 3b | 0 | yes |
| llama_3_2_1b | 32 | no |
| falcon3_1b | 36 | no |
| falcon3_3b | 44 | no |
| qwen_3_0_6b, falcon3_7b | 56 | no |
| llama_3_1_8b | 64 | no |
| qwen_3_8b, ministral_8b | 72 | no |

**Fixed in `tools/ttrt/common/util.py`** (two converters, both upstreamable):

1. `from_data_type` (input generation) now maps the six `BFP_*` formats to `torch.bfloat16`.
   The host tensor is generated in bf16; `create_tensor` declares the runtime dtype from the
   torch tensor itself and `convert_input_layouts` then applies the flatbuffer's real layout,
   which is what produces the block-float device tensor.
2. `ttrt_datatype_to_torch_dtype` (output readback) now maps them to `torch.float32`, because
   tt-metal unpacks block-float to float32 on host readback. bfloat16 there fails with
   `shape [...] is invalid for input of size N` at double the expected element count.

Caveat on (2): the returned torch tensor has the right size but reinterprets bytes, so output
*values* are not golden-comparable — adequate for profiling, not for `--enable-golden`.

With both patches all 8 models run. Three further models (phi1, phi1_5, phi2) emit
`ttnn.linear` and never take the DS path; gemma_1_1_2b carries no program config in either
variant. `ministral_8b` ran (25780.5 vs 21259.7 us, 1.21x) but its per-matmul join failed, so
it is excluded from the per-shape tables and the summary.

## Method

Durations are **not** taken from `ops_perf_results.csv`. On this card its
`DEVICE KERNEL DURATION` is computed as `max(ZONE_END across cores) - min(ZONE_START across
cores)`, and because some cores' cycle counters are zeroed at device init and others are not,
every multi-core op inherits a constant ~1.4288e13-cycle (~2.9 h) offset. Core count predicts
it exactly: 1-core ops 308/308 sane, 16-core ops 0/144 sane. Buffer overflow and
`TT_METAL_PROFILER_SYNC=1` were both ruled out. All timings here are recomputed **per core**
from `.logs/profile_log_device.csv` and joined to shapes/configs from the report on
`GLOBAL CALL COUNT`.

Controls: `qwen_2_5_3b__ds` re-measured from a fresh artifact download and fresh compile came
back at 33192.2 us against 33190.3 us in the earlier study (0.006%), and within each model the
shapes on 1D multicast in both compiles agree to 1-2%.

### Weight dtypes are not uniform — derive bandwidth per shape

Across 2418 matmul instances the fleet uses **2290 bfp8 and 128 bfp4** weight rows. Exactly one
shape is bfp4: `llama_3_1_8b` gate/up `4096x14336`. Every bandwidth figure here therefore takes
bytes-per-element from that shape's own `INPUT_1_DATATYPE` (bfp8 = 1.0625 B/elem, 1088 B/tile;
bfp4 = 0.5625 B/elem, 576 B/tile) rather than assuming bfp8.

This is not hypothetical bookkeeping. An early version of the burst analysis hardcoded bfp8 and
reported **577 GB/s** for that shape — impossible, since multicast tops out at ~390 GB/s on this
card — which is exactly how the error surfaced. Corrected, the maximum observed anywhere is
316.3 GB/s. **Use the ~390 GB/s DRAM ceiling as a sanity check on any derived bandwidth: a figure
above it means the byte accounting is wrong, not that the kernel is fast.**

Two consequences worth carrying:

- **The A/B is not confounded by precision.** Checked explicitly: every shape uses the *same*
  weight dtype in both the DS and no-DS compile, so no comparison in this study comes down to one
  side moving fewer bytes.
- **dtype changes the CB budget, and therefore the chosen `in0_block_w`.** `in1CB` scales with
  tile bytes, so a bfp4 tile costs 576 B where bfp8 costs 1088. That is why `llama_3_1_8b`'s
  gate/up can afford `w=8` against a 56-tile per-bank shard — a 448-tile burst, the largest in the
  fleet — while bfp8 shapes of comparable width are pushed down to `w=2` or `w=4`. Any future
  config heuristic scoring burst must read tile bytes from the dtype, not assume bfp8.

## Recommendations

0. **The collapse guard is measured and worth keeping.** It recovers 16.5 ms across the fleet's
   four affected models (0.2% from prediction) and brings three of them to parity with DS-off. It
   does not address the remaining ~6.3 ms, which is the ceiling rather than a config defect.
1. **Keep DS off for decode, or gate it per projection.** Nine of nine fair comparisons are
   worse (1.09x-1.88x), one is neutral, none better. If DS is wanted for its memory properties
   (not addressed by these measurements), the per-projection data says where it is affordable:
   **qkv nets -70 us fleet-wide and is the only role where DS ever wins**, o_proj costs a modest
   +1.9 ms, while **down and gate/up account for ~92% of the entire penalty (+20.9 ms)**. A rule
   that routes only qkv to DS would keep most of the memory benefit at roughly zero latency
   cost.
2. **The `numIn0Cores` fix is worth doing but is not the answer.** For Qwen2.5-3B's down_proj,
   feeding the existing search `numIn0Cores = 43` makes it pick `w=8` unaided (measured 85.6 us
   vs 236.3 us, 5.43 ms/iteration). It cannot be a constant bump: the gate at
   `MatmulRules.cpp:219` requires `(K/kTileSize) % kNumIn0Cores == 0`, and 43 would decline the
   K=2048 shapes. But it only recovers 3B's *extra* penalty; DS would still sit ~1.3x behind
   multicast there, as it does on every healthy-burst model.

3. **Do not use burst to decide *whether* a matmul should be DS.** Burst is a good diagnostic and
   a bad selector, and the distinction matters:

   - As a **diagnostic** it works: below the ~200 KB knee, corr(burst, DS GB/s) = +0.84, so a
     small burst reliably means DS is running below its own 283-316 GB/s envelope. That is how
     Qwen2.5-3B's `11008x2048` (8 KB burst, 101 GB/s) is identifiable as a broken config rather
     than a bad shape.
   - As a **selector** it fails: corr(burst, DS-vs-mcast penalty) = **-0.41**. Shapes at
     252-272 KB burst — all comfortably past the knee — span penalties from **0.90x to 1.34x**,
     decided entirely by the multicast side (258 GB/s vs 388 GB/s) while DS delivers its usual
     283-306 either way. Conversely a 51 KB burst can be perfectly fine (qwen_3_0_6b
     `3072x1024`, 0.98x) because multicast only reaches 222.8 GB/s there.

   The reason is structural: above the knee DS is comparatively flat (283-316 GB/s, stdev 43)
   while multicast swings 205-404 (stdev 56). Since penalty is essentially
   `noDS GB/s / DS GB/s`, the decision is dominated by the term burst does not predict. A real
   selection rule reduces to **"prefer DS only where multicast would come in under ~290 GB/s"**,
   which needs a multicast bandwidth model — the DS side is nearly a constant.
3. **Fix the models tt-mlir leaves unconfigured.** Qwen2.5-1.5B's no-DS compile assigns a
   program config to **none** of its 141 matmuls, and its DS compile misses 29 including
   lm_head. ttnn's runtime fallback then reached only 103.0 and 96.5 GB/s on two shapes, and
   155.5 GB/s on lm_head against 356.0 when configured. This is a larger and cheaper win than
   anything in the DS-vs-multicast argument, and it also makes future A/Bs on this model
   meaningful — as it stands, the comparison is uninterpretable.
4. **Investigate the lm_head config regression** on qwen_2_5_1_5b (`default` vs 1D mcast,
   898 us). A DS-enabling change should not move decisions for non-DS matmuls.
6. **Upstream the two ttrt block-float dtype fixes.** Without them 8 of 11 fleet models cannot
   be profiled at all, and this study would have rested on three models and reached the wrong
   general conclusion.
7. **Where the real DS gain lives.** Metal's fast DRAM-sharded path is `gather_in0` + a
   GlobalCircularBuffer on the *1D multicast* kernel (`global_cb`: 29 references in the mcast_1d
   factory, 0 in the DS factory), which tt-mlir cannot express. See the first case study's
   analysis; the ~320 GB/s ceiling measured here is the legacy DS kernel's, not DRAM sharding's
   in principle.

## Reproduction

```bash
gh run download 30768002414 --repo tenstorrent/tt-xla --name ttnn-mlir-ttnn_<model>_bs32_isl128_<ts>
# the artifact's ttnn_runtime_*_g1_*.mlir is already lowered -- translate directly
ttmlir-translate --ttnn-to-flatbuffer ttnn_runtime_<model>_..._g1_*.mlir -o g1.ttnn
export TT_METAL_PROFILER_TRACE_TRACKING=1
ttrt perf g1.ttnn --loops 1 --trace-region-size 268435456 \
    --enable-program-cache --ignore-version
python -O -c "from tracy.process_ops_logs import process_ops; process_ops(None,None,False)"
python percore_perf.py  --device-log .logs/profile_log_device.csv --ops-data .logs/tracy_ops_data.csv
python matmul_detail.py --percore percore.csv --report reports/ops_perf_results.csv
python fleet_compare.py --dir fleet --models <models>
```

`g1` is the decode graph (graph index 1). Four things are silently wrong by default and all
four are load-bearing: `--enable-program-cache` (`run.py:1249` drops the registered
`default=True`), `--trace-region-size` (`run.py:588` assigns unconditionally, so the default 0
wins), `TT_METAL_PROFILER_TRACE_TRACKING` (metal emits trace END unguarded but BEGIN/REPLAY
only under this flag), and `python -O` (`process_ops_logs.py:606` aborts the whole report over
a handful of ops with no device data).

Scripts: `static_matmul_survey.py` (config survey, no device),
`run_model_fleet.sh` (per-model measurement), `percore_perf.py` (epoch-safe durations),
`matmul_detail.py` (per-matmul join), `fleet_compare.py` (DS vs no-DS tables),
`patch_bfp8_inputs.py` (parked KV-cache workaround).
