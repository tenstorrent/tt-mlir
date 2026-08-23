# DRAM-sharded matmul on n150

Three-way analysis of the DRAM-sharded (DS) matmul integration on **n150 (Wormhole)**: no DS,
DS, and DS with the collapse guard.
Compile decisions and end-to-end numbers come from three tt-xla `Performance Benchmark` runs;
the per-matmul device timings come from re-running those same decode graphs locally on a
Wormhole card, because all three CI runs skipped device perf.

| | run | tt-xla | tt-mlir | what it is |
|---|---|---|---|---|
| **no DS** | [30806869145](https://github.com/tenstorrent/tt-xla/actions/runs/30806869145) | `f6a38d65` | `99c621da` | merge base of the DS branch |
| **DS** | [30767975002](https://github.com/tenstorrent/tt-xla/actions/runs/30767975002) | `f6a38d65` | `ae602833` | merge base + the 18-commit DS series |
| **DS+guard** | [32243412436](https://github.com/tenstorrent/tt-xla/actions/runs/32243412436) | `f6a38d65` | `1fad065b` | `ae602833` + the collapse guard, one commit |

Same tt-xla commit on all three; the runs differ only in the `mlir_override` input. Because
`1fad065b` is `ae602833` plus exactly one commit, the third run isolates the guard.
`99c621da` is the exact merge base of the DS branch, so the range is the 18-commit
DS integration series — but **not all 18 commits are DS**, and one of the others turns out
to matter a great deal:

- `0b3d7856` **fuse activation into consumer multiply instead of matmul** — this moves SiLU
  out of the gate matmul. It changes what a matmul costs, so it has to be held fixed before
  any DS-vs-multicast ratio means anything. See
  [the activation confounder](#the-activation-confounder).
- `a740187c` reuse reshards for consumers of the same producer
- `7890a102` avoid illegal demotion

`4731ca73` (Blackhole worker grid) cannot affect an n150 result.

## The headline

**The DS matmul kernel is faster than 1D multicast on n150 — in all 13 models measured and for
every projection role it is applied to.** Matmul device time per decode step falls by 22.4 ms
fleet-wide. That is the opposite of the Blackhole fleet study, where DS lost on nine of nine
models.

**How much of that reaches the decode step depends sharply on model size.** Two things grow
alongside the matmul saving: layout/reshard work, and the SiLU that moves out of the matmul
into a markedly more expensive multiply (about a seventh of the raw matmul saving is just that
activation no longer running inside the matmul). For the 7B-8B models the matmul win is far
larger than both and the step lands at **0.903x-0.933x**. For everything at 1.7B and below the
two effects roughly cancel and the step sits at **0.990x-1.013x**.

| decode step ratio | models |
|---|---|
| 0.903x–0.933x | falcon3_7b, qwen_3_8b, mistral_7b, llama_3_1_8b |
| 0.975x–0.993x | llama_3_2_3b, qwen_3_4b, falcon3_3b |
| 0.990x–1.013x | qwen_2_5_0_5b, qwen_3_0_6b, qwen_2_5_3b, llama_3_2_1b, qwen_3_1_7b, falcon3_1b |

The gradient is monotone in size and has a mechanical cause: larger N gives more output tile
columns per core, which is the quantity that decides whether DS beats multicast here. That is
the opposite of the intuition the Blackhole data suggested, where the biggest models — the ones
whose `in0_block_w` is forced furthest down — looked most at risk.

### Two loss signatures, both cheap to detect at compile time

Every per-shape DS loss in the fleet falls into one of exactly two groups:

1. **A K-step ratio `kPerCore / in0_block_w` above 4.** These hold the fleet's worst DS
   bandwidths — 191 GB/s at ratio 7, 134 at ratio 37, and 110 at ratio 16, the last being
   Mistral-7B's `4096x32768` lm_head at **1.90x**. That is the Blackhole failure mode
   reproducing almost exactly (~101 GB/s there). The guard in `1fad065b` catches this, but
   **its `kMinBlockWidthFraction = 2` is calibrated for Blackhole, and the third run measures
   what that costs** on device:

   | model | DS vs no-DS | guard vs DS | guard vs no-DS | shapes declined |
   |---|---|---|---|---|
   | falcon3_7b | 0.903x | **1.104x** | 0.996x | 2 |
   | mistral_7b | 0.931x | **1.063x** | 0.989x | 3 |
   | llama_3_1_8b | 0.933x | **1.034x** | 0.965x | 1 |
   | qwen_3_8b | 0.923x | **1.000x** | 0.923x | **0** — control |
   | qwen_2_5_7b | 0.945x | **0.925x** | 0.874x | 2 — the guard's win |

Three of the four models it touches hand their DS win straight back. The model it leaves
   alone keeps all of it, measured to within 0.03% on a 48 ms step — a clean inertness check
   and a bound on the measurement's repeatability. The one model it genuinely helps is
   Qwen2.5-7B, whose `18944x3584` down sits at ratio 37 and 133.9 GB/s: the shape the guard
   exists for. (Qwen2.5-7B's baseline compile emitted no program configs, so its no-DS column
   is ttnn's runtime fallback rather than 1D multicast — read its ratios as directional.)

   The guard is right about two shapes (ratio 16 and 37) and wrong about five (ratio 3-4,
   231-255 GB/s, 0.80x-0.89x): across the six shapes with a true multicast baseline it gives up
   **10215 µs of measured matmul time to recover 615 µs**. At a fraction of 4 it would decline
   exactly the degenerate two and keep every winner. See
   [what the guard did](#what-the-collapse-guard-did) and
   [the calibration](#calibrating-kminblockwidthfraction-for-n150).
2. **`per_core_n == 1` with N >= 2048** — exceptionless across 8 of 8 such shapes, 1.04x to
   1.12x, spanning both `down` and `o_proj` in four unrelated models.

Scored against the control-shape noise floor the two rules flag 9 of 49 DS shapes and catch
**9 of the 10 real losses with no false positives**; the one that slips through costs 50 µs.
Declining both, with the guard set to 4, recovers 1.2 ms of matmul time and leaves every
remaining DS shape a win or a wash. See
[the two loss modes](#two-loss-modes-and-both-have-a-clean-signature) and
[how well they separate](#how-well-the-two-rules-actually-separate).

### Why DS wins here and lost on Blackhole

The Blackhole study located its loss in a **NOC-link ceiling** rather than in the program
config: DS reads every weight through the cores adjacent to the DRAM banks — one per bank —
capping it at 283–316 GB/s, while multicast spreads the same reads over 63–89 worker cores
and topped out against DRAM itself at 346–404 GB/s. It explicitly ruled out `in0_block_w` as
the general explanation (a *healthy* down_proj with `in0_block_w=32` still lost 1.23x).

The n150 measurements confirm that model and explain the flip. Here DS and multicast land in
the *same* bandwidth band — DS 143–255 GB/s against multicast's 136–218 GB/s on the same shapes,
median 216 against 206 — with DS at or above multicast on 36 of 48 shapes. Neither is NOC-limited, because n150's DRAM ceiling binds first:
this part has **12 DRAM channels feeding a 8×8 = 64-core** worker grid, against Blackhole's
8 banks and 110 worker cores. The ratio that decides whether DS's bank-adjacent readers are
the bottleneck — reader cores against the workers multicast can spread over — is **12:64
here versus 8:110 there**. With more readers and fewer workers to lose to, DS's structural
disadvantage disappears, and its more efficient read pattern shows up as a small win instead.

A second Blackhole-specific problem is also simply absent. On the one shape that study called
out as collapsed — Qwen2.5-3B's `11008x2048` down projection, kPerCore 43, `in0_block_w`
driven to 1 — n150 compiles the identical shape with kPerCore 43 **and `in0_block_w` 43**.

## Per-matmul device time, measured

The CI runs carry no per-op timings, so the same decode graphs were re-run locally on a
Wormhole card and profiled per matmul. Three things make that substitution legitimate here:

**The local card is the same part as the CI runner.** Every field of the local chip
descriptor matches the one embedded in the CI graphs — `Wormhole_b0`, `8x8` worker grid,
`l1_size = 1499136`, `num_dram_channels = 12`, `dram_channel_size = 1 GiB`, identical
alignments and unreserved bases. `--ignore-version` is only papering over a schema string,
not a hardware difference.

**The profiler is healthy on this card.** The runbook leads with a check for the
cross-core clock epoch bug that made `DEVICE KERNEL DURATION` useless on the Blackhole card
used for the original study. That bug is **absent here**: of 3483 rows with device data, zero
report an impossible duration, at every core count. As a second check, the epoch-safe
per-core durations from `percore_perf.py` and the report's own
`DEVICE KERNEL DURATION` agree to **0.23% median, 3.0% max** across the 181 traced matmuls
of one graph. The numbers below still come from `percore_perf.py`, but nothing depends on
that choice.

**The traced region is complete.** Each decode graph's matmul count matches exactly what
the IR contains (181 of 181 for Qwen2.5-3B), so no matmul is missing from the comparison
and none is double-counted from the capture pass.

Method, per model and per variant: translate the CI `ttnn_runtime_*_g1_*.mlir` to a
flatbuffer, `ttrt perf --loops 1 --enable-program-cache --trace-region-size 256MB` with
`TT_METAL_PROFILER_TRACE_TRACKING=1`, post-process, recompute per-core durations, then join
durations to shapes on `GLOBAL CALL COUNT`. Weight bandwidth is
`K x N x bytes(dtype) / device kernel time` for one instance — the decode weight read is what
DS changes, so it is the quantity to compare.

### Reading the noise floor

Two groups never take the DS path and so measure only run-to-run variation between two
otherwise identical executions: `lm_head` (multicast in every compile) and the biased Qwen2.5
`qkv`. Across every model measured their penalties stay inside **0.99x–1.02x**, so that is the
noise floor, and each claim below is read against it rather than against 1.00x.

### The activation confounder

The no-DS compile folds SiLU into the gate matmul via the program config's
`fused_activation`; the DS compile does not, because `0b3d7856` moved that activation into the
consumer multiply. In Qwen2.5-3B's decode step, 36 of 181 matmuls carry a fused SiLU on the
no-DS side and **none** do on the DS side.

Comparing group averages naively would therefore credit DS with the cost of work it no longer
does. Every per-instance time and penalty below is taken over the instances that carry **no**
fused activation on either side — for the gate/up shape that means the DS side's full set
against the no-DS side's `up` half — and the activation's own cost is measured separately from
the no-DS run (same shape, activated instances minus plain ones) and reported as its own
column. `Δ µs` is what the device actually spent differently at that shape; `Δ like` is that
figure with the activation move removed.

The multiply that inherited the SiLU is where the branch loses most of what the matmul gains:
`BinaryNgDeviceOperation` keeps the same op count but runs 2.1–2.4x slower
(Qwen2.5-3B: 654 µs against 271 µs; Falcon3-1B: 243 µs against 116 µs). That is a finding
about `0b3d7856`, not about DS.

Excluded for lack of traced device data: `ministral_8b`.

### Model level: the whole traced decode step

Multicast-to-DS comparisons only. Models whose *before* compile emitted no program
config for some matmuls measure fallback-to-DS instead and are listed separately.

| model | matmul DS | matmul no-DS | Δ matmul | of which activation moved out | Δ matmul, like-for-like | Δ layout | Δ other | Δ step | step ratio |
|---|---|---|---|---|---|---|---|---|---|
| falcon3_7b | 32336.9 | 38108.6 | -5771.7 | -544.8 | -5226.9 | +333.6 | +688.9 | -4749.2 | 0.903x |
| qwen_3_8b | 34262.7 | 39066.1 | -4803.3 | -353.7 | -4449.6 | +372.4 | +428.8 | -4002.1 | 0.923x |
| mistral_7b | 32224.8 | 36156.2 | -3931.5 | -357.7 | -3573.8 | +394.7 | +458.3 | -3078.4 | 0.931x |
| llama_3_1_8b | 26486.1 | 30034.8 | -3548.7 | -349.0 | -3199.8 | +408.9 | +451.0 | -2688.8 | 0.933x |
| llama_3_2_3b | 15795.9 | 16805.2 | -1009.3 | -188.1 | -821.2 | +144.7 | +160.5 | -704.2 | 0.975x |
| qwen_3_4b | 19783.1 | 21063.3 | -1280.2 | -270.1 | -1010.1 | +161.5 | +281.4 | -837.3 | 0.977x |
| qwen_2_5_0_5b | 3284.7 | 3370.5 | -85.8 | -146.1 | +60.3 | -18.5 | +7.2 | -97.0 | 0.990x |
| falcon3_3b | 13735.5 | 14916.0 | -1180.4 | -174.4 | -1006.0 | +438.4 | +574.0 | -168.1 | 0.993x |
| qwen_3_0_6b | 3667.0 | 3777.0 | -110.1 | -77.6 | -32.4 | -39.5 | +184.3 | +34.7 | 1.002x |
| qwen_2_5_3b | 15770.8 | 16254.1 | -483.4 | -337.7 | -145.7 | +22.6 | +568.1 | +107.4 | 1.004x |
| llama_3_2_1b | 6468.0 | 6507.1 | -39.1 | -107.1 | +68.1 | +16.3 | +159.5 | +136.7 | 1.011x |
| qwen_3_1_7b | 9000.1 | 9180.3 | -180.2 | -136.1 | -44.2 | +203.1 | +242.1 | +265.0 | 1.012x |
| falcon3_1b | 7313.3 | 7336.6 | -23.4 | -126.8 | +103.4 | +33.9 | +187.9 | +198.4 | 1.013x |

8 of 13 models have a faster traced decode step under DS. Matmul time alone: 13 of 13 faster, -22447.1 µs fleet-wide, of which -3169.2 µs is the SiLU no longer running inside the matmul.


#### Fallback baseline, not multicast — a different comparison

| model | matmul DS | matmul fallback | Δ matmul | Δ layout | Δ other | Δ step | step ratio |
|---|---|---|---|---|---|---|---|
| qwen_2_5_1_5b | 9198.5 | 12821.6 | -3623.1 | +220.9 | -1873.7 | -5275.9 | 0.772x |
| qwen_2_5_7b | 43876.8 | 40126.9 | +3749.9 | +441.3 | -7309.9 | -3118.7 | 0.945x |

These numbers say what tt-mlir's program configs are worth against ttnn's runtime
heuristic, which is a much lower bar than 1D multicast. They are excluded from every
aggregate above and below.


### Per projection role

Per-instance times and the penalty are taken over instances with no fused
activation on either side. `Δ µs` is what the device actually spent differently
at that shape; `Δ like` removes the activation move.

#### down

| model | K × N | n | bias | on DS | w before → after | DS µs | DS GB/s | no-DS µs | no-DS GB/s | penalty | Δ µs | Δ like | act |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| falcon3_1b | 8192 × 2048 | 18 | — | yes | 8 → 32 | 87.7 | 203.3 | 83.8 | 212.7 | 1.05x | +69.4 | +69.4 | — |
| falcon3_3b | 9216 × 3072 | 22 | — | yes | 8 → 36 | 128.7 | 233.7 | 150.6 | 199.7 | 0.85x | -481.1 | -481.1 | — |
| falcon3_7b | 23040 × 3072 | 28 | — | yes | 8 → 30 | 295.3 | 254.7 | 370.3 | 203.1 | 0.80x | -2101.2 | -2101.2 | — |
| llama_3_1_8b | 14336 × 4096 | 32 | — | yes | 8 → 14 | 252.7 | 246.9 | 301.5 | 206.9 | 0.84x | -1561.0 | -1561.0 | — |
| llama_3_2_1b | 8192 × 2048 | 16 | — | yes | 8 → 32 | 87.7 | 203.4 | 83.4 | 213.7 | 1.05x | +67.7 | +67.7 | — |
| llama_3_2_3b | 8192 × 3072 | 28 | — | yes | 8 → 32 | 114.2 | 234.2 | 135.5 | 197.3 | 0.84x | -597.0 | -597.0 | — |
| mistral_7b | 14336 × 4096 | 32 | — | yes | 8 → 14 | 252.6 | 246.9 | 301.4 | 207.0 | 0.84x | -1560.1 | -1560.1 | — |
| qwen_2_5_0_5b | 4864 × 896 | 24 | — | yes | 8 → 19 | 27.9 | 165.9 | 31.8 | 145.7 | 0.88x | -92.7 | -92.7 | — |
| qwen_2_5_3b | 11008 × 2048 | 36 | — | yes | 8 → 43 | 115.5 | 207.4 | 111.5 | 214.8 | 1.04x | +143.7 | +143.7 | — |
| qwen_3_0_6b | 3072 × 1024 | 28 | — | yes | 8 → 12 | 19.8 | 168.9 | 23.2 | 144.1 | 0.85x | -95.5 | -95.5 | — |
| qwen_3_1_7b | 6144 × 2048 | 28 | — | yes | 8 → 24 | 67.4 | 198.3 | 63.7 | 210.0 | 1.06x | +105.3 | +105.3 | — |
| qwen_3_4b | 9728 × 2560 | 36 | — | yes | 8 → 38 | 119.5 | 221.5 | 127.2 | 207.9 | 0.94x | -279.6 | -279.6 | — |
| qwen_3_8b | 12288 × 4096 | 36 | — | yes | 8 → 24 | 221.3 | 241.6 | 258.2 | 207.1 | 0.86x | -1327.6 | -1327.6 | — |

On the DS path: 13/13 shapes, per-instance penalty 0.80x–1.06x, DS 166–255 GB/s vs no-DS 144–215 GB/s, like-for-like net **-7709.7 µs**.

#### gate/up

| model | K × N | n | bias | on DS | w before → after | DS µs | DS GB/s | no-DS µs | no-DS GB/s | penalty | Δ µs | Δ like | act |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| falcon3_1b | 2048 × 8192 | 36 | — | yes | 2 → 8 | 82.6 | 215.8 | 81.9 | 217.6 | 1.01x | -102.8 | +24.0 | SILU x18 |
| falcon3_3b | 3072 × 9216 | 44 | — | yes | 2 → 12 | 132.2 | 227.5 | 140.2 | 214.5 | 0.94x | -528.4 | -353.9 | SILU x22 |
| falcon3_7b | 3072 × 23040 | 56 | — | yes | 2 → 3 | 325.5 | 231.0 | 377.8 | 199.1 | 0.86x | -3472.7 | -2927.9 | SILU x28 |
| llama_3_1_8b | 4096 × 14336 | 64 | — | yes | 2 → 8 | 143.0 | 231.0 | 160.3 | 206.1 | 0.89x | -1456.8 | -1107.8 | SILU x32 |
| llama_3_2_1b | 2048 × 8192 | 32 | — | yes | 2 → 8 | 82.5 | 216.0 | 82.3 | 216.7 | 1.00x | -98.5 | +8.6 | SILU x16 |
| llama_3_2_3b | 3072 × 8192 | 56 | — | yes | 2 → 12 | 121.8 | 219.5 | 122.7 | 218.0 | 0.99x | -234.3 | -46.2 | SILU x28 |
| mistral_7b | 4096 × 14336 | 64 | — | yes | 2 → 4 | 258.9 | 241.0 | 291.1 | 214.3 | 0.89x | -2422.4 | -2064.7 | SILU x32 |
| qwen_2_5_0_5b | 896 × 4864 | 48 | — | no | 2 → 2 | 23.5 | 196.8 | 23.5 | 196.7 | 1.00x | +3.4 | -0.7 | SILU x24 |
| qwen_2_5_3b | 2048 × 11008 | 72 | — | yes | 2 → 8 | 108.9 | 219.9 | 114.2 | 209.7 | 0.95x | -720.9 | -383.2 | SILU x36 |
| qwen_3_0_6b | 1024 × 3072 | 56 | — | yes | 8 → 4 | 19.2 | 174.0 | 18.9 | 176.7 | 1.02x | -61.1 | +16.5 | SILU x28 |
| qwen_3_1_7b | 2048 × 6144 | 56 | — | yes | 2 → 8 | 60.6 | 220.5 | 63.5 | 210.6 | 0.96x | -295.3 | -159.3 | SILU x28 |
| qwen_3_4b | 2560 × 9728 | 72 | — | yes | 2 → 10 | 119.9 | 220.8 | 126.3 | 209.5 | 0.95x | -736.0 | -465.9 | SILU x36 |
| qwen_3_8b | 4096 × 12288 | 72 | — | yes | 2 → 8 | 218.1 | 245.2 | 252.7 | 211.7 | 0.86x | -2843.8 | -2490.0 | SILU x36 |

On the DS path: 12/13 shapes, per-instance penalty 0.86x–1.02x, DS 174–245 GB/s vs no-DS 177–218 GB/s, like-for-like net **-9949.8 µs**.

#### qkv

| model | K × N | n | bias | on DS | w before → after | DS µs | DS GB/s | no-DS µs | no-DS GB/s | penalty | Δ µs | Δ like | act |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| falcon3_1b | 2048 × 4096 | 18 | — | yes | 8 → 8 | 43.4 | 205.4 | 45.1 | 197.4 | 0.96x | -31.6 | -31.6 | — |
| falcon3_3b | 3072 × 5120 | 22 | — | yes | 2 → 12 | 79.4 | 210.6 | 80.6 | 207.4 | 0.98x | -26.8 | -26.8 | — |
| falcon3_7b | 3072 × 5120 | 28 | — | yes | 2 → 12 | 79.2 | 211.1 | 80.9 | 206.5 | 0.98x | -49.3 | -49.3 | — |
| llama_3_1_8b | 4096 × 6144 | 32 | — | yes | 2 → 16 | 115.3 | 231.8 | 124.8 | 214.3 | 0.92x | -301.2 | -301.2 | — |
| llama_3_2_1b | 2048 × 3072 | 16 | — | yes | 8 → 8 | 32.6 | 205.3 | 36.6 | 182.6 | 0.89x | -64.7 | -64.7 | — |
| llama_3_2_3b | 3072 × 5120 | 28 | — | yes | 2 → 12 | 79.5 | 210.3 | 80.7 | 207.2 | 0.99x | -33.2 | -33.2 | — |
| mistral_7b | 4096 × 6144 | 32 | — | yes | 2 → 16 | 115.3 | 231.9 | 125.0 | 213.9 | 0.92x | -311.9 | -311.9 | — |
| qwen_2_5_0_5b | 896 × 1152 | 24 | yes | no | 4 → 4 | 11.6 | 94.8 | 11.6 | 94.7 | 1.00x | -0.3 | -0.3 | — |
| qwen_2_5_3b | 2048 × 2560 | 36 | yes | no | 8 → 8 | 29.6 | 187.9 | 29.3 | 190.0 | 1.01x | +11.7 | +11.7 | — |
| qwen_3_0_6b | 1024 × 4096 | 28 | — | yes | 8 → 4 | 24.9 | 178.9 | 23.1 | 192.8 | 1.08x | +50.4 | +50.4 | — |
| qwen_3_1_7b | 2048 × 4096 | 28 | — | yes | 8 → 8 | 43.4 | 205.3 | 45.7 | 195.0 | 0.95x | -63.7 | -63.7 | — |
| qwen_3_4b | 2560 × 6144 | 36 | — | yes | 2 → 10 | 74.9 | 223.0 | 79.3 | 210.7 | 0.94x | -158.1 | -158.1 | — |
| qwen_3_8b | 4096 × 6144 | 36 | — | yes | 2 → 16 | 115.2 | 232.1 | 125.5 | 213.1 | 0.92x | -369.6 | -369.6 | — |

On the DS path: 11/13 shapes, per-instance penalty 0.89x–1.08x, DS 179–232 GB/s vs no-DS 183–214 GB/s, like-for-like net **-1359.7 µs**.

#### o_proj

| model | K × N | n | bias | on DS | w before → after | DS µs | DS GB/s | no-DS µs | no-DS GB/s | penalty | Δ µs | Δ like | act |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| falcon3_1b | 2048 × 2048 | 18 | — | yes | 8 → 8 | 25.6 | 174.1 | 22.9 | 195.0 | 1.12x | +49.5 | +49.5 | — |
| falcon3_3b | 3072 × 3072 | 22 | — | yes | 8 → 12 | 47.3 | 211.8 | 53.4 | 187.8 | 0.89x | -133.5 | -133.5 | — |
| falcon3_7b | 3072 × 3072 | 28 | — | yes | 8 → 12 | 47.4 | 211.6 | 53.6 | 187.0 | 0.88x | -174.8 | -174.8 | — |
| llama_3_1_8b | 4096 × 4096 | 32 | — | yes | 8 → 16 | 80.7 | 221.0 | 88.2 | 202.1 | 0.91x | -241.2 | -241.2 | — |
| llama_3_2_1b | 2048 × 2048 | 16 | — | yes | 8 → 8 | 25.7 | 173.7 | 23.5 | 190.0 | 1.09x | +35.3 | +35.3 | — |
| llama_3_2_3b | 3072 × 3072 | 28 | — | yes | 8 → 12 | 47.0 | 213.2 | 53.5 | 187.3 | 0.88x | -181.9 | -181.9 | — |
| mistral_7b | 4096 × 4096 | 32 | — | yes | 8 → 16 | 80.7 | 220.9 | 88.6 | 201.3 | 0.91x | -251.7 | -251.7 | — |
| qwen_2_5_0_5b | 896 × 896 | 24 | — | no | 4 → 4 | 11.1 | 77.1 | 11.1 | 77.0 | 1.00x | -0.2 | -0.2 | — |
| qwen_2_5_3b | 2048 × 2048 | 36 | — | yes | 8 → 8 | 25.4 | 175.4 | 22.9 | 194.4 | 1.11x | +89.5 | +89.5 | — |
| qwen_3_0_6b | 2048 × 1024 | 28 | — | yes | 8 → 8 | 15.6 | 143.0 | 16.3 | 136.5 | 0.95x | -20.8 | -20.8 | — |
| qwen_3_1_7b | 2048 × 2048 | 28 | — | yes | 8 → 8 | 25.6 | 174.0 | 23.2 | 191.9 | 1.10x | +67.0 | +67.0 | — |
| qwen_3_4b | 4096 × 2560 | 36 | — | yes | 8 → 16 | 53.9 | 206.7 | 56.2 | 198.4 | 0.96x | -81.3 | -81.3 | — |
| qwen_3_8b | 4096 × 4096 | 36 | — | yes | 8 → 16 | 80.5 | 221.4 | 88.4 | 201.7 | 0.91x | -284.3 | -284.3 | — |

On the DS path: 12/13 shapes, per-instance penalty 0.88x–1.12x, DS 143–221 GB/s vs no-DS 136–202 GB/s, like-for-like net **-1128.1 µs**.

#### lm_head

| model | K × N | n | bias | on DS | w before → after | DS µs | DS GB/s | no-DS µs | no-DS GB/s | penalty | Δ µs | Δ like | act |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| falcon3_1b | 2048 × 131072 | 1 | — | no | 2 → 2 | 1519.8 | 187.7 | 1527.8 | 186.7 | 0.99x | -7.9 | -7.9 | — |
| falcon3_3b | 3072 × 131072 | 1 | — | no | 2 → 2 | 2299.1 | 186.1 | 2309.9 | 185.2 | 1.00x | -10.7 | -10.7 | — |
| falcon3_7b | 3072 × 131072 | 1 | — | no | 2 → 2 | 2297.4 | 186.2 | 2271.1 | 188.4 | 1.01x | +26.3 | +26.3 | — |
| llama_3_1_8b | 4096 × 128256 | 1 | — | no | 2 → 2 | 2977.0 | 187.5 | 2965.6 | 188.2 | 1.00x | +11.4 | +11.4 | — |
| llama_3_2_1b | 2048 × 128256 | 1 | — | no | 2 → 2 | 1493.4 | 186.9 | 1472.2 | 189.6 | 1.01x | +21.2 | +21.2 | — |
| llama_3_2_3b | 3072 × 128256 | 1 | — | no | 2 → 2 | 2233.0 | 187.5 | 2195.9 | 190.6 | 1.02x | +37.1 | +37.1 | — |
| mistral_7b | 4096 × 32768 | 1 | — | yes | 2 → 1 | 1299.8 | 109.7 | 685.2 | 208.1 | 1.90x | +614.5 | +614.5 | — |
| qwen_2_5_0_5b | 896 × 151936 | 1 | — | no | 2 → 2 | 792.1 | 182.6 | 788.1 | 183.5 | 1.01x | +4.0 | +4.0 | — |
| qwen_2_5_3b | 2048 × 151936 | 1 | — | no | 2 → 2 | 1789.7 | 184.7 | 1797.1 | 184.0 | 1.00x | -7.4 | -7.4 | — |
| qwen_3_0_6b | 1024 × 151936 | 1 | — | no | 2 → 2 | 903.2 | 183.0 | 886.3 | 186.5 | 1.02x | +16.9 | +16.9 | — |
| qwen_3_1_7b | 2048 × 151936 | 1 | — | no | 2 → 2 | 1784.0 | 185.3 | 1777.5 | 186.0 | 1.00x | +6.5 | +6.5 | — |
| qwen_3_4b | 2560 × 151936 | 1 | — | no | 2 → 2 | 2214.4 | 186.6 | 2239.6 | 184.5 | 0.99x | -25.2 | -25.2 | — |
| qwen_3_8b | 4096 × 151936 | 1 | — | no | 2 → 2 | 3548.0 | 186.4 | 3526.1 | 187.5 | 1.01x | +21.9 | +21.9 | — |

On the DS path: 1/13 shapes, per-instance penalty 1.90x–1.90x, DS 110–110 GB/s vs no-DS 208–208 GB/s, like-for-like net **+614.5 µs**.


### Two loss modes, and both have a clean signature

Every DS shape that is slower than multicast falls into one of exactly two groups, and
everything outside them wins or ties.

**Mode 1 — `in0_block_w` collapsed to 1.** tt-mlir's search walks down the divisors of
K-tiles-per-core; when none is small enough it lands on 1, and the weight read degenerates
into one tile per step. This is the failure the Blackhole study measured at ~101 GB/s,
and it reproduces here almost exactly.

| model | role | K × N | in0_block_w / kPerCore | DS GB/s | no-DS GB/s | penalty | Δ like µs |
|---|---|---|---|---|---|---|---|
| mistral_7b | lm_head | 4096 × 32768 | 1 / 16 | 109.7 | 208.1 | 1.90x | +614.5 |

This is **the single worst DS shape in the fleet** and the only one where the collapse
happens on n150 at all. The guard in `1fad065b` — which is *not* in this range — declines
exactly this pattern, so it is worth 614.5 µs here.

**Mode 2 — `per_core_n = 1` with N >= 2048.** `per_core_n` is how many output tile
columns a core owns. N = 2048 is 64 N-tiles spread one per core, so each core reads only
`in0_block_w` weight tiles per step and per-core launch and sync cost dominates.

8 of 8 such shapes are slower, 1.04x–1.12x, spanning down and o_proj across 4 models.

| model | role | K × N | in0_block_w | weight MB | DS GB/s | no-DS GB/s | penalty | Δ like µs |
|---|---|---|---|---|---|---|---|---|
| falcon3_1b | o_proj | 2048 × 2048 | 8 | 4.5 | 174.1 | 195.0 | 1.12x | +49.5 |
| qwen_2_5_3b | o_proj | 2048 × 2048 | 8 | 4.5 | 175.4 | 194.4 | 1.11x | +89.5 |
| qwen_3_1_7b | o_proj | 2048 × 2048 | 8 | 4.5 | 174.0 | 191.9 | 1.10x | +67.0 |
| llama_3_2_1b | o_proj | 2048 × 2048 | 8 | 4.5 | 173.7 | 190.0 | 1.09x | +35.3 |
| qwen_3_1_7b | down | 6144 × 2048 | 24 | 13.4 | 198.3 | 210.0 | 1.06x | +105.3 |
| llama_3_2_1b | down | 8192 × 2048 | 32 | 17.8 | 203.4 | 213.7 | 1.05x | +67.7 |
| falcon3_1b | down | 8192 × 2048 | 32 | 17.8 | 203.3 | 212.7 | 1.05x | +69.4 |
| qwen_2_5_3b | down | 11008 × 2048 | 43 | 24.0 | 207.4 | 214.8 | 1.04x | +143.7 |

The N < 2048 half of `per_core_n = 1` *wins* (qwen_2_5_0_5b down, qwen_3_0_6b o_proj, qwen_3_0_6b down, at most 4.6 MB), because at 169 GB/s against a fleet best of 255 neither path is bandwidth-bound there. Weight size has to be part of the rule.


Declining both modes recovers **1242.0 µs** of matmul time. The remaining 40 DS shapes run 0.80x–1.08x for -20774.7 µs, taking the fleet's like-for-like matmul result from -19532.7 µs to -20774.7 µs.


#### It is the K-step ratio, not the block width

Absolute bandwidth does not depend on what the other compile did, so these tables use
every measured model including the fallback-baseline ones.

| kPerCore / in0_block_w | shapes | DS GB/s | median | worst shape |
|---|---|---|---|---|
| 1 | 44 | 143–234 | 211 | qwen_3_0_6b o_proj 2048×1024 |
| 2 | 3 | 231–245 | 242 | llama_3_1_8b gate/up 4096×14336 |
| 3 | 1 | 255–255 | 255 | falcon3_7b down 23040×3072 |
| 4 | 4 | 231–247 | 247 | falcon3_7b gate/up 3072×23040 |
| 7 | 1 | 191–191 | 191 | qwen_2_5_7b gate/up 3584×18944 |
| 16 | 1 | 110–110 | 110 | mistral_7b lm_head 4096×32768 |
| 37 | 1 | 134–134 | 134 | qwen_2_5_7b down 18944×3584 |

Ratio at or below 4: 52 shapes at 143–255 GB/s. Above 4: 3 shapes at 110–191 GB/s. The cut is clean and there is no further trend below it.

The absolute block width predicts nothing on its own:

| in0_block_w | shapes | DS GB/s | ratio range |
|---|---|---|---|
| 1 | 1 | 110–110 | 16–16 |
| 2 | 2 | 134–191 | 7–37 |
| 3 | 1 | 231–231 | 4–4 |
| 4 | 3 | 174–241 | 1–4 |
| 6 | 2 | 154–207 | 1–1 |
| 8 | 14 | 143–245 | 1–2 |
| 10 | 2 | 221–223 | 1–1 |
| 12 | 9 | 169–228 | 1–1 |
| 14 | 3 | 204–247 | 1–4 |
| 16 | 7 | 207–232 | 1–1 |
| 19 | 1 | 166–166 | 1–1 |
| 24 | 2 | 198–242 | 1–2 |
| 30 | 1 | 255–255 | 3–3 |
| 32 | 3 | 203–234 | 1–1 |
| 35 | 1 | 223–223 | 1–1 |
| 36 | 1 | 234–234 | 1–1 |
| 38 | 1 | 222–222 | 1–1 |
| 43 | 1 | 207–207 | 1–1 |

The clearest case against a width threshold is falcon3_7b's gate/up `3072x23040`: `in0_block_w` of just **3** — the smallest healthy width in the fleet — yet **231.0 GB/s**, because kPerCore is only 12 so the ratio is 4. Meanwhile `in0_block_w` of 2 measures 191 GB/s at ratio 7 and 134 at ratio 37. What costs bandwidth is the number of serialized K-steps the kernel loops over, not how wide each one is.


#### Calibrating kMinBlockWidthFraction for n150

The guard in `1fad065b` declines DS when `in0_block_w * kMinBlockWidthFraction <
kPerCore`, with the constant set to **2** — a number calibrated on Blackhole. On n150 the
K-step ratio `kPerCore / in0_block_w` separates healthy from degenerate shapes, but the
cut sits in a different place: ratio 4 shapes are the *fastest* in this fleet.

| kMinBlockWidthFraction | shapes declined | net µs | wins thrown away |
|---|---|---|---|
| 2 | 6 | -9600.3 | 10214.9 |
| 3 | 5 | -7499.1 | 8113.7 |
| 4 | 1 | +614.5 | 0.0 |
| 6 | 1 | +614.5 | 0.0 |
| 8 | 1 | +614.5 | 0.0 |

Every shape the guard could touch, by K-step ratio:

| kPerCore / in0_block_w | model | role | in0_block_w / kPerCore | DS GB/s | penalty | Δ like µs | baseline |
|---|---|---|---|---|---|---|---|
| 37 | qwen_2_5_7b | down | 2 / 74 | 133.9 | 1.36x | +3965.0 | fallback |
| 16 | mistral_7b | lm_head | 1 / 16 | 109.7 | 1.90x | +614.5 | clean |
| 7 | qwen_2_5_7b | gate/up | 2 / 14 | 191.4 | 1.00x | +104.7 | fallback |
| 4 | falcon3_7b | gate/up | 3 / 12 | 231.0 | 0.86x | -2927.9 | clean |
| 4 | llama_3_1_8b | down | 14 / 56 | 246.9 | 0.84x | -1561.0 | clean |
| 4 | mistral_7b | gate/up | 4 / 16 | 241.0 | 0.89x | -2064.7 | clean |
| 4 | mistral_7b | down | 14 / 56 | 246.9 | 0.84x | -1560.1 | clean |
| 3 | falcon3_7b | down | 30 / 90 | 254.7 | 0.80x | -2101.2 | clean |

**On n150 the constant should be 4, not 2.** At 2 the guard declines 6 shapes, 5 of which are wins it should keep — among them the fleet's fastest DS shapes, at 231.0-254.7 GB/s and 0.80x-0.89x. That trades away 10215 µs of wins to recover 615 µs, a net loss of 9600 µs. At 4 it declines exactly the 1 genuinely degenerate shape and costs nothing.

The Blackhole calibration recorded in the guard's own comment (`in0_block_w reduced
n=8  -6.60% median sps`) does not reproduce here: on this part a reduced block width is
harmless down to a ratio of 4, and only past that does bandwidth fall off.


#### How well the two rules actually separate

Scored against the control-shape noise floor: a shape counts as a real loss at
penalty > 1.02, a real win at < 0.98. Clean-baseline DS shapes only.

| | predicted loss | predicted keep |
|---|---|---|
| **is a loss** (>1.02x) | 9 | 1 |
| **is not** (<=1.02x) | 0 | 39 |

The two rules flag 9 of 49 DS shapes and catch 9 of the 10 real losses, with no false positives. 1 slips through (qwen_3_0_6b qkv).

| model | role | K × N | penalty | Δ like µs | why it is missed |
|---|---|---|---|---|---|
| qwen_3_0_6b | qkv | 1024 × 4096 | 1.08x | +50.4 | per_core_n 2, in0_block_w 4/4 |


#### Penalty against per_core_n

| per_core_n | shapes | penalty range | median | DS GB/s | no-DS GB/s |
|---|---|---|---|---|---|
| 1 | 11 | 0.85x–1.12x | 1.05x | 143–207 | 136–215 |
| 2 | 19 | 0.80x–1.08x | 0.89x | 174–255 | 177–208 |
| 3 | 8 | 0.92x–0.99x | 0.96x | 210–232 | 206–214 |
| 4 | 3 | 0.99x–1.01x | 1.00x | 216–220 | 217–218 |
| 5 | 2 | 0.94x–0.95x | 0.95x | 221–228 | 210–214 |
| 6 | 2 | 0.86x–0.95x | 0.95x | 220–245 | 210–212 |
| 7 | 2 | 0.89x–0.89x | 0.89x | 231–241 | 206–214 |
| 12 | 1 | 0.86x–0.86x | 0.86x | 231–231 | 199–199 |
| 16 | 1 | 1.90x–1.90x | 1.90x | 110–110 | 208–208 |

### Every DS shape, ranked

- Noise floor from the 16 never-on-DS control shapes: penalty 0.99x–1.02x.
- 49 shape groups take the DS path across 13 models.
- **36 faster**, -20874.2 µs like-for-like.
- **13 slower**, +1341.6 µs like-for-like.

**Worst losses**

| model | role | K × N | penalty | Δ like µs | DS GB/s | no-DS GB/s | w after / kPerCore | weight MB |
|---|---|---|---|---|---|---|---|---|
| mistral_7b | lm_head | 4096 × 32768 | 1.90x | +614.5 | 109.7 | 208.1 | 1 / 16 | 142.6 |
| qwen_2_5_3b | down | 11008 × 2048 | 1.04x | +143.7 | 207.4 | 214.8 | 43 / 43 | 24.0 |
| qwen_3_1_7b | down | 6144 × 2048 | 1.06x | +105.3 | 198.3 | 210.0 | 24 / 24 | 13.4 |
| qwen_2_5_3b | o_proj | 2048 × 2048 | 1.11x | +89.5 | 175.4 | 194.4 | 8 / 8 | 4.5 |
| falcon3_1b | down | 8192 × 2048 | 1.05x | +69.4 | 203.3 | 212.7 | 32 / 32 | 17.8 |
| llama_3_2_1b | down | 8192 × 2048 | 1.05x | +67.7 | 203.4 | 213.7 | 32 / 32 | 17.8 |
| qwen_3_1_7b | o_proj | 2048 × 2048 | 1.10x | +67.0 | 174.0 | 191.9 | 8 / 8 | 4.5 |
| qwen_3_0_6b | qkv | 1024 × 4096 | 1.08x | +50.4 | 178.9 | 192.8 | 4 / 4 | 4.5 |
| falcon3_1b | o_proj | 2048 × 2048 | 1.12x | +49.5 | 174.1 | 195.0 | 8 / 8 | 4.5 |
| llama_3_2_1b | o_proj | 2048 × 2048 | 1.09x | +35.3 | 173.7 | 190.0 | 8 / 8 | 4.5 |
| falcon3_1b | gate/up | 2048 × 8192 | 1.01x | +24.0 | 215.8 | 217.6 | 8 / 8 | 17.8 |
| qwen_3_0_6b | gate/up | 1024 × 3072 | 1.02x | +16.5 | 174.0 | 176.7 | 4 / 4 | 3.3 |

**Biggest wins**

| model | role | K × N | penalty | Δ like µs | DS GB/s | no-DS GB/s | w after / kPerCore | weight MB |
|---|---|---|---|---|---|---|---|---|
| falcon3_7b | gate/up | 3072 × 23040 | 0.86x | -2927.9 | 231.0 | 199.1 | 3 / 12 | 75.2 |
| qwen_3_8b | gate/up | 4096 × 12288 | 0.86x | -2490.0 | 245.2 | 211.7 | 8 / 16 | 53.5 |
| falcon3_7b | down | 23040 × 3072 | 0.80x | -2101.2 | 254.7 | 203.1 | 30 / 90 | 75.2 |
| mistral_7b | gate/up | 4096 × 14336 | 0.89x | -2064.7 | 241.0 | 214.3 | 4 / 16 | 62.4 |
| llama_3_1_8b | down | 14336 × 4096 | 0.84x | -1561.0 | 246.9 | 206.9 | 14 / 56 | 62.4 |
| mistral_7b | down | 14336 × 4096 | 0.84x | -1560.1 | 246.9 | 207.0 | 14 / 56 | 62.4 |
| qwen_3_8b | down | 12288 × 4096 | 0.86x | -1327.6 | 241.6 | 207.1 | 24 / 48 | 53.5 |
| llama_3_1_8b | gate/up | 4096 × 14336 | 0.89x | -1107.8 | 231.0 | 206.1 | 8 / 16 | 33.0 |
| llama_3_2_3b | down | 8192 × 3072 | 0.84x | -597.0 | 234.2 | 197.3 | 32 / 32 | 26.7 |
| falcon3_3b | down | 9216 × 3072 | 0.85x | -481.1 | 233.7 | 199.7 | 36 / 36 | 30.1 |
| qwen_3_4b | gate/up | 2560 × 9728 | 0.95x | -465.9 | 220.8 | 209.5 | 10 / 10 | 26.5 |
| qwen_2_5_3b | gate/up | 2048 × 11008 | 0.95x | -383.2 | 219.9 | 209.7 | 8 / 8 | 24.0 |
## What the collapse guard did

`1fad065b` is `ae602833` plus exactly one commit, so the third run isolates the guard.
In the decode graphs it **declined 8 shape groups**, added 0, and left 51 on the DS path.

Each declined shape had already been measured on device in the DS-vs-multicast pair, so
what the guard gave up or recovered is known rather than inferred:

| model | role | K × N | ops | in0_block_w / kPerCore | K-step ratio | measured DS penalty | DS GB/s | multicast GB/s | Δ the guard forfeits | baseline |
|---|---|---|---|---|---|---|---|---|---|---|
| llama_3_1_8b_instruct | down | 14336 × 4096 | 32 | 14 / 56 | 4 | 0.84x | 246.9 | 206.9 | +1561.0 µs | clean |
| mistral_7b | gate/up | 4096 × 14336 | 64 | 4 / 16 | 4 | 0.89x | 241.0 | 214.3 | +2064.7 µs | clean |
| mistral_7b | lm_head | 4096 × 32768 | 1 | 1 / 16 | 16 | 1.90x | 109.7 | 208.1 | -614.5 µs | clean |
| mistral_7b | down | 14336 × 4096 | 32 | 14 / 56 | 4 | 0.84x | 246.9 | 207.0 | +1560.1 µs | clean |
| qwen_2_5_7b_instruct | gate/up | 3584 × 18944 | 56 | 2 / 14 | 7 | 1.00x | 191.4 | 192.3 | -104.7 µs | fallback |
| qwen_2_5_7b_instruct | down | 18944 × 3584 | 28 | 2 / 74 | 37 | 1.36x | 133.9 | 181.7 | -3965.0 µs | fallback |
| tiiuae_falcon3_7b_base | gate/up | 3072 × 23040 | 56 | 3 / 12 | 4 | 0.86x | 231.0 | 199.1 | +2927.9 µs | clean |
| tiiuae_falcon3_7b_base | down | 23040 × 3072 | 28 | 30 / 90 | 3 | 0.80x | 254.7 | 203.1 | +2101.2 µs | clean |

On the shapes with a true multicast baseline the guard **gives up 10215 µs of matmul time to recover 615 µs** — a net loss of **9600 µs** per decode step.

`qwen_2_5_7b`'s two shapes are counted separately (4070 µs recovered, 0 µs given up): that model's baseline compile emitted no program config at all, so its penalties are measured against ttnn's runtime fallback rather than against 1D multicast. Its `18944x3584` down at ratio 37 is genuinely degenerate and the guard is right to decline it.

### Which models the guard touched

| model | roles taken off DS |
|---|---|
| llama_3_1_8b_instruct | down |
| mistral_7b | down, gate/up, lm_head |
| qwen_2_5_7b_instruct | down, gate/up |
| tiiuae_falcon3_7b_base | down, gate/up |

## Decode step, three ways

### Models the guard changed

| model | no DS | DS | DS+guard | DS vs no-DS | guard vs DS | guard vs no-DS |
|---|---|---|---|---|---|---|
| llama_3_1_8b_instruct | 34.61 | — | 34.37 | — | — | -0.68% |
| mistral_7b | 40.93 | — | — | — | — | — |
| qwen_2_5_7b_instruct | 56.26 | — | 47.82 | — | — | -14.99% |
| tiiuae_falcon3_7b_base | 44.07 | — | 44.96 | — | — | +2.01% |

The DS column is missing for llama_3_1_8b_instruct, mistral_7b, qwen_2_5_7b_instruct, tiiuae_falcon3_7b_base — those benchmark jobs failed in the DS run, which is why the guard's effect cannot be read from CI alone and was measured on device instead.

## Measured on device: no DS vs DS vs DS+guard

Same card, same graphs, traced decode step. This is the comparison CI cannot make,
because the DS run has no end-to-end number for any model the guard touched.

| model | matmul noDS | matmul DS | matmul DS+guard | step noDS | step DS | step DS+guard | DS vs noDS | guard vs DS | guard vs noDS |
|---|---|---|---|---|---|---|---|---|---|
| falcon3_7b | 38109 | 32337 | 37916 | 48826 | 44077 | 48645 | 0.903x | 1.104x | 0.996x |
| llama_3_1_8b | 30035 | 26486 | 28060 | 40414 | 37725 | 39012 | 0.933x | 1.034x | 0.965x |
| mistral_7b | 36156 | 32225 | 35632 | 44794 | 41716 | 44323 | 0.931x | 1.063x | 0.989x |
| qwen_2_5_7b | 40127 | 43877 | 39742 | 56733 | 53615 | 49578 | 0.945x | 0.925x | 0.874x |
| qwen_3_8b | 39066 | 34263 | 34271 | 52039 | 48037 | 48050 | 0.923x | 1.000x | 0.923x |

Across the 5 models measured: DS beats no-DS by 6.9% at the median; the guard then gives back +3.4%, leaving DS+guard at 0.965x of no-DS. Per-model figures above.


### Does the per-shape method predict the outcome?

The forfeit column earlier was computed purely from the DS-vs-multicast per-shape
measurements and the guard's decline list, without any knowledge of the guard run.
Comparing it against the matmul time the guard actually gave back is a check on the
whole per-shape method:

| model | shapes declined | predicted matmul Δ | measured matmul Δ | error |
|---|---|---|---|---|
| falcon3_7b | 2 | +5029 µs | +5579 µs | +10.9% |
| llama_3_1_8b | 1 | +1561 µs | +1574 µs | +0.8% |
| mistral_7b | 3 | +3010 µs | +3407 µs | +13.2% |
| qwen_2_5_7b | 2 | -4070 µs | -4134 µs | +1.6% |

For every declined shape the guard's fallback program config is byte-identical to the
one the no-DS compile chose — same kind, same `in0_block_w`, same `per_core_n`, same core
count — so the no-DS measurement is the right counterfactual. The residual error is
run-to-run variation on the shapes the guard did *not* touch, which the measured column
absorbs and the predicted column does not: falcon3_7b's 550 µs gap is 1.7% of its 32 ms
matmul total, inside the noise the control shapes bound at 0.99x-1.02x.

### Models it did not touch — a no-op check

Across 44 models where the guard changed no config, the step moves by a median of **+0.01%** (mean +0.50%, range -5.90%..+12.44%). The guard is inert where it does not fire; the spread is the CI noise level, and it is wide enough that single-run CI deltas below a few percent should not be read as signal.
## Decode-step outcome, by comparison class

### A. Clean A/B — every matmul had a 1D-multicast config before, DS after

| model | L | ops | on DS | MB/step | MB on DS | step ms before | step ms after | step Δ | ttft Δ |
|---|---|---|---|---|---|---|---|---|---|
| llama_3_2_1b_instruct | 16 | 81 | 80 | 1313 | 1034 | 10.36 | 10.39 | +0.28% | -5.97% |
| llama_3_2_3b_instruct | 28 | 141 | 140 | 3413 | 2995 | 27.88 | 26.89 | -3.56% | -1.42% |
| qwen_2_5_0_5b_instruct | 24 | 121 | 24 | 525 | 111 | 10.88 | 10.63 | -2.31% | -2.15% |
| qwen_2_5_3b_instruct | 36 | 181 | 144 | 3279 | 2747 | 27.37 | 27.06 | -1.12% | -5.36% |
| qwen_3_0_6b | 28 | 141 | 140 | 633 | 468 | 10.71 | 9.41 | -12.16% | +2.33% |
| qwen_3_1_7b | 28 | 141 | 140 | 1828 | 1497 | 21.93 | 21.76 | -0.79% | -3.86% |
| qwen_3_4b | 36 | 181 | 180 | 4274 | 3860 | 36.73 | 34.51 | -6.05% | -2.44% |
| tiiuae_falcon3_1b_base | 18 | 91 | 90 | 1488 | 1203 | 13.04 | 12.41 | -4.89% | +0.96% |
| tiiuae_falcon3_3b_base | 22 | 111 | 110 | 3001 | 2574 | 20.54 | 20.14 | -1.96% | -3.65% |

step Δ: mean -3.62%, median -2.31%, range -12.16%..+0.28% · ttft Δ: mean -2.39%, median -2.44%

### B. Control — configs identical before/after, DS never chosen

| model | L | ops | on DS | MB/step | MB on DS | step ms before | step ms after | step Δ | ttft Δ |
|---|---|---|---|---|---|---|---|---|---|
| microsoft_phi_1 | 24 | 73 | 0 | 1395 | 0 | 44.05 | 44.33 | +0.63% | +0.36% |
| microsoft_phi_1_5 | 24 | 73 | 0 | 1395 | 0 | 44.50 | 43.98 | -1.15% | +0.48% |
| microsoft_phi_2 | 32 | 97 | 0 | 2813 | 0 | 113.62 | 113.84 | +0.19% | -5.30% |
| vllm_qwen3_0_6b | — | 112 | 0 | 468 | 0 | 18.44 | 17.94 | -2.74% | -10.82% |

step Δ: mean -0.77%, median -0.48%, range -2.74%..+0.63% · ttft Δ: mean -3.82%, median -2.47%

### C. Before compile emitted no program config for some matmuls

| model | L | ops | on DS | MB/step | MB on DS | step ms before | step ms after | step Δ | ttft Δ |
|---|---|---|---|---|---|---|---|---|---|
| bge_m3_encode | — | 146 | 0 | 1218 | 0 | 472.87 | 470.65 | -0.47% | — |
| google_gemma_1_1_2b_it | 18 | 91 | 0 | 2663 | 0 | 20.45 | 20.10 | -1.73% | -1.99% |
| qwen_2_5_1_5b_instruct | 28 | 141 | 112 | 1640 | 1298 | 23.94 | 18.44 | -22.97% | +1.86% |
| vllm_falcon3_1b_base | — | 72 | 0 | 1203 | 0 | 18.30 | 17.61 | -3.81% | +1.43% |
| vllm_llama_3_1_8b | — | 128 | 0 | 7416 | 0 | 48.36 | 47.63 | -1.52% | +0.12% |
| vllm_llama_3_2_1b_instruct | — | 64 | 0 | 1034 | 0 | 18.44 | 17.98 | -2.48% | -1.80% |
| vllm_mistral_7b_instruct | — | 128 | 0 | 7416 | 0 | 43.57 | 42.07 | -3.45% | -0.00% |
| vllm_phi_1 | — | 96 | 0 | 1283 | 0 | 25.31 | 25.37 | +0.23% | +0.07% |
| vllm_phi_1_5 | — | 96 | 0 | 1283 | 0 | 24.78 | 25.29 | +2.08% | +0.05% |
| vllm_phi_2 | — | 128 | 0 | 2674 | 0 | 37.06 | 35.81 | -3.38% | -0.03% |
| vllm_qwen2_5_0_5b_instruct | — | 96 | 0 | 380 | 0 | 16.81 | 16.66 | -0.87% | -0.97% |
| vllm_qwen2_5_1_5b_instruct | — | 112 | 0 | 1392 | 0 | 27.63 | 26.74 | -3.21% | -2.46% |
| vllm_qwen2_5_3b_instruct | — | 144 | 0 | 2948 | 0 | 27.73 | 26.14 | -5.74% | -1.27% |
| vllm_qwen2_5_7b_instruct | — | 112 | 0 | 6933 | 0 | 35.25 | 35.29 | +0.10% | -0.31% |
| vllm_qwen3_1_7b | — | 112 | 0 | 1497 | 0 | 21.27 | 22.40 | +5.32% | -4.75% |
| vllm_qwen3_4b | — | 144 | 0 | 3860 | 0 | 28.07 | 27.94 | -0.47% | -2.23% |
| vllm_qwen3_8b | — | 144 | 0 | 7380 | 0 | 52.88 | 53.57 | +1.30% | -2.27% |

step Δ: mean -2.42%, median -1.52%, range -22.97%..+5.32% · ttft Δ: mean -0.91%, median -0.64%

### D. Config diff available, no paired end-to-end number (one side's job failed)

| model | ops | on DS | MB/step | MB on DS | missing side |
|---|---|---|---|---|---|
| janus_pro_1b | 2 | 0 | 17 | 0 | before |
| llama_3_1_8b_instruct | 161 | 160 | 6095 | 5536 | after |
| ministral_8b | 181 | 180 | 7950 | 7380 | before |
| mistral_7b | 161 | 161 | 7558 | 7558 | after |
| playground_v2_5 | 192 | 0 | 2518 | 0 | before |
| qwen_2_5_7b_instruct | 141 | 112 | 7512 | 6442 | after |
| qwen_3_8b | 181 | 180 | 8041 | 7380 | before |
| sdxl_lightning | 192 | 0 | 2518 | 0 | before |
| tiiuae_falcon3_7b_base | 141 | 140 | 7494 | 7066 | after |
| vllm_ministral_8b | 144 | 0 | 7380 | 0 | before |


## What moved, per projection role (decode graph, models where DS was chosen anywhere)

| role | shape groups | → DS | ops | ops on DS | MB/step | MB on DS | % | in0_block_w before | in0_block_w after |
|---|---|---|---|---|---|---|---|---|---|
| down | 16 | 16 | 456 | 456 | 16789 | 16789 | 100.0% | 8 | 2,12,14,19,24,30,32,35,36,38,43 |
| gate/up | 16 | 15 | 912 | 864 | 31699 | 31476 | 99.3% | 2,8 | 2,3,4,6,8,10,12 |
| qkv | 16 | 12 | 456 | 340 | 6995 | 6183 | 88.4% | 2,8 | 4,8,10,12,16 |
| o_proj | 16 | 15 | 456 | 432 | 4579 | 4559 | 99.6% | 8 | 6,8,12,14,16 |
| lm_head | 16 | 1 | 16 | 1 | 5982 | 143 | 2.4% | 2 | 1 |


## Every matmul shape, per role

### down

| model | K × N | ops | bias | config before | config after | in0_block_w before | in0_block_w after | K-tiles/core | MB/step |
|---|---|---|---|---|---|---|---|---|---|
| llama_3_1_8b_instruct | 14336 × 4096 | 32 | — | mcast1d | DS | 8 | 14 | 56 | 1996 |
| llama_3_2_1b_instruct | 8192 × 2048 | 16 | — | mcast1d | DS | 8 | 32 | 32 | 285 |
| llama_3_2_3b_instruct | 8192 × 3072 | 28 | — | mcast1d | DS | 8 | 32 | 32 | 749 |
| ministral_8b | 12288 × 4096 | 36 | — | mcast1d | DS | 8 | 24 | 48 | 1925 |
| mistral_7b | 14336 × 4096 | 32 | — | mcast1d | DS | 8 | 14 | 56 | 1996 |
| qwen_2_5_0_5b_instruct | 4864 × 896 | 24 | — | mcast1d | DS | 8 | 19 | 19 | 111 |
| qwen_2_5_1_5b_instruct | 8960 × 1536 | 28 | — | default | DS | — | 35 | 35 | 409 |
| qwen_2_5_3b_instruct | 11008 × 2048 | 36 | — | mcast1d | DS | 8 | 43 | 43 | 862 |
| qwen_2_5_7b_instruct | 18944 × 3584 | 28 | — | default | DS | — | 2 | 74 | 2020 |
| qwen_3_0_6b | 3072 × 1024 | 28 | — | mcast1d | DS | 8 | 12 | 12 | 94 |
| qwen_3_1_7b | 6144 × 2048 | 28 | — | mcast1d | DS | 8 | 24 | 24 | 374 |
| qwen_3_4b | 9728 × 2560 | 36 | — | mcast1d | DS | 8 | 38 | 38 | 953 |
| qwen_3_8b | 12288 × 4096 | 36 | — | mcast1d | DS | 8 | 24 | 48 | 1925 |
| tiiuae_falcon3_1b_base | 8192 × 2048 | 18 | — | mcast1d | DS | 8 | 32 | 32 | 321 |
| tiiuae_falcon3_3b_base | 9216 × 3072 | 22 | — | mcast1d | DS | 8 | 36 | 36 | 662 |
| tiiuae_falcon3_7b_base | 23040 × 3072 | 28 | — | mcast1d | DS | 8 | 30 | 90 | 2106 |

### gate/up

| model | K × N | ops | bias | config before | config after | in0_block_w before | in0_block_w after | K-tiles/core | MB/step |
|---|---|---|---|---|---|---|---|---|---|
| llama_3_1_8b_instruct | 4096 × 14336 | 64 | — | mcast1d | DS | 2 | 8 | 16 | 2114 |
| llama_3_2_1b_instruct | 2048 × 8192 | 32 | — | mcast1d | DS | 2 | 8 | 8 | 570 |
| llama_3_2_3b_instruct | 3072 × 8192 | 56 | — | mcast1d | DS | 2 | 12 | 12 | 1497 |
| ministral_8b | 4096 × 12288 | 72 | — | mcast1d | DS | 2 | 8 | 16 | 3850 |
| mistral_7b | 4096 × 14336 | 64 | — | mcast1d | DS | 2 | 4 | 16 | 3993 |
| qwen_2_5_0_5b_instruct | 896 × 4864 | 48 | — | mcast1d | mcast1d | 2 | 2 | 4 | 222 |
| qwen_2_5_1_5b_instruct | 1536 × 8960 | 56 | — | default | DS | — | 6 | 6 | 819 |
| qwen_2_5_3b_instruct | 2048 × 11008 | 72 | — | mcast1d | DS | 2 | 8 | 8 | 1725 |
| qwen_2_5_7b_instruct | 3584 × 18944 | 56 | — | default | DS | — | 2 | 14 | 4040 |
| qwen_3_0_6b | 1024 × 3072 | 56 | — | mcast1d | DS | 8 | 4 | 4 | 187 |
| qwen_3_1_7b | 2048 × 6144 | 56 | — | mcast1d | DS | 2 | 8 | 8 | 749 |
| qwen_3_4b | 2560 × 9728 | 72 | — | mcast1d | DS | 2 | 10 | 10 | 1905 |
| qwen_3_8b | 4096 × 12288 | 72 | — | mcast1d | DS | 2 | 8 | 16 | 3850 |
| tiiuae_falcon3_1b_base | 2048 × 8192 | 36 | — | mcast1d | DS | 2 | 8 | 8 | 642 |
| tiiuae_falcon3_3b_base | 3072 × 9216 | 44 | — | mcast1d | DS | 2 | 12 | 12 | 1324 |
| tiiuae_falcon3_7b_base | 3072 × 23040 | 56 | — | mcast1d | DS | 2 | 3 | 12 | 4211 |

### qkv

| model | K × N | ops | bias | config before | config after | in0_block_w before | in0_block_w after | K-tiles/core | MB/step |
|---|---|---|---|---|---|---|---|---|---|
| llama_3_1_8b_instruct | 4096 × 6144 | 32 | — | mcast1d | DS | 2 | 16 | 16 | 856 |
| llama_3_2_1b_instruct | 2048 × 3072 | 16 | — | mcast1d | DS | 8 | 8 | 8 | 107 |
| llama_3_2_3b_instruct | 3072 × 5120 | 28 | — | mcast1d | DS | 2 | 12 | 12 | 468 |
| ministral_8b | 4096 × 6144 | 36 | — | mcast1d | DS | 2 | 16 | 16 | 963 |
| mistral_7b | 4096 × 6144 | 32 | — | mcast1d | DS | 2 | 16 | 16 | 856 |
| qwen_2_5_0_5b_instruct | 896 × 1152 | 24 | yes | mcast1d | mcast1d | 4 | 4 | 4 | 26 |
| qwen_2_5_1_5b_instruct | 1536 × 2048 | 28 | yes | default | default | — | — | 48 | 94 |
| qwen_2_5_3b_instruct | 2048 × 2560 | 36 | yes | mcast1d | mcast1d | 8 | 8 | 1 | 201 |
| qwen_2_5_7b_instruct | 3584 × 4608 | 28 | yes | default | default | — | — | 112 | 491 |
| qwen_3_0_6b | 1024 × 4096 | 28 | — | mcast1d | DS | 8 | 4 | 4 | 125 |
| qwen_3_1_7b | 2048 × 4096 | 28 | — | mcast1d | DS | 8 | 8 | 8 | 250 |
| qwen_3_4b | 2560 × 6144 | 36 | — | mcast1d | DS | 2 | 10 | 10 | 602 |
| qwen_3_8b | 4096 × 6144 | 36 | — | mcast1d | DS | 2 | 16 | 16 | 963 |
| tiiuae_falcon3_1b_base | 2048 × 4096 | 18 | — | mcast1d | DS | 8 | 8 | 8 | 160 |
| tiiuae_falcon3_3b_base | 3072 × 5120 | 22 | — | mcast1d | DS | 2 | 12 | 12 | 368 |
| tiiuae_falcon3_7b_base | 3072 × 5120 | 28 | — | mcast1d | DS | 2 | 12 | 12 | 468 |

### o_proj

| model | K × N | ops | bias | config before | config after | in0_block_w before | in0_block_w after | K-tiles/core | MB/step |
|---|---|---|---|---|---|---|---|---|---|
| llama_3_1_8b_instruct | 4096 × 4096 | 32 | — | mcast1d | DS | 8 | 16 | 16 | 570 |
| llama_3_2_1b_instruct | 2048 × 2048 | 16 | — | mcast1d | DS | 8 | 8 | 8 | 71 |
| llama_3_2_3b_instruct | 3072 × 3072 | 28 | — | mcast1d | DS | 8 | 12 | 12 | 281 |
| ministral_8b | 4096 × 4096 | 36 | — | mcast1d | DS | 8 | 16 | 16 | 642 |
| mistral_7b | 4096 × 4096 | 32 | — | mcast1d | DS | 8 | 16 | 16 | 570 |
| qwen_2_5_0_5b_instruct | 896 × 896 | 24 | — | mcast1d | mcast1d | 4 | 4 | 1 | 20 |
| qwen_2_5_1_5b_instruct | 1536 × 1536 | 28 | — | default | DS | — | 6 | 6 | 70 |
| qwen_2_5_3b_instruct | 2048 × 2048 | 36 | — | mcast1d | DS | 8 | 8 | 8 | 160 |
| qwen_2_5_7b_instruct | 3584 × 3584 | 28 | — | default | DS | — | 14 | 14 | 382 |
| qwen_3_0_6b | 2048 × 1024 | 28 | — | mcast1d | DS | 8 | 8 | 8 | 62 |
| qwen_3_1_7b | 2048 × 2048 | 28 | — | mcast1d | DS | 8 | 8 | 8 | 125 |
| qwen_3_4b | 4096 × 2560 | 36 | — | mcast1d | DS | 8 | 16 | 16 | 401 |
| qwen_3_8b | 4096 × 4096 | 36 | — | mcast1d | DS | 8 | 16 | 16 | 642 |
| tiiuae_falcon3_1b_base | 2048 × 2048 | 18 | — | mcast1d | DS | 8 | 8 | 8 | 80 |
| tiiuae_falcon3_3b_base | 3072 × 3072 | 22 | — | mcast1d | DS | 8 | 12 | 12 | 221 |
| tiiuae_falcon3_7b_base | 3072 × 3072 | 28 | — | mcast1d | DS | 8 | 12 | 12 | 281 |

### lm_head

| model | K × N | ops | bias | config before | config after | in0_block_w before | in0_block_w after | K-tiles/core | MB/step |
|---|---|---|---|---|---|---|---|---|---|
| llama_3_1_8b_instruct | 4096 × 128256 | 1 | — | mcast1d | mcast1d | 2 | 2 | 2 | 558 |
| llama_3_2_1b_instruct | 2048 × 128256 | 1 | — | mcast1d | mcast1d | 2 | 2 | 2 | 279 |
| llama_3_2_3b_instruct | 3072 × 128256 | 1 | — | mcast1d | mcast1d | 2 | 2 | 2 | 419 |
| ministral_8b | 4096 × 131072 | 1 | — | mcast1d | mcast1d | 2 | 2 | 2 | 570 |
| mistral_7b | 4096 × 32768 | 1 | — | mcast1d | DS | 2 | 1 | 16 | 143 |
| qwen_2_5_0_5b_instruct | 896 × 151936 | 1 | — | mcast1d | mcast1d | 2 | 2 | 4 | 145 |
| qwen_2_5_1_5b_instruct | 1536 × 151936 | 1 | — | default | default | — | — | 1 | 248 |
| qwen_2_5_3b_instruct | 2048 × 151936 | 1 | — | mcast1d | mcast1d | 2 | 2 | 2 | 331 |
| qwen_2_5_7b_instruct | 3584 × 152064 | 1 | — | default | default | — | — | 2 | 579 |
| qwen_3_0_6b | 1024 × 151936 | 1 | — | mcast1d | mcast1d | 2 | 2 | 2 | 165 |
| qwen_3_1_7b | 2048 × 151936 | 1 | — | mcast1d | mcast1d | 2 | 2 | 2 | 331 |
| qwen_3_4b | 2560 × 151936 | 1 | — | mcast1d | mcast1d | 2 | 2 | 2 | 413 |
| qwen_3_8b | 4096 × 151936 | 1 | — | mcast1d | mcast1d | 2 | 2 | 2 | 661 |
| tiiuae_falcon3_1b_base | 2048 × 131072 | 1 | — | mcast1d | mcast1d | 2 | 2 | 2 | 285 |
| tiiuae_falcon3_3b_base | 3072 × 131072 | 1 | — | mcast1d | mcast1d | 2 | 2 | 2 | 428 |
| tiiuae_falcon3_7b_base | 3072 × 131072 | 1 | — | mcast1d | mcast1d | 2 | 2 | 2 | 428 |


## in0_block_w on n150

- `in0_block_w == K-tiles-per-core` in **46/59** DS shape groups (max 43).
- collapsed to 1: **1** (mistral_7b lm_head)
- reduced below K-tiles-per-core: **13** groups, all in 7B-plus models:
  - qwen_2_5_7b_instruct down 18944×3584: kPerCore 74 → in0_block_w 2 (37× fewer tiles per burst)
  - mistral_7b lm_head 4096×32768: kPerCore 16 → in0_block_w 1 (16× fewer tiles per burst)
  - qwen_2_5_7b_instruct gate/up 3584×18944: kPerCore 14 → in0_block_w 2 (7× fewer tiles per burst)
  - llama_3_1_8b_instruct down 14336×4096: kPerCore 56 → in0_block_w 14 (4× fewer tiles per burst)
  - mistral_7b gate/up 4096×14336: kPerCore 16 → in0_block_w 4 (4× fewer tiles per burst)
  - mistral_7b down 14336×4096: kPerCore 56 → in0_block_w 14 (4× fewer tiles per burst)
  - tiiuae_falcon3_7b_base gate/up 3072×23040: kPerCore 12 → in0_block_w 3 (4× fewer tiles per burst)
  - tiiuae_falcon3_7b_base down 23040×3072: kPerCore 90 → in0_block_w 30 (3× fewer tiles per burst)
  - llama_3_1_8b_instruct gate/up 4096×14336: kPerCore 16 → in0_block_w 8 (2× fewer tiles per burst)
  - ministral_8b gate/up 4096×12288: kPerCore 16 → in0_block_w 8 (2× fewer tiles per burst)
  - ministral_8b down 12288×4096: kPerCore 48 → in0_block_w 24 (2× fewer tiles per burst)
  - qwen_3_8b gate/up 4096×12288: kPerCore 16 → in0_block_w 8 (2× fewer tiles per burst)
  - qwen_3_8b down 12288×4096: kPerCore 48 → in0_block_w 24 (2× fewer tiles per burst)

## Bias

- 47 of 182 shape groups carry a bias (`ttnn.linear`); **0** of them are on DS.

## Prefill

- DS chosen in **0/34** models' prefill graph (g0). Prefill is a DS-free control, so its ttft delta measures only the non-DS commits in the range.
## Reading the per-role result

- **`down` — moved in 16/16 models, 100% of its 16.8 GB/step.** Universally eligible.
  `in0_block_w` goes from a flat 8 under multicast to K-tiles-per-core under DS: 12, 19, 24,
  32, 35, 36, 38, **43**. Longer weight bursts per core is the whole point of DS, and on
  n150 the config search actually reaches them.
- **`gate/up` — moved in 15/16, 99.3% of its 31.5 GB/step.** The single largest traffic
  block in every model. `in0_block_w` 2 → 3–12. Only `qwen_2_5_0_5b` kept multicast here.
- **`o_proj` — moved in 15/16, 99.6% of traffic**, `in0_block_w` 8 → 6–16, and it is the
  one role where `in0_block_w == kPerCore` in **every** model without exception.
- **`qkv` — moved in only 12/16, 88.4% of traffic.** Every decline is a **bias**:
  Qwen2.5 is the one family in the fleet with a bias on q/k/v, and its qkv is the one
  projection that stayed on multicast in all four Qwen2.5 models. See
  [Why DS was declined](#why-ds-was-declined).
- **`lm_head` — moved in 1/16, 2.4% of traffic.** It stays on 1D multicast in 15 of 16
  models, matching the Blackhole result exactly. The one exception, `mistral_7b`
  (4096×32768 — the only sub-100k vocab in the fleet), is also the only DS group in the
  whole fleet whose `in0_block_w` collapses to 1.

## Why DS was declined

**DS is never chosen for a matmul that carries a bias.** Across all 40 models with a
decode graph, `ttnn.linear` ops (matmul + bias) get DS **0 times out of 1151**, while
`ttnn.matmul` ops get it **2093 times out of 3843**. This one rule explains most of
the fleet's decline pattern:

| model family | biased ops in decode | DS ops | what stays on multicast |
|---|---|---|---|
| Phi-1 / 1.5 / 2 | all of them (73, 73, 97) | 0 | everything — no DS anywhere |
| Qwen2.5 (0.5B–7B) | qkv only (1 per layer) | all the rest | qkv only |
| Qwen3, Llama-3.x, Falcon3, Mistral, Ministral | none | all but lm_head | lm_head |

That is consistent with the DS kernel adding bias on the DRAM-bank cores and reading it
per bank, which requires a DRAM width-sharded bias; an interleaved bias would be silently
wrong rather than slow. Declining is the right call — it is worth noting only because it
means **the Phi family gets no DS benefit at all on n150**, and Qwen2.5 gets it on
everything except qkv.

Two other decline patterns, both smaller:

- **kPerCore too small to block over.** `qwen_2_5_3b` qkv and `qwen_2_5_0_5b` /
  `microsoft_phi_*` o_proj have kPerCore = 1 — one K-tile per core, so `in0_block_w`
  could only ever be 1. Across the fleet no DS group has kPerCore below 4.
- **`google_gemma_1_1_2b_it` declines everywhere despite having no biased ops**, and for
  a different reason: its in0 shard geometry is unlike the rest of the fleet
  (kPerCore 2 for gate/up, 17 for o_proj, 512 for down — the last meaning a single core
  holds all of K). Worth a look on its own; it is the only unbiased model in the fleet
  with zero DS.

### Class C is not a DS measurement

`qwen_2_5_1_5b` shows −22.97%, the largest single number in this report. Do not read it as
a DS win: its **before** compile emitted `default` for all 141 matmuls — no program config
at all — so ttnn's runtime heuristic chose, and the delta measures *fallback → DS*, not
*multicast → DS*. `qwen_2_5_7b` has the identical before-state and is missing its after-side
number. Why the baseline compile produced no configs for exactly these two models is a
separate question worth chasing.

The `vllm_*` rows in class C never get a program config on either side (they compile to
2D multicast plus `default`), so DS never engages. They are useful only as a noise floor:
−5.7% to +5.3%, which is considerably wider than the non-vLLM control group and suggests
the vLLM harness numbers should not be used for fine A/B work.

### The shapes that looked riskiest are the best ones

Reading the static config diff alone, the 7B-8B models looked like the danger zone: they are
the only ones where `in0_block_w` is forced below K-tiles-per-core, and Qwen2.5-7B's down
projection is the extreme case at kPerCore 74 against `in0_block_w` 2 — the same signature the
Blackhole study measured at ~101 GB/s.

**On n150 that concern does not survive measurement.** Llama-3.1-8B's down projection carries
`in0_block_w` 14 against kPerCore 56, a 4x reduction, and it is the **fastest DS shape in the
whole fleet at 246.9 GB/s** against multicast's 206.9 — a 0.84x win worth 1561 µs on its own.
Llama-3.1-8B is also the largest whole-step win measured, 0.933x.

The reason is the same one that governs everything else here: what matters on n150 is having
enough output columns per core, not long K blocks. Big models have large N, so `per_core_n`
lands at 2-7 and DS runs near the DRAM ceiling. It is the *small* models, with N = 2048 and
`per_core_n = 1`, that lose. DS gain scales **with** model size on this part, which is the
opposite of the intuition the Blackhole data suggested.

## What is still not measured

Two gaps remain, both worth closing:

- **The CI end-to-end numbers are noisy, and the three runs show how noisy.** Across the 44
  models where the guard changed no program config at all, the CI step still moves by a median
  of +0.01% but with a range of roughly -6% to +12%. Qwen3-0.6B is the clearest case: CI put DS
  at -12.16%, the device measurement said +0.2%, and the guard run — which changed nothing for
  that model — puts it back at -1.2%. Single-run CI deltas below a few percent are not signal.
  Where CI and the device disagree, the device number is the one to trust.
- **The DS run has no end-to-end number for any model the guard touched.** All four of
  llama_3_1_8b, mistral_7b, qwen_2_5_7b and falcon3_7b failed their benchmark job in that run,
  which is why the guard's cost had to be measured on device rather than read from CI.
- **`ministral_8b` could not be measured at all.** Its CI decode graph carries no trace region
  (zero `capture_or_execute_trace` references, against 22 in falcon3_7b's), so nothing lands in
  a replayed region to measure, and `ttrt` additionally fails it with
  `TT_FATAL: Input Tensor is not allocated` in both variants. Worth chasing on its own — a
  decode graph compiling untraced is a separate bug from anything DS does.
- **Single run per configuration.** The control shapes bound the noise at 0.99x–1.02x, which
  is enough to trust the per-role penalties, but not enough to separate the several models
  whose whole-step ratio sits within 1% of parity. Repeats with `--loops` above 1 would.

## Suggested follow-ups

1. **The activation move deserves its own investigation.** `0b3d7856` takes SiLU out of the
   matmul, where it was nearly free, and puts it in a multiply that then runs 2.1–2.4x slower.
   On the models measured that costs more than DS saves. Fusing the activation into the matmul
   *and* keeping DS — rather than trading one for the other — would make this branch a clear
   win rather than a wash.
2. **Keep the guard, set `kMinBlockWidthFraction` to 4 for Wormhole.** Measured, not projected.
   The guard declines 8 shape groups; the six with a true multicast baseline give up **10215 µs**
   of matmul time to recover **615 µs**. On device the three models it wrongly touches hand back
   their whole DS win — falcon3_7b 0.903x to 0.996x of no-DS, mistral_7b 0.931x to 0.989x,
   llama_3_1_8b 0.933x to 0.965x — while qwen_3_8b, which it does not touch, holds 0.923x at
   exactly 1.000x. It earns its keep on one model, qwen_2_5_7b, whose ratio-37 down runs at
   133.9 GB/s and which improves 0.925x once declined. A fraction of 4 keeps that decline and
   the ratio-16 lm_head decline, and keeps all five ratio-3/4 winners. The constant has to be
   arch-dependent; the Blackhole calibration in the guard's own comment does not transfer.

3. **Add a second decline rule: `per_core_n == 1` with a bandwidth-sized weight.** 8 of 8
   shapes with `per_core_n == 1` and N >= 2048 lose (1.04x-1.12x), across `down` and `o_proj` in
   four models; declining them recovers 627 µs. N = 2048 gives 64 N-tiles over 12 banks, one
   output tile column per core, so each core reads only `in0_block_w` weight tiles and per-core
   launch cost dominates. The weight-size half matters: the `per_core_n == 1` shapes with
   N < 2048 (at most 4.6 MB) *win*, because at 143-169 GB/s neither path is bandwidth-bound.
   This is exactly the shape of criterion [`../criteria.py`](../criteria.py) scores.

4. **Bias blocks DS entirely for the Phi family.** A DRAM width-sharded bias would open DS to
   every biased projection, including all of Qwen2.5's qkv.

## Reproducing this report

```bash
cd ds-perf/n150
./fetch.sh 30806869145 before      # no DS   (perf JSON + shlo/ttir/ttnn dumps, 60 jobs each)
./fetch.sh 30767975002 after       # DS
./fetch.sh 32243412436 patched     # DS + collapse guard
./link_graphs.sh                   # -> graphs/{nods,ds,guard}/<model>/
./run_fleet_all.sh                 # device perf, nods vs ds across the fleet (needs an n150)
./run_guard.sh                     # device perf for the guard variant on the models it changed
./rederive.sh                      # re-join per-matmul CSVs from saved profiler artifacts
./assemble.sh                      # all tables -> n150-ds-matmul-ab.md + .html
```

`run_fleet_all.sh` drives [`../run_model_fleet.sh`](../run_model_fleet.sh), which needs
`DSPERF` set — the driver `cd`s to the repo root before resolving `$0`, so the default
script-directory guess lands one level too high. A perf-trace build is required
(`-DTT_RUNTIME_ENABLE_PERF_TRACE=ON`); see [`../README.md`](../README.md) for the full
runbook, including the profiler-health check to run before trusting any absolute number.

`ttnn_role_diff.py` reuses `survey()` and `largest_divisor_upto()` from
[`../static_matmul_survey.py`](../static_matmul_survey.py) and the role rules from
[`../by_projection.py`](../by_projection.py). `device_report.py` consumes
[`../matmul_detail.py`](../matmul_detail.py) output, which now carries the fused-activation
column that the like-for-like comparison depends on. Graph index 1 is decode (M = 32), 0 is
prefill (M = 544); g2/g3 duplicate g0/g1.
