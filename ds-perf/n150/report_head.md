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
2. **`per_core_n == 1` with N >= 2048** (see the caveat below) — 8 of 8 such shapes, 1.04x to
   1.12x, spanning both `down` and `o_proj` in four unrelated models.

Both signatures were measured with DS on HiFi2 and multicast on LoFi, which the shipped
compiler does by construction. That confound is large enough to account for the smaller
losses on its own, and a matched-fidelity A/B on p150 reverses the verdict on the shape
signature 2 was drawn from. Read them as properties of the shipped configuration rather
than of the DS kernel — see [the fidelity confounder](#the-fidelity-confounder).

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

