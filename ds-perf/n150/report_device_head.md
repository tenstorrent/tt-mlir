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

### The fidelity confounder

**Every DS-vs-multicast ratio in this report varies math fidelity along with the kernel.**
`buildComputeConfig` (`MatmulProgramConfig.cpp`) attaches an explicit compute config to DS
matmuls and nothing else, so a DS matmul ran the fidelity that function picked while its
multicast counterpart took ttnn's default. Counted over every matmul instance in this
measurement record:

| variant | kernel | fidelity |
|---|---|---|
| DS | DS config | **HiFi2 97%**, LoFi 3% |
| DS | multicast | LoFi 71%, HiFi2 29% |
| no DS | multicast | **LoFi 87%**, HiFi2 13% |

The 3% LoFi on the DS side is llama_3_1_8b's bfp4 gate/up, the one weight type
`buildComputeConfig` exempted.

The asymmetry matters because MVMULs issued per tile-MAC scale with `fidelity_loops`, and
DS concentrates the output: `per_core_N` is 8–12 against multicast's 1–3 on the same shape,
so DS issues 4–8x the tile-MACs per core. HiFi2 is a tax only DS can pay, and only DS was
charged it.

**What survives and what does not.** The headline direction is safe and if anything
understated: DS won while carrying the tax, so the kernel advantage measured here is a
floor. The two per-shape loss signatures are not safe. A fidelity-matched A/B on p150
(`ds-perf/results/kernel_fidelity_matrix.csv`) reverses the verdict on all three shapes
tested — multicast wins at HiFi2, DS wins at LoFi — and on `2048x2048` o_proj, the shape
the `per_core_n == 1` rule was drawn from, matched fidelity gives parity (0.98x) rather
than the 1.11x loss recorded below. A ladder sweep at LoFi
(`ds-perf/results/ds_blockw_fidelity.csv`) further shows the DS path holding 96–98% of peak
down to a K-step ratio of 15, which no threshold of 2 or 4 would keep.

Both rules below should therefore be read as measured under the shipped configuration, not
as properties of the DS kernel.

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

