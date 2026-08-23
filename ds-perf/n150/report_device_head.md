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

