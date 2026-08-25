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
2. **Keep the guard, but do not take the threshold from this report.** The
   `kMinBlockWidthFraction = 2` finding stands: the guard as shipped declines shapes whose DS
   configs are the fleet's fastest, and on device the three models it wrongly touches hand back
   their whole win. The replacement value of 4 proposed here does **not** stand. It was fitted
   to timings taken with DS on HiFi2, and a ladder sweep at LoFi
   (`ds-perf/results/ds_blockw_fidelity.csv`) shows the DS path holding 96-98% of peak down to a
   K-step ratio of 15 — every row a fraction of 4 would decline. On that evidence the cut belongs
   near an absolute floor on `in0_block_w` (1 and 2 are bad at kPerCore 16, 43 and 90 alike;
   6 and above are flat) rather than at any fraction of kPerCore. Measured on p150; the floor
   needs the same ladder on 12 banks before a constant changes.
3. **Do not add the `per_core_n == 1` rule.** At matched fidelity the shape it was drawn from,
   `2048x2048` o_proj, is parity (0.98x) rather than a 1.11x loss
   (`ds-perf/results/kernel_fidelity_matrix.csv`). The rule was measuring the HiFi2 tax that
   `buildComputeConfig` charges DS and not multicast, so the fix is the fidelity assignment, not
   a decline.
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
