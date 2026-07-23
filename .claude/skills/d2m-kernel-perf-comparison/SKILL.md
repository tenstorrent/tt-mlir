---
name: d2m-kernel-perf-comparison
description: >-
  Establish, run, and audit defensible D2M-JIT versus compiler-TTNN or
  handwritten-TTNN performance comparisons. Use when comparing wall latency,
  submit time, input/output transfers, resident-input execution, device kernel
  rows, device makespan, program caches, fusion, Tracy/ttrt artifacts, or any
  surprising performance gap in TTMetal workloads such as Llama decoder
  blocks, SDPA, softmax, matmul, and projections.
---

# D2M Performance Comparison

## Non-Negotiable Rules

- Define the timed boundary before collecting numbers. Do not use `e2e`,
  `submit`, `device time`, or `kernel time` without a precise definition.
- Compare one semantic workload: identical shapes, dtypes, values or value
  distribution, outputs, cache mutation, causal mask, memory placement, and
  target chip/grid.
- Synchronize the start and end of a latency sample. API call duration is not
  completion time unless the call is documented or proven blocking.
- Treat uninstrumented wall latency, host tracing, and device profiling as
  separate experiments. Instrumentation can change compilation, caches,
  scheduling, and timing.
- Never label `wall - sum(device kernel rows)` as host overhead. Call it
  `unattributed synchronized time` until serialization, device makespan,
  transfers, and unprofiled device work have been established.
- Never trust a ratio until row counts, invocation boundaries, active cores,
  cache state, runtime/build identity, and generated graph structure pass audit.

## Timing Contracts

Use one of these names in commands, artifacts, and reports.

### Host-to-host latency

Start with ordinary host tensors. Include all required input conversion and
upload, graph execution, synchronization, every output readback, and the copy
into caller-owned host tensors. This is the default API comparison.

This contract does not require transfers to occur in the same API call. It
requires them to be inside the same timed envelope.

### Resident-to-resident latency

Prepare equivalent device-resident inputs before timing. Start from an idle
device, submit the graph, wait for the requested device outputs, and stop
without host readback. Use this to study steady inference after weights and
state are resident.

Do not publish this comparison unless both backends accept equivalent device
tensors. If one ABI embeds host transfers in its executable, report the mode as
unsupported rather than substituting a different boundary.

### Submit envelope

Measure `submit` followed by an explicit wait on the returned outputs. State
where input upload and output download occur for that executable. The envelope
can include host program construction, runtime-argument refresh, allocation,
command dispatch, transfers, device idle gaps, kernels, and blocking reads.

Use this as a synchronized total, not as causal attribution.

### Device operation duration

Use `DEVICE KERNEL DURATION [ns]` for one validated profiler row. State the op,
row-selection rule, active core count, loop, and profiler configuration.

### Device invocation makespan

Compute the earliest validated device start to latest validated device end for
one invocation. Include gaps in that span. Partition rows by program and loop
metadata where available; otherwise establish exact repeated row structure and
cross-check it against host invocation count.

Do not infer makespan by summing rows. Rows can overlap, omit transfers, include
one-time work, or be assigned to the wrong invocation.

## Timing Model

Use this as a checklist, not as an additive equation unless barriers isolate
the terms:

```text
host preparation
+ input conversion/upload completion
+ runtime construction/cache refresh
+ allocation and command dispatch
+ device makespan, including idle gaps and unprofiled commands
+ output download completion
+ caller-side copies/postprocessing
= synchronized wall latency
```

Asynchronous stages can overlap or charge their completion to a later wait.
Adding a barrier can isolate a stage but also changes the execution schedule.
Report both the natural latency and any barrier-based diagnostic run.

## Required Workflow

### Bottleneck attribution gates

Treat a full-graph comparison as the start of a study, not its bottleneck
analysis. Use `references/one_to_one_protocol.md` and keep a machine-readable
study file based on `references/study_template.json`.

The following claims require different evidence:

- **API latency:** exact semantic boundary and source IR hash, matching
  binary I/O, an audited build manifest linking that source to both binary
  hashes, one fingerprinted input corpus, output validation, one hashed runtime
  module, a matching single-chip/grid target, and at least two independent
  trials in each ABBA order with an acceptable order effect.
- **D2M internal transfer attribution:** flatbuffer transfer payloads and
  locations correlated with steady device-timeline gaps. This can explain the
  measured D2M executable but is not a D2M-versus-TTNN bandwidth comparison.
- **Cross-backend transfer attribution:** matched transfer-only capsules at
  representative physical payload sizes and layouts. Logical byte counts or
  differently placed API phases are insufficient.
- **Dispatch attribution:** dependent one-tile chains at multiple program
  counts. Compare the fitted steady latency slope and intercept; do not divide
  a full graph's unattributed time by its row count.
- **Math attribution:** matched semantic capsules compiled from the same TTIR
  and input corpus. Compare synchronized capsule makespan, then inspect rows.
  Full-decoder row sums cannot identify a projection or SDPA bottleneck.

Run the readiness auditor before publishing a causal claim:

```bash
python .claude/skills/d2m-kernel-perf-comparison/scripts/audit_comparison_study.py \
  path/to/study.json --output path/to/readiness.json
```

`blocked` is the expected state for a claim whose control has not been run.
Do not replace a missing control with a trace-based estimate.

### 1. Freeze the work unit

Record:

- Model boundary and logical inputs/outputs.
- Batch, sequence, head count, head dimension, hidden size, and relevant cache
  dimensions.
- Dtypes, layouts, memory spaces, and physical grid/core count.
- Weight shapes and structure. Values may be synthetic if both paths use the
  same distribution and the kernel path is value-independent.
- Mutated inputs, returned outputs, and whether every output is consumed.
- Fusion boundary and op count for each backend.

Reject a comparison that silently substitutes a smaller graph, a different
cache update, a different precision, or a different set of returned tensors.

### 2. Prove graph and binary provenance

Save the normalized IR, compiled binary, command used to generate it, git SHA,
runtime module path, system descriptor, and binary hash or unique artifact
directory. Link them with a manifest based on
`references/build_manifest_template.json`; co-location is not provenance.
Inspect the binary or IR for the expected fusion and input/output descriptors.

Do not reuse global profiler CSVs. Read results from the per-run,
per-binary artifact directory and verify timestamps plus row counts.

Use the binary auditor for a machine-readable record and parity check:

```bash
python .claude/skills/d2m-kernel-perf-comparison/scripts/audit_binaries.py \
  path/to/decoder.ttm path/to/decoder.ttnn \
  --output path/to/binary_manifest.json
```

### 3. Prove runtime and cache state

Record separately:

- Compiler/JIT cache hits and misses.
- TTMetal/TTNN device program-cache state from
  `device.is_program_cache_enabled()`.
- Cold first invocation and steady invocations.
- Whether inputs or outputs are retained across invocations.
- Runtime Python module path and linked runtime build.

Kernel-source JIT hits do not prove that runtime `Program`, mesh workload,
runtime arguments, or CB bindings are cached. Never describe these as one
generic cache.

Do not compare cache-off D2M with cache-on TTNN unless the experiment is
explicitly a cache ablation.

### 4. Validate correctness first

Validate realistic nonzero inputs at the exact benchmark shape. Record PCC or
the appropriate numerical criterion and output statistics. Revalidate after a
cache, lifetime, layout, fusion, or runtime-argument change.

Discard performance results from a measured runner with a `TT_FATAL`, timeout,
device reset, unexpected fallback, stale artifact, missing profiler row, or
failed output. Record warnings from a separate preflight process and establish
that it returned success before the measured runner started; do not silently
conflate preflight diagnostics with runner health.

### 5. Collect uninstrumented wall latency

Use a persistent device handle, exclude device open/close and compilation,
perform warmups, and retain every individual steady sample. Use
`time.perf_counter_ns()` around the full contract.

Before the first timed sample, drain prior work. At the sample end, wait on the
specific outputs and perform every action required by the selected contract.
Report median, min, max, and sample count. Investigate multimodal samples
instead of hiding them in an average.

For the Llama decoder harness:

```bash
source env/activate
python .claude/skills/d2m-kernel-perf-comparison/scripts/run_wall_trials.py \
  --d2m-binary path/to/decoder.ttm \
  --compiler-ttnn-binary path/to/decoder.ttnn \
  --output-dir path/to/wall_trials \
  --warmup 2 --loops 7 --trials-per-order 2 \
  --expected-runtime-sha256 HASH \
  --expected-input-corpus-sha256 HASH
```

The default trial schedule is `D2M/TTNN`, `TTNN/D2M`, `TTNN/D2M`,
`D2M/TTNN`. Each entry runs in a fresh process and writes its exact command,
stdout log, manifest, binary hashes, harness hash, runtime hash, and corpus
hash to `wall_trial_index.json`. The runner times out one stuck trial instead
of letting the full study stall indefinitely.

Prefer loading the same hashed binaries used for device profiling. Compiling in
the wall harness is useful during development but weakens provenance and can
hide one-time JIT work in noisy logs even though compilation is outside the
timed samples.

The exact-binary path derives deterministic inputs from the flatbuffer input
descriptors and checks logical shape/dtype parity before opening a device. See
`references/llama_decoder_baseline.md` for one fully reconciled baseline and
the interpretation limits that remain.

Treat phase output as API-envelope timing. `input_enqueue_ms` is not transfer
completion unless an explicit input wait proves it.

### 6. Collect device profiles separately

Use a fresh process and unique artifact directory. Keep shape, memory space,
grid, fusion, and binary-generation options aligned with the wall run. Record
whether profiler instrumentation changed the emitted binary or triggered new
JIT builds.

For each invocation:

1. Select rows using program/loop metadata when available.
2. Verify expected row count and op ordering.
3. Exclude const-eval and one-time input conversion only when the chosen
   contract excludes them.
4. Report individual op durations.
5. Compute device makespan from validated timestamps.
6. Compute summed row duration only as a separate diagnostic.
7. Quantify device gaps and overlap; do not assume serialized execution.

`tools/ttrt/common/perf.py` currently sums every
`DEVICE KERNEL DURATION [ns]` row per loop. Audit the rows yourself before
using its `total_device_kernel_duration_ns` as an invocation duration.

Use the skill's audit helper to keep invocation span and row sums separate:

```bash
# TTNN: recover program/loop metadata from the matching Tracy export.
python .claude/skills/d2m-kernel-perf-comparison/scripts/audit_device_profile.py \
  path/to/ops_perf_results.csv \
  --trace-data path/to/tracy_ops_data.csv \
  --clock-mhz 1000

# D2M: split a time-ordered report using the audited program-launch count.
python .claude/skills/d2m-kernel-perf-comparison/scripts/audit_device_profile.py \
  path/to/ops_perf_results.csv \
  --rows-per-invocation 305 \
  --binary-manifest path/to/binary_manifest.json \
  --clock-mhz 1000
```

With a binary manifest, the device auditor reports the largest row gaps and
kernel rows and annotates TTMetal gaps whose following operation location
matches a flatbuffer transfer-command location. Inspect the raw command order
before treating location correlation as causal.

The helper intentionally leaves unmatched rows separate. Do not assign those
rows to an invocation without additional evidence.

Legacy reports can instead use `--repetitions N` when every call-count group
contains exactly one row per invocation and rows are operation-major. The
helper validates that condition before inferring groups.

Regenerate the two Tracy CSV views from the raw capture when provenance is in
doubt. The exporter records hashes and exact arguments:

```bash
python .claude/skills/d2m-kernel-perf-comparison/scripts/export_tracy.py \
  path/to/tracy_profile_log_host.tracy path/to/exported_tracy
```

Keep `tracy_profile_log_host.tracy`; it is the visualizable source artifact.
Open it with a version-compatible `tracy-profiler`. The CSVs are derived views
for automation, not replacements for the interactive timeline.

Audit host zones inside the synchronized loop windows recovered from matching
Tracy metadata and ttrt results:

```bash
python .claude/skills/d2m-kernel-perf-comparison/scripts/audit_host_trace.py \
  path/to/tracy_ops_times.csv path/to/tracy_ops_data.csv path/to/result.json \
  --output path/to/host_timing_manifest.json
```

The report ranks inclusive zone times and computes the union of exported zone
intervals per host thread. Inclusive zones can nest, so their durations are not
additive. Per-thread zone coverage is not CPU utilization. Use both as leads
for controlled ablations and inspect the raw `.tracy` timeline before assigning
causality.

Reconcile all manifests before interpreting ratios:

```bash
python .claude/skills/d2m-kernel-perf-comparison/scripts/compare_manifests.py \
  --d2m-device path/to/d2m_device.json \
  --ttnn-device path/to/ttnn_device.json \
  --d2m-host path/to/d2m_host.json \
  --ttnn-host path/to/ttnn_host.json \
  --wall path/to/wall.json --binaries path/to/binaries.json \
  --output path/to/comparison.json
```

The reconciliation checks hashes and sample stability, keeps wall, device span,
and kernel-row sum as distinct ratios, and tests whether removing the inclusive
device-profiler read zone predicts the matching uninstrumented submit envelope.
That subtraction is a sanity check specific to the observed nesting, not a
general performance equation.

Pass the matching ttrt result file to the auditor to check that each device
span fits inside its synchronized submit-plus-output host envelope:

```bash
python .claude/skills/d2m-kernel-perf-comparison/scripts/audit_device_profile.py \
  path/to/ops_perf_results.csv \
  --trace-data path/to/tracy_ops_data.csv \
  --run-results path/to/result.json \
  --clock-mhz 1000 \
  --output path/to/device_timing_manifest.json
```

### 7. Attribute gaps with controlled ablations

When wall latency and device makespan disagree, vary one factor at a time:

- Program cache off versus on, with identical inputs and binary.
- Host-to-host versus resident-to-resident, when both backends support it.
- Natural asynchronous execution versus an explicit post-input barrier.
- Output wait only versus complete host readback.
- Full graph versus a structurally identical subgraph or single-op repeat.
- Runtime host tracing versus uninstrumented execution to quantify tracing
  perturbation.

Use low-overhead counters or Tracy zones around concrete runtime operations:
input writes, output reads, program-cache lookup, cache-hit refresh, program
construction, buffer allocation/deallocation, workload enqueue, and waits.
Require the instrumented total to remain reasonably close to the
uninstrumented run before using its percentages.

## Backend Boundary Audit

Re-check these source paths at the current revision before interpreting phases:

- `runtime/lib/ttmetal/runtime.cpp`: TTMetal `getLayout`, `toLayout`, `wait`,
  `toHost`, device-open cache policy, and `submit`.
- `runtime/lib/ttmetal/executor.cpp`: flatbuffer command execution, buffer
  lifetime, program construction/cache refresh, and workload enqueue.
- `runtime/lib/ttmetal/executor_utils.h`: blocking behavior of host writes and
  reads.
- `runtime/lib/ttnn/runtime.cpp`: TTNN layout conversion, waits, host readback,
  program executor, and cache policy.
- `tools/ttrt/common/run.py`: exact timer placement and synchronization.
- `tools/ttrt/common/perf.py`: profiler-row aggregation.

Current TTMetal behavior can make phase labels especially misleading:

- `toLayout` may return the host tensor unchanged, leaving input writes inside
  the flatbuffer `submit`.
- `EnqueueReadBufferCommand` can perform a blocking device-to-host read inside
  `submit`; later `toHost` can therefore be only a wait/no-op plus host memcpy.
- `submit + wait` can contain both transfers and all graph commands.

Current TTNN behavior can perform layout conversion and device upload before
`submit`. The host-to-host total can still be fair even though phase placement
is different. Per-phase comparisons are not fair until completion boundaries
are normalized.

`ttrt run` currently converts inputs once before its repeated loop. TTNN loops
therefore reuse device tensors. TTMetal `toLayout` can leave inputs on the host,
so TTMetal flatbuffer writes can recur inside every loop. Do not treat repeated
`ttrt perf` loops as a transfer-parity comparison between these backends.

## Device Row Audit

- State the exact metric: device duration, FW duration, per-core min/max/avg,
  TRISC, BRISC, NCRISC, or host duration.
- Compare device duration with the maximum compute/data-movement thread span.
  A large difference suggests synchronization, dispatch, or data movement.
- Use `DEVICE FW START CYCLE`, `DEVICE FW END CYCLE`, and
  `OP TO OP LATENCY [ns]` only after checking clock conversion, wraparound,
  device ID, and invocation partitioning.
- Inspect active cores in the raw device log. A declared 8x8 grid does not
  prove that all 64 cores ran useful work.
- Inspect dumped kernels for expected tile work, multicast, CB reuse, DST
  reuse, packing/unpacking, and matmul/reduction structure.
- Report both absolute time and ratio. Ratios alone amplify small or invalid
  denominators.
- Check profiler row ordering before partitioning repeated runs. D2M reports
  can be operation-major, with all loop samples for one call count adjacent,
  rather than invocation-major.
- Preserve input-layout rows that have no program metadata. They are often
  outside `submit`, but remain part of a host-to-host contract.

## SDPA-Specific Checks

- Compare fused SDPA with fused SDPA. Exclude layout/tilize/untilize rows from
  a math-kernel claim, but include them in the matching end-to-end contract.
- Validate causal masking semantics and exact sequence/cache dimensions.
- Reason by tile work: matmul scales with `S * S * D`; softmax scales with
  `S * S`. A large matmul can hide a weak softmax.
- Prefer shape-preserving ablations: QK only, scores plus softmax, softmax only,
  PV only, and full SDPA.
- Generate flatbuffers in one process and profile them in a fresh process when
  debugging profiler hangs.
- Avoid Python scalar runtime kernel arguments in saved D2M-JIT flatbuffers;
  specialize shape/grid constants through closure-captured Python integers.

## Device Recovery

Use `tt-smi -r` when a reset is needed. After a reset, run an unprofiled health
check before collecting data. Treat all caches as cold and repeat warmup.

Do not reset merely because compilation or a profiler run is slow. Distinguish
active host compilation from a device hang with process state and bounded logs.

## Reporting Template

Report in this order:

1. **Contract:** host-to-host, resident-to-resident, submit envelope, device
   operation, or device makespan.
2. **Parity:** graph boundary, shapes, dtypes, layouts, memory, grid, fusion,
   and correctness.
3. **Environment:** git SHA, runtime path, device, cache states, profiler mode,
   warmups, loops, and artifact paths.
4. **Measurements:** raw samples, median/min/max, individual device ops,
   device makespan, and summed rows as separate fields.
5. **Confirmed attribution:** effects established by controlled ablations.
6. **Unattributed time:** synchronized time not yet causally assigned.
7. **Hypotheses:** clearly labeled and paired with the next discriminating
   experiment.
8. **Invalid attempts:** hangs, stale artifacts, wrong runtime, profiler
   perturbation, or failed correctness that must not support conclusions.

When a comparison reveals a new reusable timing trap, update this skill in the
same workstream. Remove stale workload-specific assumptions rather than
accumulating exceptions.
