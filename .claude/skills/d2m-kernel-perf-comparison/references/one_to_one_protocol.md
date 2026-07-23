# One-to-One Bottleneck Protocol

The goal is not to force D2M and TTNN to emit identical kernels. The goal is
to hold one semantic work unit and one experimental variable constant at a
time so a measured difference has a defensible owner.

## Tier 0: Semantic Decoder Contract

Compile both backends from the same normalized TTIR source. Record its SHA-256,
the compiler revision, binary hashes, main-program input/output descriptors,
chip, grid, precision, memory policy, and fusion boundary.

Use a build manifest based on `build_manifest_template.json` to link the source
hash, exact D2M and TTNN compiler commands, compiler revision, system
descriptor, and output binary hashes. A source file and binaries stored in the
same artifact directory do not prove this relationship.

Record the physical device grid separately from active worker cores. An 8x8
physical grid does not prove every emitted operation uses 64 cores; establish
active-core utilization from the device profile when it matters to a kernel
claim.

Generate one deterministic input corpus. Record every tensor's shape, dtype,
byte length, and SHA-256 plus a combined corpus hash. Run an untimed validation
on both exact binaries and require finite outputs and the declared PCC for
every returned tensor.

This tier proves semantic parity. It contains no performance claim.

## Tier 1: Full-Block API Latency

Use the host-to-host contract because the current TTMetal ABI cannot accept
device-resident program inputs: TTMetal `getLayout`/`toLayout` leave inputs on
the host and `EnqueueWriteBufferCommand` performs the executable's writes.
TTNN can upload during `toLayout`. Only the total synchronized envelope is a
fair full-block metric.

Pin and hash one runtime shared object. Record program-cache state and kernel
JIT telemetry. Run at least two independent trials in each order:

```text
trial 1: D2M, TTNN
trial 2: TTNN, D2M
trial 3: TTNN, D2M
trial 4: D2M, TTNN
```

Each backend gets at least two warmups and five retained samples per trial.
Reject the aggregate when a backend's order medians differ by more than 10%,
or investigate and add more blocks when they differ by more than 5%.

This tier supports an API-latency ratio only. Phase timings show where each API
charges work; they do not compare equivalent phases.

Use `scripts/run_wall_trials.py` to create this schedule in independent
processes. Pin the expected runtime and input-corpus hashes once the first
valid run establishes them; do not rely on the active Python environment as
provenance.

## Tier 2: Transfer and Dispatch Controls

### Transfer-only capsules

Measure both backends at representative physical payloads and the layouts used
by the decoder:

- Activation/cache payload near 4-8 MiB.
- Projection weight payload near 32 MiB.
- MLP weight payload near 112 MiB.

For each size, use the same host dtype, device memory space, physical layout,
queue mode, synchronization, and readback policy. Record logical and physical
bytes separately. Fit latency versus bytes; compare slope and intercept.

Until these capsules exist, location-correlated TTMetal gaps can establish that
D2M streams weights, but cannot establish a TTNN-versus-D2M bandwidth ratio.

### Dispatch slope

Compile dependent one-tile operation chains with program counts such as 1, 8,
32, 128, and 305. Keep transfer bytes constant. For each backend, fit steady
synchronized latency versus program count. Report slope, intercept, residuals,
and device row count.

This separates per-program construction/refresh/dispatch cost from fixed
transfer and readback cost. A single 305-program graph cannot do so.

## Tier 3: Matched Semantic Capsules

Compile each capsule through D2M and compiler TTNN from one TTIR module:

| Capsule | Decoder shape or boundary |
| --- | --- |
| RMSNorm | `[32, 16, 4096]` |
| K/V projection | `512x4096` by `4096x1024` |
| Q/output projection | `512x4096` by `4096x4096` |
| SDPA plus cache update | exact 32-query-head/8-KV-head causal boundary |
| MLP expand | `512x4096` by `4096x14336` |
| MLP contract | `512x14336` by `14336x4096` |

For each capsule:

1. Hash TTIR, binaries, runtime, and inputs.
2. Validate every output.
3. Audit physical transfers and emitted op/program counts.
4. Measure counterbalanced host-to-host latency.
5. Profile separately and compare device invocation makespan.
6. Inspect kernel rows only after the capsule-level result is stable.

The capsule makespan owns the semantic operation gap even when one backend
fuses it and the other emits multiple rows. Do not compare one selected D2M row
with one selected TTNN row unless their exact work and dependencies match.

## Tier 4: Full-Block Reconciliation

Use the transfer slopes, dispatch slope, and semantic capsule results to form
testable predictions about the full block. Predictions are not additive when
commands overlap. Validate each proposed fix by changing one mechanism and
rerunning Tier 1 with the same corpus and runtime.

Report conclusions in three sets:

- **Supported:** the required control is complete and reproduces.
- **Observed but unattributed:** the synchronized gap is real but its control
  is incomplete.
- **Blocked:** parity, provenance, correctness, or stability failed.
