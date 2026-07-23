# Llama Decoder Baseline

This is a historical reconciled D2M-JIT versus compiler-TTNN decoder
measurement. It predates runtime-module hashing, input-corpus fingerprints,
output parity in the timing manifest, and counterbalanced trial order. Retain
it as evidence about that artifact set, not as the canonical performance
baseline or a bottleneck attribution.

## Contract

- Boundary: Llama 3 8B decoder hidden states through attention, K/V cache
  updates, residual paths, and MLP.
- Shape: batch 32, sequence 16, 512 tokens.
- Precision: BF16 activations and weights.
- Device: one chip, 8x8 worker grid.
- Wall samples: exact precompiled binaries, two warmups and seven measured
  host-to-host iterations, persistent device handle, program cache enabled.
- Host-to-host includes input layout/transfers, synchronized execution, every
  output readback, and copies into caller-owned torch tensors.

## Validated Results

| Metric | D2M-JIT | Compiler TTNN | D2M / TTNN |
| --- | ---: | ---: | ---: |
| Host-to-host median | 471.195 ms | 158.725 ms | 2.969x |
| Device invocation span | 447.777 ms | 168.079 ms | 2.664x |
| Sum of profiler kernel rows | 81.545 ms | 37.013 ms | 2.203x |

The two binaries have matching logical input/output shapes and matching
correctness hashes. The D2M binary performs 19 host-to-device writes totaling
470,306,816 bytes. Their inclusive Tracy zones total about 410.879 ms, an
effective diagnostic bandwidth of 1.145 GB/s. The compiler-TTNN logical inputs
total 457,195,648 bytes; its wall input-enqueue median is about 89.594 ms, or
5.103 GB/s if treated as a diagnostic bandwidth.

Subtracting the inclusive device-profiler read zone from profiled submit time
predicts the uninstrumented submit envelope within 0.642% for D2M and 1.264%
for compiler TTNN. This supports the reconciliation of instrumented and
uninstrumented runs; it does not establish a generally additive timing model.

## Interpretation

The row sums are not device execution time. Rows overlap and omit substantial
transfer, dispatch, idle, or otherwise unprofiled time. The device span is the
validated earliest-start to latest-end interval for one invocation. The wall
number is the primary API-level comparison.

The baseline establishes three separate gaps:

1. D2M is slower in the kernels represented by profiler rows.
2. D2M has a larger device span than the kernel-row sum alone explains.
3. D2M host input handling is much slower despite similar logical byte volume.

Those observations justify targeted transfer, dispatch/cache, and kernel
ablations. They do not yet identify a single root cause.

A later audit correlated 94.9% and 97.2% of the two steady D2M inter-row gap
totals with flatbuffer transfer-command locations. The five largest gaps
preceded the three 117,440,512-byte MLP-weight writes and two 33,554,432-byte
projection-weight writes. This establishes a transfer/ABI bottleneck in that
D2M executable. It does not compare D2M and TTNN transfer mechanisms; that
requires matched transfer-only capsules.

Raw Tracy captures, profiler CSVs, binaries, and generated manifests are kept
outside git because they are build artifacts. A publishable result must retain
those artifacts together with binary SHA-256 hashes, commands, runtime path,
git SHA, and system descriptor.
