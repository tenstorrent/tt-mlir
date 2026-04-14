# Milestone: tt-xla Integration

## Description

Integrate Chisel with tt-xla so that models compiled and executed through the
XLA/StableHLO → tt-mlir → TTNN pipeline can be debugged with Chisel's per-op
golden comparison. tt-xla uses a different entry point than the builder — it
compiles StableHLO to flatbuffers and executes them via TTRT. Chisel must be
usable from this path without requiring builder APIs.

## Goals

- Models compiled through tt-xla can be debugged with Chisel callbacks
- Per-op accuracy reports (PCC, atol, rtol) produced for tt-xla workloads
- Integration does not require modifying tt-xla's compilation pipeline — only
  its execution/runtime path
- Chisel remains a passive observer (no changes to compilation)

## Key Challenges

1. **Entry point**: tt-xla drives execution differently than builder. Need to
   identify where `DebugHooks` can be registered in the tt-xla runtime path
   (likely at the TTRT execution call site)
2. **MLIR availability**: Chisel reads TTNN MLIR from `TTNNBinary.mlir.source`
   in the flatbuffer. Verify that tt-xla-compiled flatbuffers include this
   field (it must be enabled during compilation)
3. **Golden mappings coverage**: tt-xla may produce TTNN ops not yet covered
   by `GOLDEN_MAPPINGS`. Audit which ops appear in typical tt-xla workloads
   (JAX models) and ensure golden implementations exist
4. **Tensor format differences**: tt-xla may use different tensor layouts or
   data types than builder-compiled models. Chisel's tensor conversion utils
   must handle these correctly
5. **Multi-program / multi-binary**: tt-xla workloads may involve multiple
   programs or binaries per inference. Chisel's hierarchical state model
   (ChiselContext → BinaryState → ProgramState) should handle this, but
   needs validation

## Prerequisites

- issue_06 through issue_10 (core Chisel PRs 1-5 complete)
- tt-xla must embed TTNN MLIR source in compiled flatbuffers
- `DebugHooks` registration point accessible from tt-xla runtime

## Integration Points

### tt-xla side
- Identify the TTRT execution call in tt-xla runtime
- Add `enable_chisel` option (or equivalent) that constructs `ChiselContext`
  and registers callbacks before execution
- Pass through configuration (output dir, report path, skip mode settings)

### Chisel side
- Ensure `ChiselContext` can be initialized without builder dependencies
- Validate that `IRModule` parsing works for tt-xla-compiled MLIR
- Add any missing golden mappings for tt-xla-specific op patterns

## Test Models

- JAX models compiled through tt-xla (e.g., JAX ResNet, JAX BERT)
- StableHLO test cases from tt-xla test suite

## Acceptance Criteria

- At least one JAX model compiled through tt-xla produces a correct Chisel
  per-op accuracy report
- `ChiselContext` initializes and callbacks fire correctly from tt-xla runtime
- No builder dependency required for tt-xla integration path
- Golden mappings cover all TTNN ops in the test model(s)
- Documentation on how to enable Chisel from tt-xla
