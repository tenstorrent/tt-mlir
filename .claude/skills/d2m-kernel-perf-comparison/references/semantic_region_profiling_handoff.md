# Semantic Region Profiling Handoff

## Goal

Attribute device profiler rows to stable, high-level semantic regions such as
`decoder.sdpa` and `decoder.mlp` without requiring D2M and TTNN to emit the
same number of programs or kernels.

This is attribution metadata, not a timing mechanism:

- Measure wall latency without Tracy capture.
- Collect device attribution in a separate Tracy capture.
- Do not add device-side markers or timers.
- Check that capture-on device totals agree with an independent profile before
  using the semantic breakdown.

## Validated Prototype

An experimental implementation attached metadata to high-level TTIR
operations using a fused location:

```mlir
loc(fused<{"tt.profile.region" = "decoder.sdpa"}>[...])
```

The label survived lowering and fusion. When an emitted operation represented
work from multiple regions, the prototype retained an ordered, de-duplicated
set of labels rather than assuming one source operation per program.

On the Llama 3 8B decoder comparison:

- D2M produced 305 profiler rows per loop, of which 282 were tagged.
- TTNN produced 338 region rows per loop plus 13 setup/teardown rows.
- D2M summed device time was approximately 81.5 ms.
- TTNN summed device time was approximately 37 ms.
- Each total agreed with its independent, untagged profile within about 0.2%.
- The semantic breakdown showed that D2M SDPA was faster than TTNN SDPA, while
  the D2M MLP accounted for much of the full-block regression.

This established that semantic attribution is useful even when backend program
and kernel counts differ. It did not establish that the experimental
implementation was suitable for upstreaming.

## Recommended Implementation

Reuse the location machinery that already exists. Do not introduce a separate
FlatBuffer `ProfileInfo` table or a new Tracy message unless the existing
location path proves insufficient.

### Existing TTNN path

- `include/ttmlir/Target/TTNN/program.fbs` stores `Operation.loc_info`.
- `runtime/lib/ttnn/program_executor.cpp` calls
  `perf::Env::tracyLogOpLocation` before executing each operation.
- `tools/ttrt/common/perf.py` associates `MLIR_OP_LOCATION` messages with
  device operations and writes the `LOC` column.

### Existing TTMetal path

- `include/ttmlir/Target/TTMetal/command.fbs` stores `Command.loc`.
- `lib/Target/TTMetal/TTMetalToFlatbuffer.cpp` serializes the command location.
- `runtime/lib/ttmetal/executor.cpp` passes the location of each
  `EnqueueProgramCommand` to `profiler::addProgramProfileHostMetadata`.

The implementation should align TTMetal with the existing TTNN location
contract at the enqueue-program boundary. Confirm whether the TTMetal profiler
metadata already provides a reliable `LOC` association in `ttrt perf`; only
add `tracyLogOpLocation(loc)` if that association is missing. Emitting a
location for every TTMetal command would be incorrect because not every
command creates a corresponding device operation.

Extract all `tt.profile.region` entries from the resulting location and add a
derived `PROFILE_REGIONS` column during `ttrt perf` post-processing, or in a
small reusable parser consumed by it. Preserve an ordered, de-duplicated list
for fused locations containing multiple regions. Keep the original `LOC`
column unchanged.

## D2M-JIT Location Preservation

The D2M-JIT rewrite path currently converts a module to text with
`str(module)`. That form can omit debug locations, which drops semantic tags
before compilation. The experimental workaround used:

```python
module.operation.get_asm(enable_debug_info=True)
```

Upstream support should make location preservation explicit and tested. An
optional `preserve_debug_info` argument or a module-preserving API is
preferable to changing every textual rewrite result without auditing callers.
Keep this change separate from runtime and profiler export plumbing when
possible.

## Required Tests

1. Parse a single `tt.profile.region` entry from a fused location.
2. Recursively parse nested fused and call-site locations.
3. Preserve order and remove duplicate region labels.
4. Handle an emitted operation carrying multiple semantic regions.
5. Leave operations without semantic metadata untagged.
6. Verify TTMetal enqueue-program location serialization and runtime emission.
7. Preserve the existing TTNN location-to-device-operation association.
8. Export `PROFILE_REGIONS` from synthetic Tracy input while preserving `LOC`.
9. Verify D2M-JIT retains debug locations through its rewrite API.
10. Run a small capture smoke test and reconcile tagged and untagged device
    totals.

## Scope Boundaries

Do not combine the semantic profiling primitive with:

- TTMetal runtime program caching.
- Profiler runtime-ID generation changes.
- Changes to when device profiler results are read.
- Llama-specific benchmark or model lowering changes.
- Sidecar attribution used to recover labels for an old binary.

The standalone program-cache work is on branch
`vwells/ttmetal-program-cache`, commit `6c2e2e8c`. It may be needed for
representative steady-state measurements, but it is not part of the profiling
metadata contract.

## Prototype Caveats

The original semantic-region prototype was uncommitted in a temporary
worktree and is not included here. It added a typed `ProfileInfo` field, an
`MLIR_PROFILE_REGIONS` Tracy tag, and a `PROFILE_REGIONS` CSV column. That
prototype proved the measurement concept but was more invasive than the
existing-location design above.

One rebuilt tagged D2M decoder failed PCC. An identically rebuilt untagged
decoder failed in the same way, so the location tag was not the cause. For the
validated performance study, regions were applied to the previously validated
binary only after all 305 enqueue signatures and the per-loop compute and data
movement kernel hashes matched by ordinal. Treat that sidecar procedure as
experimental evidence, not as an upstream implementation or correctness test.

## Suggested PR Sequence

1. Existing-location semantic extraction and TTMetal/TTNN profiler export,
   including unit tests.
2. D2M-JIT debug-location preservation.
3. Program-cache and profiler lifecycle changes as independent runtime PRs.

The first PR should contain no model-specific region names. The metadata key
and parser define a general compiler/runtime contract; model builders or
profiling harnesses decide which operations form `decoder.sdpa`,
`decoder.mlp`, or any other semantic region.
