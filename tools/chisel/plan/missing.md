# Gaps from dump.md — Open Items

## Still Needed

- **Aggregate metrics**: `postProgram` should compute and log min/max/mean PCC
  across all ops in the program. Design is in dump.md but implementation
  details (where to store, how to display) need specification.

## Still Open

- **Multi-chip handling**: Per-device tensor lists, per-device comparison.
  Not addressed by hierarchical redesign. Needs separate analysis.

- **`load_cache` and `funcCall` ops**: How to handle these special ops in the
  callback flow. Open question from original dump.md, not yet answered.
