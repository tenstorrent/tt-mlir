# Chisel PR 3: Multi-Program Tensor Sharing

## Description

Add cross-program/cross-binary tensor sharing via `global_tensor_pool`. After
this PR, Chisel correctly handles multi-program binaries where later programs
consume outputs from earlier ones.

Adds `global_tensor_pool` on `ChiselContext` with copy-in/copy-out at program
boundaries, and disk caching to `TensorPool`.

See [pr3_reporting_and_sharing.md](../docs/pr3_reporting_and_sharing.md) for
full design.



## Dependencies

- Chisel PR 2: Single Program Flow
