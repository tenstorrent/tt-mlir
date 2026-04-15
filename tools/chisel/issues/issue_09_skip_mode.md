# Chisel PR 4: Skip Mode

## Description

Add skip mode for designated ops: `preOp` stashes device input tensors before
the device executes, then `postOp` replaces the device output with
golden-computed results. This allows selectively "skipping" ops on device and
substituting CPU-correct values, isolating which op introduces numerical
divergence.

The stash pattern is critical because the device op may overwrite input buffers
in-place — the golden op in postOp needs the original pre-execution values.

See [pr4_skip_mode.md](../docs/pr4_skip_mode.md) for full design.



## Dependencies

- Chisel PR 3: Reporting + Cross-Program Sharing
