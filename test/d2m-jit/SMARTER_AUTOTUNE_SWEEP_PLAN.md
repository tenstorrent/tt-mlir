<!--
SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# Plan: Smarter Full-Sweep for Non-Elementwise Kernels

**Status:** proposed (not started)
**Area:** `test/d2m-jit/autotuner.py` + per-kernel benches
**Owner:** _unassigned_

## Motivation

`autotune_kernel(...)` without a `knobs` argument runs in **full-sweep mode**,
auto-generating the config space from tensor shapes. This works for elementwise
kernels but collapses to a near-empty space for matmul-like kernels.

Concretely, for the `matmul_tiled_multi_k` bench (lhs `(64,96)`, rhs `(96,64)`):

```
FULL-SWEEP configs (2): ['g1x1_b1x1_mL1', 'g1x1_b1x1_mDRAM']
```

whereas the intended sweep (via explicit `joint_*` knobs) is 32 configs
(4 grids × 2 block schemes × 4 per-tensor mem combos). The no-knobs path is
effectively useless for matmul today: a user must hand-write `grid_shapes`,
`joint_block_shapes`, and `joint_mem_spaces` to get any real coverage.

## Root Cause

This is **not** a tuning-heuristic bug — the Cartesian-product structure of
`generate_configs` is structurally wrong for matmul. Two properties break it:

1. **The grid maps to output dims only.** For matmul the execution grid maps to
   output M×N; K is contracted and never grid-sharded. The operands are
   transposed (lhs is 2×3 tiles, rhs is 3×2 tiles), so the current
   `valid_grid_shapes` — which takes a per-axis `gcd` across *all* tensors —
   yields `gcd(2,3)=1` on both axes and collapses to `(1,1)`. There is no way to
   recover M/N from raw tensor shapes without knowing which logical problem dims
   are griddable. (The autotuner docstring already warns: *"For non-elementwise
   kernels … supply explicit grid_shapes."*)

2. **Block axes are coupled.** `lhs.block[1] == rhs.block[0] == k_block` is a
   hard constraint — the materializer asserts it. A Cartesian product over
   independent per-tensor block candidates cannot express "these vary together,"
   which is exactly why the working example uses `joint_block_shapes` rather than
   `block_shapes`. Any auto-generator emitting independent per-axis candidates
   would produce mostly-invalid configs.

The knowledge needed to enumerate a valid space (tensor→grid mapping, the K
coupling) lives *inside the materializer* (`matmul_tiled_multi_k_materializer`),
not anywhere `generate_configs` can currently see.

## Recommended Approach: Bench-Supplied Space Hook

Let the bench — which already encodes the tensor→grid mapping in its
materializer — also describe its valid space. The generic autotuner delegates
when a bench provides this, and keeps the current elementwise auto-heuristic as
the default.

### Changes

1. **`KernelBench`** (`runner.py`): add one optional field.
   ```python
   gen_space: Optional[Callable] = None  # (bench, knobs) -> list[AutotuneConfig]
   ```
   Returning `AutotuneConfig`s directly (with per-tensor `blocks`/`mems`) means
   the result drops straight into the existing run/rank pipeline.

2. **`Autotuner.generate_configs`** (`autotuner.py`): branch at the top.
   ```python
   if bench.gen_space is not None:
       return bench.gen_space(bench, self.knobs)
   # else: existing _resolve_grids / _resolve_block_options / _resolve_mem_options
   ```
   No behavior change for existing (elementwise) benches.

3. **`test_matmul.py`**: implement the enumerator next to
   `matmul_tiled_multi_k_materializer`:
   - **grids**: `(gy, gx)` where `gy | M_tiles` and `gx | N_tiles`, filtered by
     `knobs.max_cores`.
   - **blocks** (per grid): `k_block` over divisors of `K_tiles`; `m_block` /
     `n_block` over per-core M/N tile counts; emit the *coupled* pair
     `lhs=[m_block, k_block]`, `rhs=[k_block, n_block]` (always valid by
     construction).
   - **mems**: per-tensor L1/DRAM combos (respecting any `knobs` override).
   - Honor `knobs.max_cores` / `knobs.max_block_tiles` so caps still apply.

4. The existing **`_verify_config_applied` guard** already validates that each
   emitted config actually reaches the device, so it composes unchanged and
   keeps the bench enumerator honest.

### Why this one

- Puts the space definition where the mapping knowledge already lives.
- Minimal new abstraction: one field, one branch, one bench function.
- Fully backward-compatible: elementwise benches are untouched (fallback path).
- The coupling constraint (k_block) is expressible because the bench emits whole
  configs, not independent per-axis candidates.

## Alternative: Declarative Problem-Dim Metadata

Instead of a per-bench function, annotate the bench with logical problem dims
(M, N, K), mark which are griddable, and give each `TensorSpec` a projection
from problem dims → its shape/blocking. The autotuner then derives grids and
coupled blocks generically.

- **Pro:** one engine handles matmul, conv, attention, etc.
- **Con:** pushes contraction/sharding concepts into the generic layer and is
  materially more machinery. Only worth it once several non-elementwise kernels
  exist. Defer until then.

## Also In Scope / Follow-ups

- **hill-climb** is currently elementwise-only (emits a warning when `joint_*`
  knobs are set). To tune matmul it would need to walk the bench-supplied space
  too — a follow-up, not required for a smarter *sweep*.
- **`AutotuneKnobs`**: continue passing `max_cores` / `max_block_tiles` into the
  bench enumerator so caps are respected.
- Consider a tiny fallback safeguard: if full-sweep (no `gen_space`, no knobs)
  produces `< N` configs for a multi-tensor bench, `log()` a hint pointing at
  `gen_space` / explicit knobs, so the degenerate case is visible rather than
  silent.

## Acceptance

- `autotune_kernel("test/d2m-jit/test_matmul.py", bench_names=["matmul_tiled_multi_k"])`
  with **no knobs** produces a broad, all-valid space (order of the 32 the
  explicit-knob path yields), every config passing the applied-guard.
- Elementwise benches (`exp`, `add_exp`, `rope`) generate the same configs as
  before (regression check).

## References

- `test/d2m-jit/autotuner.py` — `generate_configs`, `_resolve_grids`,
  `_resolve_block_options`, `_resolve_mem_options`, `valid_grid_shapes`,
  `valid_block_shapes`, `_verify_config_applied`.
- `test/d2m-jit/test_matmul.py` — `matmul_tiled_multi_k_materializer`,
  `KERNEL_BENCHES["matmul_tiled_multi_k"]`.
- `test/d2m-jit/runner.py` — `KernelBench`, `TensorSpec`.
- `test/d2m-jit/example_autotuner.py::d` — the 32-config explicit-knob example.
