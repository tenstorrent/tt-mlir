# d2m-jit simulator — status

**Snapshot:** 2026-08-13, `main` @ `6d27eedfdf`. Work on d2m-jit is **on hold**;
this document is the pick-up-where-we-left-off record for the simulator.

[SIMULATOR_SPEC.md](SIMULATOR_SPEC.md) is the design document and is **accurate
and current** — it already carries per-section status marks, the intended
divergences from device, and the testing strategy. This document does not repeat
it; it records where the implementation actually stands, what multichip support
exists, and what is left.

Status legend: 🔴 blocker · 🟡 missing surface · 🟢 nice to have · ✅ done.

---

## 1. What it is, where it lives

A **pure-Python / torch simulator**: it runs a `@d2m.kernel` body as *real
Python*, once per simulated core, with the in-kernel names (`core_index`,
`remote_load`, `exp`, `reduce_sum`, `@`, …) bound into a rebuilt function object.
No MLIR context, no pass pipeline, no silicon. The kernel you debug in sim is
byte-identical to the one that compiles to device — there is no second source of
truth and no AST rewriting.

| Path | Role |
| --- | --- |
| `tools/d2m-jit/_src/sim/tensors.py` | `SimTensor` (host handle) and `SimBlock` (in-kernel tile block, `(bm, bn, 32, 32)` torch tensor). |
| `tools/d2m-jit/_src/sim/host.py` | Host ops: `to_layout` / `empty` / `zeros` / `full` / `tilize` / `untilize` / `arange` / `reshape` / `spatial` / views / `mesh_gather` / `to_host`. |
| `tools/d2m-jit/_src/sim/ops.py` | `SIM_OPS` / `SIM_METHODS`: torch backings for every `@syntax` in-kernel name, plus `core_index` / `remote_load` / `remote_store` / `Semaphore`. |
| `tools/d2m-jit/_src/sim/run.py` | `SimKernel`: namespace build, SPMD loop over the grid, async driving. |
| `tools/d2m-jit/sim.py` | Shadow surface (`import d2m_jit.sim as d2m`). |
| `tools/d2m-jit/_src/layout_math.py` | MLIR-free descriptor math **shared** by device and sim (`reduction_layout`, `resolve_reshape`, mesh shard math) so the two backends cannot disagree on layout transforms. |
| `test/d2m-jit/sim/test_sim.py`, `test_backend_switch.py` | 9 + 7 tests; see §5. |

Two ways in, one implementation:

```python
import d2m_jit.sim as d2m          # shadow module (forces sim)
# or
import d2m_jit as d2m
d2m.config.backend = "sim"          # per-call, runtime-toggleable; env D2M_JIT_BACKEND=sim
```

---

## 2. Implemented ✅ (v1, landed in #9120)

- **Execution model:** SPMD over `grid=(Y, X)`; `core_index(d)` from a
  thread-local; `remote_store` mutates the output `SimTensor`'s buffer in place,
  so writes from all cores accumulate exactly like a device output tensor.
  Sequential core iteration is order-independent given the well-formed-kernel
  invariant (each core derives its blocks from `core_index`).
- **Whole in-kernel op surface:** all 41 unary and 19 binary eltwise ops, the six
  comparisons (name-only, matching the device — Python `<`/`==` stay
  unoverloaded because `visit_Compare` lowers to `arith.cmpi` for index-domain
  conditions), `where`, `clamp_scalar`, `typecast`, `tile_transpose`,
  `tile_bcast` (row/col/2d + shorthands), reductions (`reduce_sum` / `reduce_max`
  / `reduce_mean`, with the reduced result broadcast back to full block shape so
  both the store→readback and implicit-eltwise-broadcast consumers are correct),
  matmul (incl. `transpose_b`) and the in-kernel `zeros([m, n])` + `c += a @ b`
  accumulator idiom.
- **Host ops:** every op the canonical surface dispatches, including `arange` /
  `reshape` / `spatial` (host roundtrips / sequential region execution, since
  physical placement is value-neutral in sim) and `reduction_layout`.
- **Views:** `view` / `permute` (logical permutation, `is_view=True`, same
  validation and same `to_host`-on-view rejection as device); `view_layout`
  restricted to paired `(grid, tile)` permutations.
- **Async:** `async def` + `await` bodies run to completion via `_drive_async`
  (every sim awaitable resolves immediately); `Semaphore.set/inc/wait` mirror the
  device signatures as no-ops.
- **Runtime-free import:** `import d2m_jit.sim` requires neither the `ttmlir`
  bindings nor `_ttmlir_runtime`, so it works with no tt-metal build. Held true
  by two deliberate pieces of laziness (PEP 562 `__getattr__` in
  `__init__.py`, lazy `_mlir()` in `tensor_layout.py`) and **asserted by a test**
  that re-runs the import in a subprocess with both modules forced unimportable.
- **Numerics:** exact torch math in the block dtype (f32 stays f32) — more
  accurate than device, which is what lets it serve as the oracle for intended
  semantics.

---

## 3. Multichip support

**Effectively no — and this is the simulator's single largest gap.** Precisely:

| Piece | Sim status |
| --- | --- |
| `d2m.mesh(...)` (mesh declaration) | 🔴 **not dispatched.** Stays on the device builder even under `backend="sim"` (it owns the `ttcore.meshes` module attribute), so it needs the MLIR bindings. Not exported by the `d2m_jit.sim` shadow module at all. |
| `d2m.mesh_shard(...)` | 🔴 **not dispatched.** Emits `d2m.mesh_shard`; device-only. |
| `d2m.mesh_gather(...)` | 🟡 **metadata only.** Has a sim backing that derives and validates the gather mapping via the shared `validate_mesh_mapping` (so `full_shape` and the error messages match device), but performs **no data movement**. |
| Multi-device data movement / per-device buffers | 🔴 not modeled. A `SimTensor` is one logical buffer; there is no notion of N devices each holding a shard. |
| Fabric / CCL ops | 🔴 not on `main` at all (see §6). |

Consequences to know before resuming:
- `test/d2m-jit/test_mesh.py`'s shard round-trips are device tests
  (`@pytest.mark.machines("n300")`), not sim-checkable.
- `.github/test_scripts/d2m_jit.sh` **skips the whole simulator re-run on the
  `n300` and `llmbox` lanes**, with a pointer to
  [issue #9202](https://github.com/tenstorrent/tt-mlir/issues/9202) ("the sim
  backend currently does not support multi-chip topologies"). So the multi-chip
  lanes get zero simulator coverage today.
- The mesh mirror in `layout_math.py` (`set_current_mesh` / `current_mesh`) is
  the seam that a real multi-device sim would build on: the device builder
  already mirrors the declared mesh into MLIR-free state precisely so the sim can
  validate mesh ops without importing the builder.

A plausible v2 shape: give `SimTensor` an optional leading device axis (or a
list of per-device buffers), make `mesh_shard` a real split and `mesh_gather` a
real concat over it, run the SPMD core loop once per device, and dispatch `mesh`
to a sim-side no-op that only records the shape. Values, not timing — the same
scope discipline as the rest of the sim.

---

## 4. Not implemented

Everything here is already tracked in [SIMULATOR_SPEC.md](SIMULATOR_SPEC.md)
§10/§12/§13; consolidated for convenience.

### 🔴 Rejected loudly (fail, don't silently no-op)
- **Declarative generic forms** — `indexing_maps` / `iterator_types` raise
  `NotImplementedError`. Asymmetry worth closing: `block_factors` and
  `kernel_io_in_dram` are *accepted and ignored* (both are placement/blocking
  hints that do not change values in the SPMD form).
- **Async-generator kernels** (`async def` with `yield`) — producer/consumer
  handoff across concurrently scheduled threads needs an ordering model the sim
  deliberately omits.
- **`view_layout` broadcast / constant remaps** — only paired `(grid, tile)`
  permutations are modeled.
- **Rank > 2 tensors** — `tile_padded_shape`, `_apply_perm`, and
  `remote_load`/`remote_store` indices are all rank-2 only. This is why the RoPE
  kernel is `device_only`: it derives its half-roll `view_layout` from the device
  physical rank-4 shape.

### 🟡 Known gaps with device counterparts
- `!tensor.store` (belongs to the declarative form) and `__matmul_acc__`
  (supplied by native Python `+=`) — the two known holes in the
  every-syntax-name-has-a-sim-backing audit.
- DMA primitives (`dma_read`, `embedding`, `indexed_row_copy`, …) — not in
  `api.py` yet either; add sim backings as they land.
- Multi-device — §3.

### 🟢 Deferred by design
- `quirks.py` device-quirk numerics (`config.sim_device_quirks`: fp19 `full`
  fills, reduced-precision tile math and accumulation).
- `config.sim_warn_divergence` (print once per intended divergence hit).
- Stale-`LazyTensor`-after-`to_host` emulation.
- A sim↔device divergence report.
- A separate no-device CI lane — viable (`pytest test/d2m-jit/sim` is green on a
  no-device runner in ~2s, and `"runs-on": "builder"` is the ready-made hook) but
  deliberately not wired: it adds no coverage, since the hardware lane already
  runs that directory twice. §11.5 of the spec explains what it would and would
  not prove.

### Explicit non-goals (do not "fix" these)
Not a performance model, not a race/deadlock detector, not bit-exact to silicon,
not a replacement for on-device tests.

---

## 5. Testing & self-audits

The interesting design decision: there is **no hand-copied sim test suite**.
`.github/test_scripts/d2m_jit.sh` re-runs the *entire* pytest directory with
`D2M_JIT_BACKEND=sim` after the device pass. Every test carries its own torch
golden, so each kernel is checked against the same reference on both backends —
which is what catches "a new device kernel uses an op the sim lacks", something a
standalone sim file structurally cannot. Tests that cannot hold under sim carry
`@pytest.mark.device_only(reason=…)`, skipped by `conftest.py` only when the sim
backend is requested, with the reason propagated into the junit report.

An earlier `test_parity.py` (assert `pcc(device, sim)`) was **removed**: given
`|device − golden| < t` and `|sim − golden| < t`, parity follows, and agreement
with an independent golden strictly dominates mutual agreement between two
implementations that could share a misconception. The reasoning is preserved in
spec §11.2, along with the one case where parity would still earn its keep.

What the two sim-specific files cover (only what the re-run cannot reach):
- `test_sim.py` — the shadow surface itself, the intended divergences (`empty` is
  zero; matmul into a raw `empty` is the correct product), sim-only rejections
  and internals, and the runtime-free-import subprocess test.
- `test_backend_switch.py` — the backend switch, plus two **mechanical audits**
  that are the reason this stays honest over time:
  - `test_every_device_syntax_name_has_a_sim_backing` walks
    `D2MCompiler._syntax` and asserts every registered in-kernel name resolves in
    the sim registries (known gaps: `!tensor.store`, `__matmul_acc__`), with a
    guard against a vacuous pass on an unpopulated registry.
  - `test_every_dispatched_host_op_is_backed_and_exported_by_sim` asserts every
    `_dispatch`-tagged host op is backed *and* listed in both `__all__`s — the
    gap that once left `reduction_layout` silently running the device impl under
    `backend="sim"`.

---

## 6. Cleanliness assessment

**The cleanest part of d2m-jit.** The spec is genuinely current (status marks
inline, divergences tabulated, removed approaches explained rather than
deleted); the "device path is untouched, sim is additive" rule is enforced by a
test rather than convention; shared descriptor math lives in one MLIR-free module
so the oracle cannot drift from the thing it is an oracle for; failures are loud
(`NotImplementedError`) instead of silently wrong.

Rough edges:
- The `block_factors` / `kernel_io_in_dram` accept-and-ignore vs
  `indexing_maps` / `iterator_types` reject asymmetry (spec §12 notes it).
- Rank-2-only shows up as four separate `NotImplementedError` sites rather than
  one validation point.

Fixed on 2026-08-13 while writing these docs: `SIMULATOR_SPEC.md` §9 listed two
divergences that no longer exist, because the device-side bugs behind them were
fixed — *matmul into `empty`* (#8891; re-verified by a device run of the exact §9
case, PCC 1.0) and *multicast on grid > 1×1* (#8892; covered on device by
`test_matmul.py::test_mcast_overwrite_grid_2x2`). Both rows are gone, §5.1/§5.4
are corrected, the `device_only` reason on the multicast test now names its
actual root cause (the sim's ignore-mcast model, §5.1) rather than a divergence,
and `test_sim.py::test_matmul_into_empty_needs_no_prefill`'s docstring no longer
claims a device bug. `KERNEL_AUDIT.md` was corrected in the same pass.

---

## 7. Related unmerged work

`origin/vwells/d2m-jit-ccl-kernel-api-standalone` (not mine, unmerged) adds
fabric CCL kernel support to the DSL *and* the matching simulator parity —
`_src/sim/ops.py` +106 lines, new `test/d2m-jit/lit/ccl_all_gather.py` /
`ccl_ops.py`, `test_mesh.py` and `test_sim.py` additions, and a
`SIMULATOR_SPEC.md` update. If multichip simulation is the reason you are back
here, read that branch first — it is the nearest existing precedent for
multi-device semantics in the sim.

Related docs: [SIMULATOR_SPEC.md](SIMULATOR_SPEC.md) (design) ·
[README.md](README.md) · [TODO.md](TODO.md) ·
[AUTOTUNER_STATUS.md](AUTOTUNER_STATUS.md) ·
[TESTING_STATUS.md](TESTING_STATUS.md)
