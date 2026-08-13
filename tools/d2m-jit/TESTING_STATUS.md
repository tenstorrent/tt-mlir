# d2m-jit testing infrastructure — status

**Snapshot:** 2026-08-13, `main` @ `6d27eedfdf`. Work on d2m-jit is **on hold**;
this document is the pick-up-where-we-left-off record for the test harness and CI.

Status legend: 🔴 blocker · 🟡 missing surface · 🟢 nice to have · ✅ done.

---

## 1. Shape of the suite

```
test/d2m-jit/
  conftest.py            markers (device_only, machines), autouse fixtures, spec parametrization
  runner.py              declarative spec harness: PatternTest / KernelBench / e2e / discovery
  utils.py               assert_pcc, arange_tile
  test_*.py              22 pytest modules — on-device numerics + negative/error tests
  lit/                   13 Python-driven FileCheck tests (no device)
  sim/                   test_sim.py, test_backend_switch.py (no device)
  autotuner/             autotuner.py + its tests (see AUTOTUNER_STATUS.md)
  kernels/patterns/      self-contained pattern files (kernel + rewrite + declared specs)
  kernels/prefill/       rope.py (a real prefill kernel with a bench)
```

**Counts:** 186 pytest tests collected (186 pass on device; 130 pass / 56 skip
on the simulator re-run), 13 lit tests. Largest modules:
`test_reductions.py` (31), `autotuner/test_autotuner.py` (24),
`test_broadcasts.py` (16), `test_errors.py` (12), `test_matmul.py` (11).

Three test kinds, each with a different cost/coverage tradeoff:

| Kind | Where | Device? | What it locks down |
| --- | --- | --- | --- |
| lit + FileCheck | `lit/*.py` | no | IR shape / pre-pipeline contract (`REQUIRES: d2m-jit`, `parallelism_group = "d2m-jit"` so they serialize) |
| pytest on silicon | `test_*.py` | yes | numerics vs a torch golden, in-process (no ttrt subprocess, no files) |
| pytest on simulator | same files, `D2M_JIT_BACKEND=sim` | no | the same goldens, on the torch backend |

---

## 2. The declarative harness (`runner.py`) ✅

The design idea worth preserving: **a pattern file under
`test/d2m-jit/kernels/patterns/` declares its own tests as module-level data**,
so one file is the complete unit (kernel + rewrite + tests) and adding pattern
#1001 is a zero-diff change to the harness. `discover()` imports every
non-underscore-prefixed file in that directory and collects:

- `PATTERN_TESTS = [PatternTest(...)]` — rewrite correctness. The runner applies
  *only that file's* patterns via `apply_patterns_text` (which
  snapshots/clears/restores the global pattern registry, so specs stay isolated
  even with thousands of pattern files imported) and pipes the result through the
  real `FileCheck` binary. No device. Replaces hand-written `lit/*_pattern.py`.
- `KERNEL_BENCHES = {"name": KernelBench(...)}` — on-device numerics, in-process:
  drives the `@d2m.kernel` entrypoint with an explicit
  `(layout, block_shape, grid_shape)` config and PCC-compares against a torch
  golden. This is also exactly what the autotuner consumes.
- `PatternTest(..., e2e=True)` — true e2e: the rewritten module is compiled to a
  flatbuffer **held in memory** and run via the in-process tt-metal runtime; the
  device output is read straight back into a torch tensor. Disk footprint ~zero
  regardless of pattern count, one device handle reused per run. `golden` is
  *optional*: without it the runner cross-checks against the **ttnn device
  baseline** of the original (pre-pattern) TTIR, compiled `ttir → ttnn` and run
  on device, cached per `(module, inputs)`.

`conftest.py::pytest_generate_tests` parametrizes `pattern_test` /
`kernel_bench` / `e2e_spec` from that discovery, and
`test_patterns.py` is the generic three-line runner over them.

Supporting pieces: `TensorSpec` (shape/block_shape/dtype/dist/mem_space per
tensor — everything tensor-level; `grid_shape` is the only graph-level parameter,
on `KernelBench`), `layout_from_spec` (centralizes `Layout` construction so the
swept `mem_space`/`block_shape` knobs always reach a `Layout`, which the
autotuner's guard requires), `eltwise_block_run` (stock materializer with loud
divisibility assertions rather than silent under-computation), deterministic
`make_inputs`, and `parse_func_io` (the TTIR signature is the single source of
truth for e2e input shapes, so shapes are never duplicated in a spec).

One sharp edge documented in the docstring: scalar kernel args take opposite
routes on the two paths — in a rewrite scope they are baked into the kernel body
as in-region constants (so the flatbuffer has no unserialisable scalar program
args), while the in-process lazy builder makes them `index` function params so
the binary stays parameterised.

---

## 3. Markers, fixtures, and backend/machine filtering ✅

`conftest.py` implements two skip-by-marker filters. Both **skip rather than
deselect by path**, deliberately, so exclusions stay visible in the junit report
instead of silently narrowing what a lane covers.

- **`@pytest.mark.device_only(reason=…)`** — skipped when
  `D2M_JIT_BACKEND=sim`. The reason is mandatory-by-convention and must name the
  root cause, not the symptom (it drifted out of date twice before the reasons
  were audited). Four legitimate classes: intended sim divergences, error
  type/message parity, device-only machinery with no sim analog, and — the
  subtle one — tests that would pass *vacuously* under sim
  (`autotuner/test_autotuner.py` at module scope: its `error`/`pcc` assertions
  would describe the simulator, a green result that checks nothing it claims to).
  Currently used in 9 places.
- **`@pytest.mark.machines(*names)`** — CI machine types (the `RUNS_ON` value) a
  test runs on. Unmarked tests default to the single-chip lanes
  (`_DEFAULT_MACHINES = {n150, p150}`); multi-chip tests opt in.
  `_MACHINE_NUM_DEVICES = {n150: 1, p150: 1, n300: 2, llmbox: 8}`. In CI a test
  runs only on its listed machines; locally (`RUNS_ON` unset) it runs when
  `_num_devices()` — read from the resolved system descriptor — is at least the
  smallest machine listed. Unknown names and an empty marker are `UsageError`s,
  not silent passes.

Autouse fixtures: `_set_seed` (deterministic torch RNG per test) and
`_reset_builder` (drops the `_Builder` singleton after each test so a failed
negative-test compile cannot leak MLIR state). Both import lazily where needed so
the device-free simulator suite can collect with no bindings. `e2e_device` is
function-scoped so at most one device is open at a time.

---

## 4. CI ✅

`.github/settings/tests.json`:
`{"runs-on": ["n150", "p150", "n300", "llmbox"], "image": "tracy", "script": "d2m_jit.sh"}`
— the `tracy` image matters: it is the profiler-enabled build the autotuner's
smoke test needs.

`.github/test_scripts/d2m_jit.sh` does three passes, with per-pass junit reports
(`report_lit.xml`, `report.xml`, `report_sim.xml`) so they cannot clobber each
other:

1. `llvm-lit` over `$BUILD_DIR/test/d2m-jit/lit` — single-chip lanes only.
2. `pytest test/d2m-jit` — every lane. Passing the *directory* (not a `test_*.py`
   glob) is load-bearing: the glob would silently skip `sim/` and `autotuner/`.
3. `D2M_JIT_BACKEND=sim pytest test/d2m-jit` — single-chip lanes only, because
   the sim has no multi-chip support ([issue #9202](https://github.com/tenstorrent/tt-mlir/issues/9202)).

It also symlinks `$INSTALL_DIR/tt-metal` into
`third_party/tt-metal/src/tt-metal`, which the in-process runtime expects.

---

## 5. Multichip support

**Yes on `main`, but the coverage behind it is thin.** Landed in `#9201`
(machines marker + multi-chip CI lanes) on top of `#9058` (d2m-jit mesh and
global-semaphore support).

What exists:
- The `machines` marker mechanism and the local device-count fallback (§3).
- `n300` and `llmbox` in the CI matrix, with the hardware-independent lit and sim
  passes correctly skipped on them.
- `test/d2m-jit/test_mesh.py` — 4 tests at `@pytest.mark.machines("n300")`:
  mesh declaration (`ttcore.meshes` attribute), `mesh_gather` full-shape
  derivation, a `mesh_shard` round-trip on a 1x2 mesh, and a sharded sigmoid
  compute round-trip (shard → kernel → gather → PCC).
- `test/d2m-jit/lit/mesh_shard.py` and `lit/global_semaphore.py` — IR-level
  coverage, no device.

Gaps to know:
- **`llmbox` was removed from the CI matrix on 2026-08-13** (`tests.json` now
  reads `["n150","p150","n300"]`). It ran zero d2m-jit tests: every test either
  defaults to `{n150, p150}` or is marked `machines("n300")`, so `RUNS_ON=llmbox`
  skipped all 186 while booking an 8-chip machine on every PR. The marker
  mechanism still supports `llmbox` (`_MACHINE_NUM_DEVICES`) and
  `d2m_jit.sh` still guards for it, so **re-adding the lane is a one-token change
  to `tests.json`** — do that together with the first `machines("llmbox")` test
  (an 8-way shard round-trip is the obvious candidate), not before.
- 🟡 Multi-chip numerics coverage is one sigmoid round-trip on 1x2. No multi-mesh
  shapes, no matmul/reduction across a mesh, no fabric/CCL tests (those live
  unmerged on `origin/vwells/d2m-jit-ccl-kernel-api-standalone`, which adds
  `lit/ccl_all_gather.py` and `lit/ccl_ops.py`).
- 🟡 No simulator coverage on multi-chip lanes at all (§SIMULATOR_STATUS §3).
- On the unmerged `jgrim/d2m-jit-multi` branch, `test_mesh.py` also gains a
  `mesh_sigmoid` `KernelBench` + `requires_mesh` and the mesh autotuner tests —
  see [AUTOTUNER_STATUS.md](AUTOTUNER_STATUS.md) §3.

---

## 6. Not implemented

- 🟡 **No `llmbox` (8-chip) tests**, which is why that lane was dropped (above).
- 🟡 **No perf regression gating.** The autotuner runs in CI only as a
  correctness smoke test; nothing tracks `kernel_ns` over time or fails a PR on
  regression.
- 🟡 **Multi-output kernels (`num_outs > 1`) are untested** anywhere — the API
  knob exists, the harness has no bench for it.
- 🟡 **Rank > 2 is untested on the sim path** (the sim is rank-2 only), so any
  rank-4 kernel is implicitly device-only.
- 🟢 **e2e scale.** `e2e_device`'s docstring notes the intended next step: for
  large-scale CI, prefer one batch driver that opens a single device and loops
  over all specs in-process rather than one pytest case per pattern.
- 🟢 **Lit/FileCheck breadth.** `TODO.md` still wants pre-pipeline IR-shape tests
  for more DSL primitives (the builder already supports
  `print_ir_before_pipeline`); today's 13 files cover captures, error paths,
  pattern rewrites, broadcasts, reductions, spatial, matmul variants, mesh shard,
  global semaphore, DRAM kernel IO, affine `view_layout`, and rope.
- 🟢 **No separate no-device lane** — viable and deliberately not wired; see
  `SIMULATOR_SPEC.md` §11.5 for why.

---

## 7. Cleanliness assessment

**Structurally strong.** The declarative co-located spec model is the right
abstraction and has already paid for itself (it deleted a family of per-pattern
test files); the two-filter marker design keeps exclusions auditable in the junit
report; the in-process e2e path with an optional ttnn baseline removes both disk
churn and the need to hand-write goldens; the CI script's comments explain *why*
each pass is scoped the way it is.

Fixed on 2026-08-13, in the same pass that produced these docs:

1. ✅ `conftest._num_devices()` **opened a device during collection** when
   `SYSTEM_DESC_PATH` was unset — even under
   `D2M_JIT_BACKEND=sim pytest --collect-only`, the one path that is supposed to
   need no device (`_get_system_desc_path()` queries the runtime for a descriptor
   and writes `current.ttsys` into the CWD). It now short-circuits to a
   single-chip answer under the sim backend, which is also the semantically right
   result: multi-chip tests drive `mesh`/`mesh_shard`, which the switch does not
   dispatch, so they must skip under sim regardless of local hardware.
2. ✅ `runner.py`'s module docstring: corrected the pattern-file path
   (`kernels/patterns/`) and replaced "Not implemented yet: Autotuning" (landed
   in #9072) with a pointer to the autotuner and the `layout_from_spec`
   requirement its guard imposes on materializers.
3. ✅ **Duplicated PCC helpers** collapsed: the math lives in `utils.py` (the
   bindings-free module the sim suite imports from) and `runner.py` re-exports
   `assert_pcc` / `compute_pcc`, so the dependency can only point one way. The
   surviving implementation is the `.float()`-casting one, so reduced-precision
   device outputs correlate at f32 precision.
4. ✅ Doc rot in the sibling docs — see `SIMULATOR_STATUS.md` §6 for what changed
   and how it was verified.

Remaining rough edges:

5. 🟡 Test runs litter the source tree with (gitignored) artifacts:
   `test/d2m-jit/generated/{fabric,inspector,watcher}/`,
   `test/d2m-jit/current.ttsys`, `.pytest_cache/`, `__pycache__/`. Harmless but
   they make `git status`/`ls` noisy and they are written *into the test
   directory* rather than a build/tmp dir. (Item 1 stops the sim path from
   contributing `current.ttsys`; the device path still writes it, by design.)
6. 🟢 `sys.path` juggling: `autotuner.py` inserts `test/d2m-jit` into `sys.path`
   to import `runner`, and the mesh-branch test file depends on that side effect
   ("keep this import first"). Works; brittle. A `conftest`-level path fixture or
   a small package would be sturdier.

---

## 8. If you come back to this

Items 1–3 of this list (the `llmbox` lane, the collection-time device open, and
the housekeeping pass) were done on 2026-08-13; see §7. What is left:

1. If you want the `llmbox` lane back, add a `machines("llmbox")` test in the
   same change — the lane was dropped precisely because it had none.
2. If multichip is the reason you are back: read
   `origin/vwells/d2m-jit-ccl-kernel-api-standalone` (CCL kernel API + sim
   parity) and `jgrim/d2m-jit-multi` (mesh autotuning) before adding tests, then
   extend `test_mesh.py` beyond the single 1x2 sigmoid round-trip.

Related docs: [README.md](README.md) · [TODO.md](TODO.md) ·
[SIMULATOR_SPEC.md](SIMULATOR_SPEC.md) ·
[AUTOTUNER_STATUS.md](AUTOTUNER_STATUS.md) ·
[SIMULATOR_STATUS.md](SIMULATOR_STATUS.md) · [KERNEL_AUDIT.md](KERNEL_AUDIT.md)
