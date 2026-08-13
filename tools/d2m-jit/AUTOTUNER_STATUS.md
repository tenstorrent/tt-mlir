# d2m-jit autotuner — status

**Snapshot:** 2026-08-13, `main` @ `6d27eedfdf`. Work on d2m-jit is **on hold**;
this document is the pick-up-where-we-left-off record for the autotuner.

Status legend matches [TODO.md](TODO.md): 🔴 blocker · 🟡 missing surface ·
🟢 nice to have · ✅ done.

---

## 1. What it is, where it lives

| Path | Role |
| --- | --- |
| `test/d2m-jit/autotuner/autotuner.py` | The whole tuner: config model, sweep/hill-climb, profiling, guards, ranking, CLI (~1850 lines). |
| `test/d2m-jit/autotuner/test_autotuner.py` | 24 tests: 23 pure-logic + 1 on-silicon smoke test. |
| `test/d2m-jit/autotuner/example.py` | **Untracked** scratch driver (see §7). |
| `test/d2m-jit/runner.py` | `KernelBench` / `TensorSpec` / `run_bench` / stock materializers — the thing the tuner tunes. |
| `tools/perf-analyzer/perf-analyzer.py` | Profiler-CSV parser; lazily loaded by the tuner (hyphenated filename blocks a normal import). |

It sweeps execution parameters for every `KernelBench` declared in a kernel
file, runs each config on silicon in-process with the tt-metal device profiler
enabled, and ranks configs by `kernel_ns` (first `*-KERNEL` ZONE_START → last
`*-KERNEL` ZONE_END across all cores/RISCs, matching tt-metal's
`device_kernel_duration`).

Entry points: `python test/d2m-jit/autotuner/autotuner.py --kernel … --bench …`
(full CLI) or `from autotuner import autotune_kernel, AutotuneKnobs` (module
API). Both are documented in the module docstring.

---

## 2. Implemented ✅

**Knobs** (`AutotuneKnobs`) — three axes, with an `"all"` shorthand per axis:
- `grid_shapes` — graph-level `(gy, gx)` core grid.
- `block_shapes` — per-tensor tile-block `[by, bx]`.
- `mem_spaces` — per-tensor `"L1"` / `"DRAM"`.
- `joint_block_shapes` / `joint_mem_spaces` — per-tensor combos dispatched
  jointly instead of Cartesian (needed for coupled operands, e.g. matmul).
- `max_cores` / `max_block_tiles` — caps for auto-generation.
- Per-run `tensor_shapes` / `tensor_dtypes` overrides (`_override_bench_tensors`).

**Config model** — `AutotuneConfig` is *always* per-tensor (`blocks`, `mems`
have one entry per kernel input); `uniform()` broadcasts. `id` collapses uniform
axes for short stable names (`g2x2_b1x1_mL1`), dedup uses a structured key so it
can never alias distinct configs.

**Sweep narrowing** — when a knob is unset, candidates are generated from tile
dimensions reduced by GCD across all of the bench's tensors
(`valid_grid_shapes`, `valid_block_shapes`), then filtered by the caps. Two
modes: *full sweep* (nothing set) and *focused* (any knob set → unset axes stay
at bench defaults rather than exploding).

**Search strategies** — `sweep` (exhaustive Cartesian), `hill-climb`
(coordinate descent over grid → block → mem, `O(G+B+M)` per round, typically
converges in 1–2 rounds; adapts the block shape across grid changes via
`_closest_block`), `default` (single config, quick spot-check; also `--no-sweep`).

**Measurement plumbing** — `_profiling_ctx` sets `TT_METAL_PROFILER_DIR` /
`TT_METAL_DEVICE_PROFILER=1` and `config.enable_perf_trace` and restores
everything on exit; one profiler dir per session because tt-metal initialises
its profiler singleton once per process; warm-up runs happen *inside* the
profiling context (the env var must precede the first device open) with their
`.logs` wiped before the measured pass; `_silence_native_output` redirects fds
1/2 so tt-metal's JIT chatter lands in `native_logs/<config_id>.log` (deleted on
success, path appended to the error on failure).

Deliberate omission worth keeping: `insert_profiler_traces` is **not** enabled —
the inserted `DeviceZoneScopedN` scopes register 32-bit source-location hashes
process-globally and collide across many kernel compilations. Firmware
`*-KERNEL` zones suffice.

**Correctness guards** (the part that makes results trustworthy):
- `_layout_probe` + `_verify_config_applied` — wraps `Layout.__init__` and
  fails any config whose swept `block_shape` / `mem_space` never reached a
  constructed `Layout`. Without it a materializer that ignores a knob would get
  a distinct `config_id` and a timing, and the tuner would rank a configuration
  it never applied. Necessary-but-not-sufficient (the value must *appear*; it is
  not matched to the right tensor).
- PCC gate — with `check_pcc`, a config below `bench.pcc` is dropped from the
  ranking (fast-but-wrong cannot win), with the PCC retained so the summary
  shows why.
- `_rank` is the single source of truth for valid-vs-failed, so the stdout
  ranking, `summary.txt`'s table, and the `Best:` line cannot disagree.

**Artifacts** — `<output_dir>/<bench>/runs/<config_id>.json` (one per result),
`summary.txt` (best + ranked table + failed/excluded table with reasons),
optional `profiler_logs/<config_id>.csv`, `native_logs/`.

**Tests** — `test_autotuner.py` covers divisors/grid/block heuristics, config
identity & serialization, config-space generation (full-sweep, focused, joint
mem, per-tensor counts), CLI parsers, ranking, and tensor overrides; plus
`test_autotune_exp_on_device`, the one on-silicon test, which asserts
`error is None` (i.e. the knob was applied *and* PCC held) and never asserts
timing. The module is `device_only` at module scope, deliberately: under
`D2M_JIT_BACKEND=sim` its assertions would describe the simulator, a green
result that checks nothing it claims to.

CI: the autotuner runs in CI only through this test file (single-chip lanes,
`tracy` image so the profiler is present). There is no perf-tracking or
regression-gating lane.

---

## 3. Multichip support

**On `main`: no. Single device only.** Three independent reasons:

1. `AutotuneKnobs` / `AutotuneConfig` have no mesh or sharding axis.
2. `run_bench`'s materializer contract is 4-arg (`kernel, inputs, tensors,
   grid_shape`) — nowhere to pass a mesh; the stock materializer
   (`eltwise_block_run`) never calls `d2m.mesh*`.
3. `perf-analyzer` on `main` does not group profiler rows by device. Timestamps
   are "cycles since reset" of each chip's *own* clock, so a multi-chip trace
   would mix independent clocks and produce a meaningless span.

**On the unmerged branch `jgrim/d2m-jit-multi`: yes, working end-to-end on a
1x2 mesh.** Five commits (plus one `Save misc` WIP commit) on top of
`da6f1ce081`:

| Commit | What it adds |
| --- | --- |
| `6840ed5b5a` | Multi-device timing: `perf-analyzer` groups every zone/span by the "PCIe slot" column; `kernel_ns` = **max over devices** (the slowest device gates completion); `"kernel duration per device"` breakdown surfaced as `AutotuneResult.device_kernel_ns` and printed per config. |
| `8992066b9d` | Mesh in the config model: `AutotuneKnobs.mesh_shapes` / `shard_strategies`; `AutotuneConfig.mesh_shape` / `mesh_topology` / `shards` / `strategy` (graph-level vs per-tensor split mirrors `grid_shape` vs `blocks`), `id` gains `_mesh1x2_scols`, `as_dict` keys always emitted (stable JSON schema). |
| `d196f8fbcc` | `_shard_view`: grid/block candidates are re-derived from each strategy's **per-device shard shapes** — blocks valid for the full tensor routinely fail the shard's divisibility constraints. `(mesh × strategy)` is the outer axis. |
| `d7aadbb950` | `runner.MeshSpec`, `TensorSpec.shard_dims`, `shard_factors` / `shard_shape_for_spec`, and the stock `mesh_block_run` materializer (shard each input → run the same per-shard config on every device → gather); `test_mesh.py` gains the `mesh_sigmoid` `KernelBench` with declared `shard_strategies` (`cols`, `rows`). |
| `0a25e39bd5` | `_mesh_probe` + `_verify_mesh_applied` — the mesh analogue of the layout guard, and *strict* about absence (a config that requests `shards` but built zero `mesh_shard` ops is failed, not ranked); hill-climb sweeps strategy exhaustively as an outer loop and hill-climbs grid/block/mem inside each. |

Branch test coverage: `test/d2m-jit/autotuner/test_autotune_mesh.py` — one pure
config-space test plus two on-silicon mesh autotune tests (direct `Autotuner`
and via `autotune_kernel`), all `device_only` + `requires_mesh`.

Two hard constraints found while doing this, recorded in `mesh_block_run`'s
docstring and worth re-reading before resuming:
- The runtime gather path (`tensorShardToFull` → `concat_ndim` in
  `runtime/lib/ttmetal/meshshard_utils.cpp`) does **not** skip `-1` (replicate)
  entries the way the shard path does: `-1` wraps to the last tensor dim and
  fails with "dims must be unique". So map an extent-1 mesh axis to a real dim
  (factor 1) instead, and keep `tensors[0]` free of `-1`.
- Full replication is a separate `shard_type` the d2m-jit builder does not emit
  yet, so a fully-replicated baseline strategy cannot be declared.

---

## 4. Not implemented

### 🔴 / 🟡 Blocking a real tuning workflow

- **No "use the winner" path.** The tuner reports a best config; nothing feeds
  it back into a kernel/bench definition, and nothing persists it across
  sessions (no cache keyed by kernel+shape+system). Every run re-measures.
- **No feasibility model.** Configs that overflow L1 or violate a lowering
  constraint are discovered by *failing on device* (caught, recorded as
  `error`). Fine for small sweeps, expensive for large ones.
- **Grid auto-generation assumes elementwise.** It derives grids from GCDs over
  *all* tensors, which is wrong for matmul-shaped kernels where the grid maps
  only to output M×N. Documented; the workaround is explicit `grid_shapes` (and
  `joint_block_shapes` for coupled operands).
- **Hill-climb tunes uniform configs only.** `joint_*` knobs are ignored with a
  warning; per-tensor tuning requires the exhaustive sweep.

### 🟡 Missing axes

- dtype as a swept axis (only a whole-run override today).
- `block_factors`, `kernel_io_in_dram`, iterator/interchange choices, and
  anything else kernel-internal.
- Multi-output benches (`num_outs > 1`) are untested here and in the DSL.
- Mesh/sharding axes — see §3 (branch only).

### 🟢 Ergonomics / scale

- Serial by construction: one process, one device, one config at a time. No
  parallel fan-out across devices or hosts.
- No schema version on `runs/*.json`, so consumers cannot detect a format change
  (the branch adds keys; older files just lack them).
- Rank output is text/JSON only — no plots, no cross-run comparison tooling.
- Warm-up is a count (`n_warmup`), not a convergence criterion.

---

## 5. Cleanliness assessment

**Good, above the bar for a testbed.** Specific strengths worth preserving:
the module docstring documents both the metric and the CLI; every non-obvious
decision carries a *why* comment (profiler-singleton dir reuse, the
`insert_profiler_traces` hash-collision omission, fd-level silencing, dedup on a
structured key rather than the display id); the two guards
(`_verify_config_applied`, PCC gate) plus a single `_rank` mean a printed ranking
is trustworthy by construction; the pure logic is separable and tested without
silicon.

Known rough edges:
- `test/d2m-jit/autotuner/example.py` is **untracked** and unfinished: stub
  functions `a()/ab()/b()/c()`, a stale docstring path
  (`test/d2m-jit/example_autotune.py`), a commented-out `sys.path.insert`, and an
  "Example 4" comment block with no function under it. Either finish it as the
  documented module-API example and commit it, or delete it.
- On the mesh branch, `test_autotune_mesh.py` has commented-out
  `output_dir=str(tmp_path)`, `knobs=`, and `verbose=False` arguments and one
  commented-out length assertion — so those tests write artifacts into the CWD.
  Clean before any PR.
- `runner.py`'s module docstring still ends with "Not implemented yet:
  **Autotuning** — perf traces per config to rank execution parameters", which
  landed in `#9072`. Stale.
- The tuner reaches into `d2m_jit._src` internals (`_Builder.reset`,
  `tensor_layout.Layout.__init__`, `_to_mem_space`) and monkey-patches
  `Layout.__init__`. Intentional and commented, but it couples the tuner to
  private surface — a DSL refactor will break it silently-ish (the probe
  delegates via `*args/**kwargs`, so signature changes are survivable).

---

## 6. If you come back to this

Highest-value order:

1. **Land the mesh work.** `jgrim/d2m-jit-multi` is functional but needs:
   `Save misc` squashed out, the commented-out test arguments restored, and the
   `perf-analyzer` device-grouping change split out (it is independently useful
   and independently reviewable — it fixes multi-chip numbers for every
   perf-analyzer consumer, not just the tuner).
2. **Decide whether the tuner keeps its winners.** Without persistence +
   apply-back it is a measurement tool, not an autotuner. Smallest useful
   version: write `best.json` per (bench, system-desc hash, shapes) and a
   `KernelBench`-side loader.
3. **Fix grid auto-generation for non-elementwise kernels** (or make the bench
   declare which tensor dims the grid maps to), so matmul-shaped kernels stop
   needing hand-written `grid_shapes`.
4. Delete or finish `example.py`; fix the `runner.py` docstring.

Related docs: [README.md](README.md) · [TODO.md](TODO.md) ·
[SIMULATOR_STATUS.md](SIMULATOR_STATUS.md) ·
[TESTING_STATUS.md](TESTING_STATUS.md)
