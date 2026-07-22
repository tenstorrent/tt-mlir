# d2m-jit simulator spec

A design for a **pure-Python / torch simulator** that runs a `@d2m.kernel`
exactly as written, on the host, with no MLIR context, no pass pipeline, and
no device. The goal is a fast oracle/debugger: author a d2m kernel, run it as
regular Python, get a `torch.Tensor` back that matches what `to_host()` would
return from silicon — at interactive speed and with normal Python
tracebacks/`print`/`pdb` inside the kernel body.

Skim-readable for humans and LLMs, same as [README.md](README.md). Status
legend matches [TODO.md](TODO.md): 🔴 blocker · 🟡 needs design · 🟢 nice to
have · ✅ in scope for v1.

**Status (implemented):** v1 is landed. `import d2m_jit.sim as d2m` runs the
full eltwise / reduction / matmul / view surface on torch with no device
(`_src/sim/`, tests in `test/d2m-jit/test_sim.py`). The `config.backend`
switch (§2) is also wired: `import d2m_jit as d2m; d2m.config.backend = "sim"`
(or `D2M_JIT_BACKEND=sim`) dispatches the canonical surface to the simulator
(tests in `test/d2m-jit/test_backend_switch.py`).

---

## 1. Why this is feasible

The DSL already cleaves into two layers, and **both** map onto torch with no
compiler in the loop:

1. **Host orchestration** — `to_layout` / `empty` / `zeros` / `full` /
   `tilize` / `untilize` / `view` / `permute` / kernel invocation / `to_host`.
   Today these eagerly append MLIR ops (`_src/builder.py`). They are pure
   bookkeeping over tensors; re-expressing them over `torch.Tensor` is direct.

2. **Kernel body** — today `D2MCompiler` (`_src/ast.py`) parses the body's
   *AST* into MLIR; the body is **never executed as Python**. But the body
   *is* valid Python — the only reason it can't be `fn()`-called directly is
   that the in-kernel names (`core_index`, `remote_load`, `remote_store`,
   `exp`, `reduce_sum`, `@`, …) aren't bound in Python scope, and the values
   they pass around (`!tensor` tile-blocks) need operator overloads.

The simulator supplies those names and a tile-block value type, then **runs
the real Python function body** once per simulated core. No AST rewriting, no
second source of truth — the kernel the user debugs in sim is byte-identical
to the one that compiles to device.

Key consequence: the simulator reuses `CompiledKernel.fn` (the original
function) and `kernel._captures`. It does **not** reuse `D2MCompiler`.

---

## 2. Opt-in / integration

Primary surface (✅ implemented) — a **shadow module** that forces the sim
backend, so an existing test changes one import and nothing else:

```python
import d2m_jit.sim as d2m      # was: import d2m_jit as d2m
# ... rest of the test is unchanged ...
out = out_d.to_host()          # returns a torch.Tensor, host-computed
```

Convenience surface (✅ implemented) — a process backend switch on the
existing config singleton, so a single `import d2m_jit as d2m` can flip between
paths. Resolved per-call, so it can be toggled at runtime:

```python
d2m.config.backend = "sim"     # or env D2M_JIT_BACKEND=sim ; default "device"
```

Each dispatchable host function (`to_layout`, `empty`, `zeros`, `full`,
`tilize`, `untilize`, `view`, `view_layout`, `permute`, `to_host`) and the
`@d2m.kernel` decorator route to the device builder or the sim impl based on
`config.backend` at the moment they are called (`api.py`). `@d2m.kernel` is
decorated at import time — before the user sets the backend — so it builds the
concrete device/sim kernel lazily on first call and caches both. The sim
package is imported lazily, so `import d2m_jit` stays cheap and the device
package keeps importing without torch.

Design rule: **the device path is untouched.** The simulator is additive
(`_src/sim/` + a thin `d2m_jit/sim.py` re-export). `import d2m_jit.sim`
must not require the `_ttmlir_runtime` extension or an MLIR `Context`, so
sim works in environments with no tt-metal build at all. (`Layout` is a pure
descriptor and is reused as-is; sim simply never calls its `build_*` MLIR
methods.)

The two surfaces share one implementation: `d2m_jit.sim.<name>` are the
sim-bound entry points; the (deferred) `backend="sim"` switch will make the
canonical `d2m_jit.<name>` dispatch to the same ones.

---

## 3. Data model

### `SimTensor` — replaces `LazyTensor`

A host handle wrapping a real torch tensor plus its `Layout`.

| Field | Meaning |
| --- | --- |
| `.layout` | the `Layout` descriptor (reused unchanged from `_src/tensor_layout.py`) |
| `.buffer` | the backing `torch.Tensor` in **tile-padded** shape — the logical shape rounded up to the tile grid (`tile_padded_shape`); logical dtype |
| `.is_view` | `True` for `view` / `view_layout` / `permute` results — same semantics/rejection rule as device |

`.to_logical()` slices `.buffer` back to `layout.logical_shape` (a clone); that
is what `to_host` returns.

The device representation (tiled, blocked grid, sharded across a physical
grid) carries **no information that changes output values** — it's a
placement/packing detail. So a `SimTensor` only ever stores the logical data,
in a single tile-padded buffer (the pad is zero-filled and value-neutral).
Tiling, blocked vs. user grid, mem_space (`l1`/`dram`), and `collapse` are all
value-neutral and ignored for numerics (kept on `.layout` for shape math and
parity). The *only* layout fields that affect values are `dtype` (casts) and
the tiled→row-major elementwise identity (a no-op on logical form). This is the
single biggest simplification the sim buys.

### `SimBlock` — the in-kernel `!tensor` tile-block value

What `remote_load` returns and eltwise/reduce/matmul ops consume. A 2-D block
of `bm × bn` tiles of 32×32.

| Field | Meaning |
| --- | --- |
| `.tiles` | torch tensor of shape `(bm, bn, 32, 32)` — tiles are a leading pair of axes so torch broadcasting handles both tile-axis and within-tile broadcast for free |
| `.tile_grid` | property → `(bm, bn)` tile counts, needed for per-tile ops (bcast, reduce, matmul) |
| `.reduced_axes` | `frozenset[int]` mirroring `_REDUCED_AXES_ATTR`; see §5.3 |

`SimBlock` overloads `__add__/__sub__/__mul__/__truediv__/__neg__/__invert__/
__matmul__` and exposes method forms (`.exp()`, `.reduce_max(0)`, …) via
`__getattr__` dispatching to the `SIM_METHODS` registry — mirroring
`TensorBlock` in `api.py` 1:1 so the same body resolves identically.

Block ↔ 2-D reshape helpers: `to_2d()` returns the logical block `(bm*32,
bn*32)`; `from_2d(t)` reshapes `(bm*32, bn*32) → (bm, 32, bn, 32)` permuted to
`(bm, bn, 32, 32)`. Per-tile ops operate on the trailing `32×32`.

---

## 4. Kernel execution model

### SPMD over the grid

A `@d2m.kernel` call runs once **per simulated core** of `grid=(Y, X)`:

```
for y in range(Y):
    for x in range(X):
        _current_core = (y, x)
        body(*tensor_args, *scalar_args)   # the user's real Python fn
```

`core_index(d)` reads `_current_core[d]` from a thread-local. Stores from
different cores land in disjoint blocks (the well-formed-kernel invariant —
each core derives its block range from `core_index`), so sequential iteration
is order-independent. `remote_store` mutates the **output `SimTensor`'s
`.buffer` in place**, so writes from all cores accumulate into the same buffer,
exactly like the device output tensor.

### Running the body as Python

The body is executed by binding the in-kernel names into a fresh globals dict
and rebuilding the function object against it (preserving the original closure
so int captures still resolve):

```python
sim_globals = {**fn.__globals__, **SIM_OPS}     # SIM_OPS: core_index, remote_load, ...
runnable = types.FunctionType(fn.__code__, sim_globals, fn.__name__,
                              fn.__defaults__, fn.__closure__)
```

Native Python then drives everything `D2MCompiler` otherwise hand-lowers:

| DSL construct | Device (D2MCompiler) | Simulator |
| --- | --- | --- |
| `for i in range(...)` | `scf.for` | native `for` |
| `if/else` | `scf.if` | native `if` |
| `a + b`, `a @ b` | `!tensor.__add__` → `tile_*` | `SimBlock.__add__` → torch |
| `x.exp()` | `!tensor.exp` | `SimBlock.exp` → torch |
| int math on indices | `arith.*` on `index` | native int |
| closures over ints | `_captures` → `arith.constant` | real closure |

This is the crux of "runs as regular Python": control flow, indexing
arithmetic, and helper calls are just executed.

### Arg conventions (parity with device)

Same rules as `_emit_kernel_generic`: all `SimTensor` args precede all `int`
scalars; the last `num_outs` tensor args are outputs (mutated in place); extra
ints become plain Python ints in the body. `grid=(Y,X)` is required. Same
`TypeError`/`ValueError` messages where practical (see §8).

---

## 5. In-kernel op semantics (the torch backing)

All ops below are the sim implementations of the `@syntax`-registered names in
`api.py`. Everything computes in the block's torch dtype (see §6 for
fidelity).

### 5.1 Movement & indexing ✅

- `core_index(d)` → `int` from thread-local current core.
- `remote_load(src, [i, j], mcast_*=None)` → `SimBlock` for block `(i,j)` of
  `src`: slice `src.buffer[i*em:(i+1)*em, j*en:(j+1)*en]` where the per-axis
  block extent `(em,en)=block_extent(src.layout)` is `block_shape*32` (tiled)
  or `block_shape` (non-tiled), then `SimBlock.from_2d(slice)`. **Multicast
  args (`mcast_start_index`/`mcast_shape`/`mcast_dims`) are accepted and
  ignored** — in sim every core reads from the shared global buffer, so the
  result is identical. (This means the sim *runs* multicast kernels that
  currently hit the device `SplitUnifiedThread` assertion, [TODO §2] — a
  feature, flagged in output as "device-divergent: multicast".)
- `remote_store(dst, [i, j], block)` → writes `block.to_2d()` into the same
  slice of `dst.buffer` (shape-checked against the block extent). Overwrite
  (not accumulate); store index is global (no core-relative resolution in this
  version).

### 5.2 Elementwise ✅

Unary (all 41 in the README table) and binary (all 13) map to the obvious
torch op over `.tiles`. `where(c,t,f)` → `torch.where(c.tiles != 0, t.tiles,
f.tiles)`. `clamp_scalar(x,lo,hi)` → `x.tiles.clamp(lo,hi)`.
`typecast(x,dtype)` → `x.tiles.to(torch_dtype)`. `tile_transpose(x)` →
`x.tiles.transpose(2,3)`, i.e. transpose the trailing `32×32` of every tile
(distinct from logical `permute`).

Broadcast helpers operate **per tile**:
- `tile_bcast(x,"row")` → each tile's row 0 expanded down all 32 rows.
- `tile_bcast(x,"col")` → each tile's col 0 expanded across all 32 cols.
- `tile_bcast(x,"2d")` → element (0,0) expanded over the whole tile.

Broadcast-compatible binary shapes (a `(bm,1)`-tile operand against `(bm,bn)`)
follow `_broadcast_block_shape`: torch broadcasting covers it.

### 5.3 Reductions ✅ (the one subtlety worth stating precisely)

`reduce_sum/max/mean(x, dim)` with torch/numpy `dim` numbering:

| `dim` | reduces | device `reduce_dim` | logical effect |
| --- | --- | --- | --- |
| `0` / `-2` | rows (axis 0) | `C` | collapse rows |
| `1` / `-1` | cols (axis 1) | `R` | collapse cols |

Semantics that matter for matching `to_host`:

1. **Reduction spans the whole block** along that axis — all tiles in the
   reduced tile-dimension *and* within each tile. Cross-*core* reduction is
   not modeled (matches device; needs a gather op, [TODO]). So
   `reduce_sum(x,1)` = sum over this block's columns only.
2. **Result is broadcast back to full block shape**, and `reduced_axes` is
   recorded. The device keeps the reduced value in row/col 0 of the result
   tile and relies on the output *layout* (`reduction_layout`, which sets the
   reduced logical dim to 1) to select it on readback. Broadcasting the result
   to full size makes both consumers correct in sim:
   - **store → readback**: storing the full tile into a `reduction_layout`
     output and reading the logical `(rows,1)` / `(1,cols)` slice picks the
     reduced value — correct.
   - **implicit eltwise broadcast** (`x - reduce_mean(x,1)`): the reduced
     operand is already full-size, so plain torch eltwise is correct — no
     special `reduced_axes` handling needed at the eltwise site (the field is
     kept only for parity/diagnostics and the `tile_bcast` interaction).
3. `mean` divides by `32 * tile_count_along_axis` (matches the `1/(32*N)`
   scaler in `_reduce_block`).

Concretely, on the 4-D `.tiles` `(bm,bn,32,32)` the reduce spans both the
tile-axis and the within-tile axis: axis 1 (cols) reduces dims `(1,3)`, axis 0
(rows) reduces dims `(0,2)`, each `keepdim=True`, and the reduced tile-axis is
then `.expand`-ed back to full size (so tile-axis 1 → count 1, within-tile → 32)
so both consumers in point 2 see the right number. `reduce_sum(x,1)` uses
`x.tiles.sum(dim=(1,3), keepdim=True)`, `reduce_max` uses `torch.amax`, `mean`
uses `.mean`; results carry `reduced_axes={1}` / `{0}`.

This passes the existing reduction tests (atol 0.05–0.15) trivially in f32,
since sim is *more* accurate than the device tile path.

### 5.4 Matmul ✅

`matmul(lhs, rhs, transpose_b=False)` — block matmul over tiles, computed on
the logical 2-D form: `lhs.to_2d() @ rhs.to_2d()` (`(M*32,K*32) @ (K*32,N*32) →
(M*32,N*32)`); `transpose_b=True` transposes the rhs 2-D form (rhs stored
`(N,K)`). Result is re-tiled via `SimBlock.from_2d`.

The sim computes the **correct** product. It deliberately does **not**
reproduce the device's undefined-accumulator bug ([TODO §1]) — `d2m.empty`
outputs are zero in sim, so `matmul_kernel` is correct whether the caller
pre-fills with `zeros` or not. This divergence is intended (sim = oracle for
the *intended* semantics); it is noted in §9.

### 5.5 Async / semaphores ✅ (DMA 🟡)

- `async def` + `await` ✅: an `async def` body returns a coroutine; the SPMD
  driver (`run.py` `_drive_async`) runs it to completion via `.send(None)`.
  Every sim awaitable — `SimBlock`, `SimTensor`, `Semaphore` — implements
  `__await__` as `yield from (); return self`, so it resolves immediately and
  never suspends the coroutine (device ops are synchronous in the functional
  sim). Without this drive, an un-awaited coroutine would silently no-op.
- `async def` + `yield` (async-generator) 🔴 **rejected**: a `yield`-based
  body models a producer/consumer split across concurrently-scheduled threads,
  which needs an ordering model the sim deliberately omits. `_drive_async`
  detects the async-generator and raises `NotImplementedError` (fail loud, not
  silent no-op); use `await` without `yield`, or run on device.
- `Semaphore(value).set/inc/wait` ✅ (`ops.py`, injected via `SIM_OPS`):
  mirrors the device DSL signatures (`set(value, core=None, mcast=None)`,
  `inc(...)`, `wait(value, reset=None)`). Modeled as a single integer counter;
  under sequential execution the awaited condition always already holds, so
  `set`/`inc`/`wait` are no-ops (`wait` honors an explicit `reset`).
  Ordering-only: they do not affect numerics, and the sim will **not** catch a
  real deadlock/race (see §9 / §13).
- Low-level DMA primitives (`dma_read`, `embedding`, …) 🟡 are not in `api.py`
  yet; add sim backings as they land (mirror `remote_load`/`remote_store`).

---

## 6. Numeric fidelity

Default mode: **exact torch math in the block's dtype** (f32 stays f32). This
is more accurate than device and passes the PCC/atol thresholds the existing
tests use. It is the right default for "is my kernel algebraically correct?".

Optional `config.sim_device_quirks = True` (off by default) narrows the gap to
silicon for fidelity studies:

| Quirk | Device behavior | Sim model |
| --- | --- | --- |
| `full`/`zeros` f32 fill via SFPU vFloat (fp19) | low 13 mantissa bits truncated | round fills to fp19 |
| bf16/fp16 tiles | compute in reduced precision | cast operands to tile dtype around each op |
| reduction/matmul accumulation | bf16-ish accumulate | accumulate in tile dtype |

Quirk mode is best-effort, not bit-exact (no SFPU LUT modeling). v1 ships the
exact mode; quirks are a 🟢 follow-up.

---

## 7. Host ops in sim

| Symbol | Sim behavior |
| --- | --- |
| `to_layout(torch, L)` | allocate a tile-padded `.buffer` (`_alloc`), copy the logical region in cast to `L.dtype` (shape-checked vs `L.logical_shape`, same assert as device) |
| `to_layout(SimTensor, L)` | `to_logical()` the source, re-wrap into a fresh tile-padded buffer under the new layout; casts dtype if changed; clears `is_view` |
| `empty(L)` | tile-padded `torch.zeros` buffer — **zero**, not garbage, so sim is deterministic (documented divergence; see §9) |
| `zeros(L)` / `full(L,v)` | tile-padded `torch.zeros` / `torch.full` |
| `tilize/untilize(lt, dtype=None)` | `to_layout` onto `layout.replace(tiled=…)`; value-identity, optional dtype cast |
| `view(lt, fn)` / `permute(lt, *d)` | logical permutation of `.buffer`; `is_view=True`; same arity/true-permutation validation as device |
| `view_layout(lt, fn)` | **paired `(grid, tile)` permutations only** — the `2*n`-arg lambda's head permutes and the tail must mirror it (`pos == head[i]+n`); broadcast/const (literal `0`) remaps raise `NotImplementedError` (not modeled yet); `is_view=True` |
| `to_host(*lts)` | reject `is_view` args (same message as device); return `tuple` of `to_logical()` slices (logical shape + dtype). No module/pipeline/reset needed |
| `reduction_layout(L, dim, ...)` | reused unchanged (pure descriptor math) |

`view`/`permute` validation (rank, true-permutation, torch-tensor rejection)
and the `to_host`-on-view rejection are replicated; the view/permute test cases
in `test_sim.py` (`test_permute_is_view_and_materialise`,
`test_view_identity_round_trip`, `test_to_host_on_view_raises`, …) exercise them.

---

## 8. Error & parity behavior

The sim should fail the *same way* on the same mistakes so it's a faithful
front-end:

- Reuse the arg-splitting / `num_outs` / "tensors before scalars" checks and
  messages from `_emit_kernel_generic`.
- Reuse `to_layout` shape asserts, `permute`/`view` validation, the
  view-`to_host` rejection.
- Unknown in-kernel names: in sim they raise a normal Python `NameError` from
  the real interpreter (with a real traceback into the kernel body) instead of
  `D2mJitError` with a "did you mean" hint. This is *better* for debugging;
  the divergence in error *type* is acceptable and documented.
- Out-of-bounds block indices: `remote_load`/`remote_store` bounds-check
  against the blocked grid and raise `IndexError` (device relies on the
  verifier/lowering).

Staleness (`LazyTensor` reuse after `to_host`) is a device-builder artifact;
sim has no builder reset, so reuse just works. The sim may optionally emulate
the stale-after-`to_host` error for strict parity (🟢), but the default is to
allow reuse.

---

## 9. Intended divergences from device (document, don't hide)

The sim is an oracle for *intended* semantics, so it deliberately differs in a
few places. Each should be discoverable (a `config.sim_warn_divergence` flag
that prints once per occurrence):

| Area | Device | Sim |
| --- | --- | --- |
| `empty` contents | undefined | zero |
| matmul into `empty` | garbage (accumulator bug, [TODO §1]) | correct product |
| multicast on grid > 1×1 | `SplitUnifiedThread` assert ([TODO §2]) | runs correctly |
| f32 `full` precision | fp19-truncated | exact (unless quirk mode) |
| synchronization | real semaphores/threads | serialized, no-op waits |
| reduced-precision tiles | bf16/fp16 math | f32 (unless quirk mode) |

These make the sim a clean **golden reference**: a device test can assert
`pcc(device_out, sim_out)` to catch lowering regressions, and the sim result
is the algebraically-correct target.

---

## 10. Module layout

As shipped (✅):

```
tools/d2m-jit/
  sim.py                     # shadow surface: re-exports Layout/dtypes/config + the sim host+kernel API
  _src/sim/
    __init__.py              # public sim surface: kernel, to_layout, empty, zeros, full,
                             #   tilize, untilize, view, view_layout, permute, to_host,
                             #   reduction_layout, SimTensor, SimBlock
    tensors.py               # SimTensor, SimBlock (+ block<->2d reshape, dtype helpers, __await__)
    host.py                  # host-op implementations (§7) + reduction_layout
    run.py                   # SimKernel: namespace build, SPMD loop, _current_core thread-local,
                             #   _drive_async (drive async-def bodies, reject async-generators)
    ops.py                   # SIM_OPS / SIM_METHODS: torch backings for every @syntax name (§5),
                             #   core_index / remote_load / remote_store / Semaphore
```
The backend switch lives in `api.py`: `config.backend` (new field in
`_src/config.py`, env `D2M_JIT_BACKEND`) selects per call; the device path in
`_src/builder.py` is otherwise untouched. Tests: `test/d2m-jit/test_sim.py`
(shadow) and `test/d2m-jit/test_backend_switch.py` (switch), both pure pytest
with no device / no SYSTEM_DESC_PATH.

Deferred (🟡/🟢), out of v1:
- `quirks.py` — device-quirk numerics (§6).
- `config` fields `sim_device_quirks` / `sim_warn_divergence`.

---

## 11. Testing strategy

1. **Dedicated shadow suite (✅ implemented).** `test/d2m-jit/test_sim.py`
   uses `import d2m_jit.sim as d2m` and re-covers the surface in one
   device-free file: eltwise (add, fused exp+add), softmax, reductions
   (sum-cols, max-rows), implicit-broadcast centering, matmul (per-shard +
   `transpose_b`), where/clamp/tile_bcast, zeros/full/empty, bf16 eltwise,
   views (permute/view/view_layout identity + round-trip, `to_host`-on-view
   and non-permutation rejections), and arg-validation (scalar-before-tensor,
   declarative-form rejection). It reuses the golden torch computations and
   `utils.assert_pcc`, and runs with **no device** and no `SYSTEM_DESC_PATH`.
   (This is a stand-alone file rather than a reparametrization of the existing
   `test_*.py` suite — the sim redefines its own kernels against the shadow
   import.)
2. **Sim-vs-device parity (✅ implemented).** `test/d2m-jit/test_parity.py`
   runs each kernel on both backends through the `config.backend` switch and
   asserts `assert_pcc(sim, device)` (`utils.assert_parity` reseeds torch so
   both runs see identical inputs). Tagged with the `parity` marker (registered
   in `conftest.py`) and skipped via `utils.device_runtime_available()` when no
   device is present. Run with `pytest -m parity`; exclude with
   `pytest -m 'not parity'`. Doubles as a lowering-regression net; only covers
   kernels where device and sim are expected to agree (the §9 divergences are
   excluded by construction).
3. **No-runtime CI lane.** Because sim imports without `_ttmlir_runtime`, add a
   CI job that runs the sim suite on a plain Python+torch image — fast signal
   with no silicon.
4. **Kernel-author UX test.** Confirm `print(...)` / `breakpoint()` inside a
   kernel body work under sim (they can't on device), since that's a headline
   benefit.

---

## 12. Phasing

- **v1 (✅ done):** SPMD `core_index` execution model; `SimTensor`/`SimBlock`;
  all eltwise (unary/binary/where/clamp/typecast/tile_transpose/bcast);
  reductions; matmul; views/permute/tilize/untilize; host ops; `to_host`;
  `async def` + `await` bodies and no-op `Semaphore` (§5.5); shadow module
  **and** the `config.backend` switch; exact numerics; tests in
  `test/d2m-jit/test_sim.py` and `test/d2m-jit/test_backend_switch.py`.
- **v2 (🟡):** declarative generic forms (`indexing_maps` / `iterator_types` /
  `block_factors` with `iter_index`/`block_index`/`block_offset`); async-
  generator (`yield`) producer/consumer scheduling beyond pure serialization
  (currently rejected, §5.5); DMA primitives as they land in `api.py`.
- **v3 (🟢):** device-quirk numerics (fp19 fills, reduced-precision accumulate);
  optional staleness emulation; a sim↔device divergence report.

---

## 13. Non-goals

- **Not a performance model.** No cycle/bandwidth/L1-pressure estimates — it
  models *values*, not timing. (Perf belongs to the device profiler hooks in
  `config.insert_profiler_traces`.)
- **Not a race/deadlock detector.** Sequential execution hides real
  synchronization bugs; see §5.5 / §9.
- **Not bit-exact to silicon** in the default mode (and only best-effort in
  quirk mode).
- **Not a replacement for on-device tests** — it's the fast inner loop and the
  golden oracle that feeds them.
