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
(`_src/sim/`, tests in `test/d2m-jit/sim/test_sim.py`). The `config.backend`
switch (§2) is also wired: `import d2m_jit as d2m; d2m.config.backend = "sim"`
(or `D2M_JIT_BACKEND=sim`) dispatches the canonical surface to the simulator
(tests in `test/d2m-jit/sim/test_backend_switch.py`). Multi-device (mesh)
support is designed in §14 and tracked in
[#9202](https://github.com/tenstorrent/tt-mlir/issues/9202); not implemented
yet.

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
must not *require* the `ttmlir` bindings, the `_ttmlir_runtime` extension, or an
MLIR `Context`, so sim works in environments with no tt-metal build at all.
(`Layout` is a pure descriptor and is reused as-is; sim simply never calls its
`build_*` MLIR methods.)

"Not required" is the precise claim, and it is weaker than "never imported":
where the bindings *are* installed, `import d2m_jit.sim` still loads `ttmlir`,
because the dtype constants below resolve to real `ttcore` enum members so that
one `Layout` works under either backend. It never loads `_ttmlir_runtime` or
`d2m_jit.api`.

Holding that rule true takes two deliberate pieces of laziness, because
importing a submodule runs every module on the path to it:

1. `d2m_jit/__init__.py` resolves the `d2m_jit.api` device surface through a
   PEP 562 module `__getattr__` instead of star-importing it eagerly —
   otherwise `import d2m_jit.sim` would pull in the parent's `api` import (and
   with it the bindings and the runtime extension) before reaching `sim`.
2. `_src/tensor_layout.py` imports the MLIR bindings lazily (`_mlir()`), since
   every sim module needs `Layout` from it. When the bindings are present,
   `dtype` / `mem_space` resolve to the same `ttcore` enum members as before,
   so the device path is bit-identical; when they are absent they resolve to
   the pure-Python `_DataType` / `_MemorySpace` mirrors (same member names,
   values, and `str()` spellings), which only the sim ever sees — it keys off
   `.name`.

This is easy to regress by adding one module-scope import, so it is asserted by
a test rather than left to convention (§11).

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
| `.buffer` | the backing `torch.Tensor` in **tile-padded** shape — the logical shape rounded up to the tile grid (`tile_padded_shape`); logical dtype. Under a mesh this becomes a per-device resolving property over `.buffers` (§14.1) |
| `.is_view` | `True` for `view` / `view_layout` / `permute` results — same semantics/rejection rule as device |
| `.mesh` | `MeshShard` metadata (full shape / `shard_dims` / `shard_shape`) when the tensor is marked for a mesh gather; `None` otherwise (§14) |

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

Unary (all 41 in the README table) and binary (all 19) map to the obvious
torch op over `.tiles`. The six comparisons in that binary list (`eq` / `ne` /
`gt` / `ge` / `lt` / `le`) go through `_compare`, which writes 1/0 in the
*operands'* tile dtype to match `d2m.tile_eq` & co. (they take the lhs tile
type); like the device they are name-only — Python's `<` / `==` dunders stay
unoverloaded because `visit_Compare` lowers compares to `arith.cmpi` for
index-domain conditions. `where(c,t,f)` → `torch.where(c.tiles != 0, t.tiles,
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

The explicit-K-loop idiom the README prescribes — a kernel-body
`zeros([m, n])` accumulator plus `c += a @ b` — works in sim too. In-kernel
`zeros(shape)` returns an f32 zero block of `shape` *tiles* (the device op's
tile type is always f32 regardless of operand layouts), and native Python `+=`
on a `SimBlock` computes what the device lowers through the hidden
`__matmul_acc__` helper, so that helper needs no separate sim backing. This is
distinct from host-side `d2m.zeros(layout)`, which allocates a whole tensor
(§7).

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
| `reduction_layout(L, dim, ...)` | same pure descriptor math as device (duplicated, MLIR-free); dispatched so the sim copy is live under `backend="sim"` |
| `arange(L, start, step)` | host `torch.arange` over `L.logical_shape` + `to_layout` (mirrors the device host roundtrip) |
| `reshape(lt, *shape)` | host roundtrip (`to_host` → `torch.reshape` → `to_layout`); shape resolution (`-1` inference, numel/error messages) mirrors `builder.reshape` |
| `spatial(inputs, outputs, grid_ranges, region_builders)` | runs each region builder in sequence — kernels mutate their outputs in place, and physical placement (`grid_ranges`) is value-neutral (§3), so no core-range modeling is needed |

`view`/`permute` validation (rank, true-permutation, torch-tensor rejection)
and the `to_host`-on-view rejection are replicated; `test/d2m-jit/test_views.py`
exercises them on both backends, since the whole-suite sim re-run (§11.4) covers
that file unchanged.

The mesh host ops (`mesh`, `mesh_shard`, `mesh_gather`) are specified in §14.
`mesh` and `mesh_shard` are implemented (§14.2/§14.3); `mesh_gather` is still
metadata-only (§14.5).

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
few places. Making each one discoverable at runtime — a `config.sim_warn_divergence`
flag that prints once per occurrence — is a 🟢 follow-up (§10); v1 documents them
here only:

| Area | Device | Sim |
| --- | --- | --- |
| `empty` contents | undefined | zero |
| matmul into `empty` | garbage (accumulator bug, [TODO §1]) | correct product |
| multicast on grid > 1×1 | `SplitUnifiedThread` assert ([TODO §2]) | runs correctly |
| f32 `full` precision | fp19-truncated | exact (unless quirk mode) |
| synchronization | real semaphores/threads | serialized, no-op waits |
| reduced-precision tiles | bf16/fp16 math | f32 (unless quirk mode) |
| mesh gather over a replicated (`-1`) mesh axis | `TT_FATAL "dims must be unique"` (runtime bug: `concat` does not skip `-1` axes the way `shard` does) | correct gather — replicated axes contribute one copy (§14.5) |

These make the sim result the algebraically-correct target, which is what lets
the whole device suite re-run against it (§11.4). Note the sim is an oracle for
*intended semantics*, not a substitute for a reference: a test should assert
against its own torch golden on both backends rather than asserting
`pcc(device_out, sim_out)`. Two implementations can share a misconception and
agree with each other while both disagree with torch — see §11.2 for why the
device-vs-sim parity suite was removed in favor of the goldens.

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
`_src/builder.py` is otherwise untouched. Tests live under `test/d2m-jit/sim/`: `test_sim.py`
(shadow) and `test_backend_switch.py` (switch), both pure pytest
with no device / no SYSTEM_DESC_PATH.

Two existing files were made lazy so the sim import path stays binding-free
(§2) — `__init__.py` (PEP 562 `__getattr__` over `api`) and
`_src/tensor_layout.py` (`_mlir()`) — plus `test/d2m-jit/conftest.py`, whose
`_Builder` import and `runner` import (both of which reach the bindings) now
happen inside the fixture / only when a parametrized fixture is requested.

Deferred (🟡/🟢), out of v1:
- `quirks.py` — device-quirk numerics (§6).
- `config` fields `sim_device_quirks` / `sim_warn_divergence`.

---

## 11. Testing strategy

1. **Sim-specific suite (✅ implemented).** `test/d2m-jit/sim/test_sim.py` uses
   `import d2m_jit.sim as d2m`, runs with **no device** and no
   `SYSTEM_DESC_PATH`, and deliberately covers *only* what the whole-suite
   re-run (item 4) cannot reach:
   - the shadow surface itself — the re-run goes through `import d2m_jit` plus
     `config.backend`, so nothing else imports `tools/d2m-jit/sim.py`;
   - the §9 divergences, which are assertions no shared test could make:
     `empty` is zero, and matmul into a raw `empty` is the correct product;
   - simulator-only rejections and internals: `async def` + `await` with no-op
     `Semaphore`, the async-generator rejection, the declarative-form
     rejection, the sim arg-order `TypeError`, the in-kernel `zeros` block;
   - the runtime-free import property (item 3).

   It used to mirror the whole op surface (eltwise, reductions, matmul,
   comparisons, views, broadcasts) against hand-copied kernels. Item 4 covers
   all of that against the kernels people actually write, so the duplicates were
   removed — a stand-alone sim file can only ever test what someone remembered
   to copy, which is exactly how the comparisons and in-kernel `zeros` came to
   be missing in the first place.

   The mechanical counterpart lives alongside it in `test_backend_switch.py`:
   `test_every_device_syntax_name_has_a_sim_backing` walks
   `D2MCompiler._syntax` and asserts every registered in-kernel name resolves in
   the sim registries (`!d2m.semaphore.*` against the `Semaphore` class,
   operator forms against `SimBlock`, the rest against `SIM_OPS`/`SIM_METHODS`),
   with `!tensor.store` and `__matmul_acc__` as the known v2 gaps. It lives
   there rather than in `test_sim.py` because reading the device registry needs
   the bindings.
2. **Sim-vs-device parity (removed — subsumed by the goldens).** There was a
   `test_parity.py` that ran each kernel on both backends and asserted
   `assert_pcc(sim, device)`. It was deleted, along with `utils.assert_parity` /
   `_run_on_backend` / `device_runtime_available` and the `parity` marker, because
   every test in the directory carries its own torch golden and item 4 runs those
   same tests on both backends. Given `|device − golden| < t` and
   `|sim − golden| < t`, parity follows by the triangle inequality — and parity's
   threshold (PCC 0.99) was never tighter than the goldens it backstopped
   (`diff < 0.05` for reductions and unary ops, `diff < 0.01` for the where/ge
   case). So it could not fail unless a golden check already had.

   It was also the *weaker* oracle: two implementations can share a misconception
   and agree with each other while both disagree with torch. Agreement with an
   independent golden strictly dominates mutual agreement, so §9's old suggestion
   to "assert `pcc(device_out, sim_out)` to catch lowering regressions" was worse
   advice than just checking each backend against torch.

   Where parity would still earn its keep: a configuration whose torch golden is
   genuinely awkward to express (multi-block sharded reductions are the plausible
   case) but where device-vs-sim is easy to state. None of the removed cases were
   that. Recover the helper from git history if such a case turns up.
3. **Runtime-free import, asserted (✅ implemented).**
   `test_sim_imports_and_runs_without_mlir_bindings` re-runs the import and a
   one-block kernel in a subprocess with `ttmlir` / `_ttmlir_runtime` forced
   unimportable, so the §2 design rule fails a test rather than silently
   eroding the first time someone adds a module-scope import on the sim path.
4. **Whole-suite sim re-run (✅ implemented).** `d2m_jit.sh` re-runs the entire
   pytest directory with `D2M_JIT_BACKEND=sim` after the device pass, so every
   device kernel is checked against the same torch golden the device run used,
   without anyone hand-copying it into the sim suite. This is what catches "new
   device kernel uses an op the sim lacks", which a stand-alone sim file
   structurally cannot. It writes its own `_sim.xml` report so it does not clobber
   the device run's. Together with the device pass this replaces the old parity
   suite (item 2).

   Tests that cannot hold on the simulator carry
   `@pytest.mark.device_only(reason=...)`, which `conftest.py` skips only when
   `D2M_JIT_BACKEND=sim`, propagating the `reason` into the skip message.
   Marking rather than filtering paths keeps the exclusions visible in the junit
   report; carrying the reason on the marker rather than in a nearby comment
   keeps it visible in that report too — and stops it drifting out of date, which
   it did twice before the reasons were audited against actual root causes.

   The reason should name the root cause, not the symptom. Four classes appear:
   intended divergences (§9 — multicast); error type/message parity (§8 —
   `test_errors.py`, the reduce-dim rejections, staleness); device-only machinery
   with no sim analog (the pass-pipeline debug knobs, `runner`-driven rewrite and
   e2e tests, and the RoPE kernel — which derives its half-roll `view_layout`
   from the device physical rank-4 shape via `LazyTensor.value`, something §3
   deliberately does not model); and tests that would pass *vacuously* under sim
   (`autotuner/test_autotuner.py`, marked `device_only` at module scope, asserts
   an on-silicon contract — dispatching it to the simulator would make its
   `error` / `pcc` assertions describe the simulator instead, a green result that
   checks nothing it claims to).

   Every host op — `arange` / `reshape` / `spatial` included — is now dispatched
   to the simulator (§7), so there is no longer a "host API the switch does not
   dispatch" skip class.

   That last class is worth calling out: `device_only` is not only for tests that
   *fail* under sim. A test whose assertions stop meaning anything is worse than
   one that errors, because nothing draws attention to it.
5. **Separate no-device lane (🟢 viable, deliberately not wired).** Because item
   3 holds, `pytest test/d2m-jit/sim` runs green on a no-device runner in
   ~2s — verified locally, and `"runs-on": "builder"` in
   `.github/settings/tests.json` is the ready-made hook (the matrix generator
   turns it into `no-device: true`, dropping `--device /dev/tenstorrent` and
   skipping the system-descriptor and lit steps). It is not wired because it adds
   **no coverage**: `test/d2m-jit/sim/` is already run twice by the hardware lane
   (the device pass, then the sim re-run), so a separate job only buys latency and
   turns one sim bug into two red jobs. The one signal it would add that the
   hardware lane cannot give is catching the sim path accidentally acquiring a
   device, since its container has no `/dev/tenstorrent`. Worth adding if sim
   feedback latency ever starts to hurt.

   Note what such a lane would and would not prove: the `builder` runner still
   downloads the build artifacts, so the bindings are present there. It would
   demonstrate "no device", not "no tt-metal build" — the latter is what item 3's
   subprocess test covers, in every lane.
6. **Kernel-author UX test.** Confirm `print(...)` / `breakpoint()` inside a
   kernel body work under sim (they can't on device), since that's a headline
   benefit.

---

## 12. Phasing

- **v1 (✅ done):** SPMD `core_index` execution model; `SimTensor`/`SimBlock`;
  all eltwise (unary/binary/comparisons/where/clamp/typecast/tile_transpose/
  bcast); reductions; matmul plus the in-kernel `zeros` + `+=` accumulator
  idiom; views/permute/tilize/untilize; host ops; `to_host`; `async def` +
  `await` bodies and no-op `Semaphore` (§5.5); shadow module **and** the
  `config.backend` switch; exact numerics; runtime-free import (§2), asserted;
  tests in `test/d2m-jit/sim/`,
  plus the whole-suite sim re-run in CI (§11.4).
- **v2 (🟡):** multi-device (mesh) execution — per-device buffers, sim
  `mesh`/`mesh_shard`, real gather (§14, tracked in
  [#9202](https://github.com/tenstorrent/tt-mlir/issues/9202)); declarative
  generic forms (`indexing_maps` / `iterator_types` / `block_factors` with
  `iter_index`/`block_index`/`block_offset`) and the `!tensor.store` method
  that goes with them; async-generator (`yield`) producer/consumer scheduling
  beyond pure serialization (currently rejected, §5.5); DMA primitives as they
  land in `api.py`.

Note that `SimKernel.__call__` *rejects* `indexing_maps` / `iterator_types`
(`NotImplementedError`) but currently accepts and ignores `block_factors` and
`kernel_io_in_dram`; both are placement/blocking hints that do not change
values in the SPMD form, but the asymmetry is worth closing when v2 lands.
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

---

## 14. Multi-device (mesh) 🟡 — landing incrementally

Tracked in [#9202](https://github.com/tenstorrent/tt-mlir/issues/9202); each
subsection below carries its own status marker. Goal:
run the mesh surface (`d2m.mesh` / `d2m.mesh_shard` / `d2m.mesh_gather` and
kernels over sharded tensors) under the sim backend, so mesh kernels and
sharding strategies can be developed and CI-tested without multi-chip hardware.

The descriptor math is already shared: `MeshShard`, `validate_mesh_mapping`,
`shard_logical_shape`, and the `current_mesh` mirror live MLIR-free in
`_src/layout_math.py`, used by both backends. What is missing is per-device
*storage*, per-device *execution*, and actual shard/gather *data movement*.

### 14.1 Data model: per-device buffers ✅

On device, every tensor under a mesh exists once per chip — sharded tensors
hold different shards, everything else (allocations, replicated inputs) holds a
copy per chip. The sim mirrors that:

- `SimTensor` stores `.buffers`, a **list of tile-padded torch buffers in
  row-major mesh order** (`len == prod(mesh_shape)`; length 1 when no mesh is
  declared — the existing single-device case is the degenerate form).
- `.buffer` becomes a property: inside a kernel it resolves through the
  `_current_device` thread-local (§14.4), so every existing `.buffer` consumer
  in `ops.py` works unchanged. Outside a kernel it is valid only for
  single-buffer tensors and raises on per-device tensors — host code must be
  explicitly mesh-aware (fail loud, §8 style), which prevents silent
  shard-0 reads.
- `.layout` still describes the **per-device shard** (as `LazyTensor.layout`
  does on device); `.mesh` carries the full-tensor mapping exactly as today.
- Under an active mesh, `to_layout` / `empty` / `zeros` / `full` / `arange`
  allocate one buffer per device (replicated fill). Value-transforming host ops
  (`to_layout`-from-`SimTensor`, `tilize`/`untilize`, `reshape`, views/permute)
  apply uniformly to every buffer.

### 14.2 `mesh` declaration ✅

A sim-native `mesh(shape, topology=None)` in `_src/sim/host.py` that validates
and calls `layout_math.set_current_mesh` — no MLIR, no builder scope. It joins
the `_dispatch` table in `api.py` (device → `builder.mesh`, which owns the
`ttcore.meshes` module attribute; sim → the mirror-only version) and is
exported from the shadow module, which today has no `mesh` at all.

`topology` is accepted and recorded but value-neutral (like the multicast args
in §5.1): the sim models no fabric, so `("linear", "ring")` changes nothing.

Lifecycle divergence (minor, documented): the device builder pins the mesh per
lazy graph (redeclaring a different shape errors until `to_host` resets the
graph). The sim has no graph lifecycle, so `mesh()` simply replaces the current
mesh; consistency is enforced where it matters, at `mesh_shard` / `mesh_gather`
validation time.

### 14.3 `mesh_shard` — full tensor → per-device shards ✅

Sim implementation in `_src/sim/host.py`; flipped from device-only to
`_dispatch` in `api.py`. Semantics match the runtime
(`runtime/lib/ttmetal/meshshard_utils.cpp::shard`) exactly:

1. Validate with the shared `shard_logical_shape(mesh_shape, full_shape,
   shard_dims, shard_shape)` — same shapes, same error messages as device.
2. Chunk the full host tensor along each tensor dim named by a non-`-1` entry
   of `shard_dims`, producing one chunk per point of the sharded mesh axes
   (indices increment last-axis-fastest, matching `incrementIndices`).
3. Along `-1` (replicated) mesh axes, **copy**: every device on that axis gets
   the same chunk (matching the runtime's replicate fill).
4. Each chunk lands as a tile-padded buffer at its row-major device slot;
   the result carries `.mesh = MeshShard(full_shape, shard_dims, shard_shape)`
   and a shard-shaped `.layout`, exactly like the device `LazyTensor`.

### 14.4 Execution: SPMD over devices × cores 🟡 (kernels over per-device tensors currently raise `NotImplementedError`)

The device runs the *same program* on every chip; data differences come only
through the shards. The sim adds one outer loop to §4:

```
for d in range(num_devices):            # row-major mesh order
    _current_device = d
    for y in range(Y):
        for x in range(X):
            _current_core = (y, x)
            body(*tensor_args, *scalar_args)
```

`_current_device` is a thread-local next to `_current_core` in `ops.py`; it
exists so `.buffer` resolution (§14.1) picks the right per-device buffer inside
`remote_load` / `remote_store` and the block ops. **Kernel bodies need no
changes and get no device-index op** — there is none in the device DSL either.
Devices run sequentially and share nothing during a kernel call: `remote_*`
addressing stays within the current device's buffer, so cross-device
communication is impossible to express (§14.7).

### 14.5 `mesh_gather` / `to_host` — shards → full tensor 🟡 (metadata-only today)

`mesh_gather` keeps its current metadata behavior (attach/validate `MeshShard`
via the shared math), and `to_host` does the actual gather for mesh-marked
tensors, mirroring the device split (`builder.py` gathers in
`_emit_returns_and_finalise`; the runtime concatenates in
`concatDistributedHostBuffers`):

- Concatenate per-device logical shards along the tensor dims named by
  `shard_dims`, inverting the placement order of §14.3.
- Replicated (`-1`) mesh axes contribute **one** copy (shard 0 along that
  axis). This is deliberately *more correct* than the runtime, whose `concat`
  fails to skip `-1` axes and dies with `TT_FATAL "dims must be unique"` —
  a §9 divergence (see the table there). The sim is the oracle for intended
  semantics; matching a bug we want fixed upstream would be backwards, and
  correct gather is what gives sim a working "replicate" strategy baseline.
- `to_host` of a per-device tensor **without** `.mesh` metadata raises,
  directing the user to `mesh_gather` (device behavior for an ungathered
  multi-device tensor is runtime-defined; fail loud rather than guess).

### 14.6 Testing

Follows §11's structure — shared tests carry their own torch goldens and run on
both backends; the sim suite covers only what shared tests cannot:

- `test/d2m-jit/test_mesh.py` keeps its `machines("n300")` gate for the device
  backend but gains a deviceless lane (the whole-suite sim re-run covers it on
  the hardware runner; a backend-parametrized or sim-marked lane covers
  no-device runners).
- Sim-suite additions (`test/d2m-jit/sim/test_sim.py`): shard → kernel → gather
  round trips against pure-torch goldens, per-device shard placement (incl.
  replicate copies), the replicate-gather divergence assertion (a §9-class
  assertion no shared test can make), the ungathered-`to_host` rejection, and
  the `.buffer`-on-per-device-tensor rejection.
- The §11.4 `device_only` audit: any mesh test asserting on-silicon contracts
  (e.g. real multi-chip timing) gets the marker with a root-cause reason.

### 14.7 Non-goals

- **No fabric/CCL semantics** beyond shard/gather: no link topology, no
  cross-device DMA, no all-gather/reduce-scatter primitives (none exist in the
  DSL yet; add sim backings as they land, per §5.5's DMA rule).
- **No timing** — same as §13; the mesh autotuner can use sim for *correctness*
  of sharding strategies, not for picking winners.
- **No real concurrency across devices** — sequential device loop, same
  ordering caveats as §5.5/§9.
