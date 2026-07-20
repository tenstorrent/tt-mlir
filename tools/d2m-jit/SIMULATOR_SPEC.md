# d2m-jit Simulator Spec

A design for a **pure-Python, torch-backed simulator** that runs a `@d2m.kernel`
without MLIR, the d2m→ttmetal pipeline, or a device. The simulator mirrors the
`d2m_jit` public API so the *same* kernels and *the same test files* execute
either on silicon (today's path) or in the simulator (this spec), selectable by
a flag.

> Status: Phase 1 implemented (`_src/sim/`). Branch `jgrim/d2m-jit-sim`.

## Implementation status (Phase 1)

Phase 1 is implemented in `_src/sim/` (`tensor.py`, `block.py`, `ops.py`,
`runtime.py`, `host.py`, `__init__.py`) and wired via `config.simulator`
(env `D2M_JIT_SIM`) + a backend hook at the end of `api.py`. Run any numeric
test under it:

```bash
D2M_JIT_SIM=1 pytest test/d2m-jit/test_simulator.py    # sim-only acceptance + coverage guard
D2M_JIT_SIM=1 pytest test/d2m-jit/test_eltwise.py test/d2m-jit/test_matmul.py ...
```

What passes verbatim under `D2M_JIT_SIM=1` (no test edits): `test_eltwise`,
`test_matmul`, `test_reductions`, `test_broadcasts`, `test_zeros_full_where`,
`test_arange_reshape`, `test_tilize_untilize`, `test_views`, `test_round_trip`
(incl. the stale-LazyTensor lifecycle), `test_simple`, `test_ops`,
`test_compare`, `test_bespoke`, plus `test_simulator` — **111 passing, 1
failing**.

**The one failing numeric test: `test_matmul.py::test_mcast_overwrite_grid_2x2`.**
It is the quirky multicast-overwrite smoke kernel: every core issues
`remote_store(out, [0, 0], ...)` (a bare local index, no core offset) yet the
expected result has each core owning its own output quadrant. Under the
documented op semantics — `remote_store` indices are *global* grid indices, the
model every other kernel (eltwise/matmul/reduction) uses and which the sim
implements — all four cores write grid cell `[0, 0]` and the last writer wins.
Matching the expected result would require `remote_store`-to-output to be
core-relative, which contradicts the global indexing the canonical eltwise
kernels rely on; reconciling the two needs the real multicast/output-shard
ownership semantics that Phase 1 explicitly defers (§5.3, §11). Left failing
rather than modeled incorrectly.

**Out of scope by design (compiler path, not sim numerics):** `test_errors`
(asserts `D2mJitError` formatting / `did you mean?` hints / unsupported-syntax
diagnostics from the AST compiler), the pipeline-printing cases in `test_config`,
and the pattern-rewrite / e2e-device cases in `test_patterns` (they drive the
MLIR rewrite + device path). These are properties of the compiler backend, not
the simulator.

**Sim-vs-device parity harness (§12) is implemented:** `runner.run_bench` gained
a `backend=` knob (`"device"` / `"sim"`), plus `run_bench_parity` and
`device_runtime_available`; `test/d2m-jit/test_parity.py` drives every discovered
`KernelBench` through both backends. `test_sim_matches_golden` needs no device;
`test_sim_matches_device` compares against the device result of the identical
kernel and skips when no runtime is present or the process is pinned to the sim.
On hardware, the two discovered eltwise benches show PCC = 1.000000 sim-vs-device.
(Parity is scoped to `KernelBench`es because those carry a golden + materializer;
the simulator itself runs any `@d2m.kernel`, benched or not.)

Deviations from the spec below, as built:
- `config.simulator` is resolved at import time; runtime toggling is not
  supported in Phase 1 (noted in §14).
- `api.py`'s backend hook reaches the module dict via `sys.modules` because
  `from ttmlir.ir import *` shadows the builtin `globals`.
- `reduce_*` raise `D2mJitError` (not `ValueError`) on a bad dim, matching the
  compiler path's error type.
- Arithmetic `view_layout` (the rope half-roll) raises a clear
  `NotImplementedError` — it is Phase 2.

---


---

## 1. Why

The current `d2m_jit` flow (`_src/builder.py`) is: build MLIR eagerly → at
`to_host` run ~16 passes → flatbuffer → open a mesh device → submit → copy back.
Every correctness check needs silicon and a full compile.

A simulator that interprets the kernel with torch buys us:

- **No device, no compile.** Runs anywhere torch runs; unblocks laptops and
  device-less CI.
- **Fast iteration.** Millisecond kernel runs vs. seconds of pipeline + device
  open.
- **Debuggability.** The kernel body runs as ordinary Python — set a
  breakpoint, `print` a block, inspect intermediate torch tensors.
- **A second oracle.** Sim-vs-device PCC parity catches both compiler bugs (sim
  right, device wrong) and DSL-semantics bugs (they agree but both wrong vs. a
  hand golden).
- **Golden generation.** Cheap reference outputs for the pattern/e2e harness in
  `test/d2m-jit/runner.py` without a `ttnn` baseline device run.

Explicit **non-goals** (inherited from the DSL itself — see `README.md`):

- Not a performance model. No cycle/bandwidth/placement modeling. Functional
  numerics only.
- Not a second source of truth for *hardware* quirks beyond the few dtype
  effects we choose to model (§10).
- Not a productized DSL. Same "internal testbed" caveats as `d2m_jit`.

---

## 2. What the simulator must reproduce

The observable contract of a kernel run, end to end:

```python
L   = d2m.Layout(shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 2])
inp = d2m.to_layout(torch_t, L)     # host -> device
out = d2m.empty(L)
add(inp, inp2, out, 1, 1, grid=(2, 2))   # kernel dispatch
result = out.to_host()              # torch.Tensor == golden
```

So the simulator has to stand in for four things:

1. **The host API** — `Layout`, `to_layout`/`empty`/`zeros`/`full`/`arange`,
   `tilize`/`untilize`/`view`/`view_layout`/`permute`/`reshape`,
   `reduction_layout`, `to_host`, and the `LazyTensor` lifecycle (generation /
   staleness). Defined in `_src/builder.py` and `_src/tensor_layout.py`.
2. **The kernel dispatch** — `@d2m.kernel` / `CompiledKernel.__call__` with
   `grid=`, `num_outs=`, scalar args, `kernel_io_in_dram=`.
3. **The kernel body vocabulary** — everything the AST visitor injects:
   `core_index`, `remote_load`, `remote_store`, `zeros([..])`, the 60+ eltwise
   ops, `matmul`, `reduce_*`, `tile_bcast*`, `where`, `typecast`,
   `tile_transpose`, `Semaphore`, `for`/`if`/`+=`. Registered via `@syntax` in
   `_src/ast.py` + `api.py`.
4. **The tiled/blocked data model** — the mapping between a logical torch
   tensor and the grid-of-blocks-of-tiles that `remote_load`/`remote_store`
   index.

The key realization: the numeric checks in `test/d2m-jit/` all compare against
**logical torch ops** (`torch.sigmoid(t)`, `lhs @ rhs`, `t.sum(dim=1,
keepdim=True)`, per-tile broadcasts computed on the host). So the simulator only
needs to be faithful at the **logical / block-region level**, not bit-faithful
to the physical tile stream. Tile structure matters for exactly a handful of
structural ops (§7.3); everything else is pointwise and identical whether
applied per-tile or per-region.

---

## 3. Architecture at a glance

```
                       d2m_jit public API  (api.py re-exports)
                                  |
              +-------------------+--------------------+
              |                                        |
    D2M_JIT_SIM=0 (default)                    D2M_JIT_SIM=1
    real builder (_src/builder.py)      sim builder (_src/sim/*.py)
     MLIR -> pipeline -> device          torch tile-grid interpreter
              |                                        |
              +-------------------+--------------------+
                                  |
                          torch.Tensor result
```

One toggle, two backends, identical surface. The toggle lives with the existing
config knobs (`_src/config.py`) and an env var, matching the `D2M_JIT_*`
convention already there.

---

## 4. Data model

### 4.1 `SimTensor` — a device tensor

Backs a `LazyTensor` in sim mode. Holds:

| field | meaning |
| --- | --- |
| `layout` | the `d2m.Layout` (reused unchanged) |
| `tiles` | torch tensor of the data in **tile-grid** form (see below), dtype = `layout.dtype` |
| `view` | optional index-remap function for views (§8); `None` for concrete buffers |

**Tile-grid form.** For a tiled 2D layout with logical shape `(H, W)`, the tile
grid is `[H//32, W//32, 32, 32]` — a torch tensor indexed by
`(tile_row, tile_col, i, j)`. This is the natural representation because:

- `remote_load(src, [bi, bj])` slices tile rows `bi*Bm : (bi+1)*Bm` and tile
  cols `bj*Bn : (bj+1)*Bn` (block_shape `[Bm, Bn]` in tiles), then reshapes to a
  `(Bm*32, Bn*32)` region.
- structural per-tile ops (`tile_bcast`, `tile_transpose`, per-tile reduce)
  operate on the trailing `32×32` axes cleanly.
- views (§8) are just a permutation/remap of the leading tile-index axes.

Conversion logical `(H, W)` torch tensor ⇄ tile grid `[H//32, W//32, 32, 32]` is
a reshape + transpose (`view(H//32,32,W//32,32).permute(0,2,1,3)` and back).
Non-tiled layouts keep a plain logical region and treat "tile" as the element
grid (block_shape in elements).

> The blocked-grid index space used by `remote_load` is
> `blocked_grid_shape = tiles / block_shape` (see
> `tensor_layout._derive_blocked_grid_shape`). It is **independent of
> `grid_shape`** — `grid_shape` only tells the *kernel* how to split that block
> index space across cores via `core_index`. So the data model needs `tiles`
> and `block_shape`; it does not need to physically shard by `grid_shape`. This
> matches every golden in the suite, which is computed with logical row-major
> tiling.

### 4.2 `SimBlock` — a loaded shard (what `remote_load` returns)

A thin wrapper the kernel body manipulates. Holds:

| field | meaning |
| --- | --- |
| `data` | torch region tensor, shape `(Bm*32, Bn*32)` (tiled) |
| `block_shape` | `(Bm, Bn)` in tiles — needed for per-tile ops |
| `reduced_axes` | `frozenset` of logical axes already reduced (mirrors `api._REDUCED_AXES_ATTR`), drives implicit broadcast (§7.4) |

`SimBlock` implements the same operator dunders and methods as `api.TensorBlock`
(`__add__`, `__matmul__`, `__neg__`, `.sigmoid()`, `.reduce_sum()`,
`.tile_bcast_col()`, `.where()`, `.matmul(..., transpose_b=)`, …), each
delegating to the torch implementations in §7. This is what makes
`a.add(b).sigmoid()` and `a @ b` and `x - reduce_mean(x, 1)` "just work" when the
body runs as native Python.

---

## 5. Executing the kernel body as native Python

The kernel body already reads like plain Python; the only reason it needs an AST
visitor today is that names like `core_index`, `remote_load`, `sigmoid`,
`zeros`, `reduce_sum` are **not** imported in the kernel's module — the AST
compiler resolves them from its `_fn_map`. The simulator supplies them instead
as an injected globals namespace and then *calls the function directly*.

### 5.1 Rebinding globals

`CompiledKernel` already captures `fn`. In sim mode we build a shadow function:

```python
import types
sim_globals = dict(fn.__globals__)          # keep the user's real imports
sim_globals.update(SIM_BUILTINS)            # core_index, remote_load, sigmoid, ...
sim_fn = types.FunctionType(
    fn.__code__, sim_globals, fn.__name__, fn.__defaults__, fn.__closure__)
```

`SIM_BUILTINS` is the sim analog of `D2MCompiler._syntax`: one entry per
`@syntax`-registered name, implemented against `SimBlock`/torch. Because we run
the real code object, Python's own `for`/`range`/`if`/`+=`/`and`/`or` execute
natively — no re-implementation of control flow, and loop bounds / indices are
ordinary Python ints (exactly the "index domain" the AST compiler lowers to
`arith`).

This sidesteps every AST corner case (chained methods, augmented matmul
accumulation, nested loops, closures over int captures) for free: they are
Python semantics.

### 5.2 Per-core loop

A kernel runs once per core. The dispatcher iterates the grid and binds
`core_index`:

```python
def run_kernel_sim(kernel, lazy_args, scalar_args, grid, num_outs):
    gy, gx = grid
    inputs  = lazy_args[:len(lazy_args) - num_outs]
    outputs = lazy_args[len(lazy_args) - num_outs:]
    for cy in range(gy):
        for cx in range(gx):
            ctx = CoreContext(cy, cx, inputs, outputs)   # backs remote_load/store/core_index
            with _active_core(ctx):
                sim_fn(*inputs, *outputs, *scalar_args)   # writes into output SimTensors
```

`core_index(0) -> cy`, `core_index(1) -> cx`. `remote_load(src, idx)` reads a
block from `src`'s `SimTensor.tiles`; `remote_store(dst, idx, block)` writes into
`dst`'s tiles. The active `CoreContext` is a thread/context-local so the injected
builtins can find it (mirrors how `_get_scope()` works in the real builder).

Because all cores read inputs and write disjoint (or, for the intentional
overwrite tests, well-defined last-writer) output blocks, sequential iteration
produces the same result a real many-core run would — there is no cross-core
data hazard in the current op set except explicit multi-pass accumulation, which
the kernels already sequence host-side (see the cross-tile reduction tests).

### 5.3 Multicast

`remote_load(..., mcast_start_index=, mcast_shape=, mcast_dims=)` describes NOC
routing. The **data** loaded is still the block at `indices` (see
`mcast_overwrite_kernel` in `test_matmul.py`: the value is `lhs[cy*M+m, k]`
regardless of the mcast args). The simulator therefore ignores the mcast kwargs
for value computation and just reads `indices`. (A later phase could assert the
mcast region is consistent, but it does not change numerics.)

---

## 6. Host API in sim mode

All of these mint/consume `LazyTensor`s exactly as today; only the backing
storage changes from an `ir.Value` to a `SimTensor`.

| API | sim behavior |
| --- | --- |
| `to_layout(torch, L)` | tilize the logical tensor into `L`'s tile grid, cast to `L.dtype`; wrap as `LazyTensor(SimTensor)` |
| `to_layout(lt, L)` | re-tile/re-cast the source `SimTensor` into `L` (grid/tile/dtype/mem_space changes are no-ops except dtype + tiled flag) |
| `empty(L)` | uninitialized `SimTensor` (use `torch.empty`) |
| `zeros(L)` / `full(L, v)` | `torch.zeros`/`torch.full` in tile grid |
| `arange(L, start, step)` | `torch.arange(...).reshape(logical)` then tilize (matches today's host-roundtrip impl) |
| `tilize/untilize(lt, dtype=)` | toggle `layout.tiled`; apply `.to(dtype)`; data is already logical so mostly metadata + cast |
| `view/view_layout/permute(lt, …)` | build a `SimTensor` with a `view` remap (§8); mark `is_view=True` |
| `reshape(lt, *shape)` | `to_host` → `torch.reshape` → `to_layout` (same host roundtrip as real) |
| `reduction_layout(L, dim)` | pure `Layout` math — reuse the real function verbatim |
| `to_host(*lts)` | untile each `SimTensor` to logical torch, cast to output dtype, return tuple; reset builder generation |
| `LazyTensor.to_host()` | `to_host(self)[0]` |

**Views and `to_host`.** Keep the real rule: `to_host` on an `is_view` tensor
raises; the user must `to_layout(v, v.layout)` first. In sim, materializing a
view = gather tiles through the remap into a concrete `SimTensor`.

**Lazy vs. eager.** The real builder is lazy (accumulate, run at `to_host`). The
simulator can be **eager** — execute each kernel call immediately against the
`SimTensor` buffers — while preserving the *observable* lazy contract:

- `LazyTensor.generation` and the stale-tensor `RuntimeError` are reproduced by
  keeping the same generation counter and resetting it in `to_host`. A tensor
  from a prior generation that was `to_host`-materialized re-enters via
  `to_layout(self.materialized, ...)`; one that wasn't raises "Stale
  LazyTensor". This is a straight port of `LazyTensor._resolve`.
- Passing multiple outputs to one `to_host(a, b, c)` returns them together, as
  `test_broadcasts.py::test_tile_broadcasts` relies on.

Eager execution is simpler and sufficient because sim has no cross-kernel fusion
to gain from laziness.

---

## 7. Op semantics (the `SIM_BUILTINS` table)

Each maps to torch. Grouped by how much tile structure they respect.

### 7.1 Elementwise unary/binary/ternary — pure pointwise

Identical per-tile or per-region, so just apply torch to `SimBlock.data`:

| DSL | torch |
| --- | --- |
| `exp, log, sqrt, rsqrt, sin, cos, tanh, sigmoid, relu, gelu, erf, abs, sign, floor, ceil, square, recip, negative, …` | matching `torch.*` (see mapping table below) |
| `add, sub, mul, div, pow, maximum, minimum` | `+ - * / **`, `torch.maximum/minimum` |
| `eq, ne, gt, ge, lt, le` | `(a <cmp> b).to(dtype)` — write 1.0/0.0 into lanes, matching the dialect |
| `bitwise_and/or/xor/not`, shifts | torch bitwise on integer-typed tiles |
| `where(cond, t, f)` | `torch.where(cond != 0, t, f)` |
| `clamp_scalar(x, lo, hi)` | `x.clamp(lo, hi)` |
| `typecast(x, dtype)` | `x.to(torch_dtype)` — changes block dtype |

A few need care:
- `recip` → `1/x`; `rsqrt` → `torch.rsqrt`; `silu` → `x*torch.sigmoid(x)`;
  `hardsigmoid`, `selu`, `softsign`, `expm1`, `log1p`, `exp2`, `frac`, `trunc`,
  `signbit`, `eqz/nez/gtz/gez/ltz/lez` → their obvious torch forms. The set is
  closed and enumerated in `api.py`; the sim table has one line each.

### 7.2 Matmul — block-of-tiles = region matmul

`a @ b` / `matmul(a, b, transpose_b=)`: `a.data (Bm*32, Bk*32)`,
`b.data (Bk*32, Bn*32)` → `a.data @ b.data`. `transpose_b=True` ⇒
`a.data @ b.data.T` (b stored `(N, K)`). Accumulation `c += a @ b` is native
Python `+=` on `SimBlock` (delegates to `add`), matching the loop-carried
accumulator kernels in `test_matmul.py`. Standalone `a @ b` needs no explicit
zero-init in sim (torch matmul is already accumulator-free).

### 7.3 Structural per-tile ops — respect the `32×32` tile

Operate on the trailing tile axes of the `[Bm, Bn, 32, 32]` view of the block:

| DSL | per-tile torch |
| --- | --- |
| `tile_bcast(x, "row")` / `tile_bcast_row` | broadcast each tile's row 0 across the tile: `tile[:1, :].expand(32, 32)` |
| `tile_bcast(x, "col")` / `tile_bcast_col` | broadcast each tile's col 0: `tile[:, :1].expand(32, 32)` |
| `tile_bcast(x, "2d")` / `tile_bcast_2d` | broadcast element (0,0): `tile[:1, :1].expand(32, 32)` |
| `tile_transpose(x)` | transpose each `32×32` tile in place (not the block layout) |

The `expected_tile_bcast` helper in `test_broadcasts.py` is exactly this
per-tile loop — the sim implementation and the golden are the same computation,
so parity is trivially exact there.

### 7.4 Reductions — keepdim, with implicit broadcast

`reduce_sum/max/mean(x, dim)` are keepdim reductions over the whole block along
the tile axis (`dim=0/-2` = rows → `#reduce_dim C`, `dim=1/-1` = cols →
`#reduce_dim R`). Sim implementation:

```python
axis = normalize(dim)               # 0 or 1 (rows/cols of the block region)
out.data = reduce_fn(x.data, dim=axis, keepdim=True)   # sum / amax / mean
out.reduced_axes = x.reduced_axes | {axis}
```

- `reduce_mean` divides by the reduced element count (`block_dim * 32`) — matches
  `api.reduce_mean`'s scaler.
- The result keeps shape `(..., 1)` along the reduced axis. **Implicit
  broadcast**: when a reduced `SimBlock` is combined with an unreduced one in an
  eltwise op (`x - reduce_mean(x, 1)`), torch broadcasting over the size-1 axis
  reproduces `api._eltwise_block`'s `tile_bcast`-of-reduced-operand behavior.
  `test_reduce_mean_cols_implicit_broadcast` computes precisely
  `block - block.mean(dim=1, keepdim=True)`, so keepdim broadcasting is exact.

Cross-core reductions are out of scope (the real DSL rejects them via
`reduction_layout` unless `allow_cross_tile=True`, and the cross-tile tests do
explicit host-sequenced accumulation passes, which the sim runs as ordinary
repeated kernel calls).

### 7.5 Kernel-body init

`zeros([Bm, Bn])` → `SimBlock(torch.zeros(Bm*32, Bn*32))`. (A future `full([..],
v)` maps to `torch.full`; see `TODO.md`.)

### 7.6 Mapping completeness

`SIM_BUILTINS` must cover every key in `D2MCompiler._syntax`. Generate a test
that asserts `set(SIM_BUILTINS) >= set(D2MCompiler._syntax) - {async/semaphore/
DMA phase-2 set}` so new `@syntax` ops can't silently fall through to a
device-only path.

---

## 8. Views

Views (`view`, `view_layout`, `permute`) are metadata remaps of the tile grid,
no data movement. The lambda passed to `view_layout` takes the physical
(blocked) index tuple and returns a remapped tuple — and it is an **ordinary
Python lambda**. The real builder runs it with `AffineExprProxy` sentinels to
build an `AffineMap`; the simulator runs it with **concrete ints** to get the
source index for each destination index.

Representation: a `SimTensor` view stores `remap: dst_index_tuple ->
src_index_tuple` plus a reference to the source `SimTensor`. `remote_load` on a
view maps each requested tile coordinate through `remap` and gathers from the
source tile grid; materializing a view (`to_layout(v, v.layout)`) builds a
concrete tile grid by gathering all tiles once.

- **`permute` / simple `view`** — a pure axis permutation of the tile grid;
  trivial (`torch.permute` on the leading tile axes).
- **Affine-arithmetic `view_layout`** (the rope half-roll,
  `kernels/prefill/rope.py`) — evaluate the lambda per physical index. For rope
  the lambda rolls the feature-tile index by half; evaluating it concretely per
  `(d0, d1, d2, d3)` yields the rolled source coordinate, and the gather
  reproduces `torch.cat([x_hi, x_lo], dim=-1)` — exactly `rope._golden`'s
  `roll_half`. The `_derive_perm_layout` logic that permutes
  `logical/block/grid` shapes is reused unchanged to give the view its `Layout`.

Broadcast-to-1 (`lambda …: (…, 0, …)`) maps the destination axis to source index
0.

---

## 9. Toggle & integration

Goal: the existing `test/d2m-jit/*.py` run unmodified under the simulator.

- **Env / config:** `D2M_JIT_SIM=1` or `d2m.config.simulator = True`
  (add to `_src/config.py` alongside the other flags). When set, `api.py`
  re-exports the sim implementations of `to_layout`/`empty`/…/`to_host` and
  `kernel`, instead of `_src/builder.py`'s.
- **Single dispatch point:** the cleanest seam is `CompiledKernel.__call__` and
  the free constructors. Factor the builder-facing functions behind a small
  backend interface (`Backend.to_layout`, `.empty`, `.run_kernel`, `.to_host`,
  …) selected once at import from the config. `api.py`'s `@syntax` registrations
  stay as-is for the real path; the sim path uses `SIM_BUILTINS`.
- **conftest:** add a pytest option / marker so CI can run the whole suite twice
  (device and sim), and a `@pytest.mark.device_only` for tests that genuinely
  need silicon (none currently do for numerics; the `lit/` tests are compiler IR
  checks and stay device/compiler-side).

---

## 10. Numerics & fidelity

The suite checks with `assert_pcc(..., threshold≈0.96–0.99)` or `max abs diff <
0.05–0.15`, i.e. tolerant of low-precision hardware. The simulator computes in
torch and will usually be *more* accurate than the device. Decisions:

- **Default:** compute in the tile dtype (`f32`/`bf16`/`f16`) so bf16 kernels
  round like the device roughly does. This keeps PCC comparisons meaningful.
- **Optional `sim_high_precision`** config to compute in f32 and cast at
  stores, for a clean logical oracle.
- **fp19 `tile_fill` caveat** (see `builder.full` docstring): the device SFPU
  truncates f32 fills to fp19. The sim does *not* model this by default; note it
  as a known sim-vs-device divergence for arbitrary-valued fills. Model behind a
  flag only if a test needs it.
- **Reduction order:** torch reductions differ from the device's tile-by-tile
  accumulation in float rounding; within the tolerances used, this is a
  non-issue.

---

## 11. Scope / phasing

**Phase 1 (covers `test_eltwise`, `test_matmul`, `test_reductions`,
`test_broadcasts`, `test_zeros_full_where`, `test_arange_reshape`,
`test_tilize_untilize`, `test_simple`, most of `test_views`):**
tiled 2D layouts, unified single-thread kernels, all eltwise/matmul/reduce/
bcast/where/typecast/tile_transpose ops, `for`/`if`/`+=`, multicast-as-plain-
load, permute + simple views, host API + LazyTensor lifecycle.

**Phase 2:** affine-arithmetic `view_layout` (rope), non-tiled layouts, richer
`view`/`view_layout` shape changes.

**Phase 3 (only if a use case lands — mirrors `TODO.md`):** `async`/`yield`/
`await` multi-thread datamovement kernels and `Semaphore` (Phase 1 treats these
as sequential no-ops since single-threaded execution yields the same data), DMA
primitives, cross-core gather/redistribute reductions, multi-region generics.

Anything unsupported must **raise a clear `NotImplementedError`** naming the op —
never silently under-compute (same principle as the `assert`-heavy materializers
in `runner.py`).

---

## 12. Testing the simulator

1. **Reuse the suite.** Run `pytest test/d2m-jit/` with `D2M_JIT_SIM=1`. Every
   numeric test should pass at the same thresholds. This is the primary
   acceptance gate.
2. **Sim-vs-device parity harness.** A small driver that, for each `KernelBench`
   in `test/d2m-jit/kernels/`, runs the kernel through both backends and asserts
   PCC(sim, device) ≥ threshold. Extends `runner.run_bench` with a `backend=`
   knob. Catches compiler regressions cheaply.
3. **Golden source for the e2e harness.** Where `PatternTest`/`KernelBench` has
   no hand golden, allow the sim result to stand in for the current `ttnn`
   device baseline (`runner.ttnn_baseline_outputs`), removing a device
   dependency from `run_e2e`.
4. **Op-coverage test** (§7.6): `SIM_BUILTINS` keys ⊇ supported `_syntax` keys.

---

## 13. Proposed file layout

```
tools/d2m-jit/_src/sim/
  __init__.py         # Backend selection, SIM_BUILTINS assembly
  tensor.py           # SimTensor (tile-grid), logical<->tiled conversions
  block.py            # SimBlock + operator dunders / method forms
  ops.py              # torch implementations of every @syntax op
  runtime.py          # CoreContext, per-core loop, remote_load/store, core_index
  host.py             # sim to_layout/empty/zeros/full/arange/tilize/.../to_host
  views.py            # concrete-index remap evaluation for view/view_layout/permute
```

`api.py` and `_src/builder.py` gain a backend seam (§9) but their real-path
behavior is unchanged. `config.py` gains `simulator` (env `D2M_JIT_SIM`) and
`sim_high_precision`.

---

## 14. Open questions

- **Backend seam shape.** Cleanest is a `Backend` protocol chosen at import;
  alternative is per-function `if config.simulator` branches. Protocol is
  tidier but touches more of `builder.py`. Recommend the protocol.
- **`kernel_io_in_dram`.** A no-op for sim numerics (mem space doesn't change
  values). Accept and ignore, or assert it's unset? Recommend accept+ignore for
  drop-in compatibility.
- **bf16 rounding parity.** Is per-op bf16 rounding close enough to the device's
  fused accumulation for the 0.96 matmul thresholds? Validate on the
  `bf16-tall` matmul case early; fall back to `sim_high_precision` if PCC drifts.
- **Native-exec vs. AST-interpret.** This spec recommends native-exec (§5) for
  simplicity and fidelity to "run as regular Python." An AST interpreter sharing
  `D2MCompiler`'s dispatch tables is the fallback if native-exec hits a wall
  (e.g. some future syntax the real compiler special-cases that Python evaluates
  differently — none known today).
```
