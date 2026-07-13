<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# d2m-jit Language Specification

This is the **normative reference** for the `d2m-jit` language: the legal
types, operations, and syntax of both the host-scope eager API and the
`@d2m.kernel` body sub-language. It complements [README.md](README.md)
(tutorial / overview) and [TODO.md](TODO.md) (tracker). Where the README
explains *how to use* the DSL, this document defines *what is legal* and is
the place to **propose new language constructs before they are implemented**.

`d2m-jit` is an internal testbed for the `d2m` MLIR dialect, not a productized
DSL (see [README.md](README.md) "Goals"). The surface defined here can change
between commits without deprecation. This spec tracks the *current* contract
and the *intended* one.

## Status conventions

Every type, operation, and grammar production in this document carries a
status marker:

- ✅ **implemented** — wired end to end (`api.py` / `_src/`) and exercised by
  tests. Safe to use.
- ⚠️ **implemented, constrained** — works only within stated limits; using it
  outside them is undefined or hits a known bug (cross-referenced to
  [TODO.md](TODO.md)).
- 🚧 **proposed** — specified here but **not implemented**. The grammar/op is
  reserved; calling it today raises a diagnostic. This is the extension area.

When you implement a 🚧 item, flip its marker to ✅/⚠️ in this file in the same
change, and remove the matching entry from [TODO.md](TODO.md).

---

## 1. Execution model and the two scopes

A `d2m-jit` program has **two distinct language scopes**, each with its own
type system and legal syntax. Keep them separate when reading this spec.

| Scope | Where | What it is | Compiled by |
| --- | --- | --- | --- |
| **Host scope** | Module-level Python; calls to `d2m.*` | Ordinary Python that eagerly appends MLIR ops to an open `func.func` via the singleton builder. The graph *is* the MLIR module. | The Python interpreter (eager); no AST rewriting. |
| **Kernel scope** | The body of a `@d2m.kernel` function | A restricted Python sub-language (§5) parsed into MLIR by `D2MCompiler` (an `ast.NodeVisitor`). The function is **never executed as Python** — its source is parsed and walked. | `D2MCompiler` in `_src/ast.py`. |

Host scope is lazy at the device level: ops accumulate until `to_host(*lts)`
runs the pass pipeline, executes the resulting flatbuffer, and resets the
builder. See [README.md](README.md) "The implicit builder" for the lifecycle
and "spent LazyTensor" rules.

A name registered with the `@syntax` decorator (§6) is legal **only in kernel
scope**. A host-side function (`to_layout`, `empty`, …) is legal **only in
host scope**. A handful of names exist in both with different meaning (e.g.
`zeros`/`empty` — host-side allocate a `LazyTensor`; kernel-side allocate an
in-kernel block; see §6.7).

---

## 2. Types

### 2.1 Element data types ✅

The scalar element types a tensor / tile may carry.

| Constant | Aliases accepted by `_to_data_type` | MLIR scalar | Notes |
| --- | --- | --- | --- |
| `d2m.float32` | `"fp32"`, `"torch.float32"`, `torch.float32` | `f32` | Default accumulator dtype for in-kernel `zeros`/`empty`/`matmul`. |
| `d2m.float16` | `"fp16"`, `"torch.float16"`, `torch.float16` | `f16` | |
| `d2m.bfloat16` | `"bf16"`, `"torch.bfloat16"`, `torch.bfloat16` | `bf16` | |
| `d2m.uint32` | `"uint32"`, `"u32"`, `"torch.uint32"`, `torch.uint32` | `ui32` | Primarily the backing type for semaphore buffers. |

Any other dtype raises `TypeError("Unsupported dtype ...")`. `dtype` arguments
everywhere (`Layout`, `tilize`, `typecast`, …) accept any spelling in the
table.

### 2.2 Memory spaces ✅

| Value | Aliases | MLIR |
| --- | --- | --- |
| `ttcore.MemorySpace.DeviceL1` | `"l1"`, `"sram"` | `#ttcore.memory_space<l1>` |
| `ttcore.MemorySpace.DeviceDRAM` | `"dram"` | `#ttcore.memory_space<dram>` |

### 2.3 Tile type ✅

The device element type when `tiled=True` is `!ttcore.tile<32x32, dtype>`.
The tile shape is fixed at **32×32**; no other tile shape is legal. A
"block" / `TensorBlock` is a `tensor<...x!ttcore.tile<32x32, dtype>>` — a
tensor *of tiles*, which is the kernel-scope value type (§2.5).

### 2.4 Host-scope types

| Type | Status | Definition |
| --- | --- | --- |
| `Layout` | ✅ | Pure descriptor of a tensor's device layout. Carries no data. Constructor: `Layout(shape, dtype, block_shape, grid_shape=None, tiled=True, collapse=True, mem_space=DeviceL1)`. `len(block_shape)` must equal `len(shape)`; each logical dim must be divisible by its block factor. `grid_shape` defaults to the derived blocked grid. Method `Layout.replace(**overrides)` returns a copy with fields swapped. |
| `LazyTensor` | ✅ | Host handle for a value in the lazy graph. Public fields: `.layout`, `.value`, `.generation`, `.materialized`, `.is_view`, `.mesh`. Method `.to_host()` ≡ `to_host(self)[0]`. A view (`is_view=True`) cannot be `to_host`'d directly. |
| `GlobalSemaphore` | ✅ | Allocated by `global_semaphore(grid_shape=(8,8), init=0)`. Fields `.value`, `.generation`, `.grid_shape`, `.init`. Passed as a `@kernel` argument → typed `!d2m.global_semaphore` inside the body. |
| `MeshShard` | ✅ | Metadata `(full_shape, shard_dims, shard_shape)` describing how a tensor is sharded across a device mesh. |
| `config` | ✅ | Process-level debug singleton (§3.6). |

### 2.5 Kernel-scope types

These are the only value types that may flow through a `@kernel` body. The
"MLIR type tag" column is the string `_get_type_str` produces; it is the key
under which `@syntax` methods/dunders are registered and dispatched (§7).

| Type | MLIR type tag | Status | What it is |
| --- | --- | --- | --- |
| Tile block | `!tensor` | ✅ | A `tensor<...x!ttcore.tile<32x32, dtype>>`. The class `TensorBlock` (`@syntax("!tensor")`) holds its dunders/methods. This is the type all elementwise/matmul ops operate on. |
| Semaphore | `!d2m.semaphore` | ✅ | A local/global semaphore. Methods `.set`/`.inc`/`.wait` (class `Semaphore`, `@syntax("!d2m.semaphore")`). |
| Index scalar | `index` | ✅ | Integer-typed runtime value: loop induction vars, `core_index(...)`, `mesh_position(...)`, scalar `int` kernel args, arithmetic results. Used for indices, loop bounds, and `scf.if` conditions. |

> Kernel-scope booleans are `i1` values produced by comparisons (§5.2) and
> consumed by `if` / `and` / `or`. There is no first-class `bool` block type.

---

## 3. Host-scope language (the eager API)

All names below are `d2m.<name>` after `import d2m_jit as d2m`.

### 3.1 Constructors and materialization ✅

| Symbol | Signature | Semantics |
| --- | --- | --- |
| `to_layout` | `to_layout(x, layout)` | Bring a `torch.Tensor` onto device, or convert a `LazyTensor` to a different device layout (emits `d2m.to_layout`, a real data shuffle). |
| `empty` | `empty(layout)` | Allocate an uninitialised device tensor. |
| `zeros` | `zeros(layout)` | Zero-initialised device tensor (≡ `full(layout, 0)`). |
| `full` | `full(layout, value)` | Device tensor filled with scalar `value`. Tiled → device-side `d2m.tile_fill`; non-tiled → host roundtrip. |
| `arange` | `arange(layout, start=0, step=1)` | Device tensor of arange values (host `torch.arange` → `to_layout`). |
| `tilize` | `tilize(lt, dtype=None)` | Convert a `LazyTensor` to a tiled (`tiled=True`) layout; optional dtype override. Accepts `LazyTensor` only. |
| `untilize` | `untilize(lt, dtype=None)` | Convert a `LazyTensor` to row-major (`tiled=False`); optional dtype override. |
| `to_host` | `to_host(*lts)` | Compile + execute; return a tuple of `torch.Tensor`. Resets the builder. Rejects views. |

### 3.2 Layout transforms / views ✅

| Symbol | Signature | Semantics |
| --- | --- | --- |
| `view` | `view(lt, lambda d0, d1: ...)` | Logical-rank permutation. Lambda arity = source logical rank. Result `is_view=True`. |
| `view_layout` | `view_layout(lt, lambda d0…dN: ...)` | Low-level: lambda arity = source MLIR rank (typically `2 × logical_rank` for tiled). Each result expr is a parameter or literal `0`. |
| `permute` | `permute(lt, *dims)` | `torch.permute`-style positional permutation. |
| `reshape` | `reshape(lt, *new_shape)` | ⚠️ Logical-shape change. Currently a host roundtrip (pays a DRAM transfer); use only for shapes not expressible as a `view`. |

`view`/`view_layout`/`permute` emit `d2m.view_layout` (metadata only). Views
must be run through `to_layout` before `to_host`.

### 3.3 Mesh / multi-device ✅

| Symbol | Signature | Semantics |
| --- | --- | --- |
| `mesh` | `mesh(shape, topology=None)` | Declare the device mesh shape; sets the module `ttcore.meshes` attr. `topology` e.g. `("linear","ring")` for CCL. |
| `mesh_shard` | `mesh_shard(input_, layout, shard_dims, shard_shape)` | Distribute a full host tensor across the mesh (`d2m.mesh_shard`, full→shard). |
| `mesh_gather` | `mesh_gather(lt, shard_dims=None, shard_shape=None)` | Mark a shard for gather on `to_host`. |
| `reblock` | `reblock(lt, grid)` | Reblock to a different worker grid (metadata view, no data movement). |
| `MeshShard` | `MeshShard(full_shape, shard_dims, shard_shape)` | Shard metadata (§2.4). |

### 3.4 Semaphores (host allocation) ✅

| Symbol | Signature | Semantics |
| --- | --- | --- |
| `global_semaphore` | `global_semaphore(grid_shape=(8,8), init=0)` | Allocate a `GlobalSemaphore` (ui32-backed). `grid_shape` must match the device worker grid. |

### 3.5 Fabric / CCL config ✅

| Symbol | Signature | Semantics |
| --- | --- | --- |
| `fabric_config` | `fabric_config(cluster_axis, topology="ring", num_links=1, noc="noc0", routing="unidir_ring_torus")` | Build a `#ttcore.fabric_connection_config` attribute for a CCL kernel; pass as `@kernel(..., fabric=...)`. |

### 3.6 Debug knobs `d2m.config` ✅

Each flag also reads a `D2M_JIT_*` env var:
`print_pipeline`, `print_ir_before_pipeline`, `print_ir_after_pipeline`,
`print_ir_after_each_pass`, `print_ir_debug_info`, `verify_passes`
(default `True`), `save_flatbuffer_path`.

---

## 4. The `@d2m.kernel` decorator

```python
@d2m.kernel
def k(in0, in1, out, m_blocks, n_blocks):
    ...

k(a_lt, b_lt, out_lt, M, N, grid=(2, 2))
```

`@kernel` wraps the function in a `CompiledKernel`. Calling it appends a
`d2m.GenericOp` to the open host func and parses the body with `D2MCompiler`.

### 4.1 Argument conventions ✅

A kernel call takes positional arguments followed by keyword-only knobs.

**Positional arguments** must be ordered: all "tensor-like" args first, then
all "extra" args. Mixing raises `TypeError`.

| Positional arg type | Becomes | Rule |
| --- | --- | --- |
| `LazyTensor` | A blocked device-tensor operand of the `GenericOp`; typed `tensor<...x!ttcore.tile<...>>` inside the body. | Must precede all extras. The **last `num_outs`** lazy-tensor args are outputs; the rest are inputs. |
| `int` | An `index`-typed func arg (`additionalArgs`). | An "extra"; must follow all lazy tensors. |
| `GlobalSemaphore` | A `!d2m.global_semaphore` body argument (via the `SEMAPHORE_ARG` sentinel). | An "extra"; must follow all lazy tensors. |

**Keyword-only knobs** of `CompiledKernel.__call__`:

| Kwarg | Default | Status | Meaning |
| --- | --- | --- | --- |
| `grid` | *(required)* | ✅ | Physical worker grid, e.g. `(Y, X)` / `[8, 8]`. → `ttcore.ir.GridAttr`. |
| `num_outs` | `1` | ✅ | Number of trailing lazy-tensor args treated as outputs. |
| `block_factors` | `None` | ⚠️ | Per-grid-dim tiling factors. |
| `indexing_maps` | `None` | ⚠️ | List of lambdas → `AffineMap` for the `GenericOp`. |
| `iterator_types` | `None` | ⚠️ | List of `"parallel"`/`"reduction"`, one per loop dim. |
| `fabric` | `None` | ✅ | A `fabric_config(...)` attribute for CCL kernels. |

### 4.2 Thread model ✅

All kernels are authored as the **`unified`** compute+datamovement form
(`CompiledKernel.thread_type = "unified"`). The backend splits the unified
body into per-thread regions (compute / datamovement). CCL kernels
(`device_synchronize`, cross-device `remote_store`, semaphores) are authored
the same way; the split pins the barrier to a single datamovement thread.

The internal `ThreadAttr` accepts `{None, "datamovement", "noc", "compute",
"unified"}`, but the DSL only ever emits `"unified"`. Authoring a specific
thread type is **🚧 not exposed**.

---

## 5. Kernel-body grammar

The kernel body is a **restricted subset of Python**. `D2MCompiler` rejects
any AST node outside its supported set with
`NotImplementedError("unsupported Python syntax inside @kernel: <Node>")`,
pinned to the source line.

### 5.1 Supported statements ✅

The legal statement grammar (derived from `_SUPPORTED_NODES` and the
`visit_*` methods):

```
kernel        ::= "def" NAME "(" params ")" ":" suite
suite         ::= statement+
statement     ::= assign | aug_assign | for_stmt | if_stmt
                | expr_stmt | return_stmt
assign        ::= NAME "=" expr                      # single target only
aug_assign    ::= NAME augop expr                    # += -= *= /= etc.
for_stmt      ::= "for" NAME "in" "range" "(" args ")" ":" suite
if_stmt       ::= "if" expr ":" suite ("else" ":" suite)?
return_stmt   ::= "return" expr?
expr_stmt     ::= expr                               # usually a call
```

Statement-level rules:

- **`for`** ✅ — the iterable **must** be `range(...)` (1, 2, or 3 args).
  Lowers to `scf.for`. The induction var is `index`-typed. See §5.3 for the
  loop-carried accumulator rule.
- **`if` / `else`** ✅ — lowers to `scf.if`. The condition must be an
  integer/`i1` value (a comparison or a `BoolOp`); non-`i1` integers are
  compared `!= 0`. There is **no `elif`** node distinct from nested
  `if`/`else` (Python desugars `elif` to nested `if`, which works).
- **`Assign`** ✅ — single `NAME = expr` only. Tuple/list unpacking and
  multiple targets (`a = b = …`) are **🚧 not supported**.
- **`AugAssign`** ✅ — `x op= y`; reads `x` from its defining scope and writes
  back. `c += a @ b` is special-cased to in-place matmul accumulation (§6.8).
- **`return`** ✅ — kernels are `() -> ()`; a bare `return` or `return expr`
  is accepted but the host func is always emitted with no results.

Statements **not** in the grammar (each 🚧 / rejected): `async def`, `await`,
`yield`, `pass`, `while`, `break`, `continue`, `with`, `try`/`except`,
`raise`, nested `def`/`lambda`, `global`/`nonlocal`, `del`, `import`,
comprehensions, `match`, annotated assignments, starred/keyword unpacking in
assignment.

### 5.2 Expressions and operators ✅

```
expr     ::= NAME | const | call | attribute | subscript
           | binop | unaryop | boolop | compare | ifexp
           | list | tuple
```

- **Constants** ⚠️ — only `int` and `bool` literals are legal as bare
  kernel-scope values (both lower to `index` constants). **Float literals are
  rejected** as runtime values (`NotImplementedError`); floats are legal
  *only* as attribute arguments to ops that declare `args_as_attr`
  (`clamp_scalar`, `typecast`, …) — see §6.6.
- **`BinOp`** ✅ — dispatch is type-driven (§7): for a `!tensor` operand the op
  routes to the registered `__add__`/`__mul__`/… (block elementwise / matmul);
  for `index` operands it falls back to the `arith.*` integer op. Mapping:

  | Python | `!tensor` (block) | `index` fallback |
  | --- | --- | --- |
  | `+` `-` `*` | `add` `sub` `mul` | `arith.addi` `subi` `muli` |
  | `/` | `div` | 🚧 unimplemented |
  | `//` `%` | 🚧 | `arith.divsi` `remsi` |
  | `@` | `matmul` | 🚧 unimplemented |
  | `**` | `pow` | 🚧 unimplemented |
  | `<<` `>>` | `logical_left_shift` / `right_shift`* | `arith.shli` `shrsi` |
  | `&` `\|` `^` | `bitwise_and` `bitwise_or` `bitwise_xor` | `arith.andi` `ori` `xori` |

  \* `>>` on a block maps to arithmetic `right_shift` via the dunder; see §6.4.

- **`UnaryOp`** ✅ — `-x` → `negative` (block) / `emitc.unary_minus`; `~x` →
  `bitwise_not` (block) / `emitc.bitwise_not`; `+x`, `not x` route to
  `__pos__`/`__not__` or emitc fallbacks.
- **`BoolOp`** ✅ — `and`/`or` over `i1`/integer values → `arith.andi`/`ori`
  (chained left-to-right). Operands must be integer-typed.
- **`Compare`** ⚠️ — **single** comparator only (`a < b`, not `a < b < c`).
  Lowers to `arith.cmpi` (signed) on `index`/integer operands — this is the
  **index domain** (loop bounds, `if` conditions). Comparisons are therefore
  **not overloaded for tensor blocks**: to compare blocks use the named ops
  `eq`/`ne`/`gt`/`ge`/`lt`/`le` or their methods (§6.3), which write 1/0 into
  each tile lane.
- **`IfExp`** ⚠️ — `a if c else b` is in the supported-node set; for blocks
  prefer the explicit `where` op (§6.3).
- **`Subscript`** ✅ — `arr[i]` / `arr[i, j]` on a `tensor` value; constant
  indices are bounds-checked at compile time. Returns an `(array, indices)`
  pair consumed by ops like `store`.
- **`List` / `Tuple`** ✅ — evaluate elementwise to a Python tuple of values;
  used for index lists (`[m, n]`) and variadic op operands.

### 5.3 Loop-carried accumulators ✅

A variable assigned inside a `for` body that **already exists in an enclosing
scope** becomes an `scf.for` `iter_arg`: its updated value is threaded through
the loop's `yield` and visible after the loop. Fresh per-iteration locals stay
region-local. This is what makes the canonical matmul K-reduction work:

```python
c = zeros([1, 1])
for k in range(k_blocks):
    c += remote_load(lhs, [m, k]) @ remote_load(rhs, [k, n])
# c now holds the reduced result
```

---

## 6. Kernel-body operations

Every op below is registered via `@syntax("<name>")` and is callable in
kernel scope as a free function `name(...)`. Elementwise/tensor ops are
*also* methods on a block (`x.name(...)`) and, where noted, Python operators
(§7). Each op wraps one or more `d2m.tile_*` builders in a `linalg.generic`
over the tensor of tiles (`_eltwise_block` / `_matmul_block`).

### 6.1 Iteration / position queries

| Op | Signature | Status | Semantics |
| --- | --- | --- | --- |
| `core_index` | `core_index(dim)` | ✅ | Current core's grid index along `dim` (0=y, 1=x). `dim` is a compile-time literal → `index`. |
| `mesh_position` | `mesh_position(dim)` | ✅ | This device's mesh position along `dim` (0=y, 1=x). `index`. 0 on a 1×1 mesh. |
| `iter_index` / `block_index` / `block_offset` | — | 🚧 | Alternative iteration-position queries. Meaningful only with `block_factors`/`indexing_maps` forms (§4.1). |

### 6.2 Data movement (point-to-point)

| Op | Signature | Status | Semantics |
| --- | --- | --- | --- |
| `remote_load` | `remote_load(src, indices, *, mcast_start_index=None, mcast_shape=None, mcast_dims=None)` or `remote_load(buf, src, indices, ...)` | ✅ / ⚠️ mcast | Load a shard from a remote (device-laid-out) tensor into an L1 buffer (allocated, or explicit `buf` from `empty`). Multicast forms hit a `SplitUnifiedThread` assertion on grids > 1×1 ([TODO.md](TODO.md)). |
| `remote_store` | `remote_store(dst, indices, src, *, start_device=None, device_mcast_shape=None, semaphore=None, semaphore_indices=None)` | ✅ | Store a shard from a local buffer into a remote tensor. Cross-device kwargs drive a mesh store. |
| `core_read` | `core_read(dst, src, *, core)` | ✅ | Direct core→core L1 read: read core `[y,x]`'s `src` into local `dst` (no device layout). Caller owns synchronization. See [core_read_write_spec.md](core_read_write_spec.md). |
| `core_write` | `core_write(src, dst, *, core)` | ✅ | Push dual of `core_read`: write local `src` into core `[y,x]`'s `dst`. |
| `dma_read`/`dma_write`/`dma_wait`/`local_copy`/`indexed_row_copy`/`embedding`/`null_tx` | — | 🚧 | Lower-level DMA primitives (exist in `D2MGenericRegionOps.td`, unexposed). |

### 6.2.1 Core collectives (MPI-style) 🚧

A proposed family of **collective** data-movement ops that coordinate data
*between cores on the device grid*, with MPI semantics. They are the
core-grid analogue of the cross-device CCL ops (`device_synchronize`,
mesh-level all-gather): same collective shapes, but the "communicator" is a
set of worker cores rather than a set of mesh devices, and the transport is
the on-chip NoC (the `core_read`/`core_write` mechanism, §6.2) rather than the
fabric. None are implemented yet; this subsection fixes the intended surface.

**Communicator model.** Each collective runs over a *group* of cores:

- **Default group** — all cores on the kernel's `grid` (one communicator).
- `axis=` (0=y, 1=x) — restrict the collective to one grid dimension: cores
  sharing the *other* coordinate form one independent group (so `axis=1`
  gives per-row collectives, `axis=0` per-column). Mirrors `cluster_axis` in
  `fabric_config`.
- `group=(start_index, shape)` — an explicit rectangular sub-grid range
  (same shape as the `mcast_start_index`/`mcast_shape` operands of
  `remote_load`), for collectives over a sub-tile of the grid.

**Rank ordering.** Within a group, a core's rank is row-major over the group's
extent (`rank = local_y * group_width + local_x`); for an `axis` collective it
is the core's position along that axis. Gather/scatter/all-to-all chunk order
follows this rank.

**Reduction op.** `op` is a compile-time literal (an attribute arg, like
`clamp_scalar`): one of `"sum"`, `"max"`, `"min"`, `"prod"`. Reductions are
elementwise over the tile block.

**Buffers.** `src`/`dst` are in-kernel L1 blocks (tile-tensors, §2.5) — no
device layout, exactly like `core_read`/`core_write`. Send/receive buffer
sizing follows MPI: gather/scatter/reduce-scatter/all-to-all relate `src` and
`dst` by a `group_size` factor as noted per op.

**Synchronization.** Unlike the raw `core_read`/`core_write` primitives (where
the caller owns all semaphore handshakes), a collective is **self-contained**:
it emits the internal NoC handshakes needed for correctness, and (except where
noted) implies a barrier across the group on completion. This is the primary
value-add over hand-rolling collectives out of `core_read`/`core_write`.

`root` (rooted collectives) is a core coordinate `[y, x]` (an `axis`
collective takes the root's index along that axis).

| Op | MPI analogue | Signature | Semantics |
| --- | --- | --- | --- |
| `core_broadcast` | `MPI_Bcast` | `core_broadcast(buf, *, root, axis=None, group=None)` | `root`'s `buf` is copied to every core's `buf` (in place). |
| `core_gather` | `MPI_Gather` | `core_gather(dst, src, *, root, axis=None, group=None)` | Every core sends `src`; `root`'s `dst` (size `group_size × src`) holds the rank-ordered concatenation. `dst` is meaningful only on `root`. |
| `core_all_gather` | `MPI_Allgather` | `core_all_gather(dst, src, *, axis=None, group=None)` | Like `core_gather`, but *every* core's `dst` receives the full concatenation. |
| `core_scatter` | `MPI_Scatter` | `core_scatter(dst, src, *, root, axis=None, group=None)` | `root`'s `src` (size `group_size × dst`) is split into rank-ordered chunks; chunk `r` lands in rank `r`'s `dst`. |
| `core_reduce` | `MPI_Reduce` | `core_reduce(dst, src, op, *, root, axis=None, group=None)` | Elementwise `op`-reduction of every core's `src` across the group; result in `root`'s `dst` (same shape as `src`). |
| `core_all_reduce` | `MPI_Allreduce` | `core_all_reduce(dst, src, op, *, axis=None, group=None)` | Like `core_reduce`, but the reduced result lands in *every* core's `dst`. |
| `core_reduce_scatter` | `MPI_Reduce_scatter` | `core_reduce_scatter(dst, src, op, *, axis=None, group=None)` | `src` is `group_size` chunks; reduce across cores chunk-wise, then rank `r` receives reduced chunk `r` in `dst`. |
| `core_all_to_all` | `MPI_Alltoall` | `core_all_to_all(dst, src, *, axis=None, group=None)` | `src`/`dst` are `group_size` chunks each; chunk `j` of rank `i` is delivered to slot `i` of rank `j`'s `dst` (transpose). |
| `core_synchronize` | `MPI_Barrier` | `core_synchronize(*, axis=None, group=None)` | Barrier: every core in the group blocks until all have arrived. Cross-core analogue of `device_synchronize` (§6.3). |

### 6.3 Synchronization

| Op | Signature | Status | Semantics |
| --- | --- | --- | --- |
| `semaphore_set` | `semaphore_set(sem, value, core=None, mcast=None)` | ✅ | Set a local/global semaphore. Also `sem.set(...)`. |
| `semaphore_inc` | `semaphore_inc(sem, value, core=None, mcast=None, compute=False)` | ✅ | Increment. `compute=True` marks a producer-done signal (`d2m.compute_signal`); see [unified_semaphore_design.md](unified_semaphore_design.md). Also `sem.inc(...)`. |
| `semaphore_wait` | `semaphore_wait(sem, value, reset=None)` | ✅ | Block until the semaphore reaches `value`; optional reset-after. Also `sem.wait(...)`. |
| `device_synchronize` | `device_synchronize(sem, start_device=None, mcast_shape=None, num_receivers=0, core_indices=None)` | ✅ | Cross-device CCL barrier. `num_receivers` must be a compile-time literal. |
| `synchronized_region` | — | 🚧 | Explicit synchronized scope. |

### 6.4 Elementwise ops

All take and return same-typed `!tensor` blocks (a tile-elementwise
`linalg.generic`). Each is a free function **and** a `TensorBlock` method.

**Unary (41) ✅:** `recip` `exp` `exp2` `expm1` `log` `log1p` `negative`
`cos` `acos` `sin` `asin` `tan` `atan` `tanh` `sqrt` `square` `rsqrt`
`sigmoid` `hardsigmoid` `silu` `softsign` `selu` `relu` `gelu` `erf` `erfc`
`sign` `signbit` `ceil` `floor` `frac` `trunc` `abs` `bitwise_not`
`logical_not` `eqz` `nez` `gtz` `gez` `ltz` `lez`.

**Binary (19) ✅:** `add` `sub` `mul` `div` `pow` `maximum` `minimum`
`bitwise_and` `bitwise_or` `bitwise_xor` `logical_left_shift`
`logical_right_shift` `right_shift` `eq` `ne` `gt` `ge` `lt` `le`.

**Comparisons** (`eq`/`ne`/`gt`/`ge`/`lt`/`le`) ✅ write 1/0 (matching the
input tile dtype) into each lane; pair with `where`. They are **free-function
/ method only** — Python `<`/`>=`/`==` are *not* overloaded (those go to the
index-domain `arith.cmpi`, §5.2).

**Ternary** ✅: `where(cond, t, f)` / `cond.where(t, f)` — elementwise select;
all three blocks same type; `cond` non-zero ⇒ `t`.

**Tile broadcast** ✅: `tile_bcast(x, "row"|"col"|"2d")` plus shorthands
`tile_bcast_row` / `tile_bcast_col` / `tile_bcast_2d` (free + method). `row`
broadcasts the tile's 0-row, `col` its 0-column, `2d`/`scalar` element (0,0).

### 6.5 Per-tile structural

| Op | Signature | Status | Semantics |
| --- | --- | --- | --- |
| `tile_transpose` | `tile_transpose(x)` / `x.tile_transpose()` | ✅ | Per-tile (32×32) transpose, in place. Distinct from host-side `permute`/`view`. |

### 6.6 Attribute-carrying ops ⚠️

These take **Python literals** that lower to MLIR attributes (not runtime
values). They are **free-function only** — no method form, because the
`args_as_attr` mechanism only fires on `visit_Call` of a `Name` target.

| Op | Signature | Status | Semantics |
| --- | --- | --- | --- |
| `clamp_scalar` | `clamp_scalar(x, min, max)` | ✅ | Clamp to literal `[min, max]`. Attr type follows tile dtype (F32 vs I32). |
| `typecast` | `typecast(x, dtype)` | ✅ | Per-tile typecast to a `d2m` dtype (or string alias). Same-dtype is a no-op. In-kernel analogue of host `tilize(dtype=)`. |
| `tile_bcast` | `tile_bcast(x, bcast_type)` | ✅ | See §6.4 (the `bcast_type` is an attribute). |

### 6.7 In-kernel block allocation ⚠️

| Op | Signature | Status | Semantics |
| --- | --- | --- | --- |
| `zeros` | `zeros([m_tiles, n_tiles])` | ✅ | Zero-initialised tile-tensor block (f32). Matmul-accumulator init pattern. Shape is a compile-time literal list/tuple. |
| `empty` | `empty([m_tiles, n_tiles])` | ✅ | Uninitialised L1 scratch block via `tensor.empty` (f32). Explicit `remote_load` destination. |

> These shadow the host-side `zeros`/`empty` *names* but are distinct
> kernel-scope ops (registered under the same string). Tile dtype is fixed at
> f32; a dtype override is 🚧.

### 6.8 Matmul ✅ / ⚠️

| Form | Status | Semantics |
| --- | --- | --- |
| `matmul(lhs, rhs)` / `lhs.matmul(rhs)` / `lhs @ rhs` | ⚠️ | `tensor<MxKx!tile> @ tensor<KxNx!tile> -> tensor<MxNx!tile>`. Standard parallel/parallel/reduction maps + `d2m.tile_matmul`. **Caveat:** a standalone `@` accumulates into a fresh (zero-init) buffer; the fill-init path breaks `d2m→ttkernel` lowering ([TODO.md](TODO.md)). Pre-fill the out-param with `d2m.zeros(L)`. |
| `c += a @ b` | ✅ | In-place accumulation into `c` (routed to internal `__matmul_acc__`). Canonical K-reduction accumulator; required for a loop-carried `c` to bufferize (§5.3). |

### 6.9 Reductions 🚧

`reduce_sum` / `reduce_max` / `reduce_mean(block, dim)` (numpy-axis `dim`),
with same-shape-broadcast (default) and `*_collapse` flavours, float/int
auto-dispatch. **Blocked**: float reductions need a scaler `b` operand and the
inline-`tile_fill` shortcut fails to lower; int-only (`tile_sfpu_reduce_*`) is
the viable near-term path. Full design and cost in [TODO.md](TODO.md).

### 6.10 Init / debug 🚧

`arange_block(layout)`, `fill_arange_tile()` (device-side arange fills) and a
kernel-side `print` (thin `d2m.print` wrapper) are proposed; all need the same
host-scope `linalg.generic` materialization the device-side fill needs.

---

## 7. Dispatch / overloading rules

How a kernel-scope expression resolves to a registered op:

1. **Free-function call** `name(args)` → `D2MCompiler._fn_map["name"]`. Unknown
   names raise `NameError` with a `did you mean?` hint.
2. **Method call** `recv.name(args)` → `_fn_map["<type-tag>.name"]` where the
   type tag is `_get_type_str(recv.type)` (e.g. `!tensor.exp`,
   `!d2m.semaphore.wait`). Unknown methods raise `AttributeError` + hint.
3. **Operator** `a <op> b` / `<op> a` → for tensor operands, the dunder
   `_fn_map["!tensor.__add__"]` etc.; otherwise the `arith.*`/`emitc.*`
   fallback (§5.2). `c += a @ b` is intercepted before dispatch (§6.8).
4. **Attribute args** — an op registered with `args_as_attr` / `kwargs_as_attr`
   pulls those positions out of the **AST node** as literals instead of
   evaluating them (enables float/shape/enum literals; §6.6).

The `@syntax` decorator (`_src/ast.py`) is the single registration point:
`@syntax("name")` for a free function, `@syntax("!type")` for a class whose
`ast_self`-first methods register as `!type.method`. Registration happens at
import time and populates `D2MCompiler._syntax`.

---

## 8. Diagnostics

Every error raised while compiling a kernel body is wrapped in `D2mJitError`
(`_src/errors.py`), pinned to `(file, line, col)` in the user's Python source,
with a code excerpt and an optional `did you mean?` suggestion (Levenshtein
over registered names / locals). The original exception is preserved as
`__cause__`, so `pytest.raises(ValueError)` etc. still match. Unsupported AST
nodes raise `NotImplementedError("unsupported Python syntax inside @kernel:
<Node>")` through the same path.

---

## 9. Proposed / not-yet-implemented (extension area)

This is the canonical place to **specify a language feature before building
it**. Add a subsection here (status 🚧), agree the surface, then implement and
flip the marker. Items already scoped in [TODO.md](TODO.md) are summarized
above at their natural home (§6.9 reductions, §6.2 DMA, §6.1 index queries,
§6.10 init/print); list *new* proposals below.

### 9.1 Grammar extensions under consideration 🚧

Not currently parseable in a kernel body; each would need a `visit_*` method
and an entry in `_SUPPORTED_NODES`:

- `while` loops (→ `scf.while`); `break` / `continue`.
- Tuple/list unpacking and multi-target assignment (`a, b = ...`).
- Chained comparisons (`lo <= x < hi`).
- Float runtime constants (today floats are attribute-only, §5.2).
- Tensor-domain comparison operators (overloading `<`/`==` for blocks) — blocked
  by the index-domain use of `visit_Compare`; would need a type-directed split.

### 9.2 Thread-type authoring 🚧

Expose `datamovement` / `compute` / `noc` kernel types (the `ThreadAttr`
already accepts them) instead of always emitting `unified`.

### 9.3 Op surface 🚧

Track new ops against `D2MGenericRegionOps.td`: dst/scratch register ops
(`acquire_dst`, `dst_reinterpret_cast`, `scratch_allocate`, …), CB management
(`push`/`pop`/`reserve`), masks (`write_col_mask_tile`,
`write_row_mask_tile`), runtime-arg queries (`get_arg`, `get_block_factor`,
`get_cb`). See [TODO.md](TODO.md) "Lower-level kernel primitives".

The **MPI-style core collectives** (`core_broadcast`, `core_gather`,
`core_all_gather`, `core_scatter`, `core_reduce`, `core_all_reduce`,
`core_reduce_scatter`, `core_all_to_all`, `core_synchronize`) are scoped in
§6.2.1; they need new `d2m` ops (or a lowering built on `core_read`/
`core_write` + semaphores) before they can be exposed.

### 9.4 Template — adding a new kernel-scope op

1. Implement the wrapper in `api.py` decorated with `@syntax("name")` (and a
   `TensorBlock` method + dunder if it's a tensor op). Use `_eltwise_block` /
   `_matmul_block` / a fresh `linalg.generic` helper.
2. If it carries literal attributes, declare `args_as_attr` / `kwargs_as_attr`
   and write the `node → Attribute` callback (model on `_const_value`,
   `_shape_literal`, `_tile_bcast_type_attr`).
3. Add a row to the relevant table in §6 with status ⚠️ (pending tests), and a
   lit IR-shape test under `test/d2m-jit/lit/` plus an on-device test.
4. Flip to ✅ once both pass; delete the [TODO.md](TODO.md) entry.

### 9.5 Template — adding a new statement form

1. Add the `ast` node(s) to `D2MCompiler._SUPPORTED_NODES`.
2. Implement `visit_<Node>` emitting the target `scf`/`d2m`/`arith` ops,
   managing `symbol_tables` scopes as `visit_If`/`visit_For` do.
3. Extend the §5.1 grammar and add lit + on-device tests.

---

## Related

- [README.md](README.md) — tutorial, API reference tables, pipeline, gotchas.
- [TODO.md](TODO.md) — active bugs/blockers and missing-surface tracker.
- [CCL_SPEC.md](CCL_SPEC.md) — collective-communication design (`all_gather`).
- [core_read_write_spec.md](core_read_write_spec.md) — `core_read`/`core_write`.
- [unified_semaphore_design.md](unified_semaphore_design.md) — producer-done
  semaphore fence.
- d2m dialect: `include/ttmlir/Dialect/D2M/`, `lib/Dialect/D2M/`.
