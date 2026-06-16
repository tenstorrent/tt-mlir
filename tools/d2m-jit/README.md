# d2m-jit

`d2m-jit` is a Python DSL for authoring **block-level Tenstorrent kernels**
that compile through the `d2m` MLIR dialect and execute on silicon via
`tt-metal`. It is the JIT/eager front-end that sits on top of d2m; kernels are
written in normal Python, parsed into MLIR by an AST visitor, and dispatched
lazily so multiple kernel calls accumulate into a single program before being
compiled and submitted to the device.

This README is intended to be skim-readable both by humans and by LLMs that
need to understand or extend the DSL.

## Goals

`d2m-jit` is a **testbed and prototyping framework** for the `d2m` MLIR
dialect, used by compiler developers to exercise and iterate on the dialect
quickly from Python.

Explicit non-goals:

- **Not a productized DSL.** This is internal tooling. Surface, semantics,
  and IR shape can change between commits without deprecation.
- **Not competing with `tt-lang` or other Tenstorrent end-user DSLs.** If you
  are looking for a stable Python authoring surface for production kernels,
  use those instead.
- **API stability is not promised.** Function names, argument orders,
  Layout fields, and the pass pipeline are all subject to change as the
  dialect evolves. The DSL exists to make the dialect easier to develop
  against, not to be locked down.

Use `d2m-jit` if you are working *on* the d2m dialect or its pipeline and
want a fast Python-driven way to construct, lower, and execute kernels.

## At a glance

```python
import torch
import d2m_jit as d2m

# 1. Describe the data layout the kernel expects.
L = d2m.Layout(
    shape=(512, 512),
    dtype=d2m.float32,
    block_shape=[1, 1],
    grid_shape=[2, 2],
)

# 2. Define a kernel that operates on blocks (tile-typed tensors).
@d2m.kernel
def add(lhs, rhs, out, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            a = remote_load(lhs, [m_off + m, n_off + n])
            b = remote_load(rhs, [m_off + m, n_off + n])
            remote_store(out, [m_off + m, n_off + n], a + b)

# 3. Materialise inputs onto the device, run the kernel, fetch the result.
lhs = torch.randn(512, 512)
rhs = torch.randn(512, 512)
out = d2m.empty(L)
add(d2m.to_layout(lhs, L), d2m.to_layout(rhs, L), out, 8, 8, grid=(2, 2))
print(out.to_host())  # torch.Tensor matching `lhs + rhs`.
```

## Concepts

### `Layout`

A `d2m.Layout` is a pure descriptor of how a logical tensor lays out on the
device. It carries no data. Construct one with the logical shape, dtype, the
per-grid block shape, the physical grid, plus `tiled` / `collapse` / `mem_space`
options.

```python
L = d2m.Layout(
    shape=(64, 64),
    dtype=d2m.float32,           # also: d2m.float16, d2m.bfloat16, "fp32", torch.float32
    block_shape=[1, 1],
    grid_shape=[2, 2],
    tiled=True,                  # tile element type is !ttcore.tile<32x32, dtype>
    mem_space="l1",              # also: "dram"
)
```

`Layout.replace(**overrides)` returns a new layout with selected fields
swapped (used internally by `tilize` / `untilize` / `view_layout`).

### `LazyTensor`

A `LazyTensor` is a host-side handle for a value being built into the lazy
MLIR graph. Its fields:

- `.layout`: the `Layout` describing this value.
- `.value`: the underlying `ir.Value` at host-func scope (cleared once
  materialised).
- `.generation`: which builder generation produced it (see lifecycle below).
- `.materialized`: the `torch.Tensor` populated after `to_host()`.
- `.is_view`: `True` if produced by `view` / `view_layout` / `permute`.

Most users never inspect these fields directly; they just pass `LazyTensor`s
to other API calls.

### The implicit builder

`d2m_jit` keeps a process-level singleton `_Builder` that owns the current
MLIR `Context`, `Module`, and an open `func.func`. Every call to
`to_layout` / `empty` / `view` / a `@kernel` invocation eagerly appends MLIR
ops to that open function. The graph **is** the MLIR module — there is no
separate Python-side IR.

The builder lifecycle is:

1. Lazy-created on the first lazy-tensor construction.
2. Reset by `to_host(*lts)`: emits returns, runs the pass pipeline, executes
   the resulting flatbuffer on a mesh device, drops the module.

After reset, every `LazyTensor` produced by the dropped builder generation is
"spent":

- If `.materialized` is set (the tensor was passed to `to_host`), re-using it
  in a new op transparently re-enters the builder via `to_layout`.
- If `.materialized` is `None`, using it raises `RuntimeError("Stale
  LazyTensor: ...")`. Either include it in the `to_host` call, or re-build
  from the source.

### Block model

The DSL works at the **block** level. Inputs are `tensor<...x!ttcore.tile<...>>`
values — a tensor of tiles. The `d2m.tile_*` ops in the MLIR dialect operate
on single tile scalars, so elementwise ops in the DSL are wrapped in
`linalg.generic` over the tensor of tiles (`_eltwise_block` in `api.py`).
Matmul follows the same pattern but with parallel/parallel/reduction iterators
over (M, N, K) and `d2m.tile_matmul` in the body (`_matmul_block`).
Float reductions use `d2m.tile_reduce_*` with an automatically generated
device-local scaler generic (`_reduce_block`).

### Views

`view`, `view_layout`, and `permute` emit `d2m.view_layout` — a metadata
reinterpretation of the underlying buffer, not a data shuffle. Their results
have `is_view=True`, and `to_host` rejects them directly:

```python
v = d2m.permute(lt, 1, 0)
v.to_host()   # ValueError: argument 0 is a view ...
```

To materialise a view, run it through `to_layout` first (which emits a real
`d2m.to_layout` that shuffles data):

```python
out = d2m.to_layout(v, v.layout).to_host()
```

## API reference

All names below are accessible as `d2m.<name>` after `import d2m_jit as d2m`,
and (where they are method-style) as `tensor_block.<name>(...)` from inside a
kernel body.

### Constructors and materialisation

| Symbol | What |
| --- | --- |
| `d2m.to_layout(x, layout)` | Bring a `torch.Tensor` onto the device, or convert a `LazyTensor` to a different device layout. |
| `d2m.empty(layout)` | Allocate an uninitialised device tensor. |
| `d2m.zeros(layout)` | Allocate a zero-initialised device tensor (host-side `torch.zeros` + `to_layout`). |
| `d2m.full(layout, value)` | Allocate a device tensor initialised to a scalar `value` (host-side `torch.full` + `to_layout`). |
| `d2m.reduction_layout(layout, dim, allow_cross_tile=False)` | Build the keepdim output layout for a row/column reduction. Set `allow_cross_tile=True` only when the kernel has a cross-core gather/redistribute strategy for the reduced dimension. |
| `d2m.tilize(lt, dtype=None)` | Convert a `LazyTensor` to a tile-typed (`tiled=True`) layout; optional dtype override. |
| `d2m.untilize(lt, dtype=None)` | Convert a `LazyTensor` to row-major (`tiled=False`); optional dtype override. |
| `d2m.view(lt, lambda d0, d1: ...)` | Logical-rank permutation. The lambda's parameter count matches the source's logical rank. Result is a view (`is_view=True`). |
| `d2m.view_layout(lt, lambda d0, d1, d2, d3: ...)` | Low-level: lambda parameter count matches the source's MLIR rank (typically `2 * logical_rank` for tiled tensors). Each result expression may be a parameter or the literal `0`. |
| `d2m.permute(lt, *dims)` | `torch.permute`-style positional permutation. |
| `d2m.to_host(*lts)` | Compile and execute; return a tuple of `torch.Tensor`s. Resets the builder. |
| `LazyTensor.to_host()` | Sugar for `to_host(self)[0]`. |

### `@d2m.kernel`

Decorates a Python function so that calling it appends a `d2m.GenericOp` to
the open host func. The function body is parsed into MLIR by `D2MCompiler`.

```python
@d2m.kernel
def k(a, b, out, m_blocks, n_blocks):
    ...

k(a_lt, b_lt, out_lt, M, N, grid=(2, 2))
```

Argument conventions:

- The first arguments are `LazyTensor`s. By default, the **last** lazy-tensor
  argument is the output (i.e. `num_outs=1`); earlier lazy-tensor arguments
  are inputs.
- After all lazy-tensor arguments, additional `int` arguments become
  index-typed func args of the GenericOp (`additionalArgs`). LazyTensor and
  scalar arguments cannot be interleaved.
- `grid=(Y, X)` is required; it controls the physical grid the kernel runs on.
- `num_outs`, `block_factors`, `indexing_maps`, `iterator_types` are optional
  advanced knobs (see `CompiledKernel.__call__`).
- Set `kernel_io_in_dram=True` on a call, or
  `d2m.config.kernel_io_in_dram = True` globally, to convert every tensor input
  and out-param for the kernel to `mem_space="dram"` before emitting the
  `d2m.generic`. The process-wide default can also be set with
  `D2M_JIT_KERNEL_IO_IN_DRAM=1`.

Inside a kernel body, the following names are available (registered via the
`@syntax` decorator and dispatched by `D2MCompiler`):

| Symbol | Meaning |
| --- | --- |
| `core_index(0)` / `core_index(1)` | Current core's y / x grid index. |
| `remote_load(src, indices, mcast_start_index=None, mcast_shape=None, mcast_dims=None)` | Load a shard from a remote tensor; optional multicast. |
| `remote_store(dst, indices, src)` | Store a shard into a remote tensor. |
| `for i in range(...)` | Compiles to `scf.for`. |
| `if cond: ... else: ...` | Compiles to `scf.if`. |
| `async def` + `yield` / `await` | Compiles to `d2m.YieldOp` / `d2m.AwaitOp` (for multi-thread kernels). |
| `Semaphore` methods (`set`, `inc`, `wait`) | Sync primitives. |

### Elementwise block-level ops

Each op is registered as both a free function (`d2m.exp(x)`) and a
`TensorBlock` method (`x.exp()`). Both routes wrap `d2m.tile_*` in a
`linalg.generic` over the tensor of tiles.

Unary (41):

`recip`, `exp`, `exp2`, `expm1`, `log`, `log1p`, `negative`, `cos`, `acos`,
`sin`, `asin`, `tan`, `atan`, `tanh`, `sqrt`, `square`, `rsqrt`, `sigmoid`,
`hardsigmoid`, `silu`, `softsign`, `selu`, `relu`, `gelu`, `erf`, `erfc`,
`sign`, `signbit`, `ceil`, `floor`, `frac`, `trunc`, `abs`, `bitwise_not`,
`logical_not`, `eqz`, `nez`, `gtz`, `gez`, `ltz`, `lez`.

Binary (13):

`add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum`, `bitwise_and`,
`bitwise_or`, `bitwise_xor`, `logical_left_shift`, `logical_right_shift`,
`right_shift`.

Ternary:

| Symbol | What |
| --- | --- |
| `d2m.where(cond, t, f)` | Elementwise select: `cond ? t : f`. All three blocks must have the same type. `cond` is interpreted elementwise (non-zero ⇒ `t`, zero ⇒ `f`). Also available as `cond.where(t, f)`. |

Tile broadcast:

| Symbol | What |
| --- | --- |
| `d2m.tile_bcast(x, "row")` | Broadcast the tile's 0-row across each tile. |
| `d2m.tile_bcast(x, "col")` | Broadcast the tile's 0-column across each tile. |
| `d2m.tile_bcast(x, "2d")` | Broadcast element (0, 0) across each tile. |
| `d2m.tile_bcast_row(x)` / `x.tile_bcast_row()` | No-argument row-broadcast shorthand. |
| `d2m.tile_bcast_col(x)` / `x.tile_bcast_col()` | No-argument column-broadcast shorthand. |
| `d2m.tile_bcast_2d(x)` / `x.tile_bcast_2d()` | No-argument row-and-column broadcast shorthand. |

### Float reduction block-level ops

`reduce_sum`, `reduce_max`, and `reduce_mean` are registered as free functions
and `TensorBlock` methods. They are keepdim reductions: the reduced logical
dimension becomes size 1 in the destination layout.

```python
row_sum = d2m.reduce_sum(x, 1)  # or x.reduce_sum(1)
col_max = x.reduce_max(0)
row_avg = x.reduce_mean(-1)
```

`dim` follows torch/numpy axis numbering for a 2D tile block:

| `dim` | D2M reduce dim | Result layout |
| --- | --- | --- |
| `0` / `-2` | `#d2m<reduce_dim C>` | `shape[0] == 1` |
| `1` / `-1` | `#d2m<reduce_dim R>` | `shape[1] == 1` |

These ops are float-only (`f32`, `bf16`) and use a device-local scaler generic,
matching the D2M float tile-reduce signature `reduce(a * b, c)`. Sum/max use a
unit scaler; mean scales by the number of elements reduced in the local block.

Use `d2m.reduction_layout(input_layout, dim)` for the output tensor:

```python
L_in = d2m.Layout(shape=(64, 32), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[2, 1])
L_out = d2m.reduction_layout(L_in, 1)  # shape=(64, 1), grid_shape=[2, 1]

@d2m.kernel
def row_sum_kernel(in_t, out_t, m_blocks, n_blocks):
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x = remote_load(in_t, [m_off + m, n_off + n])
            remote_store(out_t, [m_off + m, 0], reduce_sum(x, 1))
```

Elementwise ops implicitly broadcast a reduced operand when it is combined with
an unreduced block, so softmax/layernorm-style code can stay natural:

```python
y = x - x.reduce_max(1)
```

Reductions that span multiple cores need a core gather/redistribute op: gather
partials from the cores that own the reduced dimension, complete the reduction,
then redistribute the reduced values to the cores that own the output layout.

### Python operators on `TensorBlock`

`+` `-` `*` `/` `@` map to `add`, `sub`, `mul`, `div`, `matmul`. Unary `-`
and `~` map to `negative` and `bitwise_not`. All five Python dunders delegate
to the matching free function.

### `d2m.matmul`

```python
d2m.matmul(lhs, rhs)   # tensor<MxKx!tile> @ tensor<KxNx!tile> -> tensor<MxNx!tile>
lhs.matmul(rhs)        # method form
lhs @ rhs              # operator form
```

Emits `linalg.generic` with the standard matmul indexing maps
(parallel/parallel/reduction over M/N/K) and `d2m.tile_matmul` in the body.

> **Caveat:** the generated `linalg.generic` accumulates into the output
> operand, which is currently a fresh `d2m.empty` (undefined-init). Pre-fill
> the output with `d2m.zeros(L)` and pass it as the out-param to the kernel
> that calls `@` to get correct values. A device-side fill inside
> `_matmul_block` is tracked but blocked on a downstream
> `d2m → ttkernel` materialisation issue.

### Debug knobs (`d2m.config`)

A process-level singleton. Each flag also reads a `D2M_JIT_*` env var.

```python
d2m.config.print_pipeline           # bool: print the pipeline string
d2m.config.print_ir_before_pipeline # bool: dump the module before passes
d2m.config.print_ir_after_pipeline  # bool: dump the module after passes
d2m.config.print_ir_after_each_pass # bool: enable_ir_printing(print_after_all=True)
d2m.config.print_ir_debug_info      # bool: include locations
d2m.config.verify_passes            # bool, default True
d2m.config.save_flatbuffer_path     # str | None: write fbb to disk before submit
```

```bash
D2M_JIT_PRINT_IR_AFTER=1 python my_test.py
```

## Source layout

| File | Purpose |
| --- | --- |
| `api.py` | Public surface: all `@syntax`-registered ops, the `TensorBlock` and `Semaphore` host classes, and re-exports from `_src`. |
| `_src/builder.py` | `_Builder` singleton, `LazyTensor`, `to_layout`/`empty`/`view`/`view_layout`/`permute`/`tilize`/`untilize`/`to_host`, `@kernel`/`CompiledKernel`, pipeline + runtime execution. |
| `_src/ast.py` | `D2MCompiler` — the AST → MLIR visitor that parses kernel bodies. Plus the `@syntax(...)` registration decorator. |
| `_src/tensor_layout.py` | `Layout` descriptor and the `float32` / `float16` / `bfloat16` dtype constants. |
| `_src/config.py` | The `config` debug singleton. |
| `_src/utils.py` | Internal helpers (`_discover_dialect_ops`, `_cast`, `_asindex`, `_get_type_str`, `_cleanup_source_code`). |
| `__init__.py` | `from d2m_jit.api import *`. |

## Building

```bash
source env/activate
cmake -G Ninja -B build -DTTMLIR_ENABLE_D2M_JIT=ON
cmake --build build --target d2m-jit
```

`d2m-jit` is independent of `pykernel`; it has no runtime dependency on it.
The Python module installs to `<build>/python_packages/d2m_jit/`.

## Testing

`test/d2m-jit/` follows the `test/ttnn-jit/` split:

- **End-to-end on-device tests** (`test/d2m-jit/test_*.py`) are pytest. Run
  them with `pytest test/d2m-jit/`.
- **Compiler / no-device tests** (`test/d2m-jit/lit/*.py`) are lit. Run them
  with `llvm-lit build/test/d2m-jit/`.

`test/d2m-jit/conftest.py` provides an autouse `set_seed` fixture so torch
RNG is deterministic per test. `test/d2m-jit/utils.py` provides
`assert_pcc(golden, actual)` and `arange_tile(...)` helpers.

## Pipeline

`to_host` runs the following passes on the open module before flatbuffering:

```
ttcore-register-device,
canonicalize,
d2m-lower-to-layout,
canonicalize,
ttir-bufferization-pipeline,
d2m-insert-scratch-buffers,
d2m-generic-apply-interchange,
d2m-generate-outer-loops,
d2m-allocate,
d2m-lower-multicast-loads,
d2m-generic-lower-to-explicit-form,
canonicalize,
d2m-be-pipeline{use-tile-matmul=0},
d2m-to-ttkernel-pipeline,
d2m-to-ttmetal-pipeline
```

The three legalisation passes that the older eager d2m_jit relied on
(`convert-elementwise-to-linalg`, `arith-to-d2m-tile-ops`, `ttir-to-d2m`)
have been dropped — the DSL emits the post-legalisation form directly.

## Gotchas

- **Views need `to_layout` before `to_host`.** A view is metadata, not data.
  `to_host(view)` raises; convert via `to_layout(view, view.layout)` first.
- **`tilize` / `untilize` only accept `LazyTensor`.** Use `to_layout(torch_t, L)`
  for the initial host → device step.
- **Spent `LazyTensor`s.** After `to_host`, anything from the previous
  generation that was not passed to `to_host` raises on re-use. If you need
  multiple values out, pass them all to a single `to_host`.
- **Matmul accumulator is currently undefined.** See the matmul note above.
- **Float reductions are per tile/per core.** Use `reduction_layout` for reduced
  outputs. Reductions spanning multiple cores need a core gather/redistribute
  op.
- **Argument order in a `@kernel` call.** All `LazyTensor` arguments first,
  then any `int` scalar arguments. Mixing raises a `TypeError`. The last
  `num_outs` `LazyTensor`s (default 1) are treated as outputs.

## Related

- The d2m MLIR dialect lives at `include/ttmlir/Dialect/D2M/` and
  `lib/Dialect/D2M/`.
- The `ttir → d2m → ttmetal` lowering pipeline is shared with the rest of
  tt-mlir; `d2m-jit` only constructs IR and runs the pipeline, it does not
  define passes.
- `tools/ttnn-jit/` is the sibling JIT for the TTNN backend; the test split
  convention used here mirrors it.

## Known issues / TODO

See [TODO.md](TODO.md) for active pipeline gaps (matmul accumulator init,
host-scope `linalg.generic`), missing API surface (in-kernel typecast, DMA
primitives, ...), and other follow-ups.
