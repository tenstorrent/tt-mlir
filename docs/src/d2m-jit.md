# d2m-jit

`d2m-jit` is a Python DSL testbed for authoring block-level D2M kernels and
running them through the D2M to TTMetal pipeline. It is aimed at compiler
developers who need to prototype D2M dialect behavior directly, not at general
model authors.

The DSL operates on tensors of `!ttcore.tile` values inside `@d2m.kernel`
functions. It supports remote shard load/store, Python `for`/`if` lowering,
async `yield`/`await`, semaphores, block-level elementwise ops, `where`,
metadata views (`view`, `view_layout`, `permute`), layout conversion helpers,
single-tile matmul with an explicit zero-filled output, and float per-tile
reductions (`reduce_sum`, `reduce_max`, `reduce_mean`).

Float reductions reduce one tile axis at a time using torch/numpy-style dims:
`dim=0` or `-2` reduces rows, and `dim=1` or `-1` reduces columns. The result
keeps the same tile shape with the hardware reduce result lanes populated.
Cross-tile reductions, collapsed-output helpers, integer SFPU reductions, and
broadcast-back are still follow-up work.

Build and test it with:

```bash
source env/activate
cmake -G Ninja -B build -DTTMLIR_ENABLE_D2M_JIT=ON
cmake --build build --target d2m-jit
pytest test/d2m-jit/
llvm-lit build/test/d2m-jit/
```

See the full tool documentation in [`tools/d2m-jit/README.md`](../../tools/d2m-jit/README.md)
and the active gap list in [`tools/d2m-jit/TODO.md`](../../tools/d2m-jit/TODO.md).
