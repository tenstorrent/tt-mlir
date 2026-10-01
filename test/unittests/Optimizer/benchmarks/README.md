# Rotary reshard search benchmark (issue #7799)

This opt-in executable runs the real `MemoryLayoutPropagation` implementation
against a **synthetic, permissive backend**. It measures CPU optimizer search
cost. It does not execute rotary embedding, validate Tenstorrent kernel support,
check numerical results, or predict accelerator runtime.

## Run in the existing WSL environment

From `/home/jaumevaquer/tt-mlir`:

```bash
/home/jaumevaquer/tenstorrent-sim/dev-run cmake --build build --target RotaryReshardSearchBenchmark -j 4
/home/jaumevaquer/tenstorrent-sim/dev-run build/test/unittests/Optimizer/RotaryReshardSearchBenchmark > build/issue-7799-search-benchmark.csv
```

For a separately configured development environment, activate `env/activate`,
then run the same build and executable commands without the `dev-run` prefix.
The target requires Linux, GNU/LLVM linker wrapping, and
`TTMLIR_ENABLE_OPMODEL=OFF`. It is excluded from the default build and test suite.
The wrapper symbol names are ABI-dependent and must be updated if the wrapped
function signatures change.

## What is compared

- `disabled`: emulate the original rotary policy, which explores no reshards.
- `all`: explore reshards for input, cosine, and sine.
- `per_operand`: call the actual patched production policy (input only).

Link wrapping changes only this benchmark executable. Production libraries and
source code are not patched between measurements. The operation validator accepts
all combinations and returns the layout of input 0 as its output layout. Reshard
validation uses the op-model-disabled backend's permissive stubs. Neither models
real kernel restrictions or validation cost. In particular, allowing every cache
layout is deliberately an upper-bound search-space experiment, not a statement
that those layouts would be legal on hardware.

The fixture is one verified `ttnn.rotary_embedding` with three independent
function arguments of shape `[1, 1, 2048, 64]`, BF16, initially DRAM interleaved.
It provides seven height-sharded layouts, on 1, 2, 4, 8, 16, 32, and 64 cores.
No tensor data is allocated. Beam width is 8; maximum input candidates is 64.
The per-type reshard cap varies over 0, 1, 4 (the production default), and 7.
Function arguments intentionally avoid constant-derived-operand pruning: graphs
where that pruning already excludes the caches may see less or no benefit.

Each measurement includes `MemoryLayoutPropagation::run()`: candidate generation,
validation calls, scoring, sorting, beam processing, and applying layouts to IR.
Fixture construction and MLIR verification are outside the timer. Ten warmups
and 101 measured runs per policy/cap are interleaved in rotating order, in one
process. CSV reports median, p10, and p90 in microseconds. Do not run competing
builds/tests while collecting timings. Timings are descriptive, not pass/fail
thresholds.

Every run has deterministic checks which exit nonzero on failure:

- Operand candidate counts match the selected policy and cap.
- The observed cross product equals the number of evaluations and mock calls.
- The search produces a candidate and valid output MLIR.
- With the production policy and positive cap, an input reshard is selected under
  mock scoring, while both cache operands remain their original function arguments.
- Cap zero provides a control with one combination for all three policies.

These assertions exercise the actual optimizer call site, not just the rule-book
return value. They complement `OptimizerTests`' dispatch and compatibility tests.

## Recorded WSL result, 2026-09-24

Source baseline: `b5f1f345ee7104e00868054913dde6e53b49f7f2`, plus the local
issue-7799 patch and this benchmark. Release build, Clang 20, Intel Core Ultra 7
268V, Docker limited to 4 CPUs and 10 GiB RAM. No accelerator; no simulator used.
All benchmark assertions passed; all 305 `OptimizerTests` also passed.

At the production default cap of four:

| Policy | Candidates: input / cos / sin | Evaluations | Median search time |
| --- | --- | --- | --- |
| Original, disabled | 1 / 1 / 1 | 1 | 1.72 us |
| Explore all | 5 / 5 / 5 | 125 | 98.99 us |
| Patched, input only | 5 / 1 / 1 | 5 | 14.49 us |

Input-only exploration performed 25 times fewer evaluations and took about
6.8 times less CPU search time than all-operand exploration in this fixture.
At cap seven, counts were 512 versus 8, with medians 462.15 versus 17.34 us.
The CSV under `build/issue-7799-search-benchmark.csv` contains all caps and timing
percentiles (it is generated output, not committed source).

The arithmetic is simple: four new layouts plus the existing layout gives five
choices. Exploring all three operands tries `5 * 5 * 5 = 125` combinations;
exploring only the input tries `5 * 1 * 1 = 5`. Runtime ratios differ from count
ratios because setup, reshard checks, IR changes, and other work remain.

**This does not demonstrate a speedup over the original disabled policy**, which
is cheapest but misses the input optimization opportunity. Nor does it establish
an end-to-end compilation or model execution speedup. Real-backend compiler
measurements and device execution with numerical checks remain necessary for
those claims. The benchmark covers ordinary rotary; the existing rule-book unit
tests additionally cover rotary_embedding_llama's four-operand policy.
