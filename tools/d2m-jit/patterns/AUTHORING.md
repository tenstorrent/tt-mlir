<!--
SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
-->

# Authoring a d2m-jit pattern

A pattern file in this directory is a **self-contained unit**: the fused
kernel, the rewrite that emits it, and the tests that verify it all live in
one `.py`. The generic runner (`test/d2m-jit/test_patterns.py`) discovers the
tests automatically — you never edit the harness to add a pattern.

This contract is designed to be emitted and tweaked by agentic pipelines:
specs are plain dataclasses, discovery is filename-based, and each pattern has
a stable per-name test id.

## Quick start

```bash
cp _template.py eltwise_mul_relu_to_kernel.py        # pick a descriptive name
# fill in the TODOs, then register the file for install:
#   add `patterns/eltwise_mul_relu_to_kernel.py` to tools/d2m-jit/CMakeLists.txt
cmake --build build --target d2m-jit                 # or symlink for fast local iter
pytest test/d2m-jit/test_patterns.py -k mul_relu -v
```

Discovery skips any `_`-prefixed file, so `_template.py` and shared helpers are
never collected. Renaming the copy to a non-underscore name activates it.

## What a file declares

Four parts (see `_template.py` for a runnable skeleton):

1. **`@d2m.kernel`** — the tile-level computation. For the stock device runner
   the signature is `(in..., out, m_blocks, n_blocks)`.
2. **`@d2m.pattern(root=..., benefit=..., match=...)`** — match a TTIR op and
   `return` the replacement value. Higher `benefit` wins ties; for DAG matches
   pass a pure `match` predicate and root on the tail op.
3. **`PATTERN_TESTS`** — rewrite correctness (FileCheck, no device).
4. **`KERNEL_BENCHES`** — on-device numerics (PCC vs torch).

## `PatternTest` fields

| field | required | meaning |
|---|---|---|
| `name` | ✓ | unique test id; used in `pytest -k <name>` |
| `ttir` | ✓ | input MLIR module. Its **func signature is the source of truth** for input shapes/dtypes — don't duplicate them elsewhere |
| `check` | — | FileCheck directives (see below). Omit for a pure no-crash smoke test |
| `golden` | — | torch fn over the func args; carried for the future e2e device path, ignored by today's FileCheck-only runner |
| `inputs` | — | `InputSpec` (default `uniform(-1,1)`, seed 0) |
| `pcc` | — | threshold (default 0.99) |
| `expect_match` | — | `False` for negative cases the pattern must not fire on |
| `tags` | — | free-form labels for filtering/manifests |

Each spec is rewritten with **only its own file's patterns loaded**
(`apply_patterns_text` snapshots/clears/restores the global registry), so a
negative case won't accidentally hit another file's fallthrough lowering.

## `KernelBench` fields

| field | required | meaning |
|---|---|---|
| `name` | ✓ | unique bench id |
| `kernel` | ✓ | the `@d2m.kernel` entrypoint |
| `golden` | ✓ | torch fn; args match `input_shapes` order |
| `input_shapes` | ✓ | one shape per kernel input, in order |
| `run` | ✓ | `run(kernel, inputs, cfg) -> host tensor`. Use `eltwise_block_run` for the common elementwise-block shape; write a custom one otherwise |
| `inputs` | — | `InputSpec` |
| `default_cfg` | — | `{block_shape, grid_shape, dtype}`; defaults to `[1,1] / [1,1] / "float32"` |
| `space` | — | autotuning axes (`TuneAxis`); **not swept yet** — reserved |
| `pcc` | — | threshold (default 0.99) |

## FileCheck directives

`check` is dedented and piped to the real `FileCheck` binary with the
rewritten IR on stdin. Common directives:

- `CHECK-LABEL:` — anchor on a line (e.g. the func), resync FileCheck.
- `CHECK:` — the IR must contain this (in order, after the previous match).
- `CHECK-NOT:` — must NOT appear between the surrounding `CHECK`s.
- `{{...}}` — regex; `%{{.*}}` matches any SSA value.

Positive case shape: assert the matched op is gone and the fused subgraph
appears.

```
CHECK-LABEL: func.func @f
CHECK-NOT:   ttir.exp        # matched op replaced
CHECK:       d2m.generic     # fused kernel emitted
CHECK:       d2m.tile_add    # ... body ops, in order
CHECK:       d2m.tile_exp
```

Negative case shape (`expect_match=False`): assert the root op survives and no
kernel was emitted.

```
CHECK:       ttir.exp
CHECK-NOT:   d2m.generic
```

## True e2e device tests (`e2e=True`)

Set `e2e=True` on a `PatternTest` (with a `golden`) to additionally run the
*rewritten* module on silicon: it's compiled to a flatbuffer held **in memory**
and executed **in-process** (no ttrt subprocess, no files), then the device
output is PCC-checked against the golden. Inputs are generated deterministically
from the `ttir` signature and the golden is computed from those same inputs. See
`eltwise_exp_e2e_to_kernel.py` for a worked example.

One requirement: **no runtime scalar args** in the kernel — bake block counts /
loop bounds as Python constants. Runtime scalars lower to inline
`arith.constant`s the flatbuffer translator can't serialize. (The
`eltwise_*_to_kernel.py` kernels take `m_blocks, n_blocks` params, so they are
*not* e2e-ready; their direct-kernel `KERNEL_BENCHES` path handles them.)

Runs in the normal suite — no separate invocation or marker needed:

```bash
pytest test/d2m-jit/test_patterns.py -v
```

For large-scale CI, prefer a single batch driver that opens one device and
loops over all e2e specs in-process, rather than one pytest case per pattern.

## What's deferred (don't rely on yet)

- **Autotuning** — `KernelBench.space` is parsed but not swept. The eventual
  autotuner will iterate the config product, take perf traces, and compare.
