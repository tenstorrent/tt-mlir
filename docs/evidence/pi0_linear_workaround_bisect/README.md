# Pi0 e2e bisect: `LinearOpOutputShapeRewritePattern` workaround

Evidence for closing [tt-mlir#8459](https://github.com/tenstorrent/tt-mlir/pull/8459) without merge.

Related issues: [tt-xla#4633](https://github.com/tenstorrent/tt-xla/issues/4633), [tt-metal#44094](https://github.com/tenstorrent/tt-metal/issues/44094).

## Summary

Three **tt-mlir** commits were built into the **same tt-xla** tree (`main` at `f491c9dd1`) using:

```bash
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUSTOM_TT_MLIR_VERSION=ON \
  -DTT_MLIR_VERSION=<commit>
cmake --build build
```

Test (whole-model Pi0 e2e):

```bash
pytest -svv tests/runner/test_models.py::test_all_models_torch[pi_0/pytorch-pi0_base-single_device-inference]
```

| Run | tt-mlir commit | `LinearOpOutputShapeRewritePattern` | Pi0 `pi0_base` result |
|-----|----------------|-------------------------------------|------------------------|
| 1 | `bb1deb417cef0e2c60147072cf7b6926a49ccca7` | **Present** (pre-#8328) | **FAILED** — `ttnn::concat` / `ShapeBase[] index out of range` |
| 2 | `142346d3697b7f963dcf039584b5371e91835a5f` | **Removed** ([#8328](https://github.com/tenstorrent/tt-mlir/pull/8328)) | **PASSED** |
| 3 | `eb9005fa360a80e44607e2dfd4404137b510092e` | **Removed** (current tt-xla pin) | **PASSED** |

**Conclusion:** The `LinearOpOutputShapeRewritePattern` workaround is the culprit for this Pi0 e2e failure. Removing it in #8328 fixes the regression; current main remains healthy. **Do not re-land the workaround** via #8459.

## Root cause (recap)

- On Pi0’s fused TT graph (biased `action_in_proj` in `embed_suffix`), rank-mismatched operands reached `ttnn.concat` → `ShapeBase[] index out of range. 2 not in [-4, 2)`.
- The workaround reshaped fused-linear outputs in ways that disagreed with downstream concat expectations on this graph.
- #8328 removed the invalid workaround; Pi0 e2e passes on the removal commit and on latest main.

## Key log excerpts

### Run 1 — FAIL (`bb1deb417`, workaround present)

```
TT_THROW: ShapeBase[] index out of range. 2 not in [-4, 2)
 --- ttnn::concat(...)
...
FAILED ... RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
```

Full log: [logs/pi0_base_bb1deb41.log](logs/pi0_base_bb1deb41.log)

### Run 2 — PASS (`142346d36`, #8328 removal)

```
| e2e_perf-avg_time: 0.016693990968633443
PASSED
```

Full log: [logs/pi0_base_142346d3.log](logs/pi0_base_142346d3.log)

### Run 3 — PASS (`eb9005fa3`, current main pin)

```
| e2e_perf-avg_time: 0.016800821060314775
PASSED
```

Full log: [logs/pi0_base_eb9005fa.log](logs/pi0_base_eb9005fa.log)

## Environment


- **tt-xla:** `f491c9dd16f42271549e66343484dcedb72521ab` (main)
- **Hardware:** Wormhole (single device)
- **Date:** 2026-05-18 / 2026-05-19

## Close #8459

PR [#8459](https://github.com/tenstorrent/tt-mlir/pull/8459) re-extends `LinearOpOutputShapeRewritePattern` for Pi0. This bisect shows Pi0 **fails with the workaround** and **passes after #8328 removal** and on **current main**. Close as **not needed** — issue resolved on main by removing the workaround, not by restoring it.
