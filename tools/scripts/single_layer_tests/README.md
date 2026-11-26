# Single-block / single-layer perf tests

These perf tests live in `test/ttmlir/Silicon/TTNN/{n150,llmbox,galaxy}/optimizer/single_block_layer_perf_tests/` and consume model fixtures from `test/ttmlir/models/single_blocks_and_layers/`. The fixtures are generated upstream in `tt-xla`. The scripts in this directory pull them across and keep the lit tests in sync.

> Today only the `n150/...` subtree is checked in; the `llmbox/...` and `galaxy/...` dirs (and their `lit.local.cfg`) are created on demand by `update_lit_tests.sh` the first time a matching fixture is scaffolded.

## Refresh every fixture

On a silicon host:

```bash
export HF_TOKEN=hf_xxx
PYTHONUNBUFFERED=1 tools/scripts/single_layer_tests/regen.sh 2>&1 | tee /tmp/regen.log
```

What it does:

1. Locates a tt-xla checkout (`--ttxla-dir` if given, else sibling `../tt-xla`, else shallow-clones `main` into `/tmp`) and reads the pinned `TT_MLIR_VERSION` from it. That's the SHA your HEAD will be rebased onto in a throwaway worktree under `/tmp/ttmlir-regen-<pid>` (your working tree is never touched).
2. Pushes the rebased branch to `origin` as `regen/<label>-on-<base8>` so tt-xla can fetch it by SHA from the github URL it has hardcoded. (`--local` skips the push; see the `--local` section below.)
3. Builds tt-xla against that SHA and runs the single-layer sweep on silicon.
4. Copies the regenerated `*_1lyr_*.mlir` into the models dir.
5. Runs `llvm-lit` over each per-device lit dir under the build-tree mirror (`build/test/ttmlir/Silicon/TTNN/{n150,llmbox,galaxy}/optimizer/single_block_layer_perf_tests/`), then runs `ttrt run` over the same dirs to execute the generated flatbuffers on silicon — same path the CI PR workflow follows for optimizer tests (`op_model_ttrt.sh`).

The pushed branch is deleted from origin on success; pass `--keep-branch` to keep it (e.g. for CI handoff). On failure the branch is always kept and the final log lines print the `git push origin --delete` command to clean it up manually. Lit and ttrt failures at step 5 are reported but don't abort the run.

### Following along

The Python runner inside tt-xla is block-buffered on pipes, so `PYTHONUNBUFFERED=1` is essential — without it the log streams in chunks rather than line-by-line. With it, the runner emits one `Finished <group>::<name> -> <status>` line per test.

```bash
tail -F /tmp/regen.log
grep "Finished" /tmp/regen.log | tail        # per-test progress
```

For long sweeps, detach with tmux:

```bash
tmux new -s regen
HF_TOKEN=hf_xxx PYTHONUNBUFFERED=1 tools/scripts/single_layer_tests/regen.sh 2>&1 | tee /tmp/regen.log
# Ctrl-b d to detach; `tmux attach -t regen` to come back.
```

## `--local`

`regen.sh` defaults to pushing the rebased branch to `origin`; pre-flight requires `origin` to resolve to `tenstorrent/tt-mlir`. With `--local`, the push is skipped and tt-xla is invoked with `TT_MLIR_LOCAL_PATH=<your checkout>` instead. The worktree shares git objects with the parent checkout, so tt-xla can resolve the rebased SHA from there.

`--local` has a persistent side effect on tt-xla's checkout. tt-xla's `scripts/rebuild_for_custom_mlir.sh` rewrites `origin` inside the vendored clone at `third_party/tt-mlir/src/tt-mlir` to the local path via `git remote set-url`, and a comment in that script notes that CMake's ExternalProject update step does not refresh `origin` on subsequent runs. A later push-mode regen will therefore continue to fetch from the local checkout. To reset, either re-run with `--local` (the rewrite is idempotent) or delete `third_party/tt-mlir/src/tt-mlir` to trigger a fresh clone on the next build.

## Add a new test for a new fixture

`regen.sh` refreshes model fixtures but doesn't add lit tests. To add a lit test for a fixture without coverage (newly-arrived or just previously untested), scaffold it with `update_lit_tests.sh`:

```bash
tools/scripts/single_layer_tests/update_lit_tests.sh falcon_3_3b_decode_layer
```

The argument is the fixture basename (with or without `.mlir`); multiple names are fine. The script routes by filename pattern to the right per-device dir (see [routing](#routing)) and writes a minimal scaffold:

```
// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="..." -o <name>_ttnn.mlir %models/single_blocks_and_layers/<name>.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn <name>_ttnn.mlir
```

Customise in place: add `// CHECK:`, extra `// RUN:`, `// REQUIRES:`, `// UNSUPPORTED:` lines. Subsequent `regen.sh` runs overwrite only the *model fixture*, never the lit file (existing lit files are reported `HAVE` and skipped).

To see which fixtures are missing a lit test:

```bash
tools/scripts/single_layer_tests/update_lit_tests.sh --status
```

## Re-apply an existing tt-xla output (no silicon)

If you already have a tt-xla output dir on disk (a previous `regen.sh` run, a colleague's artefact, a partial sweep) and just want the fixtures copied across:

```bash
tools/scripts/single_layer_tests/update_models.sh /path/to/single_layer/generated_<sha>/ttir/
```

This is exactly what `regen.sh` calls at step 4. It accepts `*_1lyr_*` (current tt-xla naming) plus legacy `*_block` / `*_layer`; everything else is silently skipped. Relative paths resolve against the tt-mlir repo root.

## File layout

| Path | Written by | Overwrite policy |
|---|---|---|
| `test/ttmlir/models/single_blocks_and_layers/*.mlir` | `update_models.sh` | Overwritten every run |
| `…/Silicon/TTNN/{n150,llmbox,galaxy}/optimizer/single_block_layer_perf_tests/*.mlir` | `update_lit_tests.sh` | Never overwritten (`HAVE` → skip) |
| `…/optimizer/single_block_layer_perf_tests/lit.local.cfg` | `update_lit_tests.sh` | Created once if missing; never touched after |

The split means you can edit lit files freely — add CHECKs, gate with `UNSUPPORTED`, change the pipeline — and the next regen won't clobber your changes. Only the upstream-owned fixture content is regenerable.

### Routing

`update_lit_tests.sh` picks the per-device dir from the fixture name. `update_models.sh` does no routing — every fixture lands in the single models dir.

| Filename pattern | Lit dir |
|---|---|
| `*galaxy*` | `…/galaxy/optimizer/single_block_layer_perf_tests/` |
| `*tp*` | `…/llmbox/optimizer/…` |
| anything else | `…/n150/optimizer/…` |

## Flag reference

Run any script with `-h` for the full listing. Highlights:

**`regen.sh`**

| Flag | Default | Notes |
|---|---|---|
| `--subset name[,...]` | empty (delegates to tt-xla, currently `single`) | Pick from `single,llmbox,galaxy`. `llmbox` and `galaxy` don't combine (different meshes). |
| `--ttxla-dir <path>` | `../tt-xla`, else shallow clone of `main` into `/tmp` | |
| `--skip-lit` | off | Skip the lit + ttrt run at the end. |
| `--local` | off | Skip the push; pass `TT_MLIR_LOCAL_PATH=<your checkout>` to tt-xla instead. See the `--local` section. |
| `--keep-branch` | off | Keep the pushed branch on origin after a successful run. Failures always keep it. |

**`update_lit_tests.sh`**

- `<fixture> [<fixture>...]` — create lit files for those fixtures (default).
- `--status` — read-only; walk every fixture and report `HAVE` / `MISSING`.
- No args is a usage error.
