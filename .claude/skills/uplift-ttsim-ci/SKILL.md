---
name: uplift-ttsim-ci
description: Uplift the TTSim version used by tt-mlir CI and refresh WH/BH simulator skips. Use when updating `.github/workflows/call-test-ttsim.yml`, changing `ttsim-version`, validating WH or BH TTSim golden tests, or triaging simulator-specific pytest skips.
---

# Uplift TTSim CI

Follow the stages below in order. Do not move the regression sweep ahead of
unskip triage, and do not stop after unskip triage.

## Definition of Done

An uplift is complete only when all of these are true:

1. The release tag and both WH/BH assets are verified.
2. Every applicable simulator-skip node is inventoried before testing.
3. Every executable candidate is run in its own pytest process on each
   applicable architecture, and the skip edits are resolved.
4. The temporary `noop` edit is restored.
5. Every workflow-listed file is run, one file per pytest process, on both WH
   and BH after the skip edits.
6. Changed skip files are rerun on both architectures, the workflow is
   syntax-checked, and the final diff contains no triage-only edits.

If any stage is incomplete, report the coverage gap; do not call the uplift
done.

## Ground Rules

- `.github/workflows/call-test-ttsim.yml` is the source of truth for the test
  list, matrix, descriptors, and assets. Re-read it at the start; never reuse a
  stale test list copied into this skill.
- Cover `wormhole_b0` and `blackhole`. Ignore Quasar/QSR unless requested.
- A TTSim error may terminate pytest with no summary, sometimes with exit code
  `1` rather than a signal-derived code. During unskip triage, one exact node
  per pytest process is mandatory. A whole-file run with skips disabled is not
  valid unskip evidence.
- Serialise candidate nodes within one architecture. WH and BH may run in
  parallel because they have separate simulator homes, libraries, descriptors,
  artifact paths, and logs.
- Use `/opt/ttmlir-toolchain/venv/bin/python -m pytest`; the local `pytest`
  wrapper may hide useful output.
- Before local commands, activate without `set -u`; activation expects some
  variables to be unset:

```bash
unset BUILD_DIR
source env/activate
export BUILD_DIR="$PWD/build"
```

## Stage 1: Freeze the Release and CI Scope

1. Find the latest public release of `tenstorrent/ttsim`.
2. Verify that it contains both `libttsim_wh.so` and `libttsim_bh.so`.
3. Record the release tag, publication time, asset names, current tt-mlir SHA,
   and current tt-metal SHA.
4. Record the initial `git status --short`; preserve unrelated user changes.
5. Extract the exact `.py` arguments from the `Run golden pytest on TTSim`
   step into a shell `TEST_FILES` array and print it for review:

```bash
mapfile -t TEST_FILES < <(
  awk '/^[[:space:]]+pytest /,/--sys-desc/ {
    if ($1 ~ /\.py$/) print $1
  }' .github/workflows/call-test-ttsim.yml
)
((${#TEST_FILES[@]} > 0)) || {
  echo "No TTSim workflow tests found"
  exit 1
}
printf '%s\n' "${TEST_FILES[@]}"
```

6. Update only the workflow's `ttsim-version` default unless the current
   release requires a justified matrix/setup change. Keep this matrix:

```yaml
- arch: wormhole_b0
  soc_desc: wormhole_b0_80_arch.yaml
  ttsim_asset: libttsim_wh.so
- arch: blackhole
  soc_desc: blackhole_140_arch.yaml
  ttsim_asset: libttsim_bh.so
```

## Stage 2: Create Independent WH/BH Environments

Use a versioned root and separate result directories:

```bash
export TTSIM_VERSION=vX.Y.Z
export TTSIM_ROOT="${TMPDIR:-/tmp}/ttmlir-ttsim/$TTSIM_VERSION"
mkdir -p \
  "$TTSIM_ROOT/wormhole_b0" \
  "$TTSIM_ROOT/blackhole" \
  "$TTSIM_ROOT/results/wormhole_b0" \
  "$TTSIM_ROOT/results/blackhole"

curl -L --fail --retry 3 \
  -o "$TTSIM_ROOT/wormhole_b0/libttsim_wh.so" \
  "https://github.com/tenstorrent/ttsim/releases/download/$TTSIM_VERSION/libttsim_wh.so"
curl -L --fail --retry 3 \
  -o "$TTSIM_ROOT/blackhole/libttsim_bh.so" \
  "https://github.com/tenstorrent/ttsim/releases/download/$TTSIM_VERSION/libttsim_bh.so"

cp third_party/tt-metal/src/tt-metal/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml \
  "$TTSIM_ROOT/wormhole_b0/soc_descriptor.yaml"
cp third_party/tt-metal/src/tt-metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml \
  "$TTSIM_ROOT/blackhole/soc_descriptor.yaml"
```

Override `TTSIM_ROOT` with a durable path when logs must survive a reboot, but
keep generated binaries, artifacts, and reports outside the git working tree.

Set common variables once:

```bash
export TT_MLIR_HOME="$PWD"
export TT_METAL_HOME="$PWD/third_party/tt-metal/src/tt-metal"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
export TT_METAL_INSPECTOR=0
export TT_METAL_INSPECTOR_RPC=0
export LD_LIBRARY_PATH="$PWD/build/lib:${TTMLIR_TOOLCHAIN_DIR}/lib:${LD_LIBRARY_PATH:-}"
```

Build the current checkout before testing; results from stale tt-mlir or
tt-metal binaries are invalid:

```bash
cmake --build "$BUILD_DIR"
```

Generate and retain one system descriptor per architecture. These commands may
run in parallel:

```bash
export TT_METAL_SIMULATOR_HOME="$TTSIM_ROOT/wormhole_b0"
export TT_METAL_SIMULATOR="$TT_METAL_SIMULATOR_HOME/libttsim_wh.so"
ttrt query --save-artifacts \
  --artifact-dir "$TT_METAL_SIMULATOR_HOME/ttrt-artifacts" --quiet

export TT_METAL_SIMULATOR_HOME="$TTSIM_ROOT/blackhole"
export TT_METAL_SIMULATOR="$TT_METAL_SIMULATOR_HOME/libttsim_bh.so"
ttrt query --save-artifacts \
  --artifact-dir "$TT_METAL_SIMULATOR_HOME/ttrt-artifacts" --quiet
```

Do not share `TT_METAL_SIMULATOR_HOME`, `--sys-desc`, `--path`, or result logs
between architectures.

## Stage 3: Plan the Unskip Matrix

Complete the candidate plan before executing candidates.

1. Search only the workflow-listed files for `sim` skip sites. Record file,
   line, mark, parameter scope, and intended architecture in
   `sim-mark-sites.txt`.

```bash
rg -n '\bsim\b' "${TEST_FILES[@]}" > sim-mark-sites.txt
```

2. With the normal environment detection (`"sim"`), collect exact skipped node
   IDs on WH and BH without executing tests:

```bash
COLUMNS=10000 /opt/ttmlir-toolchain/venv/bin/python -m pytest \
  --setup-plan -vv --color=no "${TEST_FILES[@]}" \
  --sys-desc "$TT_METAL_SIMULATOR_HOME/ttrt-artifacts/system_desc.ttsys" \
  2>&1 |
  awk '/ SKIPPED/ {sub(/[[:space:]]+SKIPPED.*/, ""); print}' |
  LC_ALL=C sort -u
```

Save this once per architecture as `all-skipped-before-noop.txt`.

3. Temporarily change `_get_current_environment()` in
   `test/python/golden/conftest.py` to return `"noop"` when
   `TT_METAL_SIMULATOR` is set. Do not add `noop` to `ALL_ENVIRONMENTS`.
   Simulator marks no longer match, while unconditional skips, non-simulator
   skips, and `only_config` behavior remain active.
4. Repeat the setup-plan collection into `still-skipped-with-noop.txt`.
5. Both files must be sorted. Produce the executable candidate list with:

```bash
comm -23 \
  all-skipped-before-noop.txt \
  still-skipped-with-noop.txt \
  > candidates.txt
```

Review every source mark against the resulting WH/BH lists. Record sim-marked
nodes that remain skipped under `noop` in `not-executable.txt`; do not count
them as tested or remove their simulator mark without other evidence. Freeze
the two candidate lists before execution and record their counts.

## Stage 4: Run Every Candidate in Isolation

Keep the `noop` edit active. For each architecture, run the corresponding
`candidates.txt` serially. Quote each complete node ID:

```bash
timeout --signal=TERM --kill-after=10s 180s \
  env PYTHONUNBUFFERED=1 \
  /opt/ttmlir-toolchain/venv/bin/python -m pytest -svvv "$node_id" \
  --sys-desc "$TT_METAL_SIMULATOR_HOME/ttrt-artifacts/system_desc.ttsys" \
  --path "$TT_METAL_SIMULATOR_HOME/pytest_artifacts_candidates" \
  --tb=short -rs
```

Write one log per node plus a TSV/JSON manifest containing architecture,
node ID, elapsed time, exit code, and result. Continue after nonzero exits and
print progress after every node. WH and BH loops may run concurrently, but
never group multiple node IDs into one pytest invocation. Keep `-s` enabled so
pytest capture cannot swallow the last TTSim error when the process exits.

Classify results from both exit status and the pytest terminal summary:

- `passed`: pytest reports the node passed. This is the only automatic unskip
  evidence.
- `failed`: pytest prints a normal failure/error summary.
- `aborted`: the process ends without a pytest terminal summary, even if its
  exit code is only `1`.
- `timed_out`: the 180-second timeout expires.
- `skipped`, `xfailed`, or `xpassed`: record separately; none proves a clean
  simulator pass.

For an empty or unclear abort log, rerun that exact node with
`PYTHONUNBUFFERED=1 -svvv`. Do not rerun a group.

After the matrix is complete:

- Remove a simulator mark only when every architecture to which it applies
  passed.
- Narrow mixed results to the smallest dtype, shape, op, or architecture
  scope. Single-chip WH is `n150`; single-chip BH is `p150`.
- If a mark on one parametrization axis mixes passing and failing combinations
  from another axis, restructure the parametrization so the mark names the
  exact combinations; do not retain the broad axis-level skip.
- A config list is AND'd; separate groups are OR'd.
  `skip_config(["n150", "sim"])` means WH simulator only, while
  `SkipIf("n150", "sim")` means `n150` **or** any simulator.
- Preserve unrelated skips. For example, if only `sim` is obsolete in
  `SkipIf("ttnn", "emitc", "emitpy", "sim")`, keep
  `SkipIf("ttnn", "emitc", "emitpy")`.

If behavior differs from the previous TTSim version, run the smallest failing
node against both versions with the same tt-mlir checkout and corresponding
system descriptors.

## Stage 5: Restore Triage State

Restore `_get_current_environment()` to return `"sim"` before any final
validation. Inspect the diff rather than using a destructive checkout, and
verify that `conftest.py` has no triage-only change. Keep the candidate
manifests as validation artifacts, not source changes.

## Stage 6: Run the Full Regression Sweep

This stage is mandatory and happens after skip updates and `noop` restoration.

Run every file in `TEST_FILES`, one file per pytest process, on WH and BH. Do
not use one aggregate pytest command: an abort would hide the remaining files.
Continue after failures. The two architecture loops may run in parallel.

```bash
timeout --signal=TERM --kill-after=15s 3600s \
  env PYTHONUNBUFFERED=1 \
  /opt/ttmlir-toolchain/venv/bin/python -m pytest -vv "$test_file" \
  --sys-desc "$TT_METAL_SIMULATOR_HOME/ttrt-artifacts/system_desc.ttsys" \
  --path "$TT_METAL_SIMULATOR_HOME/pytest_artifacts_regression" \
  --tb=short -rs
```

Some files may take more than 30 minutes, but an individual node should not
take more than about three minutes. Monitor each live `-vv` log at least every
three minutes. If progress stops, terminate the file run, isolate the last
reported node with the 180-second candidate command, update the smallest
necessary skip, and rerun that file. A generous file timeout is not a
substitute for progress monitoring.

Record exactly one final status per file per architecture:

- `passed`: pytest exits 0 with a terminal summary; expected skips/xfails are
  allowed.
- `failed`: pytest reports normal failures.
- `aborted`: no pytest terminal summary, regardless of exit code.
- `timed_out`: the file timeout expires.

Any skip change made during regression requires a fresh run of that file on
both architectures. Do not finish with an untriaged failed, aborted, or timed
out file.

## Stage 7: Final Verification and Handoff

1. Run focused WH and BH pytest for every file whose skips changed.
2. Verify all workflow-listed files have a final WH and BH status.
3. Syntax-check `.github/workflows/call-test-ttsim.yml` with available local
   tooling.
4. Run `git diff --check` and inspect `git status` and the complete diff.
5. Confirm `conftest.py` is restored and generated simulator assets/results
   are not in the source diff.
6. Summarize:
   - release tag and verified asset names;
   - descriptor YAML and generated `.ttsys` used per architecture;
   - candidate counts and result totals per architecture;
   - simulator skip removals, additions, and narrowings;
   - per-file regression status for WH and BH;
   - focused reruns and workflow validation;
   - any explicit remaining coverage gap.
