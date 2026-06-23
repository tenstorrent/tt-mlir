---
name: uplift-ttsim-ci
description: Uplift the TTSim version used by tt-mlir CI and refresh WH/BH simulator skips. Use when updating `.github/workflows/call-test-ttsim.yml`, changing `ttsim-version`, validating WH or BH TTSim golden tests, or triaging simulator-specific pytest skips.
---

# Uplift TTSim CI

## Scope

Focus on `.github/workflows/call-test-ttsim.yml` and the WH/BH golden D2M tests listed there. Do not spend uplift time on Quasar/QSR unless the user asks.

## Release Update

1. Find the latest public TTSim release from `tenstorrent/ttsim` tags or releases.
2. Confirm the release has both `libttsim_wh.so` and `libttsim_bh.so`.
3. Update the workflow input default:

```yaml
ttsim-version:
  default: "vX.Y.Z"
```

4. Keep the workflow matrix focused on:

```yaml
- arch: wormhole_b0
  soc_desc: wormhole_b0_80_arch.yaml
  ttsim_asset: libttsim_wh.so
- arch: blackhole
  soc_desc: blackhole_140_arch.yaml
  ttsim_asset: libttsim_bh.so
```

5. Make these simulator env vars available to locally validate & refresh pytest ttsim skips:

```bash
TT_METAL_HOME="$PWD/third_party/tt-metal/src/tt-metal"
TT_METAL_SLOW_DISPATCH_MODE=1
TT_METAL_DISABLE_SFPLOADMACRO=1
TT_METAL_SIMULATOR=/path/to/libttsim_wh.so-or-libttsim_bh.so
```

## Local Setup

Use one folder per arch so WH/BH results can run independently:

```bash
mkdir -p ttsim/vX.Y.Z/wormhole_b0 ttsim/vX.Y.Z/blackhole
curl -L --fail -o ttsim/vX.Y.Z/wormhole_b0/libttsim_wh.so \
  https://github.com/tenstorrent/ttsim/releases/download/vX.Y.Z/libttsim_wh.so
curl -L --fail -o ttsim/vX.Y.Z/blackhole/libttsim_bh.so \
  https://github.com/tenstorrent/ttsim/releases/download/vX.Y.Z/libttsim_bh.so
cp third_party/tt-metal/src/tt-metal/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml \
  ttsim/vX.Y.Z/wormhole_b0/soc_descriptor.yaml
cp third_party/tt-metal/src/tt-metal/tt_metal/soc_descriptors/blackhole_140_arch.yaml \
  ttsim/vX.Y.Z/blackhole/soc_descriptor.yaml
```

Before every local command, avoid re-activation path corruption. Do not source `env/activate` under `set -u`; the activation script expects unset variables during initialization.

```bash
unset BUILD_DIR
source env/activate
export BUILD_DIR="$PWD/build"
```

Generate one system descriptor per arch:

```bash
export TT_METAL_HOME="$PWD/third_party/tt-metal/src/tt-metal"
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
export LD_LIBRARY_PATH="$PWD/build/lib:${TTMLIR_TOOLCHAIN_DIR}/lib:${LD_LIBRARY_PATH}"

export TT_METAL_SIMULATOR_HOME="$PWD/ttsim/vX.Y.Z/wormhole_b0"
export TT_METAL_SIMULATOR="$TT_METAL_SIMULATOR_HOME/libttsim_wh.so"
ttrt query --save-artifacts --artifact-dir "$TT_METAL_SIMULATOR_HOME/ttrt-artifacts" --quiet

export TT_METAL_SIMULATOR_HOME="$PWD/ttsim/vX.Y.Z/blackhole"
export TT_METAL_SIMULATOR="$TT_METAL_SIMULATOR_HOME/libttsim_bh.so"
ttrt query --save-artifacts --artifact-dir "$TT_METAL_SIMULATOR_HOME/ttrt-artifacts" --quiet
```

## Refresh Skips

1. Inspect simulator skips in the exact files passed by the workflow.
2. Triage with existing `sim` skips disabled, but do not leave triage-only edits in the final diff. A temporary edit to `test/python/golden/conftest.py` can make `_get_current_environment()` return `"silicon"` while `TT_METAL_SIMULATOR` remains set; restore it before final verification.
3. Prefer precise marks: skip a dtype, shape, op, or arch-specific case instead of an entire file/test.
4. A single config list is AND'd (subset match), while separate args/groups are OR'd: `skip_config(["n150", "sim"])` skips only WH-sim (single-chip WH `board_id` is `n150`, not `wh`), whereas `SkipIf("n150", "sim")` would skip on WH *or* sim — keep the inner brackets.
5. Keep non-simulator skips intact. For example, `SkipIf("ttnn", "emitc", "emitpy", "sim")` should usually become `SkipIf("ttnn", "emitc", "emitpy")` if the op now passes on WH/BH TTSim.
6. If a run aborts inside TTSim, isolate with a single pytest node or a narrow `-k` expression and keep/add the smallest simulator skip that avoids the abort.
7. When validating a skip refresh, collect the simulator-only node IDs and run them in isolated pytest subprocesses. This keeps one TTSim `UndefinedBehavior` abort from hiding the rest of the skip list. Exclude unconditional `pytest.mark.skip` cases from the executable checklist, but record them separately so they are not mistaken for validated simulator passes.

Run raw pytest through the venv Python for readable output; the local `pytest` console script may summarize through tools like `rtk` and hide details:

```bash
/opt/ttmlir-toolchain/venv/bin/python -m pytest -q \
  test/python/golden/d2m/test_unary.py \
  --sys-desc "$TT_METAL_SIMULATOR_HOME/ttrt-artifacts/system_desc.ttsys" \
  --path "$TT_METAL_SIMULATOR_HOME/pytest_artifacts" \
  --tb=short
```

Use the latest list of tests in the workflow file for full validation.

## Final Checks

- Restore all temporary triage edits, especially `conftest.py`.
- Run focused WH and BH pytest for every changed skip file.
- Run or at least syntax-check the workflow YAML if local CI tooling is available.
- Summarize the release tag, assets tested, descriptors used, skip changes, and any full-suite coverage gaps.
